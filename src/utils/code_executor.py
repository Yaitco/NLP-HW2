"""Execute untrusted solve() functions in disposable Linux namespaces.

This requires Bubblewrap and unprivileged user namespaces. There is no fallback
that executes code in the training process. Inputs/outputs use bounded JSON,
never pickle. The parent owns the wall-clock deadline.
"""

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    value: Fraction | None = None


_WORKER = r'''
import contextlib
import decimal
import fractions
import json
import math
import numbers
import random
import resource
import sys

sys.path.insert(0, '/opt/site-packages')
payload = json.load(sys.stdin)
for kind, value in (
    (resource.RLIMIT_AS, payload['memory_bytes']),
    (resource.RLIMIT_CPU, payload['cpu_seconds']),
    (resource.RLIMIT_FSIZE, 16384),
    (resource.RLIMIT_CORE, 0),
    (resource.RLIMIT_NPROC, 1),
    (resource.RLIMIT_NOFILE, 32),
):
    resource.setrlimit(kind, (value, value))
random.seed(0)

class Discard:
    def write(self, text):
        return len(text)
    def flush(self):
        pass

def numeric(value):
    if isinstance(value, bool):
        raise TypeError()
    if isinstance(value, numbers.Integral):
        return fractions.Fraction(int(value))
    if isinstance(value, fractions.Fraction):
        return value
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            raise ValueError()
        return fractions.Fraction(value)
    if type(value).__module__.startswith('sympy.'):
        if value.is_number is not True or value.is_real is not True:
            raise TypeError()
        if value.is_finite is not True:
            raise ValueError()
        if value.is_Rational:
            return fractions.Fraction(int(value.p), int(value.q))
        value = float(value.evalf(17))
    if not isinstance(value, numbers.Real):
        raise TypeError()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError()
    return fractions.Fraction.from_float(value)

status, result = 'runtime_error', None
try:
    try:
        program = compile(payload['code'], '<solution>', 'exec')
    except (SyntaxError, ValueError, TypeError):
        status = 'compile_error'
    else:
        namespace = {'__name__': '__solution__'}
        with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
            exec(program, namespace, namespace)
            solve = namespace.get('solve')
            if not callable(solve):
                status = 'missing_solve'
            else:
                value = solve()
                try:
                    result = numeric(value)
                    status = 'ok'
                except TypeError:
                    status = 'non_numeric'
                except (ValueError, OverflowError):
                    status = 'non_finite'
except BaseException:
    status = 'runtime_error'

output = {'status': status}
if status == 'ok':
    try:
        output.update(numerator=str(result.numerator), denominator=str(result.denominator))
    except (ValueError, OverflowError):
        output = {'status': 'output_limit'}
print(json.dumps(output))
'''


class SafeCodeExecutor:
    """A fresh, resource-limited sandbox per call, including reference calls."""

    max_code_bytes = 65536
    max_output_bytes = 16384
    max_integer_digits = 4096
    statuses = frozenset({
        "ok", "compile_error", "runtime_error", "missing_solve", "non_numeric",
        "non_finite", "output_limit",
    })

    def __init__(self, timeout=2.0, memory_mb=512):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(memory_mb, int) or memory_mb < 64:
            raise ValueError("memory_mb must be an integer >= 64")
        # RLIMIT_NPROC is exempt for the host root user. Drop to an
        # unprivileged host identity before starting Bubblewrap in that case.
        self.process_identity = {}
        if os.geteuid() == 0:
            self.process_identity = {"user": 65534, "group": 65534, "extra_groups": []}
        elif os.getuid() == 0:
            raise RuntimeError("Refusing code execution with a root real UID")
        self.timeout = timeout
        self.memory_bytes = memory_mb * 1024**2
        self.command = self._command()
        probe = self.run("def solve(): return 42")
        if probe != ExecutionResult("ok", Fraction(42)):
            raise RuntimeError(
                f"Code sandbox unavailable (probe: {probe.status}). Requires Linux, "
                "Bubblewrap with --disable-userns, and permitted user namespaces. "
                "No unsandboxed fallback is allowed."
            )

    @staticmethod
    def _command():
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError("Install Bubblewrap: apt install bubblewrap. No unsafe fallback.")
        command = [bwrap, "--unshare-all", "--unshare-user", "--die-with-parent", "--new-session",
                   "--cap-drop", "ALL", "--disable-userns", "--assert-userns-disabled"]
        for path in ("/usr", "/lib", "/lib64"):
            if Path(path).exists():
                command += ["--ro-bind", path, path]
        command += [
            "--proc", "/proc", "--remount-ro", "/proc", "--dev", "/dev",
            "--remount-ro", "/dev", "--dir", "/tmp", "--dir", "/work",
            "--chdir", "/work", "--dir", "/opt",
            "--ro-bind", sys.base_prefix, "/opt/python", "--dir", "/opt/site-packages",
        ]
        # Numerical dependencies only: no project modules, torch, or credentials.
        site = Path(sysconfig.get_paths()["purelib"])
        for name in ("numpy", "numpy.libs", "scipy", "scipy.libs", "sympy", "mpmath"):
            if (site / name).exists():
                command += ["--ro-bind", str(site / name), f"/opt/site-packages/{name}"]
        command += [
            "--remount-ro", "/", "--clearenv", "--setenv", "PATH", "/opt/python/bin:/usr/bin",
            "--setenv", "LD_LIBRARY_PATH", "/opt/python/lib",
            "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
            f"/opt/python/bin/python{sys.version_info.major}.{sys.version_info.minor}",
            "-I", "-S", "-c", _WORKER,
        ]
        return command

    def run(self, code):
        if not isinstance(code, str) or not code.strip():
            return ExecutionResult("missing_code")
        if len(code.encode("utf-8")) > self.max_code_bytes:
            return ExecutionResult("code_limit")
        payload = json.dumps({"code": code, "memory_bytes": self.memory_bytes,
                              "cpu_seconds": max(1, math.ceil(self.timeout))}).encode()
        # A regular file, bounded by RLIMIT_FSIZE, avoids unbounded PIPE capture.
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=output,
                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
                **self.process_identity,
            )
            try:
                process.communicate(payload, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                return ExecutionResult("timeout")
            finally:
                # A live/unreaped PID cannot be reused by another process.
                # Exiting the PID namespace also terminates its descendants.
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
                if process.stdin is not None:
                    process.stdin.close()
            if process.returncode != 0:
                return ExecutionResult("worker_failure")
            output.seek(0)
            data = output.read(self.max_output_bytes + 1)
        if len(data) > self.max_output_bytes:
            return ExecutionResult("output_limit")
        try:
            result = json.loads(data)
            if not isinstance(result, dict) or result.get("status") not in self.statuses:
                raise ValueError()
            status = result["status"]
            if status != "ok":
                return ExecutionResult(status)
            parts = [result["numerator"], result["denominator"]]
            if any(not isinstance(p, str) or len(p) > self.max_integer_digits
                   or not p.removeprefix("-").isascii()
                   or not p.removeprefix("-").isdigit() for p in parts):
                raise ValueError()
            numerator, denominator = map(int, parts)
            if denominator <= 0:
                raise ValueError()
            return ExecutionResult("ok", Fraction(numerator, denominator))
        except (ValueError, KeyError, TypeError, UnicodeError, RecursionError):
            return ExecutionResult("invalid_output")
