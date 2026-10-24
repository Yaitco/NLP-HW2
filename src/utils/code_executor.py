"""Execute solve() in a fresh process, using Linux namespaces by default.

Explicit sandbox=False is available for notebook experiments without Bubblewrap.
It retains resource limits but does not isolate files or networking. Inputs and
outputs use bounded JSON; the parent owns the wall-clock deadline.
"""

import json
import math
import os
import shutil
import signal
import site
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


_WORKER_SETUP = r'''
import contextlib
import decimal
import fractions
import json
import math
import numbers
import random
import resource
import sys

payload = json.load(sys.stdin)
sys.path[:0] = payload['site_packages']
for kind, value in (
    (resource.RLIMIT_AS, payload['memory_bytes']),
    (resource.RLIMIT_CPU, payload['cpu_seconds']),
    (resource.RLIMIT_FSIZE, payload.get('max_output_bytes', 16384)),
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
'''

_WORKER = _WORKER_SETUP + r'''
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


_BATCH_WORKER = _WORKER_SETUP + r'''
import ast
import signal
import types

class CandidateTimeout(BaseException):
    pass

def expired(*args):
    raise CandidateTimeout()

allowed_calls = set('print abs round int float bool pow min max sum len range list tuple set dict sorted all any divmod enumerate zip reversed'.split())
math_names = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
forbidden = (ast.Import, ast.ImportFrom, ast.Attribute, ast.Global, ast.Nonlocal,
             ast.ClassDef, ast.AsyncFunctionDef, ast.Try, ast.TryStar, ast.With,
             ast.AsyncWith, ast.Raise)
trees, outputs, environments = [], [None] * len(payload['codes']), {}
for index, code in enumerate(payload['codes']):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        outputs[index] = {'status': 'compile_error'}
        continue
    except BaseException:
        outputs[index] = {'status': 'use_single'}
        continue
    function = tree.body[0] if len(tree.body) == 1 else None
    bindings, math_aliases, math_calls = {}, set(), set()
    assignments = []
    # Only leading math imports are supported in the batch path. Preserve their
    # local bindings (e.g. gcd = gcd(...)) without importing for each candidate.
    if isinstance(function, ast.FunctionDef):
        while function.body and isinstance(function.body[0], (ast.Import, ast.ImportFrom)):
            node = function.body[0]
            if isinstance(node, ast.Import) and all(a.name == 'math' for a in node.names):
                imported = [(a.asname or 'math', math) for a in node.names]
                math_aliases.update(name for name, _ in imported)
            elif (isinstance(node, ast.ImportFrom) and node.module == 'math'
                  and not node.level and all(a.name in math_names for a in node.names)):
                imported = [(a.asname or a.name, math_names[a.name]) for a in node.names]
                math_calls.update(name for name, value in imported if callable(value))
            else:
                break
            function.body.pop(0)
            for name, value in imported:
                internal = f'_batch_import_{len(bindings)}'
                bindings[internal] = value
                assignments.append(ast.Assign(
                    targets=[ast.Name(id=name, ctx=ast.Store())],
                    value=ast.Name(id=internal, ctx=ast.Load())))
    nodes = list(ast.walk(tree))
    def math_attribute(node):
        return (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name) and node.value.id in math_aliases
                and node.attr in math_names)
    if (not isinstance(function, ast.FunctionDef) or function.name != 'solve'
        or function.decorator_list or function.returns or getattr(function, 'type_params', [])
        or function.args.args or function.args.posonlyargs or function.args.kwonlyargs
        or function.args.vararg or function.args.kwarg
        or sum(isinstance(node, ast.FunctionDef) for node in nodes) != 1
        or any(isinstance(node, forbidden) and not math_attribute(node) for node in nodes)
        or any(isinstance(node, ast.Name) and node.id.startswith(('__', '_batch_import_')) for node in nodes)
        or any(isinstance(target, ast.Name) and target.id.startswith(('__', '_batch_import_'))
               for assignment in assignments for target in assignment.targets)
        or any(isinstance(node, ast.Call) and
               not ((isinstance(node.func, ast.Name) and node.func.id in allowed_calls | math_calls)
                    or (math_attribute(node.func) and callable(math_names[node.func.attr])))
               for node in nodes)):
        outputs[index] = {'status': 'use_single'}
        continue
    function.name = f'_candidate_{index}'
    function.body = assignments + function.body
    environments[index] = bindings
    trees.append(function)

namespace = {'__name__': '__solution__'}
try:
    module = ast.fix_missing_locations(ast.Module(body=trees, type_ignores=[]))
    exec(compile(module, '<candidates>', 'exec'), namespace, namespace)
except BaseException:
    outputs = [output or {'status': 'use_single'} for output in outputs]

for index, output in enumerate(outputs):
    if output is None:
        # Each function gets its own globals, including its original solve name.
        function = types.FunctionType(namespace[f'_candidate_{index}'].__code__,
                                      {'__name__': '__solution__', **environments[index]}, 'solve')
        function.__globals__['solve'] = function
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, payload['timeout'])
        try:
            with contextlib.redirect_stdout(Discard()), contextlib.redirect_stderr(Discard()):
                value = function()
                try:
                    result = numeric(value)
                    output = {'status': 'ok'}
                except TypeError:
                    output = {'status': 'non_numeric'}
                except (ValueError, OverflowError):
                    output = {'status': 'non_finite'}
                if output['status'] == 'ok':
                    try:
                        output.update(numerator=str(result.numerator), denominator=str(result.denominator))
                    except (ValueError, OverflowError):
                        output = {'status': 'output_limit'}
        except CandidateTimeout:
            output = {'status': 'timeout'}
        except BaseException:
            output = {'status': 'runtime_error'}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            value = result = None
    print(json.dumps(output), flush=True)
'''


class SafeCodeExecutor:
    """Fresh, resource-limited processes; filesystem/network sandbox by default."""

    max_code_bytes = 65536
    max_output_bytes = 16384
    max_integer_digits = 4096
    statuses = frozenset({
        "ok", "compile_error", "runtime_error", "missing_solve", "non_numeric",
        "non_finite", "output_limit", "timeout",
    })

    def __init__(self, timeout=2.0, memory_mb=512, sandbox=True):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(memory_mb, int) or memory_mb < 64:
            raise ValueError("memory_mb must be an integer >= 64")
        # RLIMIT_NPROC is exempt for the host root user. Drop to an
        # unprivileged host identity before starting Bubblewrap in that case.
        self.process_identity = {}
        if sandbox and os.geteuid() == 0:
            self.process_identity = {"user": 65534, "group": 65534, "extra_groups": []}
        elif sandbox and os.getuid() == 0:
            raise RuntimeError("Refusing code execution with a root real UID")
        self.timeout = timeout
        self.memory_bytes = memory_mb * 1024**2
        self.sandbox = sandbox
        self.command = (
            self._command() if sandbox else [sys.executable, "-I", "-S", "-c", _WORKER]
        )
        self.site_packages = (
            ["/opt/site-packages"]
            if sandbox
            else list(
                dict.fromkeys(
                    [sysconfig.get_paths()["purelib"], *site.getsitepackages()]
                )
            )
        )
        self.environment = (
            None
            if sandbox
            else {
                "PATH": os.defpath,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        probe = self.run("def solve(): return 42")
        if probe != ExecutionResult("ok", Fraction(42)):
            if not sandbox:
                raise RuntimeError(f"Code worker unavailable (probe: {probe.status})")
            raise RuntimeError(
                f"Code sandbox unavailable (probe: {probe.status}). Requires Linux, "
                "Bubblewrap with --disable-userns, and permitted user namespaces. "
                "No automatic unsandboxed fallback is allowed."
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
        payload = {"code": code, "site_packages": self.site_packages, "memory_bytes": self.memory_bytes,
                   "cpu_seconds": max(1, math.ceil(self.timeout))}
        data, error = self._run_process(payload, self.command, self.timeout, self.max_output_bytes)
        return ExecutionResult(error) if error else self._decode_result(data)

    def run_batch(self, codes):
        """One process for arithmetic/math imports; complex code uses run()."""
        if len(codes) <= 1:
            return [self.run(code) for code in codes]
        results, valid = [None] * len(codes), []
        for index, code in enumerate(codes):
            if not isinstance(code, str) or not code.strip():
                results[index] = ExecutionResult("missing_code")
            elif len(code.encode("utf-8")) > self.max_code_bytes:
                results[index] = ExecutionResult("code_limit")
            else:
                valid.append(index)
        if not valid:
            return results
        output_limit = self.max_output_bytes * len(valid)
        payload = {
            "codes": [codes[index] for index in valid],
            "site_packages": self.site_packages,
            "memory_bytes": self.memory_bytes,
            "cpu_seconds": max(1, math.ceil(self.timeout * len(valid))) + 1,
            "timeout": self.timeout,
            "max_output_bytes": output_limit,
        }
        # Keep the same Bubblewrap isolation, replacing only its Python program.
        command = self.command[:-1] + [_BATCH_WORKER]
        data, _ = self._run_process(payload, command,
                                    self.timeout * len(valid) + 1, output_limit)
        lines = data.splitlines()
        for position, index in enumerate(valid):
            line = lines[position] if position < len(lines) else b''
            if line == b'{"status": "use_single"}' or not line:
                results[index] = self.run(codes[index])
            else:
                results[index] = self._decode_result(line)
                if results[index].status == "invalid_output":
                    results[index] = self.run(codes[index])
        return results

    def _run_process(self, payload, command, timeout, output_limit):
        error = None
        # A regular file, bounded by RLIMIT_FSIZE, avoids unbounded PIPE capture.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryDirectory() as workdir:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=output,
                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
                env=self.environment, cwd=None if self.sandbox else workdir,
                **self.process_identity,
            )
            try:
                process.communicate(json.dumps(payload).encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                error = "timeout"
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
            if process.returncode != 0 and error is None:
                error = "worker_failure"
            output.seek(0)
            data = output.read(output_limit + 1)
        return data, error

    def _decode_result(self, data):
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
