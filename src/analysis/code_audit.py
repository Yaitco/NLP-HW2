"""Check reference solutions in a resource-limited Bubblewrap process."""

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from collections import Counter
from pathlib import Path

STATUS = {
    "numeric_finite": 0,
    "compile_error": 1,
    "runtime_error": 2,
    "non_numeric": 3,
    "missing_solve": 4,
    "timeout": 5,
    "numeric_nonfinite": 6,
    "numeric_complex": 7,
    "dependency_error": 8,
    "worker_failure": 9,
    "environment_error": 10,
    "missing_code": 11,
}

# No project imports or credentials enter this interpreter. A fresh namespace
# is used for every solution; the enclosing OS process is recycled per batch.
WORKER = r"""
import collections
import contextlib
import decimal
import fractions
import json
import math
import numbers
import resource
import signal
import sys

sys.path.insert(0, '/opt/site-packages')

payload = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

class Deadline(BaseException):
    pass

def deadline(signum, frame):
    raise Deadline()

class DiscardOutput:
    def write(self, text):
        return len(text)
    def flush(self):
        pass

signal.signal(signal.SIGALRM, deadline)
statuses, compiled_flags, examples = [], [], []
example_counts = collections.Counter()
error_types = collections.Counter()
result_types = collections.Counter()
output = DiscardOutput()

for offset, code in enumerate(payload['codes']):
    status, error, result_type = 0, None, None
    compiled_flag = 0  # 0: unknown/not attempted, 1: success, 2: failed
    phase = 'compile'
    signal.setitimer(signal.ITIMER_REAL, payload['timeout'])
    try:
        if not isinstance(code, str) or not code.strip():
            status = 11
        else:
            program = compile(code, '<reference>', 'exec')
            compiled_flag = 1
            phase = 'execute'
            namespace = {'__name__': '__reference__'}
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(program, namespace, namespace)
                solve = namespace.get('solve')
                if not callable(solve):
                    status = 4
                else:
                    value = solve()
                    result_type = type(value).__name__
                    if type(value).__module__.startswith('sympy.') and getattr(value, 'is_number', False):
                        if value.is_finite is False or value is sys.modules['sympy'].nan:
                            status = 6
                        elif value.is_real is False:
                            status = 7
                        else:
                            status = 0 if value.is_real is True and value.is_finite is True else 3
                    elif isinstance(value, bool) or not isinstance(value, numbers.Number):
                        status = 3
                    elif isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
                        status = 7
                    elif isinstance(value, decimal.Decimal):
                        status = 0 if value.is_finite() else 6
                    elif isinstance(value, (int, fractions.Fraction)):
                        status = 0
                    else:
                        status = 0 if math.isfinite(value) else 6
    except Deadline:
        status, error = 5, 'Per-example wall-clock timeout'
    except BaseException as exc:
        if phase == 'compile':
            status = 1
            compiled_flag = 2
        elif isinstance(exc, (ImportError, ModuleNotFoundError)):
            status = 8
        elif isinstance(exc, (PermissionError, FileNotFoundError, MemoryError)):
            status = 10
        else:
            status = 2
        error = type(exc).__name__ + ': ' + str(exc)[:240]
        error_types[str(status) + ':' + type(exc).__name__] += 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    statuses.append(status)
    compiled_flags.append(compiled_flag)
    if result_type is not None:
        result_types[str(status) + ':' + result_type] += 1
    if status and example_counts[status] < payload['examples_per_status']:
        examples.append({'offset': offset, 'status': status,
                         'error': error, 'result_type': result_type})
        example_counts[status] += 1

print('AUDIT_RESULT:' + json.dumps({'statuses': statuses, 'compiled': compiled_flags, 'examples': examples,
                                   'error_types': dict(error_types), 'result_types': dict(result_types)}))
"""


def sandbox_command():
    """Mount only Python/system libraries; hide home, project, network and env."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "Code execution requires Bubblewrap (Linux: apt install bubblewrap). "
            "There is deliberately no unsandboxed fallback."
        )
    command = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for path in ("/usr", "/lib", "/lib64"):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/work",
            "--chdir",
            "/work",
            "--dir",
            "/opt",
            "--ro-bind",
            sys.base_prefix,
            "/opt/python",
            "--ro-bind",
            sysconfig.get_paths()["purelib"],
            "/opt/site-packages",
            "--clearenv",
            "--setenv",
            "PATH",
            "/opt/python/bin:/usr/bin",
            "--setenv",
            "LD_LIBRARY_PATH",
            "/opt/python/lib",
            "--setenv",
            "OPENBLAS_NUM_THREADS",
            "1",
            "--setenv",
            "OMP_NUM_THREADS",
            "1",
            f"/opt/python/bin/python{sys.version_info.major}.{sys.version_info.minor}",
            "-I",
            "-S",
            "-c",
            WORKER,
        ]
    )
    return command


def check_codes(codes, timeout=0.25, examples_per_status=3, batch_timeout=45):
    """Return one status per row; bisect a failed worker to locate crash cases."""
    command = sandbox_command()

    def run(items):
        payload = json.dumps(
            {
                "codes": items,
                "timeout": timeout,
                "examples_per_status": examples_per_status,
            }
        )
        try:
            result = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=batch_timeout,
                check=True,
            )
            line = next(
                line
                for line in reversed(result.stdout.splitlines())
                if line.startswith("AUDIT_RESULT:")
            )
            report = json.loads(line.removeprefix("AUDIT_RESULT:"))
            if len(report["statuses"]) != len(items) or len(report["compiled"]) != len(
                items
            ):
                raise ValueError("Incomplete worker response")
            return report
        except (subprocess.SubprocessError, ValueError, StopIteration) as exc:
            if len(items) == 1:
                return {
                    "statuses": [STATUS["worker_failure"]],
                    "compiled": [0],
                    "examples": [
                        {
                            "offset": 0,
                            "status": STATUS["worker_failure"],
                            "error": type(exc).__name__,
                            "result_type": None,
                        }
                    ],
                    "error_types": {"9:" + type(exc).__name__: 1},
                    "result_types": {},
                }
            middle = len(items) // 2
            left, right = run(items[:middle]), run(items[middle:])
            for example in right["examples"]:
                example["offset"] += middle
            return {
                "statuses": left["statuses"] + right["statuses"],
                "compiled": left["compiled"] + right["compiled"],
                "examples": left["examples"] + right["examples"],
                "error_types": dict(
                    Counter(left["error_types"]) + Counter(right["error_types"])
                ),
                "result_types": dict(
                    Counter(left["result_types"]) + Counter(right["result_types"])
                ),
            }

    if not codes:
        return {
            "statuses": [],
            "compiled": [],
            "examples": [],
            "error_types": {},
            "result_types": {},
        }
    return run(list(codes))


def check_sandbox():
    """Fail before a full scan if namespaces/Python mounts are unavailable."""
    command = sandbox_command()
    probe = subprocess.run(
        command,
        input=json.dumps(
            {"codes": ["def solve(): return 2"], "timeout": 1, "examples_per_status": 1}
        ),
        text=True,
        capture_output=True,
        timeout=10,
    )
    if probe.returncode or '"statuses": [0]' not in probe.stdout:
        raise RuntimeError("Sandbox probe failed: " + probe.stderr[-1000:])
    return {
        "python": sys.version.split()[0],
        "backend": "bubblewrap",
        "network": "isolated",
        "memory_per_worker_mib": 512,
    }
