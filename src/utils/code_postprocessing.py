"""Restore generated solve bodies with a return inferred from their final assignment."""

import ast

from src.code_preprocessing import restore_code


def restore_generated_code(body, add_imports=False):
    code = restore_code(body, add_imports=add_imports)
    try:
        statements = ast.parse(code).body[0].body
    except (SyntaxError, ValueError, RecursionError):
        # Invalid generations must reach the executor, which reports the error.
        return code
    last = statements[-2] if len(statements) > 1 else None
    if isinstance(last, ast.Assign) and len(last.targets) == 1:
        target = last.targets[0]
    elif isinstance(last, (ast.AnnAssign, ast.AugAssign)) and last.value is not None:
        target = last.target
    else:
        return code
    if isinstance(target, ast.Name):
        return code.removesuffix('    return result\n') + f'    return {target.id}\n'
    return code
