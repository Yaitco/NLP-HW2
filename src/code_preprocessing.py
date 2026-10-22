"""Compact AST-normalized targets containing only the body of solve()."""

import ast


def normalize_code(code):
    """Remove comments/wrapper and retain the final value in result. No execution."""
    if code is None:  # Public test has no reference solutions.
        return None
    tree = ast.parse(code)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    ]
    if len(functions) != 1:
        raise ValueError("Expected exactly one module-level def solve()")
    function = functions[0]
    used_names = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for extra in tree.body:
        if extra is function:
            continue
        if not isinstance(extra, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise ValueError("Cannot discard module-level statements outside solve()")
        if extra.name in used_names:
            raise ValueError(f"solve() depends on module-level function {extra.name}")
        if used_names & {"globals", "locals", "vars", "eval", "exec"}:
            raise ValueError("Cannot discard functions when solve() uses dynamic names")
        # Even an unused definition can have effects through decorators/defaults.
        arguments = extra.args.posonlyargs + extra.args.args + extra.args.kwonlyargs
        arguments += [arg for arg in (extra.args.vararg, extra.args.kwarg) if arg]
        annotations = [extra.returns] + [arg.annotation for arg in arguments]
        if (
            extra.decorator_list
            or extra.args.defaults
            or any(value is not None for value in extra.args.kw_defaults)
            or any(
                isinstance(node, (ast.Call, ast.NamedExpr))
                for annotation in annotations
                if annotation is not None
                for node in ast.walk(annotation)
            )
        ):
            raise ValueError(f"Cannot discard executable definition of {extra.name}")
    tree.body = [function]
    args = function.args
    if (
        function.name != "solve"
        or function.decorator_list
        or args.posonlyargs
        or args.args
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
    ):
        raise ValueError("Expected an undecorated def solve() without arguments")

    body = function.body
    if isinstance(body[-1], ast.Return):
        value = body.pop().value or ast.Constant(value=None)
        if not (isinstance(value, ast.Name) and value.id == "result"):
            body.append(
                ast.Assign(
                    targets=[ast.Name(id="result", ctx=ast.Store())], value=value
                )
            )
    else:
        # Preserve implicit return None. Earlier returns inside branches stay intact.
        body.append(
            ast.Assign(
                targets=[ast.Name(id="result", ctx=ast.Store())],
                value=ast.Constant(value=None),
            )
        )
    body.append(ast.Return(value=ast.Name(id="result", ctx=ast.Load())))
    function.returns = None
    function.type_comment = None
    ast.fix_missing_locations(tree)

    # AST removes comments without touching '#' or whitespace inside string values.
    # Keep the body indented: plain dedent could change multiline string contents.
    formatted = ast.unparse(tree)
    lines = formatted.splitlines(keepends=True)
    if lines[0].rstrip() != "def solve():" or lines[-1].rstrip() != "    return result":
        raise ValueError("Unexpected formatted solve() wrapper")
    return "".join(lines[1:-1])


def restore_code(body):
    """Wrap an indented generated target for execution by the existing sandbox."""
    return "def solve():\n" + body.rstrip("\n") + "\n    return result\n"


def normalize_batch(batch, indices, code_column):
    normalized = []
    for index, code in zip(indices, batch[code_column]):
        try:
            normalized.append(normalize_code(code))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Cannot normalize row {index}: {exc}") from exc
    return {code_column: normalized, "original_code": batch[code_column]}
