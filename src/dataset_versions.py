"""Conservative dead-code removal, local variable renaming and answer voting."""

import ast
from collections import Counter
from functools import lru_cache

from src.code_preprocessing import normalize_code, restore_code
from src.text_cleaning import import_prelude
from src.utils.beam_voting import same_result


class UnsupportedCode(ValueError):
    pass


@lru_cache(maxsize=1)
def imported_names():
    return frozenset(alias.asname or alias.name
                     for node in ast.parse(import_prelude()).body for alias in node.names)


def loads(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def pure(node):
    """No calls, mutation, introspection or attribute/subscript access."""
    return all(isinstance(n, (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
                              ast.Compare, ast.BoolOp, ast.IfExp, ast.operator,
                              ast.unaryop, ast.cmpop, ast.boolop)) for n in ast.walk(node))


def trim_block(statements, needed):
    """Backward liveness for straight-line code and if; loops remain conservative."""
    reachable = []
    for node in statements:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            branch = node.body if bool(node.test.value) else node.orelse
            reachable.extend(trim_unreachable(branch))
        else:
            reachable.append(node)
        if reachable and isinstance(reachable[-1], (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
    output, needed = [], set(needed)
    for node in reversed(reachable):
        if isinstance(node, ast.Pass):
            continue
        if isinstance(node, ast.Return):
            needed = loads(node)
        elif isinstance(node, ast.Expr):
            # Training targets only encode the returned numeric value.
            print_call = (isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Name)
                          and node.value.func.id == 'print'
                          and not any(isinstance(n, (ast.NamedExpr, ast.Yield, ast.YieldFrom))
                                      for n in ast.walk(node)))
            if pure(node.value) or print_call:
                continue
            needed |= loads(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            names = {t.id for t in node.targets}
            if not names & needed and pure(node.value):
                continue
            needed = (needed - names) | loads(node.value)
        elif isinstance(node, ast.If):
            node.body, left = trim_block(node.body, needed)
            node.orelse, right = trim_block(node.orelse, needed)
            if not node.body and not node.orelse and pure(node.test):
                continue
            node.body = node.body or [ast.Pass()]
            needed = left | right | loads(node.test)
        else:
            # Do not infer loop iteration counts, aliasing or exception semantics.
            # Preserve all potential inputs/outputs at these boundaries.
            needed |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        output.append(node)
    return list(reversed(output)), needed


def trim_unreachable(statements):
    out = []
    for node in statements:
        out.append(node)
        if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
            break
    return out


def prune_code(body):
    function = ast.parse(restore_code(body)).body[0]
    # Introspection can observe supposedly dead bindings or removed output.
    dynamic = {'eval', 'exec', 'globals', 'locals', 'vars', 'dir'}
    if any(isinstance(n, ast.Name) and n.id in dynamic for n in ast.walk(function)):
        return body
    function.body, _ = trim_block(function.body, set())
    ast.fix_missing_locations(function)
    return normalize_code(ast.unparse(function))


def rename_code(body):
    """Rename locals of solve; reject constructs needing additional scope analysis."""
    tree = ast.parse(restore_code(body))
    function = tree.body[0]
    nodes = list(ast.walk(function))
    unsupported = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda,
                   ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                   ast.Global, ast.Nonlocal, ast.Import, ast.ImportFrom,
                   ast.Match)
    if any(n is not function and isinstance(n, unsupported) for n in nodes):
        raise UnsupportedCode('nested scopes, imports, match or global/nonlocal')
    if any(isinstance(n, ast.Name) and n.id in {'eval', 'exec', 'globals', 'locals', 'vars', 'dir'}
           for n in nodes):
        raise UnsupportedCode('dynamic name lookup')
    names = []
    # Source order, including exception-handler bindings stored as strings in AST.
    bindings = [n for n in nodes
                if (isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)))
                or (isinstance(n, ast.ExceptHandler) and n.name)]
    for node in sorted(bindings, key=lambda n: (n.lineno, n.col_offset)):
        name = node.id if isinstance(node, ast.Name) else node.name
        if name != 'result' and name not in names:
            names.append(name)
    mapping = {name: f'v{i}' for i, name in enumerate(names)}

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_ExceptHandler(self, node):
            node.name = mapping.get(node.name, node.name)
            return self.generic_visit(node)

    # This AST was freshly parsed for this call and is never shared.
    function = Rename().visit(function)
    # The runtime imports math names inside solve. Preserve those initial bindings
    # if a program later reassigns e.g. gcd = gcd(15, 9).
    imported = imported_names()
    initial = [ast.Assign(targets=[ast.Name(id=mapping[name], ctx=ast.Store())],
                          value=ast.Name(id=name, ctx=ast.Load()))
               for name in names if name in imported]
    function.body = initial + function.body
    ast.fix_missing_locations(function)
    return normalize_code(ast.unparse(function))


def winning_codes(variants, answers, rtol=1e-5, atol=1e-6):
    """Rows vote before deduplication. Return all winning codes, or [] on a tie."""
    valid = sorted((code for code in variants if answers[code].status == 'ok'),
                   key=lambda code: (answers[code].value, code))
    groups = []
    for code in valid:
        if not groups or not same_result(answers[code].value, answers[groups[-1][0]].value,
                                         rtol=rtol, atol=atol):
            groups.append([])
        groups[-1].append(code)
    if not groups:
        return []
    weights = [sum(len(variants[code]) for code in group) for group in groups]
    maximum = max(weights)
    if Counter(weights)[maximum] > 1:
        return []
    return groups[weights.index(maximum)]
