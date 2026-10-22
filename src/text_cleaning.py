"""Question prefix cleanup and import-free code targets; no execution."""
import ast
import re

def import_prelude():
    """Only math, with the bare names and aliases observed in the dataset."""
    return (
        "import math\n"
        "from math import acos, atan, ceil, comb, comb as choose, cos, degrees, erf, exp, factorial, factorial as fact, floor, gcd, hypot, isclose, log, log2, perm, pi, pow, radians, sin, sqrt, tan\n"
    )


def clean_question(question):
    matches = list(re.finditer(r': (?=[^a-z])', question))
    return question[matches[-1].end():] if matches else question


def remove_imports(code):
    """Delete import statements without changing names or adding assignments."""
    tree = ast.parse(code)
    if not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        return code

    class Strip(ast.NodeTransformer):
        def visit_Import(self, node):
            return None

        def visit_ImportFrom(self, node):
            return None

        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, ast.stmt) and hasattr(node, 'body') and not node.body:
                node.body = [ast.Pass()]
            return node

    return ast.unparse(ast.fix_missing_locations(Strip().visit(tree)))
