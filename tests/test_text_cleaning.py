import ast
import unittest

from src.code_preprocessing import normalize_batch, normalize_code, restore_code
from src.text_cleaning import clean_question, remove_imports, import_prelude
from src.utils.code_executor import SafeCodeExecutor


class TextCleaningTests(unittest.TestCase):
    def test_last_prefix_match(self):
        self.assertEqual(clean_question('Human: Problem: A train moves.'), 'A train moves.')
        self.assertEqual(clean_question('Student: A rate: per hour'), 'A rate: per hour')
        self.assertEqual(clean_question('Task: 12 apples'), '12 apples')
        self.assertEqual(clean_question('No prefix here'), 'No prefix here')
        self.assertEqual(clean_question('Task: A\nQuestion: How many?'), 'How many?')

    def test_imports_are_deleted_without_assignments_or_renaming(self):
        code = 'def solve():\n    import math\n    from math import sqrt\n    return math.ceil(sqrt(8))'
        cleaned = remove_imports(code)
        self.assertEqual(cleaned, 'def solve():\n    return math.ceil(sqrt(8))')
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign)) for n in ast.walk(ast.parse(cleaned))))

    def test_empty_block_keeps_valid_python_and_strings(self):
        code = 'def solve():\n    if True:\n        import math\n    result = "import math"\n    return 1'
        self.assertIn('pass', remove_imports(code))
        self.assertIn("'import math'", remove_imports(code))

    def test_reassigned_gcd_works_with_imports_inside_solve(self):
        executor = SafeCodeExecutor(sandbox=False, timeout=4)
        code = 'def solve():\n    from math import gcd\n    gcd = gcd(15, 9)\n    return gcd'
        body = normalize_code(remove_imports(code))
        self.assertNotIn('_lib_', body)
        self.assertNotIn('import', body)
        self.assertIn('gcd = gcd(15, 9)', body)
        restored = restore_code(body, add_imports=True)
        self.assertEqual(executor.run(restored), executor.run(code))
        self.assertTrue(all(isinstance(node, ast.FunctionDef) for node in ast.parse(restored).body))

    def test_all_splits_including_public_test(self):
        output = normalize_batch({'question':['Student: A question'], 'code':[None]}, [0], 'code', True, 'question')
        self.assertEqual(output['question'], ['A question'])
        self.assertEqual(output['code'], [None])

    def test_math_names_and_aliases_are_restored(self):
        executor = SafeCodeExecutor(sandbox=False, timeout=4)
        codes = [
            'def solve():\n    import math\n    from math import sqrt\n    return math.ceil(sqrt(8))',
            'def solve():\n    from math import comb as choose\n    return choose(5,2)',
            'def solve():\n    from math import factorial as fact\n    return fact(5)',
        ]
        for code in codes:
            with self.subTest(code=code):
                body = normalize_code(remove_imports(code))
                self.assertEqual(executor.run(code), executor.run(restore_code(body, add_imports=True)))
        for node in ast.parse(import_prelude()).body:
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name == 'math' for alias in node.names))
            else:
                self.assertIsInstance(node, ast.ImportFrom)
                self.assertEqual(node.module, 'math')
