import ast
import unittest

from src.code_preprocessing import restore_code
from src.utils.code_postprocessing import restore_generated_code


class CodePostprocessingTests(unittest.TestCase):
    def test_final_assignment_overrides_earlier_result(self):
        code = restore_generated_code('    result = 1\n    smth = 2 + 3\n')
        self.assertTrue(code.endswith('    return smth\n'))
        namespace = {}
        exec(code, namespace)
        self.assertEqual(namespace['solve'](), 5)

    def test_multiline_assignment_comments_and_blank_lines(self):
        body = '    answer = (\n        10 / 2\n    ) # answer\n\n    # trailing comment\n'
        self.assertTrue(restore_generated_code(body).endswith('    return answer\n'))

    def test_annotation_augmented_assignment_and_math_imports(self):
        for body in ('    answer: float = sqrt(9)\n', '    answer = 2\n    answer += 1\n'):
            code = restore_generated_code(body, add_imports=True)
            namespace = {}
            exec(code, namespace)
            self.assertEqual(namespace['solve'](), 3)

    def test_result_and_non_assignment_fallback(self):
        for body in ('    result = 3\n', '    result = 3\n    print(result)\n',
                     '    if True:\n        answer = 3\n', '    a, b = 2, 3\n'):
            self.assertEqual(restore_generated_code(body), restore_code(body))

    def test_invalid_generation_is_preserved(self):
        body = '    answer = (\n'
        self.assertEqual(restore_generated_code(body), restore_code(body))

    def test_explicit_return_and_multiline_strings_are_preserved(self):
        body = '    answer = """a\n b\n"""\n    return 7\n'
        self.assertEqual(restore_generated_code(body), restore_code(body))
        ast.parse(restore_generated_code(body))
