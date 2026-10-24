import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import datasets
import numpy as np

from build_dataset_versions import build, prepare_variants, semantic_signature
from src.code_preprocessing import normalize_code, restore_code
from src.dataset_versions import prune_code, rename_code, winning_codes, UnsupportedCode
from src.utils.code_executor import SafeCodeExecutor, ExecutionResult
from fractions import Fraction


class DatasetVersionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = SafeCodeExecutor(sandbox=False)

    def check_equivalent(self, before, after):
        self.assertEqual(self.executor.run(restore_code(before, add_imports=True)),
                         self.executor.run(restore_code(after, add_imports=True)))

    def test_dead_assignments_print_and_constant_branches(self):
        code = '    unused = 17\n    x = 6\n    if False:\n        x = 999\n    else:\n        y = 2\n    print(x)\n    result = x / y\n'
        cleaned = prune_code(code)
        self.assertNotIn('unused', cleaned)
        self.assertNotIn('999', cleaned)
        self.assertNotIn('print', cleaned)
        self.check_equivalent(code, cleaned)

    def test_if_loop_and_early_return_dependencies(self):
        for code in [
            '    x = 3\n    if x > 2:\n        y = x + 1\n    else:\n        y = 9\n    result = y\n',
            '    x = 0\n    for i in range(4):\n        x += i\n    result = x\n',
            '    x = 3\n    if x > 2:\n        return x\n    result = 99\n',
            '    result = 5\n    return result\n    result = 99\n',
        ]:
            self.check_equivalent(code, prune_code(code))

    def test_renaming_collisions_math_bindings_and_result(self):
        for code in [
            '    v0 = 2\n    apples = v0 + 3\n    result = apples\n',
            '    gcd = gcd(15, 9)\n    result = gcd\n',
            '    a = math.sqrt(4)\n    math = 3\n    result = a + math\n',
            '    total = 0\n    for i in range(4):\n        total += i\n    result = total\n',
        ]:
            renamed = rename_code(code)
            self.assertIn('result', renamed)
            self.check_equivalent(code, renamed)
        self.assertNotIn('apples', rename_code('    apples = 3\n    result = apples\n'))

    def test_complex_scopes_are_not_silently_renamed(self):
        for code in ('    result = sum(x for x in range(4))\n',
                     '    result = len(locals())\n',
                     '    def helper(x):\n        return x\n    result = helper(4)\n'):
            with self.assertRaises(UnsupportedCode):
                rename_code(code)

    def test_row_votes_ties_errors_and_numeric_tolerance(self):
        answers = {'a': ExecutionResult('ok', Fraction(4)),
                   'b': ExecutionResult('ok', Fraction('4.0000001')),
                   'c': ExecutionResult('ok', Fraction(8)),
                   'bad': ExecutionResult('runtime_error')}
        self.assertEqual(winning_codes({'a': [0, 1], 'c': [2], 'bad': [3, 4, 5]}, answers), ['a'])
        self.assertEqual(winning_codes({'a': [0], 'c': [1]}, answers), [])
        self.assertEqual(set(winning_codes({'a': [0], 'b': [1], 'c': [2]}, answers)), {'a', 'b'})
        self.assertEqual(winning_codes({'bad': [0]}, answers), [])

    def test_failed_pruning_rebuilds_renaming_from_original(self):
        from collections import Counter
        original = '    apples = 4\n    result = apples\n'
        wrong = '    apples = 5\n    result = apples\n'
        executor = self.executor
        class Answers:
            def get(self, codes):
                return dict(zip(codes, executor.run_batch([
                    restore_code(code, add_imports=True) for code in codes])))
        before = {original: ExecutionResult('ok', Fraction(4))}
        with patch('build_dataset_versions.transform_code', return_value=(wrong, rename_code(wrong))):
            clean, renamed, skipped = prepare_variants([original], Answers(), Counter(), original=before)
        self.assertEqual(clean[original], original)
        self.assertEqual(renamed[original], rename_code(original))
        self.assertEqual(skipped, set())

    def test_only_performance_and_recomputed_transforms_can_change_on_restart(self):
        import copy
        old = dict(source='same', transforms='old', execution_hash='same',
                   config=dict(batch_size=8192, execution=dict(workers=8, batch_size=512, timeout=2)))
        new = copy.deepcopy(old)
        new['transforms'] = 'new'
        new['config']['ast_workers'] = 4
        new['config']['execution']['workers'] = 12
        self.assertEqual(semantic_signature(old), semantic_signature(new))
        new['config']['execution']['timeout'] = 3
        self.assertNotEqual(semantic_signature(old), semantic_signature(new))

    def test_small_pipeline_repairs_raw_questions_and_keeps_splits_paired(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def code(value):
                return f'def solve():\n    unused = 99\n    apples = {value}\n    return apples'
            questions = ['Student: A ratio is 3 : 2?']*3 + ['Student: Tie?']*2 + ['Student: Single?', 'Student: Validation?']
            raw_train = datasets.Dataset.from_dict(dict(question=questions, code=[code(v) for v in (6, 6, 8, 1, 2, 4, 9)]))
            raw_test = datasets.Dataset.from_dict(dict(question=['Student: Test ratio 4 : 3?'], code=[None]))
            def normalized(raw):
                return datasets.Dataset.from_dict(dict(question=['WRONG OLD QUESTION']*len(raw),
                    code=[normalize_code(c) if c else None for c in raw['code']], original_code=raw['code']))
            source = datasets.DatasetDict(train=normalized(raw_train.select(range(6))),
                                          validation=normalized(raw_train.select([6])), test=normalized(raw_test))
            source.save_to_disk(str(root/'source'))
            audit_dir = root/'audit'
            audit_dir.mkdir()
            np.save(audit_dir/'status.npy', np.zeros(7, dtype=np.int8))
            (audit_dir/'manifest.json').write_text(json.dumps(dict(complete=True, signature=dict(fingerprint=raw_train._fingerprint))))
            recovered = root/'recovered.json'
            recovered.write_text(json.dumps(dict(raw_fingerprint=raw_train._fingerprint, recovered_rows=[])))
            preparation = dict(raw_fingerprint=raw_train._fingerprint, raw_rows=7,
                source_dataset=dict(path='fixture'), cleaning=dict(audit_dir=str(audit_dir),
                rechecked_manifest=str(recovered), validation_start=6, validation_size=1))
            (root/'source'/'preparation.json').write_text(json.dumps(preparation))
            config = dict(source=str(root/'source'), output=str(root/'output'), work=str(root/'work'),
                answer_cache=str(root/'no-cache.sqlite'), local_files_only=True, batch_size=4, ast_workers=2,
                voting=dict(rtol=1e-5, atol=1e-6),
                execution=dict(timeout=2, memory_mb=512, sandbox=False, workers=2, batch_size=64),
                hub=dict(v1='unused/v1', v2='unused/v2'), private=False)
            with patch('build_dataset_versions.datasets.load_dataset', return_value=dict(train=raw_train, test=raw_test)):
                build(config)
            first = datasets.load_from_disk(str(root/'output'/'v1'))
            second = datasets.load_from_disk(str(root/'output'/'v2'))
            self.assertEqual(first['train']['question'], ['A ratio is 3 : 2?', 'Single?'])
            self.assertEqual(first['train']['raw_index'], second['train']['raw_index'])
            self.assertEqual(first['train']['code'], ['    apples = 6\n    result = apples\n', '    apples = 4\n    result = apples\n'])
            self.assertTrue(all('v0' in c and 'apples' not in c for c in second['train']['code']))
            self.assertEqual(len(first['validation']), 1)
            self.assertEqual(first['test']['question'], ['Test ratio 4 : 3?'])
            self.assertEqual(second['test']['code'], [None])
            summary = json.loads((root/'output'/'summary.json').read_text())['stats']
            self.assertEqual(summary['ties_rows'], 2)
            self.assertEqual(summary['exact_pair_copies_removed'], 1)
            self.assertEqual(summary['losing_or_invalid_rows'], 1)
