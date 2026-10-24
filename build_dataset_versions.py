"""Build paired cleaned datasets; publication is a separate --push command."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
from itertools import groupby
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from time import perf_counter

import datasets
from datasets.arrow_writer import ArrowWriter
from huggingface_hub import DatasetCard, HfApi
import numpy as np
from tqdm.auto import tqdm
import yaml

from src.code_preprocessing import restore_code
from src.dataset_versions import prune_code, rename_code, winning_codes, UnsupportedCode
from src.text_cleaning import clean_question, import_prelude
from src.utils.code_executor import SafeCodeExecutor, ExecutionResult
from src.utils.io_utils import ROOT_PATH


FEATURES = datasets.Features({
    'question': datasets.Value('string'), 'code': datasets.Value('string'),
    'original_question': datasets.Value('string'), 'original_code': datasets.Value('string'),
    'raw_index': datasets.Value('int64'),
})


def path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT_PATH / value


def digest(value):
    return hashlib.sha256(value.encode()).digest()


class Answers:
    def __init__(self, work, seed, settings):
        self.db = sqlite3.connect(work / 'answers.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS answers '
                        '(key BLOB PRIMARY KEY, code TEXT, status TEXT, numerator TEXT, denominator TEXT) WITHOUT ROWID')
        self.seed = (sqlite3.connect(seed.resolve().as_uri() + '?mode=ro', uri=True)
                     if seed is not None and seed.exists() else None)
        self.executor = SafeCodeExecutor(timeout=settings['timeout'], memory_mb=settings['memory_mb'],
                                         sandbox=settings['sandbox'])
        self.pool = ThreadPoolExecutor(max_workers=settings['workers'])
        self.batch_size = settings['batch_size']

    def get(self, codes):
        codes = list(dict.fromkeys(codes))
        keys = {digest(code): code for code in codes}
        result, copied = {}, []
        for db in (self.db, self.seed):
            if db is None:
                continue
            missing = [key for key, code in keys.items() if code not in result]
            for start in range(0, len(missing), 500):
                chunk = missing[start:start+500]
                marks = ','.join('?' for _ in chunk)
                for row in db.execute(f'SELECT key,code,status,numerator,denominator FROM answers WHERE key IN ({marks})', chunk):
                    key, code, status, numerator, denominator = row
                    if code != keys[key]:
                        raise ValueError('Code hash collision')
                    value = Fraction(int(numerator), int(denominator)) if status == 'ok' else None
                    result[code] = ExecutionResult(status, value)
                    if db is self.seed:
                        copied.append(row)
        missing = [code for code in codes if code not in result]
        runnable = [restore_code(code, add_imports=True) for code in missing]
        chunks = [runnable[i:i+self.batch_size] for i in range(0, len(runnable), self.batch_size)]
        executed = [value for batch in self.pool.map(self.executor.run_batch, chunks) for value in batch]
        for code, value in zip(missing, executed):
            result[code] = value
            copied.append((digest(code), code, value.status,
                           str(value.value.numerator) if value.status == 'ok' else None,
                           str(value.value.denominator) if value.status == 'ok' else None))
        self.db.executemany('INSERT OR IGNORE INTO answers VALUES (?,?,?,?,?)', copied)
        self.db.commit()
        return result

    def close(self):
        self.pool.shutdown(wait=True)
        self.db.close()
        if self.seed is not None:
            self.seed.close()


def question_groups(raw_train, normalized, work, batch_size):
    """External sort: restored raw questions, never the old truncated questions."""
    groups = work / 'groups.jsonl'
    if not groups.exists():
        unsorted, ordered = work / 'questions.tmp', work / 'sorted.tmp'
        with unsorted.open('w', encoding='utf-8') as out:
            batches = zip(raw_train.select_columns(['question', 'code']).iter(batch_size=batch_size),
                          normalized.select_columns(['original_code']).iter(batch_size=batch_size))
            offset = 0
            for raw, saved in tqdm(batches, total=(len(normalized)+batch_size-1)//batch_size,
                                   desc='Restore and clean raw questions'):
                if raw['code'] != saved['original_code']:
                    raise ValueError(f'Raw/normalized code alignment mismatch at {offset}')
                out.writelines(json.dumps(clean_question(q), ensure_ascii=False) + '\t' + str(offset+i) + '\n'
                               for i, q in enumerate(raw['question']))
                offset += len(raw['question'])
            assert offset == len(normalized)
        subprocess.run(['sort', '-S', '512M', '--parallel=2', '-T', str(work),
                        '-t', '\t', '-k1,1', '-k2,2n', '-o', str(ordered), str(unsorted)],
                       env={**os.environ, 'LC_ALL': 'C'}, check=True)
        temporary = work / 'groups.tmp'
        with ordered.open() as source, temporary.open('w') as out:
            records = ((q, int(i)) for q, i in (line.split('\t', 1) for line in source))
            for question, rows in groupby(records, key=lambda row: row[0]):
                out.write(question + '\t' + json.dumps([row[1] for row in rows]) + '\n')
        temporary.replace(groups)
        unsorted.unlink()
        ordered.unlink()
    with groups.open() as source:
        batch, count = [], 0
        for line in source:
            question, ids = line.split('\t', 1)
            ids = json.loads(ids)
            batch.append((json.loads(question), ids))
            count += len(ids)
            if count >= batch_size:
                yield batch
                batch, count = [], 0
        if batch:
            yield batch


def transform_code(code):
    """CPU-only transformation; safe to distribute to separate processes."""
    clean = prune_code(code)
    try:
        renamed = rename_code(clean)
    except UnsupportedCode:
        renamed = None
    return clean, renamed


def prepare_variants(codes, answers, stats, original=None, transform_pool=None):
    """A transformation is accepted only if the exact numeric value is preserved."""
    codes = list(dict.fromkeys(codes))
    original = answers.get(codes) if original is None else original
    transformed = (map(transform_code, codes) if transform_pool is None else
                   transform_pool.map(transform_code, codes, chunksize=64))
    clean, renamed, skipped = {}, {}, set()
    for code, (value, canonical) in zip(codes, transformed):
        clean[code] = value
        if canonical is None:
            skipped.add(code)
            stats['unsupported_scope_programs'] += 1
        else:
            renamed[code] = canonical
    # Run both transformations together instead of two sequential process waves.
    check_codes = []
    for code in codes:
        if original[code].status != 'ok':
            continue
        if clean[code] != code:
            check_codes.append(clean[code])
        if code in renamed:
            check_codes.append(renamed[code])
    checked = answers.get(check_codes)
    retry = []
    for code, value in list(clean.items()):
        if value != code:
            before = original[code]
            if before.status != 'ok' or checked[value] != before:
                clean[code] = code
                stats['pruning_fallback_programs'] += 1
                # Recompute canonical code from the original if pruning failed.
                try:
                    renamed[code] = rename_code(code)
                    skipped.discard(code)
                    if before.status == 'ok':
                        retry.append(renamed[code])
                except UnsupportedCode:
                    skipped.add(code)
                    renamed.pop(code, None)
    if retry:
        checked.update(answers.get(retry))
    for code in list(renamed):
        before = original[code]
        if before.status == 'ok' and checked[renamed[code]] != before:
            skipped.add(code)
            stats['renaming_mismatch_programs'] += 1
        if before.status != 'ok':
            skipped.add(code)
    return clean, renamed, skipped


def write_rows(writers, rows):
    for writer, records in zip(writers, rows):
        if records:
            writer.write_batch({key: [row[key] for row in records] for key in FEATURES})


def semantic_signature(signature):
    """Performance settings may change while reusing sorted questions/answers."""
    signature = json.loads(json.dumps(signature))
    # Transformed outputs are regenerated on every run. The work cache stores
    # sorted raw questions and numeric answers keyed by exact code, not ASTs.
    signature.pop('transforms', None)
    config = signature['config']
    config.pop('ast_workers', None)
    config.pop('batch_size', None)
    config['execution'].pop('workers', None)
    config['execution'].pop('batch_size', None)
    return signature


def build(config):
    output, work = path(config['output']), path(config['work'])
    if output.exists():
        raise FileExistsError(f'{output} already exists; use --push or choose another output')
    work.mkdir(parents=True, exist_ok=True)
    source = datasets.load_from_disk(str(path(config['source'])))
    preparation = json.loads((path(config['source'])/'preparation.json').read_text())
    cleaning = preparation['cleaning']
    audit = json.loads((path(cleaning['audit_dir'])/'manifest.json').read_text())
    if not audit.get('complete'):
        raise ValueError('The original numeric audit must be complete')
    raw = datasets.load_dataset(preparation['source_dataset']['path'],
        download_config=datasets.DownloadConfig(local_files_only=config['local_files_only']))
    if raw['train']._fingerprint != preparation['raw_fingerprint'] or len(raw['train']) != preparation['raw_rows']:
        raise ValueError('Raw dataset differs from normalized dataset provenance')
    if audit['signature']['fingerprint'] != raw['train']._fingerprint:
        raise ValueError('Numeric audit fingerprint differs from raw data')
    keep = np.load(path(cleaning['audit_dir'])/'status.npy', allow_pickle=False) == 0
    checked = json.loads(path(cleaning['rechecked_manifest']).read_text())
    if checked['raw_fingerprint'] != raw['train']._fingerprint:
        raise ValueError('Recovered rows belong to a different dataset')
    keep[checked['recovered_rows']] = True
    start, size = cleaning['validation_start'], cleaning['validation_size']
    keep[start:start+size] = False
    raw_indices = np.flatnonzero(keep)
    if len(raw_indices) != len(source['train']):
        raise ValueError('Raw/normalized train length mismatch')
    raw_train = raw['train'].select(raw_indices)
    execution_hash = hashlib.sha256((
        (ROOT_PATH/'src/utils/code_executor.py').read_text()
        + inspect.getsource(restore_code) + inspect.getsource(import_prelude)
    ).encode()).hexdigest()
    signature = dict(source={s: d._fingerprint for s, d in source.items()}, config=config,
                     cleaner=inspect.getsource(clean_question), execution_hash=execution_hash,
                     transforms=hashlib.sha256((ROOT_PATH/'src/dataset_versions.py').read_bytes()).hexdigest())
    manifest = work/'signature.json'
    if manifest.exists() and semantic_signature(json.loads(manifest.read_text())) != semantic_signature(signature):
        raise ValueError('Configuration/source changed; use a fresh work directory')
    manifest.write_text(json.dumps(signature, indent=2)+'\n')
    seed = path(config['answer_cache']) if config['answer_cache'] else None
    if seed is not None and seed.exists():
        old = json.loads((seed.parent/'source.json').read_text())
        if old['source_fingerprint'] != source['train']._fingerprint:
            raise ValueError('Answer cache belongs to another normalized dataset')
        executor_files = b''.join((ROOT_PATH/f).read_bytes() for f in
                                 ('src/utils/code_executor.py', 'src/code_preprocessing.py'))
        cleaner = (ROOT_PATH/'src/text_cleaning.py').read_text()
        # The previous audit predates only the question-regex fix. That change
        # does not affect code execution; accept both exact source signatures.
        before_regex_fix = cleaner.replace("r'(?<=\\S): (?=[^a-z])'", "r': (?=[^a-z])'")
        accepted = {hashlib.sha256(executor_files+text.encode()).hexdigest()
                    for text in (cleaner, before_regex_fix)}
        if old['execution_sha256'] not in accepted:
            raise ValueError('Seed executor differs; set answer_cache: null to execute afresh')
    answers = Answers(work, seed, config['execution'])
    ast_workers = config.get('ast_workers', 0)
    transform_pool = (ProcessPoolExecutor(max_workers=ast_workers,
                      mp_context=multiprocessing.get_context('spawn')) if ast_workers > 0 else None)
    stats = Counter(input_train_rows=len(source['train']))
    writers = {}
    for split in source:
        writers[split] = [ArrowWriter(path=str(work/f'{version}_{split}.arrow'), features=FEATURES)
                          for version in ('v1', 'v2')]
    try:
        progress = tqdm(total=len(source['train']), desc='Vote, prune AST, verify, rename')
        for batch in question_groups(raw_train, source['train'], work, config['batch_size']):
            tick = perf_counter()
            ids = [i for _, group in batch for i in group]
            saved = source['train'].select_columns(['code', 'original_code'])[ids]
            raw_questions = raw_train.select_columns(['question'])[ids]['question']
            positions = {row: i for i, row in enumerate(ids)}
            stats['seconds_read'] += perf_counter() - tick
            tick = perf_counter()
            original = answers.get(saved['code'])
            stats['seconds_original_answers'] += perf_counter() - tick
            tick = perf_counter()
            chosen, offset = [], 0
            for question, group in batch:
                variants = {}
                for i, code in zip(group, saved['code'][offset:offset+len(group)]):
                    variants.setdefault(code, []).append(i)
                offset += len(group)
                stats['question_groups'] += 1
                winner = winning_codes(variants, original, **config['voting'])
                if not winner:
                    reason = 'ties' if any(original[c].status == 'ok' for c in variants) else 'no_numeric_answer'
                    stats[reason+'_groups'] += 1
                    stats[reason+'_rows'] += len(group)
                    continue
                stats['losing_or_invalid_rows'] += sum(len(v) for c, v in variants.items() if c not in winner)
                stats['exact_pair_copies_removed'] += sum(len(variants[c])-1 for c in winner)
                chosen.extend((question, variants[code][0], code) for code in winner)
            stats['seconds_voting'] += perf_counter() - tick
            tick = perf_counter()
            clean, renamed, skipped = prepare_variants([code for _, _, code in chosen], answers, stats,
                                                       original=original, transform_pool=transform_pool)
            stats['seconds_transform_and_verify'] += perf_counter() - tick
            tick = perf_counter()
            rows = [[], []]
            seen = set()
            for question, row_id, code in chosen:
                if code in skipped:
                    stats['unsupported_or_failed_transform_rows'] += 1
                    continue
                key = (question, clean[code])
                if key in seen:
                    stats['pairs_collapsed_by_ast_cleanup'] += 1
                    continue
                seen.add(key)
                j = positions[row_id]
                row = dict(question=question, original_question=raw_questions[j],
                           original_code=saved['original_code'][j], raw_index=int(raw_indices[row_id]))
                rows[0].append(dict(row, code=clean[code]))
                rows[1].append(dict(row, code=renamed[code]))
            write_rows(writers['train'], rows)
            stats['seconds_write'] += perf_counter() - tick
            stats['output_train_rows'] += len(rows[0])
            progress.update(len(ids))
            (work/'progress.json').write_text(json.dumps(stats, indent=2)+'\n')
        progress.close()
        for split in ('validation', 'test'):
            original_split = raw['train'].select(range(start, start+size)) if split == 'validation' else raw['test']
            if len(original_split) != len(source[split]):
                raise ValueError(f'{split} length mismatch')
            for offset in tqdm(range(0, len(source[split]), config['batch_size']), desc=split):
                saved = source[split][offset:offset+config['batch_size']]
                originals = original_split[offset:offset+config['batch_size']]
                codes = [code for code in saved['code'] if code is not None]
                clean, renamed, skipped = prepare_variants(codes, answers, stats, transform_pool=transform_pool)
                rows = [[], []]
                for j, code in enumerate(saved['code']):
                    if originals['code'][j] != saved['original_code'][j]:
                        raise ValueError(f'{split} reference alignment mismatch')
                    row = dict(question=clean_question(originals['question'][j]),
                               original_question=originals['question'][j], original_code=originals['code'][j],
                               raw_index=start+offset+j if split == 'validation' else offset+j)
                    # Keep all validation/test questions. Unsupported/invalid
                    # validation programs retain their original target.
                    for version, mapping in enumerate((clean, renamed)):
                        value = None if code is None else code if code in skipped else mapping[code]
                        rows[version].append(dict(row, code=value))
                    stats[split+'_unchanged_targets'] += int(code in skipped)
                write_rows(writers[split], rows)
    finally:
        if transform_pool is not None:
            transform_pool.shutdown(wait=True, cancel_futures=True)
        for pair in writers.values():
            for writer in pair:
                writer.finalize()
        answers.close()
    ready = work/'ready'
    if ready.exists():
        shutil.rmtree(ready)
    ready.mkdir()
    for version in ('v1', 'v2'):
        data = datasets.DatasetDict({split: datasets.Dataset.from_file(str(work/f'{version}_{split}.arrow'))
                                     for split in source})
        data.save_to_disk(str(ready/version), max_shard_size='500MB')
        card = ('---\nlanguage: en\n---\n'
                f'# TinySMS cleaned {version}\n\n'
                'Source: SlavaYus/tinysms_qa_12m. License: see upstream dataset.\n\n'
                'Train: numeric-reference selection, repaired question prefixes, answer-frequency voting; '
                'ties dropped, winning programs deduplicated, conservative AST pruning. '
                'Numerical equivalence of modified programs is checked by isolated execution.\n\n'
                'Code format: indented body of solve(), final value in result. '
                'Restore using this repository’s restore_code(body, add_imports=True). '
                'Only the math prelude is provided.\n\n'
                + ('Train locals renamed to v0, v1, ...; result unchanged.\n\n' if version == 'v2' else '')
                + 'Both versions share selected raw rows and split boundaries. Validation/test rows are preserved; '
                'unsupported or invalid validation programs retain their original code. Public test code is null. '
                'No tokenizer training or token-length filtering was applied.\n')
        (ready/version/'README.md').write_text(card)
    (ready/'summary.json').write_text(json.dumps(dict(stats=stats, provenance=signature), indent=2)+'\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    ready.rename(output)
    print(json.dumps(stats, indent=2))
    print(f'Saved: {output}/v1 and {output}/v2')


def publish(config):
    root = path(config['output'])
    summary = json.loads((root/'summary.json').read_text())
    api = HfApi()
    for version, repo in config['hub'].items():
        data = datasets.load_from_disk(str(root/version))
        data.push_to_hub(repo, private=config['private'], max_shard_size='500MB')
        # Retain dataset_info/configs generated by datasets.push_to_hub.
        card = DatasetCard.load(repo, repo_type='dataset')
        card.text = DatasetCard((root/version/'README.md').read_text()).text
        card.data.language = ['en']
        api.upload_file(path_or_fileobj=str(card).encode(), path_in_repo='README.md',
                        repo_id=repo, repo_type='dataset')
        api.upload_file(path_or_fileobj=json.dumps(summary, indent=2).encode(), path_in_repo='cleaning_summary.json',
                        repo_id=repo, repo_type='dataset')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/configs/dataset_versions.yaml')
    parser.add_argument('--push', action='store_true', help='Publish already prepared datasets; do not rebuild')
    args = parser.parse_args()
    settings = yaml.safe_load(path(args.config).read_text())
    publish(settings) if args.push else build(settings)
