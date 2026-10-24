"""Audit exact question/code copies and conflicting numeric labels, without filtering data.

Saved row IDs address the full normalized/tokenized train before length filtering,
the same index space as artifacts/length128_noimports_16k/train_indices.npy.
"""

import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm

from src.code_preprocessing import restore_code
from src.utils.beam_voting import same_result
from src.utils.code_executor import SafeCodeExecutor, ExecutionResult
from src.utils.io_utils import ROOT_PATH


OUTPUT = ROOT_PATH / 'report/duplicate_code_audit'
SOURCE = ROOT_PATH / 'artifacts/code_noimports_full_clean/train'
INDICES = ROOT_PATH / 'artifacts/length128_noimports_16k/train_indices.npy'
BATCH_SIZE, WORKERS = 512, 8


def repeated_groups(source, indices):
    """Disk sort question + original row ID, then keep only repeated questions."""
    destination = OUTPUT / 'repeated_groups.jsonl'
    if destination.exists():
        return destination
    questions = source.select_columns(['question']).select(indices)
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix='sort_') as temporary:
        temporary = Path(temporary)
        unsorted, sorted_path = temporary / 'questions', temporary / 'sorted'
        offset = 0
        with unsorted.open('w', encoding='utf-8') as file:
            for batch in tqdm(questions.iter(batch_size=50000), total=(len(indices)+49999)//50000, desc='Read questions'):
                rows = batch['question']
                file.writelines(json.dumps(question, ensure_ascii=False) + '\t' + str(int(row_id)) + '\n'
                                for question, row_id in zip(rows, indices[offset:offset+len(rows)]))
                offset += len(rows)
        print('Sort questions: 512 MB, 2 threads', flush=True)
        subprocess.run(['sort', '-S', '512M', '--parallel=2', '-T', str(temporary),
                        '-t', '\t', '-k1,1', '-k2,2n', '-o', str(sorted_path), str(unsorted)],
                       env={'PATH': os.defpath, 'LC_ALL': 'C'}, check=True)
        temporary_groups = OUTPUT / 'repeated_groups.tmp'
        with sorted_path.open('rb') as file, temporary_groups.open('wb') as out:
            records = ((key, int(row)) for key, row in (line.split(b'\t', 1) for line in file))
            for question, group in groupby(records, key=lambda row: row[0]):
                ids = [row[1] for row in group]
                if len(ids) > 1:
                    out.write(question + b'\t' + json.dumps(ids).encode() + b'\n')
        temporary_groups.replace(destination)
    return destination


def group_batches(path, max_rows=8192):
    batch, size = [], 0
    with path.open(encoding='utf-8') as file:
        for line in file:
            question, ids = line.split('\t', 1)
            ids = json.loads(ids)
            batch.append((json.loads(question), ids))
            size += len(ids)
            if size >= max_rows:
                yield batch
                batch, size = [], 0
    if batch:
        yield batch


def fresh_stats(rows):
    return dict(rows=rows, repeated_question_groups=0, unique_question_code_pairs=rows,
                repeated_pair_groups=0, rows_in_repeated_pairs=0, extra_exact_pair_copies=0,
                questions_with_multiple_codes=0, multi_code_groups_with_execution_errors=0,
                multi_code_groups_with_no_valid_answer=0, exact_numeric_conflict_groups=0,
                numeric_conflict_groups=0, rows_in_numeric_conflict_groups=0)


def analyze_group(variants, answers, stats):
    """variants maps exact code -> original row IDs; only multi-code groups execute."""
    rows = sum(map(len, variants.values()))
    if rows < 2:
        return False
    stats['repeated_question_groups'] += 1
    frequencies = [len(ids) for ids in variants.values()]
    extra = rows - len(variants)
    stats['extra_exact_pair_copies'] += extra
    stats['unique_question_code_pairs'] -= extra
    stats['repeated_pair_groups'] += sum(n > 1 for n in frequencies)
    stats['rows_in_repeated_pairs'] += sum(n for n in frequencies if n > 1)
    if len(variants) == 1:
        return False
    stats['questions_with_multiple_codes'] += 1
    results = [answers[code] for code in variants]
    valid = [result.value for result in results if result.status == 'ok']
    stats['multi_code_groups_with_execution_errors'] += any(result.status != 'ok' for result in results)
    stats['multi_code_groups_with_no_valid_answer'] += not valid
    stats['exact_numeric_conflict_groups'] += len(set(valid)) > 1
    conflict = len(valid) > 1 and not same_result(min(valid), max(valid))
    if conflict:
        stats['numeric_conflict_groups'] += 1
        stats['rows_in_numeric_conflict_groups'] += rows
    return conflict


def cached_answers(database, codes, executor, pool):
    keys = {hashlib.sha256(code.encode()).digest(): code for code in codes}
    answers = {}
    all_keys = list(keys)
    for start in range(0, len(all_keys), 500):
        chunk = all_keys[start:start+500]
        marks = ','.join('?' for _ in chunk)
        for key, code, status, numerator, denominator in database.execute(
            f'SELECT key,code,status,numerator,denominator FROM answers WHERE key IN ({marks})', chunk
        ):
            if code != keys[key]:
                raise ValueError('Code hash collision')
            value = Fraction(int(numerator), int(denominator)) if status == 'ok' else None
            answers[code] = ExecutionResult(status, value)
    missing = [code for code in codes if code not in answers]
    runnable = [restore_code(code, add_imports=True) for code in missing]
    chunks = [runnable[i:i+BATCH_SIZE] for i in range(0, len(runnable), BATCH_SIZE)]
    results = [result for batch in pool.map(executor.run_batch, chunks) for result in batch]
    assert len(results) == len(missing)
    answers.update(zip(missing, results))
    database.executemany('INSERT INTO answers VALUES (?,?,?,?,?)', [
        (hashlib.sha256(code.encode()).digest(), code, result.status,
         str(result.value.numerator) if result.status == 'ok' else None,
         str(result.value.denominator) if result.status == 'ok' else None)
        for code, result in zip(missing, results)
    ])
    database.commit()
    return answers, len(missing)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_from_disk(str(SOURCE))
    tokenized = load_from_disk(str(ROOT_PATH / 'artifacts/tokenized_noimports_full_clean_16k/train'))
    manifest = json.loads(INDICES.with_suffix('.json').read_text())
    assert manifest['source_fingerprint'] == tokenized._fingerprint
    assert len(source) == len(tokenized) == manifest['source_rows']
    indices = np.load(INDICES, allow_pickle=False, mmap_mode='r')
    assert len(indices) == manifest['kept_rows']
    signature = dict(source_fingerprint=source._fingerprint, tokenized_fingerprint=tokenized._fingerprint,
                     indices_sha256=hashlib.sha256(INDICES.read_bytes()).hexdigest(),
                     execution_sha256=hashlib.sha256(b''.join((ROOT_PATH / p).read_bytes() for p in [
                         'src/utils/code_executor.py', 'src/code_preprocessing.py', 'src/text_cleaning.py'])).hexdigest())
    metadata = OUTPUT / 'source.json'
    if metadata.exists() and json.loads(metadata.read_text()) != signature:
        raise ValueError('Audit source/execution changed; use a new report directory')
    metadata.write_text(json.dumps(signature, indent=2)+'\n')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT_PATH / 'artifacts/tokenizer_noimports_16k.json'))
    sample = indices[np.linspace(0, len(indices)-1, 256, dtype=int)].tolist()
    encoded = [entry.ids for entry in tokenizer.encode_batch(source[sample]['code'], add_special_tokens=False)]
    labels = tokenized[sample]['labels']
    assert all(label == code or label == code+[tokenizer.token_to_id('<eos>')] for label, code in zip(labels, encoded))
    groups_file = repeated_groups(source, indices)
    mini_path = ROOT_PATH / 'artifacts/retrieval_val64/train_indices.npy'
    mini = set(map(int, np.load(mini_path, allow_pickle=False))) if mini_path.exists() else set()
    positions = np.searchsorted(indices, sorted(mini))
    assert np.all(positions < len(indices)) and np.array_equal(indices[positions], sorted(mini))
    stats = dict(full_train=fresh_stats(len(indices)), retrieval_train=fresh_stats(len(mini)))
    database = sqlite3.connect(OUTPUT / 'answers.sqlite')
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('PRAGMA cache_size=-65536')
    database.execute('CREATE TABLE IF NOT EXISTS answers (key BLOB PRIMARY KEY, code TEXT, status TEXT, numerator TEXT, denominator TEXT) WITHOUT ROWID')
    executor = SafeCodeExecutor(sandbox=True)
    executed = 0
    conflict_ids, extra_pair_ids = [], []
    summary = dict(complete=False, definition='Exact cleaned question and current normalized body code',
                   index_space='Row IDs in full normalized/tokenized train before length filtering; same as parent train_indices.npy',
                   comparison=dict(rtol=1e-5, atol=1e-6), execution='Current restore_code(add_imports=True), Bubblewrap, timeout=2s',
                   note='Single-code groups cannot have conflicting deterministic answers and are not executed', **stats)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, (OUTPUT/'conflicts.jsonl').open('w', encoding='utf-8') as conflicts, tqdm(total=796628, desc='Audit repeated questions', mininterval=10) as progress:
        for batch in group_batches(groups_file):
            flat_ids = [row for _, ids in batch for row in ids]
            code_rows = source.select_columns(['code'])[flat_ids]['code']
            prepared, needed = [], {}
            offset = 0
            for question, ids in batch:
                variants = {}
                for code, row_id in zip(code_rows[offset:offset+len(ids)], ids):
                    variants.setdefault(code, []).append(row_id)
                offset += len(ids)
                if len(variants) > 1:
                    needed.update(dict.fromkeys(variants))
                prepared.append((question, variants))
            answers, count = cached_answers(database, list(needed), executor, pool)
            executed += count
            for question, variants in prepared:
                for ids in variants.values():
                    extra_pair_ids.extend(ids[1:])
                conflict = analyze_group(variants, answers, stats['full_train'])
                mini_variants = {code: [row for row in ids if row in mini] for code, ids in variants.items()}
                mini_variants = {code: ids for code, ids in mini_variants.items() if ids}
                analyze_group(mini_variants, answers, stats['retrieval_train'])
                if conflict:
                    conflict_ids.extend(row for ids in variants.values() for row in ids)
                    conflicts.write(json.dumps(dict(question=question, variants=[
                        dict(code=code, train_indices=ids, count=len(ids), status=answers[code].status,
                             answer=str(answers[code].value) if answers[code].status=='ok' else None)
                        for code, ids in variants.items()
                    ]), ensure_ascii=False)+'\n')
            progress.update(len(batch))
            summary.update(processed_question_groups=progress.n, new_unique_programs_executed=executed)
            (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    summary['complete'] = True
    summary['cached_unique_programs'] = database.execute('SELECT COUNT(*) FROM answers').fetchone()[0]
    summary['program_statuses'] = dict(database.execute('SELECT status,COUNT(*) FROM answers GROUP BY status'))
    database.close()
    np.save(OUTPUT/'conflicting_train_indices.npy', np.sort(np.asarray(conflict_ids, dtype=np.int64)))
    np.save(OUTPUT/'extra_pair_indices.npy', np.sort(np.asarray(extra_pair_ids, dtype=np.int64)))
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
