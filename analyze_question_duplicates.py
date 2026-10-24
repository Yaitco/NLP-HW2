"""Exact question duplicate counts with a bounded-memory external sort."""

import argparse
from collections import Counter
import heapq
import json
import os
from pathlib import Path
import subprocess
import tempfile

from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm

from src.utils.io_utils import ROOT_PATH


def summarize_counts(counts):
    histogram = Counter(counts)
    total = sum(frequency * groups for frequency, groups in histogram.items())
    unique = sum(histogram.values())
    repeated = unique - histogram.get(1, 0)
    repeated_rows = total - histogram.get(1, 0)
    return dict(rows=total, unique_questions=unique, repeated_question_groups=repeated,
                rows_in_repeated_groups=repeated_rows, extra_copies=total - unique,
                extra_copies_percent=100 * (total - unique) / total if total else 0,
                rows_with_duplicates_percent=100 * repeated_rows / total if total else 0,
                max_frequency=max(histogram, default=0),
                frequency_histogram=dict(sorted(histogram.items())))


def sorted_frequencies(lines):
    previous, count = None, 0
    for line in lines:
        if line == previous:
            count += 1
        else:
            if previous is not None:
                yield previous, count
            previous, count = line, 1
    if previous is not None:
        yield previous, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='report/question_duplicates')
    args = parser.parse_args()
    output = ROOT_PATH / args.output
    output.mkdir(parents=True, exist_ok=True)
    data_path = ROOT_PATH / 'artifacts/tokenized_noimports_full_clean_16k'
    source_path = ROOT_PATH / 'artifacts/code_noimports_full_clean'
    tokenized = load_from_disk(str(data_path / 'train'))
    source = load_from_disk(str(source_path / 'train')).select_columns(['question'])
    manifest = json.loads((ROOT_PATH / 'artifacts/length128_noimports_16k/train_indices.json').read_text())
    assert manifest['source_fingerprint'] == tokenized._fingerprint
    assert manifest['source_rows'] == len(tokenized) == len(source)
    indices = np.load(ROOT_PATH / 'artifacts/length128_noimports_16k/train_indices.npy', allow_pickle=False, mmap_mode='r')
    assert len(indices) == manifest['kept_rows']
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(ROOT_PATH / 'artifacts/tokenizer_noimports_16k.json'))
    sample = indices[np.linspace(0, len(indices) - 1, 256, dtype=int)].tolist()
    assert [e.ids for e in tokenizer.encode_batch(source[sample]['question'], add_special_tokens=False)] == tokenized[sample]['input_ids'], 'Question/tokenized row alignment differs'
    report = dict(definition='Exact cleaned question string, including case, whitespace and punctuation',
                  normalized_source=str(source_path), tokenized_fingerprint=tokenized._fingerprint)
    mini_path = ROOT_PATH / 'artifacts/retrieval_val64/train_indices.npy'
    if mini_path.exists():
        mini = np.load(mini_path, allow_pickle=False)
        positions = np.searchsorted(indices, mini)
        assert np.all(positions < len(indices)) and np.array_equal(indices[positions], mini)
        counts = Counter(source[mini.tolist()]['question'])
        report['retrieval_train'] = summarize_counts(counts.values())
        report['retrieval_top_duplicates'] = [dict(question=q, count=n) for q, n in counts.most_common(20) if n > 1]
        del counts
        print('Retrieval train:', json.dumps(report['retrieval_train']), flush=True)
    selected = source.select(indices)
    with tempfile.TemporaryDirectory(dir=output, prefix='sort_') as temporary:
        temporary = Path(temporary)
        unsorted, sorted_path = temporary / 'questions.jsonl', temporary / 'sorted.jsonl'
        with unsorted.open('w', encoding='utf-8') as file:
            for batch in tqdm(selected.iter(batch_size=50000), total=(len(selected) + 49999) // 50000, desc='Read cleaned train'):
                for question in batch['question']:
                    if not isinstance(question, str):
                        raise ValueError('Non-string question')
                    file.write(json.dumps(question, ensure_ascii=False) + '\n')
        print('External sort (512 MB, 2 CPU threads)', flush=True)
        subprocess.run(['sort', '-S', '512M', '--parallel=2', '-T', str(temporary),
                        '-o', str(sorted_path), str(unsorted)],
                       env={'PATH': os.defpath, 'LC_ALL': 'C'}, check=True)
        histogram, top = Counter(), []
        with sorted_path.open('rb') as file:
            for question, count in sorted_frequencies(file):
                histogram[count] += 1
                if count > 1:
                    heapq.heappush(top, (count, question))
                    if len(top) > 20:
                        heapq.heappop(top)
        # Summarize the histogram without allocating a list of millions of counts.
        report['full_train'] = summarize_counts(
            frequency for frequency, groups in histogram.items() for _ in range(groups)
        )
        assert report['full_train']['rows'] == len(indices)
        report['top_duplicates'] = [dict(question=json.loads(question), count=count)
                                    for count, question in sorted(top, reverse=True)]
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('Full train:', json.dumps(report['full_train']), flush=True)
    print('Report:', output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
