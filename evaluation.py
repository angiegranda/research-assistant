"""Transparent page-level retrieval evaluation; no model judges or source-text exports."""
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import library


def evaluate(db, cases_path, k=5, hybrid=False):
    suite = json.loads(Path(cases_path).read_text(encoding='utf-8'))
    cases = suite.get('cases') if isinstance(suite, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError('Evaluation needs a nonempty cases list.')
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get('id'), str)
                or not isinstance(case.get('query'), str) or not case['query'].strip()
                or not isinstance(case.get('relevant'), list)):
            raise ValueError('Each evaluation case needs id, query, and a relevant list.')
        for source in case['relevant']:
            if (not isinstance(source, dict) or not isinstance(source.get('file'), str)
                    or type(source.get('page')) is not int or source['page'] < 1):
                raise ValueError('Each relevant source needs a filename and positive PDF page.')
    if not isinstance(suite.get('source_sha256', {}), dict):
        raise ValueError('source_sha256 must map filenames to checksums.')
    inventory = library.status(db, verify=True)
    versions = {s['name']: s['sha'] for s in inventory['files']}
    if inventory['new_files'] or any(s['state'] != 'unchanged' for s in inventory['files']):
        raise ValueError('Evaluation needs a current index. Rebuild after library changes.')
    for name, checksum in suite.get('source_sha256', {}).items():
        if versions.get(name) != checksum:
            raise ValueError(f'Evaluation source missing or different: {name}. Review labels before evaluating.')
    rows, reciprocal, negatives, elapsed = [], [], [], []
    for case in cases:
        expected = {(s['file'], s['page']) for s in case['relevant']}
        if any(name not in versions for name, _ in expected):
            raise ValueError(f"Missing labeled source for {case['id']}. Index the papers library first.")
        started = perf_counter()
        hits = library.search(db, case['query'], k, hybrid=hybrid, file=case.get('file'))
        elapsed.append((perf_counter()-started)*1000)
        rank = next((i for i,h in enumerate(hits, 1) if (h['name'], h['page']) in expected), None)
        if expected:
            reciprocal.append(1/rank if rank else 0)
        else:
            negatives.append(not hits)
        rows.append({'id': case['id'], 'query': case['query'], 'first_relevant_rank': rank,
                     'no_match_expected': not expected, 'returned': [dict(file=h['name'], page=h['page']) for h in hits]})
    return {'suite': suite.get('name', Path(cases_path).name), 'k': k,
            'mode': 'hybrid' if hybrid else 'lexical', 'positive_cases': len(reciprocal),
            'hit_at_k': mean(bool(r) for r in reciprocal) if reciprocal else None,
            'mrr_at_k': mean(reciprocal) if reciprocal else None,
            'negative_cases': len(negatives), 'no_match_accuracy': mean(negatives) if negatives else None,
            'mean_query_ms': round(mean(elapsed), 2), 'indexed_files': inventory['counts']['indexed_files'],
            'index_built_at': inventory['built_at'], 'source_sha256': versions, 'cases': rows,
            'limitations': 'Small curated development set; page hits do not establish excerpt relevance, completeness, or scientific truth. Negative cases only test lexical no-match behavior.'}
