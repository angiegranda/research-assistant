"""Opt-in checks against a prebuilt private papers index; never required by CI."""
import os
from pathlib import Path
import unittest

import evaluation
import library


@unittest.skipUnless(os.environ.get('PAPER_ASSISTANT_TEST_DB'),
                     'Set PAPER_ASSISTANT_TEST_DB to a freshly built real-papers index.')
class RealPapersTests(unittest.TestCase):
    def test_retrieval_and_source_provenance(self):
        db = Path(os.environ['PAPER_ASSISTANT_TEST_DB'])
        cases = Path(__file__).with_name('papers_cases.json')
        report = evaluation.evaluate(db, cases, k=5)
        # Regression floors, not a claim about unseen queries or model accuracy.
        self.assertGreaterEqual(report['hit_at_k'], .85)
        self.assertGreaterEqual(report['mrr_at_k'], .8)
        self.assertEqual(report['no_match_accuracy'], 1)
        for case in report['cases']:
            for hit in library.search(db, case['query'], 5):
                page = library.read_page(db, hit['name'], hit['page'], context=0)
                self.assertEqual(hit['sha'], page['sha'])
                self.assertIn(hit['passage'], page['pages'][0]['text'])
