"""Online discovery contracts tested offline; no credentials or private PDFs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import literature
import paper
import presentation


ITEM = {'DOI': '10.1234/example', 'title': ['<i>Interpretable</i> recommendations'],
        'author': [{'given':'Ada', 'family':'Lovelace'}, {'name':'Research Group'}],
        'published': {'date-parts': [[2024, 3]]}, 'container-title': ['A Journal'],
        'type': 'journal-article'}


class LiteratureTests(unittest.TestCase):
    def discover(self, topics, **kwargs):
        with redirect_stderr(io.StringIO()):
            return literature.discover(topics, **kwargs)

    def response(self, payload):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        return response

    def test_metadata_and_doi_link(self):
        result = literature.normalize(ITEM)
        self.assertEqual(result['title'], 'Interpretable recommendations')
        self.assertEqual(result['authors'], ['Ada Lovelace', 'Research Group'])
        self.assertEqual(result['year'], 2024)
        self.assertEqual(result['url'], 'https://doi.org/10.1234/example')

    def test_missing_metadata_is_not_invented(self):
        self.assertIsNone(literature.normalize({}))
        result = literature.normalize({'title':['Only a title'], 'DOI':'javascript:bad',
                                       'published':{'date-parts':[[None]]}})
        self.assertEqual(result['authors'], [])
        self.assertIsNone(result['year'])
        self.assertEqual(result['url'], '')

    def test_url_encodes_doi_special_characters(self):
        result = literature.normalize({**ITEM, 'DOI':'10.1234/a?b#c'})
        self.assertEqual(result['url'], 'https://doi.org/10.1234/a%3Fb%23c')

    def test_fetch_request_filters_and_metadata_only(self):
        with patch('literature.urlopen', return_value=self.response({'message':{'items':[ITEM]}})) as api:
            result = literature.fetch('tags & explanations', 3, 2020, 2025, 'proceedings-article')
        req = api.call_args.args[0]
        query = parse_qs(urlparse(req.full_url).query)
        self.assertEqual(urlparse(req.full_url).netloc, 'api.crossref.org')
        self.assertEqual(query['query.bibliographic'], ['tags & explanations'])
        self.assertEqual(query['rows'], ['3'])
        self.assertIn('from-pub-date:2020-01-01', query['filter'][0])
        self.assertIn('until-pub-date:2025-12-31', query['filter'][0])
        self.assertIn('type:proceedings-article', query['filter'][0])
        self.assertEqual(api.call_count, 1)  # No requests to DOI/publisher/PDF URLs.
        self.assertEqual(result, [ITEM])

    def test_duplicate_dois_across_topics(self):
        with patch('literature.fetch', return_value=[ITEM, {**ITEM,'DOI':'10.1234/EXAMPLE'}]):
            result = self.discover(['interpretability','recommendations'])
        self.assertEqual(len(result['works']), 1)
        self.assertEqual(result['works'][0]['topics'], ['interpretability','recommendations'])

    def test_same_titles_with_different_dois_are_kept(self):
        with patch('literature.fetch', return_value=[ITEM,{**ITEM,'DOI':'10.1234/second'}]):
            result = self.discover(['recommendations'])
        self.assertEqual(len(result['works']), 2)

    def test_partial_failure_preserves_success(self):
        with patch('literature.fetch', side_effect=[[ITEM], ValueError('Rate limited')]):
            result = self.discover(['one','two'])
        self.assertEqual(result['outcome'], 'partial')
        self.assertEqual(len(result['works']), 1)
        self.assertEqual(result['errors'][0]['topic'], 'two')

    def test_all_failed_is_not_empty_success(self):
        with patch('literature.fetch', side_effect=ValueError('offline')):
            with self.assertRaisesRegex(ValueError, 'Literature search failed'):
                self.discover(['topic'])

    def test_successful_empty_response(self):
        with patch('literature.fetch', return_value=[]):
            self.assertEqual(self.discover(['topic'])['outcome'], 'no_match')

    def test_invalid_input_never_calls_network(self):
        with patch('literature.fetch') as api:
            for topics, kwargs in [([],{}),([' '],{}),(['x']*6,{}),(['x'],{'limit':0}),
                                   (['x'],{'since':2025,'until':2020}),(['x'],{'since':True})]:
                with self.subTest(topics=topics,kwargs=kwargs), self.assertRaises(ValueError):
                    self.discover(topics, **kwargs)
            api.assert_not_called()

    def test_rate_limit_and_offline_are_actionable(self):
        for error, expected in [(HTTPError('url',429,'slow',{},None),'rate limit'),
                                (HTTPError('url',503,'down',{},None),'HTTP 503'),
                                (URLError('offline'),'internet connection')]:
            with patch('literature.urlopen', side_effect=error), self.assertRaisesRegex(ValueError, expected):
                literature.fetch('topic')

    def test_malformed_api_response(self):
        for payload in ({}, [], {'message':{'items':None}}, {'message':{'items':[None]}}):
            with patch('literature.urlopen', return_value=self.response(payload)), self.assertRaises(ValueError):
                literature.fetch('topic')
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'not JSON'
        with patch('literature.urlopen', return_value=response), self.assertRaisesRegex(ValueError, 'unreadable'):
            literature.fetch('topic')

    def test_literature_cli_does_not_need_local_index(self):
        stdout = io.StringIO()
        with patch('literature.fetch', return_value=[ITEM]), patch('paper.library.search') as local:
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                paper.main(['--db','/missing.sqlite','literature','recommendations','--format','json'])
        result = json.loads(stdout.getvalue())
        self.assertEqual(result['works'][0]['doi'], '10.1234/example')
        local.assert_not_called()

    def test_export_refuses_overwrite_before_network(self):
        with TemporaryDirectory() as tmp:
            destination = Path(tmp)/'reading.md'
            with patch('literature.fetch', return_value=[ITEM]):
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    paper.main(['literature','topic','--output',str(destination)])
            original = destination.read_text()
            self.assertIn('Download manually and read', original)
            with patch('literature.fetch') as api, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                paper.main(['literature','topic','--output',str(destination)])
            api.assert_not_called()
            self.assertEqual(destination.read_text(), original)


class PresentationTests(unittest.TestCase):
    def result(self):
        hit = dict(name='paper.pdf', page=2, part=1, label='iv', sha='abc',
                   passage='A complete source excerpt with clear boundaries.', url='file:///paper.pdf#page=2')
        return dict(question='What does the paper say?', hits=[hit], selection=None,
                    outcome='candidates', mode='lexical', retrieval_ms=2)

    def test_excerpt_is_separate_from_source_and_commands(self):
        result = self.result()
        text = paper.render_results(result, Path('index.sqlite'))
        self.assertLess(text.index('EXCERPT\n'), text.index('  | A complete'))
        self.assertLess(text.index('END EXCERPT'), text.index('SOURCE\n'))
        self.assertLess(text.index('SOURCE\n'), text.index('READ THE SURROUNDING PAGES'))
        self.assertNotIn('Retrieval time', text)
        self.assertNotIn('SHA-256', text)

    def test_selected_quotation_comes_before_candidates(self):
        result = self.result()
        result['selection'] = dict(source=1, quote='A complete source excerpt')
        text = paper.render_results(result, Path('index.sqlite'))
        self.assertLess(text.index('END QUOTATION'), text.index('EXCERPT\n'))
        self.assertIn('Source [1]: paper.pdf | PDF page 2', text)

    def test_details_are_opt_in(self):
        text = paper.render_results(self.result(), Path('index.sqlite'), details=True)
        self.assertIn('SEARCH DETAILS', text)
        self.assertIn('Retrieval time: 2 ms', text)

    def test_quote_wraps_to_terminal_without_mutating_evidence(self):
        result = self.result()
        original = result['hits'][0]['passage']
        with patch('presentation.width', return_value=40):
            text = paper.render_results(result, Path('index.sqlite'))
        quote_lines = [line for line in text.splitlines() if line.startswith('  | ')]
        self.assertTrue(all(len(line) <= 40 for line in quote_lines))
        self.assertEqual(' '.join(line[4:] for line in quote_lines), original)
        self.assertEqual(result['hits'][0]['passage'], original)

    def test_terminal_control_characters_are_not_emitted(self):
        self.assertNotIn('\x1b', presentation.paragraph('PDF title\x1b[31m'))

    def test_markdown_quote_and_context_are_separate(self):
        text = paper.render_results(self.result(), Path('index.sqlite'), markdown=True)
        self.assertIn('> A complete source excerpt', text)
        self.assertIn('**END EXCERPT**', text)
        self.assertIn('## READ THE SURROUNDING PAGES', text)
