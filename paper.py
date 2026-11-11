"""Small reading-first CLI; use ./reader --help."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

if sys.version_info < (3, 11):
    sys.exit('Paper assistant requires Python 3.11 or newer.')

import library
import models
import presentation

HERE = Path(__file__).resolve().parent
READING_REMINDER = ("Candidate excerpts are reading leads, not verified thesis claims. Read the source "
                    "section and check wording, attribution, and printed page before citing.")


def parser_for_cli():
    parser = argparse.ArgumentParser(description="Find evidence and return to reading your research PDFs.")
    parser.add_argument('--db', type=Path, default=HERE/'state'/'papers.sqlite',
                        help='Index path (put this option before the command).')
    commands = parser.add_subparsers(dest='action', required=True)
    index = commands.add_parser('index', help='Build a fresh snapshot of a PDF folder.')
    index.add_argument('folder', nargs='?', type=Path, default=HERE.parent/'papers')
    index.add_argument('--semantic', action='store_true', help='Also embed passages with local Ollama.')
    index.add_argument('--exclude', action='append', default=[], metavar='GLOB',
                       help='Exclude a filename or relative-path pattern; repeat as needed.')
    status = commands.add_parser('status', help='List papers, extraction gaps, and index details.')
    status.add_argument('--verify', action='store_true', help='Check for new, changed, and missing PDFs.')
    status.add_argument('--json', action='store_true')
    read = commands.add_parser('read', help='Read a full page plus its neighboring pages.')
    read.add_argument('file', help='Exact relative filename shown in search/status.')
    read.add_argument('page', type=int, help='Physical PDF page, starting at 1.')
    read.add_argument('--context', type=int, choices=range(4), default=1)
    read.add_argument('--json', action='store_true')
    for name in ('search', 'ask'):
        cmd = commands.add_parser(name, help='Find candidate excerpts' if name == 'search'
                                  else 'Find excerpts and ask a local model to select one quotation.')
        cmd.add_argument('question')
        cmd.add_argument('-k', type=int, choices=range(1,21), default=5, metavar='1..20')
        cmd.add_argument('--file', help='Search only this exact relative filename (see status).')
        cmd.add_argument('--details', action='store_true', help='Show timing, index labels, and source hashes separately.')
        cmd.add_argument('--format', choices=('text','json','markdown'), default='text')
        cmd.add_argument('--output', type=Path, help='Save a NEW .md or .json file; never overwrite existing files.')
        modes = cmd.add_mutually_exclusive_group()
        modes.add_argument('--phrase', action='store_true', help='Match adjacent stemmed tokens in order.')
        modes.add_argument('--hybrid', action='store_true', help='Combine lexical and local embedding ranks.')
    literature = commands.add_parser('literature', help='Search online for paper metadata and links; download manually.')
    literature.add_argument('topics', nargs='+', help='One to five quoted topics; each is searched separately.')
    literature.add_argument('-k', type=int, choices=range(1,21), default=5, metavar='1..20',
                            help='Maximum records per topic (default: 5).')
    literature.add_argument('--since', type=int, help='Earliest publication year, inclusive.')
    literature.add_argument('--until', type=int, help='Latest publication year, inclusive.')
    literature.add_argument('--type', dest='work_type',
                            choices=('journal-article','proceedings-article','posted-content','book-chapter'))
    literature.add_argument('--format', choices=('text','json','markdown'), default='text')
    literature.add_argument('--output', type=Path, help='Save a NEW .md or .json reading list.')
    evaluate = commands.add_parser('evaluate', help='Measure retrieval against labeled paper/page cases.')
    evaluate.add_argument('--cases', type=Path, default=HERE/'tests'/'papers_cases.json')
    evaluate.add_argument('-k', type=int, choices=range(1,21), default=5, metavar='1..20')
    evaluate.add_argument('--hybrid', action='store_true')
    commands.add_parser('test', help='Run the offline regression suite; no model or private papers required.')
    return parser


def render_results(result, db, markdown=False, details=False):
    return presentation.render_results(result, db, markdown, details)


def prepare_output(args, parser):
    if args.output:
        expected = {'.md': 'markdown', '.json': 'json'}.get(args.output.suffix.lower())
        if expected is None:
            parser.error('--output needs a .md or .json filename.')
        if args.output.exists():
            parser.error('--output already exists. Choose a new filename.')
        args.format = expected


def emit(output, destination):
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(output)
        print(f'Saved reading leads to {destination}')
    else:
        print(output, end='')


def main(argv=None):
    parser = parser_for_cli()
    args = parser.parse_args(argv)
    started = perf_counter()
    if args.action == 'test':
        import unittest
        suite = unittest.defaultTestLoader.discover(str(HERE/'tests'), top_level_dir=str(HERE))
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.action == 'index':
        print(json.dumps(library.build(args.folder, args.db, args.semantic, args.exclude), indent=2))
        print('Use status to inspect extraction gaps.', file=sys.stderr)
        return 0
    if args.action == 'status':
        result = library.status(args.db, args.verify)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Built: {result['built_at']}\nLibrary: {result['folder']}\nEmbedding model: {result['model'] or 'none (offline lexical search)'}")
            print('Excluded patterns: '+', '.join(result['excludes']))
            print(json.dumps(result['counts']))
            for source in result['files']:
                print(f"{source['name']} | {source['page_count']} pages | {source['unreadable_pages']} unreadable | {source.get('state','not checked')}")
            for issue in result['file_issues']:
                print(f"SKIPPED {issue['name']}: {issue['message']}")
            for issue in result['page_issues']:
                print(f"GAP {issue['name']}, PDF page {issue['page']}: {issue['issue']}")
            for name in result['new_files']:
                print(f'NEW {name}: rebuild to include this paper.')
        return 0
    if args.action == 'read':
        result = library.read_page(args.db, args.file, args.page, args.context)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(presentation.render_page(result), end='')
        return 0
    if args.action == 'evaluate':
        import evaluation
        print(json.dumps(evaluation.evaluate(args.db, args.cases, args.k, args.hybrid), indent=2))
        return 0
    prepare_output(args, parser)
    if args.action == 'literature':
        import literature
        result = literature.discover(args.topics, args.k, args.since, args.until, args.work_type)
        output = (json.dumps(result, indent=2, ensure_ascii=False)+'\n' if args.format == 'json'
                  else presentation.render_literature(result, args.format == 'markdown'))
        emit(output, args.output)
        return 0
    hits = library.search(args.db, args.question, args.k, args.phrase, args.hybrid, args.file)
    result = {'question': args.question, 'mode': 'hybrid' if args.hybrid else 'phrase' if args.phrase else 'lexical',
              'hits': hits, 'selection': None, 'outcome': 'candidates' if hits else 'no_match',
              'retrieval_ms': round((perf_counter()-started)*1000, 1), 'reading_reminder': READING_REMINDER}
    # A model failure is explicit, but candidates and exports remain useful offline.
    if args.action == 'ask' and hits:
        try:
            selected = models.select_quote(args.question, hits)
            result['outcome'] = 'quote' if selected else 'abstained'
            if selected:
                result['selection'] = dict(source=selected[0], quote=selected[1])
        except (ValueError, OSError) as exc:
            result['outcome'] = 'model_unavailable_or_invalid'
            result['warning'] = f'Quote selection failed: {exc} Retrieved candidates are available for reading.'
    result['total_ms'] = round((perf_counter()-started)*1000, 1)
    output = json.dumps(result, indent=2, ensure_ascii=False)+'\n' if args.format == 'json' else render_results(result, args.db, args.format == 'markdown', args.details)
    emit(output, args.output)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, sqlite3.Error) as exc:
        sys.exit(f'Paper assistant: {exc}')
    except KeyboardInterrupt:
        sys.exit('Paper assistant: cancelled; a previous index remains usable.')
