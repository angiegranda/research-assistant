"""Readable terminal and Markdown output, separate from retrieval and source text."""
import html
from pathlib import Path
import shlex
import shutil
import textwrap
import unicodedata


def clean(value):
    # Terminal escape/control characters from a PDF or remote metadata are not commands.
    return ' '.join(''.join(c if unicodedata.category(c) not in {'Cc', 'Cf'} else ' '
                            for c in str(value)).split())


def width():
    return max(40, min(88, shutil.get_terminal_size((88, 24)).columns))


def paragraph(text, prefix=''):
    return textwrap.fill(clean(text), width=width(), initial_indent=prefix,
                         subsequent_indent=prefix, break_long_words=False, break_on_hyphens=False)


def block(text, label='EXCERPT', markdown=False):
    if markdown:
        return ['**'+label+'**', '', '> '+html.escape(clean(text)), '', '**END '+label+'**']
    return [label, '', paragraph(text, '  | '), '', 'END '+label]


def command(db, hit):
    return shlex.join(['sh', str(Path(__file__).resolve().parent/'reader'), '--db',
                       str(Path(db).resolve()), 'read', hit['name'], str(hit['page'])])


def render_results(result, db, markdown=False, details=False):
    rule = '-'*width()
    heading = lambda title: ('## '+title) if markdown else title
    lines = ['# Reading leads' if markdown else 'READING LEADS', '',
             paragraph('Question: '+result['question']),
             paragraph('Read the original section before citing. Excerpts may start or end mid-sentence.')]
    selection = result['selection']
    if selection:
        hit = result['hits'][selection['source']-1]
        lines += ['', rule, heading('MODEL-SELECTED QUOTATION'),
                  paragraph(f"Source [{selection['source']}]: {hit['name']} | PDF page {hit['page']}"), '',
                  *block(selection['quote'], 'QUOTATION', markdown), '',
                  paragraph('Wording matched the extracted source text. Relevance and meaning still need your review.')]
    elif result['outcome'] == 'abstained':
        lines += ['', 'No quotation selected: the model abstained. Candidate excerpts follow.']
    if result.get('warning'):
        lines += ['', paragraph(result['warning'])]
    for i, hit in enumerate(result['hits'], 1):
        lines += ['', rule, heading(f"[{i}] {clean(hit['name'])}"),
                  f"PDF page {hit['page']}", '', *block(hit['passage'], markdown=markdown), '',
                  'SOURCE', hit['url']]
        if markdown:
            lines += [f"PDF label: {clean(hit['label'])} (verify printed pagination)",
                      f"Source SHA-256: {hit['sha']}", '',
                      '- [ ] Read this section in the original PDF',
                      '- [ ] Verify author, year, title, and printed page in my reference manager',
                      '- My interpretation and thesis relevance: ', '- Limitations or contrasting evidence: ']
    if not result['hits']:
        lines += ['', 'No matching passage. Try another term or check index coverage with status.']
    else:
        lines += ['', rule, heading('READ THE SURROUNDING PAGES'), 'Copy a command below:']
        for i, hit in enumerate(result['hits'], 1):
            lines += ['', f"Source [{i}]", 'Read context: '+command(db, hit)]
    if details:
        lines += ['', rule, heading('SEARCH DETAILS'), f"Mode: {result['mode']}",
                  f"Outcome: {result['outcome']}", f"Retrieval time: {result['retrieval_ms']} ms"]
        for i, hit in enumerate(result['hits'], 1):
            lines += [f"[{i}] PDF label: {clean(hit['label'])} | passage: {hit['part']}",
                      f"    SHA-256: {hit['sha']}"]
    return '\n'.join(lines)+'\n'


def render_page(result):
    lines = ['READING CONTEXT', paragraph(result['name']), result['url']]
    for page in result['pages']:
        lines += ['', '-'*width(), f"PDF PAGE {page['page']} | PDF label {clean(page['label'])}", '']
        lines += [paragraph(page['issue'])] if page['issue'] else block(page['text'], 'PAGE TEXT')
    lines += ['', paragraph('Check the original PDF for columns, equations, tables, and printed page numbers.')]
    return '\n'.join(lines)+'\n'


def render_literature(result, markdown=False):
    lines = ['# Literature to explore' if markdown else 'LITERATURE TO EXPLORE', '',
             paragraph('Topics: '+'; '.join(result['topics'])),
             paragraph('Crossref metadata matches. Read the paper to assess relevance; inclusion does not establish peer review.'),
             paragraph('Open a DOI link in your browser, sign in through your institution if needed, and download manually.')]
    for i, item in enumerate(result['works'], 1):
        lines += ['', '-'*width(), paragraph(f"[{i}] {item['title']}"),
                  paragraph('Authors: '+('; '.join(item['authors']) or 'Not supplied')),
                  paragraph(f"Year: {item['year'] or 'Not supplied'} | Type: {item['type'] or 'Not supplied'}"),
                  paragraph('Published in: '+(item['venue'] or 'Not supplied')),
                  paragraph('Matched topics: '+'; '.join(item['topics'])),
                  'DOI: '+(item['doi'] or 'Not supplied'), 'Open: '+(item['url'] or 'No link supplied')]
        if markdown:
            lines += ['', '- [ ] Check relevance and publication status', '- [ ] Download manually and read',
                      '- My notes: ']
    if not result['works']:
        lines += ['', 'No literature records found. Try broader or different terms.']
    if result['errors']:
        lines += ['', 'INCOMPLETE SEARCH']
        lines += [paragraph(f"{e['topic']}: {e['message']}") for e in result['errors']]
    lines += ['', '-'*width(), paragraph(f"{len(result['works'])} unique records | Retrieved {result['retrieved_at']}"),
              paragraph('After saving PDFs to your papers folder, rebuild the local index to search their text.')]
    return '\n'.join(lines)+'\n'
