"""Explicit online discovery via Crossref. Fetch metadata only; never download papers."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from presentation import clean

API = 'https://api.crossref.org/works'
MAX_RESPONSE = 4_000_000


class PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def plain(value):
    if not isinstance(value, str):
        return ''
    parser = PlainText()
    parser.feed(value)
    return clean(' '.join(parser.parts))


def first(value):
    return plain(value[0]) if isinstance(value, list) and value else ''


def normalize(item):
    title = first(item.get('title'))
    if not title:
        return None
    doi = item.get('DOI', '')
    doi = doi.strip() if isinstance(doi, str) else ''
    if not re.fullmatch(r'10\.\d{4,9}/\S+', doi, re.IGNORECASE):
        doi = ''
    authors = []
    for author in item.get('author', []) if isinstance(item.get('author'), list) else []:
        if isinstance(author, dict):
            name = plain(author.get('name')) or ' '.join(filter(None, [plain(author.get('given')), plain(author.get('family'))]))
            if name:
                authors.append(name)
    year = None
    for key in ('published', 'published-print', 'published-online', 'issued'):
        date = item.get(key)
        parts = date.get('date-parts') if isinstance(date, dict) else None
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            candidate = parts[0][0]
            if type(candidate) is int and 1000 <= candidate <= 9999:
                year = candidate
                break
    return {'title': title, 'authors': authors, 'year': year,
            'venue': first(item.get('container-title')), 'type': plain(item.get('type')),
            'doi': doi, 'url': 'https://doi.org/'+quote(doi, safe='/') if doi else '',
            'provider': 'Crossref'}


def fetch(topic, limit=5, since=None, until=None, work_type=None):
    params = {'query.bibliographic': topic, 'rows': limit, 'sort': 'relevance',
              'select': 'DOI,title,author,published,published-print,published-online,issued,container-title,type'}
    filters = []
    if since is not None:
        filters.append(f'from-pub-date:{since}-01-01')
    if until is not None:
        filters.append(f'until-pub-date:{until}-12-31')
    if work_type:
        filters.append('type:'+work_type)
    if filters:
        params['filter'] = ','.join(filters)
    req = Request(API+'?'+urlencode(params), headers={
        'User-Agent': 'ThesisPaperAssistant/1.0 (personal literature discovery)',
        'Accept': 'application/json'})
    try:
        with urlopen(req, timeout=25) as response:
            raw = response.read(MAX_RESPONSE+1)
        if len(raw) > MAX_RESPONSE:
            raise ValueError('Crossref response too large. Request fewer results.')
        payload = json.loads(raw)
    except HTTPError as exc:
        if exc.code == 429:
            raise ValueError('Crossref rate limit reached. Wait and try again later.') from exc
        raise ValueError(f'Crossref returned HTTP {exc.code}. Try again later.') from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ValueError('Cannot reach Crossref. Check your internet connection; local PDF search still works.') from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('Crossref returned an unreadable metadata response. Try again later.') from exc
    message = payload.get('message') if isinstance(payload, dict) else None
    items = message.get('items') if isinstance(message, dict) else None
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        raise ValueError('Crossref returned an unexpected metadata response.')
    return items[:limit]


def discover(topics, limit=5, since=None, until=None, work_type=None):
    if not isinstance(topics, (list, tuple)) or not 1 <= len(topics) <= 5:
        raise ValueError('Supply one to five quoted topics.')
    if any(not isinstance(t, str) or not t.strip() or len(t) > 500 for t in topics):
        raise ValueError('Each topic must contain 1 to 500 characters.')
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('Use between 1 and 20 results per topic.')
    for year in (since, until):
        if year is not None and (type(year) is not int or not 1000 <= year <= 9999):
            raise ValueError('Use a four-digit publication year.')
    if since is not None and until is not None and since > until:
        raise ValueError('--since must not be later than --until.')
    if work_type not in (None, 'journal-article', 'proceedings-article', 'posted-content', 'book-chapter'):
        raise ValueError('Unsupported publication type.')
    topics = list(dict.fromkeys(t.strip() for t in topics))
    records, errors, successes = {}, [], 0
    for topic in topics:
        print(f'Searching Crossref: {clean(topic)}', file=sys.stderr, flush=True)
        try:
            items = fetch(topic, limit, since, until, work_type)
        except ValueError as exc:
            errors.append({'topic': topic, 'message': str(exc)})
            continue
        successes += 1
        for item in items:
            record = normalize(item)
            if record is None:
                continue
            # Deduplicate DOI records only; similar titles can be different editions.
            key = record['doi'].casefold() if record['doi'] else f'missing-doi:{len(records)}'
            if key not in records:
                records[key] = {**record, 'topics': []}
            if topic not in records[key]['topics']:
                records[key]['topics'].append(topic)
    if not successes:
        raise ValueError('Literature search failed: '+'; '.join(e['message'] for e in errors))
    return {'provider': 'Crossref', 'topics': topics, 'works': list(records.values()),
            'errors': errors, 'outcome': 'partial' if errors else 'results' if records else 'no_match',
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'filters': {'since': since, 'until': until, 'type': work_type},
            'results_per_topic': limit, 'downloads': 'manual',
            'notice': 'Bibliographic metadata search, not full-text evidence or a comprehensive literature review.'}
