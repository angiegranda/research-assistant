"""Page-aware, local PDF retrieval. Source text is evidence, never instructions."""
from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unicodedata

from pypdf import PdfReader
import models

INDEX_ID = 0x53524543
SCHEMA_VERSION = 2
STOP = set("a an the and or of for at to as by in on it is are was were be this that with from do does did what how why where which who can me tell paper papers author authors say says about mention mentions mentioned find show according".split())


def windows(text, size=95, overlap=18):
    """Whitespace-normalized excerpts, preserving spelling and page boundaries."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    words = text.split()
    for start in range(0, len(words), size-overlap):
        yield " ".join(words[start:start+size])
        if start+size >= len(words):
            break


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def discover(folder, excludes=()):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() == ".pdf"
                  and p.is_file() and not any(fnmatch(p.relative_to(folder).as_posix(), pattern)
                                              or fnmatch(p.name, pattern) for pattern in excludes))


def extraction_issue(text):
    if not text.strip():
        return "No extractable text; inspect the PDF or OCR a copy."
    # Some old PDFs map glyphs to control codes. Do not present these as quotations.
    bad = sum(unicodedata.category(c) == "Cc" and not c.isspace() or c == "\ufffd" for c in text)
    if bad / len(text) > .01:
        return "Unreliable character encoding; inspect the PDF or OCR a copy."
    return None


@contextmanager
def connect(db):
    path = Path(db).resolve()
    if not path.is_file():
        raise ValueError("No index found. Run the index command first.")
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        if conn.execute("PRAGMA application_id").fetchone()[0] != INDEX_ID:
            raise ValueError("Not a paper-reader index. Choose the correct --db.")
        if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Index format changed. Rebuild with the index command.")
        yield conn


def build(folder, db, semantic=False, excludes=()):
    """Build atomically; failed extraction or embeddings cannot destroy a snapshot."""
    if Path(db).is_symlink():
        raise ValueError("The index destination must not be a symbolic link.")
    folder, db = Path(folder).resolve(), Path(db).resolve()
    if db.suffix.lower() != ".sqlite":
        raise ValueError("Use a .sqlite filename for the index.")
    if db.exists():
        with closing(sqlite3.connect(db.as_uri()+"?mode=ro", uri=True)) as previous:
            if previous.execute("PRAGMA application_id").fetchone()[0] != INDEX_ID:
                raise ValueError("Destination is not a paper-reader index. Choose a new .sqlite file.")
    if not folder.is_dir():
        raise ValueError("Choose an existing folder containing PDFs.")
    files = discover(folder, excludes)
    if not files:
        raise ValueError("No PDFs found in that folder. Existing index preserved.")
    db.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with TemporaryDirectory(dir=db.parent) as tmp:
        new = Path(tmp)/"new.sqlite"
        with closing(sqlite3.connect(new)) as conn:
            conn.execute(f"PRAGMA application_id = {INDEX_ID}")
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.executescript("""
                CREATE TABLE files(id INTEGER PRIMARY KEY, name TEXT UNIQUE, path TEXT,
                    sha TEXT, title TEXT, author TEXT, page_count INTEGER);
                CREATE TABLE pages(file_id INTEGER, page INTEGER, label TEXT, text TEXT,
                    issue TEXT, PRIMARY KEY(file_id,page));
                CREATE VIRTUAL TABLE chunks USING fts5(search_text, passage UNINDEXED,
                    file_id UNINDEXED, page UNINDEXED, part UNINDEXED,
                    tokenize='porter unicode61');
                CREATE TABLE vectors(id INTEGER PRIMARY KEY, value TEXT);
                CREATE TABLE config(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE issues(name TEXT, message TEXT);
            """)
            config = {"model": models.EMBED_MODEL if semantic else "", "folder": str(folder),
                      "excludes": list(excludes), "built_at": datetime.now(timezone.utc).isoformat()}
            dimension = None
            for file_id, path in enumerate(files, 1):
                name = path.relative_to(folder).as_posix()
                print(f"Indexing {name}", file=sys.stderr, flush=True)
                try:
                    checksum = digest(path)
                    reader = PdfReader(path)
                    if reader.is_encrypted and not reader.decrypt(""):
                        raise ValueError("password required")
                    pages = [p.extract_text() or "" for p in reader.pages]
                    labels = reader.page_labels
                    metadata = reader.metadata or {}
                    if digest(path) != checksum:
                        raise ValueError("file changed during indexing")
                except Exception as exc:
                    conn.execute("INSERT INTO issues VALUES (?,?)", (name, str(exc)))
                    print(f"Skipped {name}: {exc}", file=sys.stderr)
                    counts["skipped_files"] += 1
                    continue
                conn.execute("INSERT INTO files VALUES (?,?,?,?,?,?,?)",
                             (file_id, name, str(path), checksum,
                              str(metadata.get('/Title') or ''), str(metadata.get('/Author') or ''), len(pages)))
                indexed = False
                for page, text in enumerate(pages, 1):
                    issue = extraction_issue(text)
                    normalized = " ".join(text.split())
                    conn.execute("INSERT INTO pages VALUES (?,?,?,?,?)",
                                 (file_id, page, labels[page-1], normalized, issue))
                    if issue:
                        counts["unreadable_pages"] += 1
                        continue
                    indexed = True
                    for part, passage in enumerate(windows(text), 1):
                        row = conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                                           (unicodedata.normalize('NFKC', passage), passage, file_id, page, part))
                        if semantic:
                            vec = models.embedding(passage)
                            if dimension is not None and len(vec) != dimension:
                                raise ValueError("Embedding dimensions changed during indexing.")
                            dimension = len(vec)
                            conn.execute("INSERT INTO vectors VALUES (?,?)", (row.lastrowid, json.dumps(vec)))
                        counts["passages"] += 1
                counts["indexed_files" if indexed else "skipped_files"] += 1
            if not counts["passages"]:
                raise ValueError("No extractable pages. Try OCR on a copy. Previous index preserved.")
            config.update(counts=dict(counts), dimension=dimension)
            conn.executemany("INSERT INTO config VALUES (?,?)", [(k, json.dumps(v)) for k,v in config.items()])
            conn.commit()
        new.replace(db)
    return dict(counts)


def settings(conn):
    return {row['key']: json.loads(row['value']) for row in conn.execute('SELECT * FROM config')}


def verify_source(row):
    path = Path(row['path'])
    if not path.is_file() or digest(path) != row['sha']:
        raise ValueError(f"Source moved or changed: {row['name']}. Rebuild the index.")


def search(db, question, k=3, phrase=False, hybrid=False, file=None):
    if not question.strip():
        raise ValueError("Enter a question or phrase.")
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20.")
    if phrase and hybrid:
        raise ValueError("Choose phrase or hybrid retrieval, not both.")
    with connect(db) as conn:
        if file is not None and not conn.execute("SELECT 1 FROM files WHERE name=?", (file,)).fetchone():
            raise ValueError("Paper not found. Use status to find its exact relative filename.")
        terms = re.findall(r"[^\W_]+", unicodedata.normalize('NFKC', question).casefold())
        if phrase:
            expression = '"'+" ".join(terms)+'"' if terms else ""
        else:
            expression = " OR ".join('"'+t+'"' for t in dict.fromkeys(terms) if t not in STOP)
        pool = max(100, k*20)
        ranked = []
        if expression:
            ranked = [r[0] for r in conn.execute("""
                SELECT chunks.rowid FROM chunks JOIN files ON files.id=chunks.file_id
                WHERE chunks MATCH ? AND (? IS NULL OR files.name=?)
                ORDER BY rank,chunks.rowid LIMIT ?""", (expression, file, file, pool))]
        if hybrid:
            config = settings(conn)
            if config['model'] != models.EMBED_MODEL:
                raise ValueError("Build with index --semantic before using --hybrid.")
            q = models.embedding(question)
            if len(q) != config['dimension']:
                raise ValueError("Embedding dimensions changed. Rebuild the semantic index.")
            dense = []
            for row in conn.execute("""SELECT vectors.id,value FROM vectors
                    JOIN chunks ON chunks.rowid=vectors.id JOIN files ON files.id=chunks.file_id
                    WHERE (? IS NULL OR files.name=?)""", (file, file)):
                vec = json.loads(row['value'])
                dense.append((sum(x*y for x,y in zip(q, vec, strict=True)), row['id']))
            dense.sort(key=lambda pair: (-pair[0], pair[1]))
            scores = Counter()
            for ranking in (ranked, [i for _,i in dense[:pool]]):
                for rank, chunk_id in enumerate(ranking, 1):
                    scores[chunk_id] += 1/(60+rank)
            ranked = sorted(scores, key=lambda i: (-scores[i], i))
        hits, seen_pages, checked = [], set(), set()
        for chunk_id in ranked:
            row = dict(conn.execute("""
                SELECT passage,chunks.page,part,name,path,sha,title,author,label,page_count
                FROM chunks JOIN files ON files.id=chunks.file_id
                JOIN pages ON pages.file_id=chunks.file_id AND pages.page=chunks.page
                WHERE chunks.rowid=?""", (chunk_id,)).fetchone())
            key = (row['name'], row['page'])
            if key in seen_pages:
                continue
            if row['path'] not in checked:
                verify_source(row)
                checked.add(row['path'])
            seen_pages.add(key)
            row['url'] = Path(row['path']).as_uri()+f"#page={row['page']}"
            hits.append(row)
            if len(hits) == k:
                break
    return hits


def read_page(db, name, page, context=1):
    """Return whole physical pages, never a generated summary."""
    if type(page) is not int or page < 1 or type(context) is not int or not 0 <= context <= 3:
        raise ValueError("Use a positive PDF page and context between 0 and 3.")
    with connect(db) as conn:
        source = conn.execute("SELECT * FROM files WHERE name=?", (name,)).fetchone()
        if source is None:
            raise ValueError("Paper not found. Use status to find its exact relative filename.")
        if page > source['page_count']:
            raise ValueError(f"This PDF has only {source['page_count']} pages.")
        verify_source(source)
        pages = [dict(row) for row in conn.execute("""SELECT page,label,text,issue FROM pages
                 WHERE file_id=? AND page BETWEEN ? AND ? ORDER BY page""",
                 (source['id'], max(1, page-context), page+context))]
        return {"name": name, "path": source['path'], "sha": source['sha'], "pages": pages,
                "url": Path(source['path']).as_uri()+f"#page={page}"}


def status(db, verify=False):
    with connect(db) as conn:
        config = settings(conn)
        files = [dict(row) for row in conn.execute("""SELECT files.*,
                    (SELECT COUNT(*) FROM pages WHERE file_id=files.id AND issue IS NOT NULL) AS unreadable_pages
                    FROM files ORDER BY name""")]
        for source in files:
            if verify:
                path = Path(source['path'])
                source['state'] = 'missing' if not path.is_file() else (
                    'unchanged' if digest(path) == source['sha'] else 'changed')
        issues = [dict(row) for row in conn.execute('SELECT * FROM issues')]
        page_issues = [dict(row) for row in conn.execute("""SELECT name,page,issue FROM pages
                       JOIN files ON files.id=pages.file_id WHERE issue IS NOT NULL ORDER BY name,page""")]
        folder = Path(config['folder'])
        known = {s['name'] for s in files} | {s['name'] for s in issues}
        added = sorted(p.relative_to(folder).as_posix() for p in discover(folder, config['excludes'])
                       if p.relative_to(folder).as_posix() not in known) if verify and folder.is_dir() else []
        return {**config, 'files': files, 'file_issues': issues, 'page_issues': page_issues,
                'new_files': added, 'verified': verify}
