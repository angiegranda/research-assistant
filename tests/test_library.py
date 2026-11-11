"""Offline behavioral tests. Model-boundary tests use explicit test doubles."""
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import library
import models

EXAMPLES = Path(__file__).resolve().parents[1]/"example_papers"

class PaperTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/"papers"
        shutil.copytree(EXAMPLES, self.folder)
        self.db = Path(self.temp.name)/"index.sqlite"
        self.counts = library.build(self.folder, self.db)

    def test_gold_pages(self):
        # Judgments refer to physical PDF pages, independently of model output.
        cases = [("chronological split","Evaluation-notes.pdf",1),
                 ("tag precision","Evaluation-notes.pdf",2),
                 ("artifact bundle","Serving-notes.pdf",1)]
        for query, file, page in cases:
            with self.subTest(query=query):
                hit = library.search(self.db, query, 1)[0]
                self.assertEqual((hit["name"], hit["page"]), (file, page))
        self.assertEqual(self.counts, {"indexed_files":2,"passages":3})

    def test_unknown_and_phrase(self):
        self.assertEqual(library.search(self.db, "photosynthesis"), [])
        self.assertEqual(library.search(self.db, "precision tag", phrase=True), [])
        self.assertEqual(len(library.search(self.db, "tag precision", phrase=True)), 1)

    def test_rebuild_source_lifecycle(self):
        changed = self.folder/"Evaluation-notes.pdf"
        changed.write_bytes(changed.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            library.search(self.db, "chronological")
        changed.unlink()
        library.build(self.folder, self.db)
        self.assertEqual(library.search(self.db, "chronological"), [])

    def test_source_pdf_and_unrelated_db_cannot_be_replaced(self):
        source = self.folder/"Serving-notes.pdf"
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, ".sqlite"):
            library.build(self.folder, source)
        self.assertEqual(source.read_bytes(), original)
        other = self.folder/"other.sqlite"
        with closing(sqlite3.connect(other)) as conn:
            conn.execute("CREATE TABLE important(value)")
        with self.assertRaisesRegex(ValueError, "not a paper-reader index"):
            library.build(self.folder, other)

    def test_failed_rebuild_preserves_snapshot(self):
        with patch("models.embedding", side_effect=ValueError("interrupted")):
            with self.assertRaisesRegex(ValueError, "interrupted"):
                library.build(self.folder, self.db, semantic=True)
        self.assertEqual(library.search(self.db, "tag precision",1)[0]["page"],2)

    def test_quote_contract(self):
        hits = library.search(self.db,"artifact",1)
        excerpt = "The serving application loads an exported artifact bundle."
        self.assertEqual(models.validated_quote({"source":1,"quote":excerpt},hits), (1,excerpt))
        self.assertIsNone(models.validated_quote({"source":0,"quote":""},hits))
        for bad in ({"source":1,"quote":"Invented detail."},{"source":2,"quote":excerpt}):
            with self.assertRaises(ValueError):
                models.validated_quote(bad,hits)

    def test_local_chat_wiring(self):
        hits = library.search(self.db,"artifact",1)
        with patch("models.request", return_value={"message":{"content":
                '{"source":1,"quote":"an exported artifact bundle"}'}}) as api:
            self.assertEqual(models.select_quote("what is loaded?",hits)[0],1)
            self.assertEqual(api.call_args.args[0],"chat")

    def test_hybrid_plumbing_with_test_vectors(self):
        def vector(sentence):
            return [1.,0.] if "bundle" in sentence or "package" in sentence else [0.,1.]
        with patch("models.embedding", side_effect=vector):
            library.build(self.folder,self.db,semantic=True)
            self.assertEqual(library.search(self.db,"package",1,hybrid=True)[0]["name"],
                             "Serving-notes.pdf")

    def test_overlap_keeps_boundary_terms(self):
        chunks = list(library.windows("one two three four five six",4,2))
        self.assertEqual(chunks,["one two three four","three four five six"])

if __name__=="__main__":
    unittest.main()
