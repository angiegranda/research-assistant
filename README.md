# Research paper assistant

Find relevant passages in your own PDFs, check the source page, and write your own
thesis notes. This is a local command-line reading aid for any research work. Ordinary search needs neither an API key nor a language model.

**Start with `search`; use `read` to return to the paper.** Results are candidate
excerpts, not verified claims. `ask` optionally uses retrieval-augmented generation
(RAG) to select a quotation; it does not write a literature review.

## Everyday commands

Run these from the `assistant/` directory:

| What you want | Command |
| --- | --- |
| Find excerpts in downloaded PDFs | `./reader search "implicit feedback confidence" -k 3` |
| Read a page in context | `./reader read cf.pdf 2` |
| Ask a local model to select a quotation | `./reader ask "Why is implicit feedback noisy?"` |
| Find publications online | `./reader literature "interpretable recommender systems" -k 5` |
| Save an online reading list | `./reader literature "sparse autoencoders" --output notes/literature.md` |
| Show optional technical details | `./reader search "implicit feedback" --details` |

## Quick start

From the project root, using Python 3.11 or newer with SQLite FTS5:

```sh
cd assistant
./reader --help
./reader index [PAPERS PATH]
./reader search "Usage of sparse autoencoders for recommendation systems"
./reader read cf.pdf 2
```

The launcher uses the included, pinned pypdf wheel directly, so this path works
offline without installing packages. If execution permission was lost in a download,
use `sh reader` in place of `./reader`. Select another interpreter with
`PYTHON=python3.12 ./reader --help`.

Indexing the supplied library takes longer than a search because it includes an
801-page book. Progress goes to the terminal; the final counts go to standard output.
The source PDFs are never modified. Check extraction coverage afterwards:
