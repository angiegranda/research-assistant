#!/bin/sh
# Locate our project even when launched from another directory. The bundled wheel
# is pure Python: the interpreter can import it directly as a ZIP archive.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$here/vendor/pypdf-6.10.0-py3-none-any.whl${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -B "$here/paper.py" "$@"
