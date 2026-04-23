"""Top-level shim for convenience: ``python eval.py --config ... --checkpoint ...``.

Equivalent to ``python -m src.eval``. The real implementation lives in
:mod:`src.eval`.
"""

from src.eval import main

if __name__ == "__main__":
    main()
