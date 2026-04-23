"""Top-level shim for convenience: ``python train.py --config ...``.

Equivalent to ``python -m src.train``. The real implementation lives in
:mod:`src.train`.
"""

from src.train import main

if __name__ == "__main__":
    main()
