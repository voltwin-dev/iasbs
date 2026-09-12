"""Path bootstrap for the one-shot diagnostics in this directory.

They are run as scripts (`python iasbs/analysis/<name>.py`) rather than imported
as a package, so `iasbs/`, the repository root and `rasbs/` have to be on
`sys.path` before `common`, `sphere`, `stiefel`, `occupation`, `remeasure` or
`rasbs_port` can be reached.  Importing this module does that and exposes
`ROOT`, which every diagnostic uses to build `json/` and `ckpt/` paths.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

for _p in (HERE, os.path.join(ROOT, "iasbs"), ROOT, os.path.join(ROOT, "rasbs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
