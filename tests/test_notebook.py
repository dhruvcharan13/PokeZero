import json
from pathlib import Path

import torch

from pokezero.train.learner import resolve_device

NB = Path(__file__).parents[1] / "notebooks" / "playground.ipynb"


def test_playground_notebook_is_clean_and_compiles():
    nb = json.loads(NB.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code
    for c in code:
        assert not c["outputs"] and c["execution_count"] is None  # committed without outputs
        src = "".join(c["source"])
        src = "\n".join(  # blank out IPython magics, keeping indentation valid
            ln.replace(ln.lstrip(), "pass") if ln.lstrip().startswith("%") else ln
            for ln in src.splitlines()
        )
        compile(src, NB.name, "exec")


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
