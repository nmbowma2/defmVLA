import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from pick_checkpoint import pick  # noqa: E402


def make(tmp_path, val, saved):
    for s in saved:
        (tmp_path / f"checkpoint_{s:06d}").mkdir()
        (tmp_path / f"checkpoint_{s:06d}_deploy").mkdir()  # must be ignored
    (tmp_path / "train_log.jsonl").write_text("".join(json.dumps({"step": s, "val_loss": v}) + "\n" for s, v in val.items()))
    return str(tmp_path)


def test_best_only_considers_saved_steps(tmp_path):
    d = make(tmp_path, {1000: 0.01, 2000: 0.3, 4000: 0.2, 6000: 0.25}, [2000, 4000, 6000])
    assert pick(d) == "checkpoint_004000" and pick(d, "last") == "checkpoint_006000"


def test_missing_validation_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        pick(make(tmp_path, {2000: 0.3}, [2000, 4000]))
