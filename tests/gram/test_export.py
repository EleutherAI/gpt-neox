import subprocess
import json
import sys
from pathlib import Path

import torch
import pytest


def test_native_rank_export_removes_auxiliary_and_training_state(tmp_path):
    source = tmp_path / "rank.pt"
    output = tmp_path / "export.pt"
    torch.save({"module": {"sequential.2.mlp.core.linear1.weight": torch.ones(2, 2),
                           "sequential.2.mlp.gram_aux.0.weight": torch.ones(2, 2),
                           "sequential.0.word_embeddings.weight": torch.ones(3, 2)},
                "optimizer": {"state": "must not survive"}, "iteration": 12}, source)
    script = Path(__file__).resolve().parents[2] / "tools/gram/export_core.py"
    subprocess.run([sys.executable, str(script), str(source), str(output)], check=True)
    result = torch.load(output, weights_only=True)
    assert set(result) == {"module", "iteration", "configuration_overrides"}
    geometry = result["configuration_overrides"]
    assert geometry["intermediate_size"] == 2
    assert geometry["gram_enabled"] is False
    assert geometry["gram_absorb_auxiliary_width"] is False
    assert json.loads(output.with_suffix(".pt.config.json").read_text()) == geometry
    assert result["iteration"] == 0
    assert "sequential.2.mlp.linear1.weight" in result["module"]
    assert not any("gram_aux" in key or "mlp.core" in key for key in result["module"])
    repeat = subprocess.run([sys.executable, str(script), str(source), str(output)],
                            capture_output=True)
    assert repeat.returncode != 0
    assert b"already exists" in repeat.stderr


@pytest.mark.parametrize("level", [0, 1, 2])
def test_nested_rank_export_contains_only_requested_prefix(tmp_path, level):
    source, output = tmp_path / "nested.pt", tmp_path / f"level{level}.pt"
    torch.save({"module": {
        "sequential.2.mlp.core.linear1.weight": torch.ones(2, 2),
        "sequential.2.mlp.gram_aux.0.weight": torch.ones(2, 2),
        "sequential.2.mlp.gram_inner.0.weight": torch.ones(2, 2),
    }, "optimizer": {"state": "discard"}, "iteration": 6}, source)
    script = Path(__file__).resolve().parents[2] / "tools/gram/export_core.py"
    subprocess.run([sys.executable, str(script), str(source), str(output),
                    "--level", str(level)], check=True)
    result = torch.load(output, weights_only=True)
    assert set(result) == {"module", "iteration", "configuration_overrides"}
    geometry = result["configuration_overrides"]
    assert geometry["intermediate_size"] == 2
    assert geometry["gram_enabled"] == (level > 0)
    assert geometry["gram_nested"] == (level == 2)
    assert geometry["gram_eval_level"] == level
    if level:
        assert geometry["gram_aux_output_bias"] is False
    assert json.loads(output.with_suffix(".pt.config.json").read_text()) == geometry
    assert any("gram_aux" in key for key in result["module"]) == (level >= 1)
    assert any("gram_inner" in key for key in result["module"]) == (level == 2)
    assert any("mlp.core" in key for key in result["module"]) == (level >= 1)
