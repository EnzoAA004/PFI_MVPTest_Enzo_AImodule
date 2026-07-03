from __future__ import annotations

from pfi_ai_service.readiness import build_readiness


def test_readiness_includes_mvp_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PFI_MODEL_DIR", str(tmp_path / "models"))

    result = build_readiness(tmp_path / "outputs")

    assert "mvpCompletion" in result
    assert result["mvpCompletion"]["totalItems"] == 6
    assert result["mvpCompletion"]["completionPercent"] >= 0
    assert result["mvpCompletion"]["completeItems"] <= 6
