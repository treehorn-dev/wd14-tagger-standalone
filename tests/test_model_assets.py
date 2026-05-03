from pathlib import Path

from tagger.model_assets import resolve_local_asset_pair


def test_resolve_local_asset_pair_uses_model_root(tmp_path, monkeypatch):
    model_dir = tmp_path / "wd14-convnextv2.v1"
    model_dir.mkdir()
    model_path = model_dir / "model.onnx"
    tags_path = model_dir / "selected_tags.csv"
    model_path.write_bytes(b"model")
    tags_path.write_text("name,category")

    monkeypatch.setenv("WD14_MODEL_ROOT", str(tmp_path))

    resolved = resolve_local_asset_pair(
        model_key="wd14-convnextv2.v1",
        model_filename="model.onnx",
        tags_filename="selected_tags.csv",
    )

    assert resolved == (model_path, tags_path)


def test_resolve_local_asset_pair_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("WD14_MODEL_ROOT", str(tmp_path))

    resolved = resolve_local_asset_pair(
        model_key="wd14-convnextv2.v1",
        model_filename="model.onnx",
        tags_filename="selected_tags.csv",
    )

    assert resolved is None
