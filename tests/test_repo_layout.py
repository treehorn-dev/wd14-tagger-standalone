from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cpu_dockerfile_exists():
    dockerfile = REPO_ROOT / 'Dockerfile.cpu'
    assert dockerfile.is_file()
    text = dockerfile.read_text()
    assert 'WD14_MODEL_ROOT=/models' in text
    assert 'scripts/fetch-hf-assets.sh' in text
    assert 'ENTRYPOINT ["wd14-tagger"]' in text


def test_hf_fetch_script_exists():
    script = REPO_ROOT / 'scripts' / 'fetch-hf-assets.sh'
    assert script.is_file()
    text = script.read_text()
    assert 'huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/model.onnx' in text
    assert 'selected_tags.csv' in text
