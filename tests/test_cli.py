import json

from tagger import cli


def test_root_command_returns_command_tree(capsys):
    exit_code = cli.main([])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"]["parsed"]["path"] == []
    assert payload["result"]["description"] == "WD14 tagger CLI"
    assert any(command["name"] == "tag" for command in payload["result"]["commands"])
    assert payload["next_actions"]


def test_tag_file_command_returns_json_envelope(monkeypatch, capsys, tmp_path):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake")

    def fake_execute_tag(parsed):
        assert parsed["path"] == ["tag"]
        assert parsed["options"]["file"] == str(image_path)
        return {
            "mode": "file",
            "model": parsed["options"]["model"],
            "tags": ["1girl", "solo"],
            "tag_scores": {"1girl": 0.98, "solo": 0.95},
        }

    monkeypatch.setattr(cli, "execute_tag", fake_execute_tag)

    exit_code = cli.main(["tag", "--file", str(image_path)])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"]["parsed"]["path"] == ["tag"]
    assert payload["result"]["mode"] == "file"
    assert payload["result"]["tags"] == ["1girl", "solo"]
    assert payload["next_actions"]


def test_tag_error_returns_structured_error(monkeypatch, capsys, tmp_path):
    image_path = tmp_path / "missing.jpg"

    def fake_execute_tag(parsed):
        raise FileNotFoundError(parsed["options"]["file"])

    monkeypatch.setattr(cli, "execute_tag", fake_execute_tag)

    exit_code = cli.main(["tag", "--file", str(image_path)])
    assert exit_code == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "FILE_NOT_FOUND"
    assert str(image_path) in payload["error"]["message"]
    assert payload["fix"]
