from docsum.cli import main


def test_cli_end_to_end(sample_folder, tmp_path):
    out, js = tmp_path / "summary.md", tmp_path / "summary.json"
    code = main([str(sample_folder), "--provider", "fake", "-o", str(out), "--json", str(js),
                 "--cache-dir", str(tmp_path / "cache")])
    assert code == 0
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# Саммари содержимого папки")
    assert "## contract.docx" in md and "## Не обработаны" in md and "photo.png" in md
    assert js.exists()


def test_cli_missing_path(tmp_path):
    assert main([str(tmp_path / "nope"), "--provider", "fake"]) == 2
