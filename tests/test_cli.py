from docsum.cli import main
from docsum.ocr import _find_tesseract


def test_cli_end_to_end(sample_folder, tmp_path):
    out, js = tmp_path / "summary.md", tmp_path / "summary.json"
    code = main([str(sample_folder), "--provider", "fake", "-o", str(out), "--json", str(js),
                 "--cache-dir", str(tmp_path / "cache")])
    assert code == 0
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# Саммари содержимого папки")
    assert "## contract.docx" in md and "## Не обработаны" in md and "broken.jpg" in md
    if _find_tesseract():   # картинка и скан распознаны OCR и попали в саммари
        assert "## photo.png" in md and "OCR: 1 изобр." in md
        assert "## sub/scan.pdf" in md.replace("\\", "/")
    else:                   # без Tesseract — в «Не обработаны» с подсказкой по установке
        assert "`photo.png` — нужен OCR" in md
    assert js.exists()


def test_cli_no_ocr(sample_folder, tmp_path):
    out = tmp_path / "s.md"
    assert main([str(sample_folder), "--provider", "fake", "--no-ocr", "-o", str(out),
                 "--no-cache"]) == 0
    assert "`photo.png` — OCR выключен" in out.read_text(encoding="utf-8")


def test_cli_missing_path(tmp_path):
    assert main([str(tmp_path / "nope"), "--provider", "fake"]) == 2
