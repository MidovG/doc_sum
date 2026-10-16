from pathlib import Path

from docsum.extract import extract_bytes, load_folder


def test_load_folder_reads_all_formats(sample_folder):
    docs = {d.path: d for d in load_folder(sample_folder)}

    assert "Ромашка" in docs["contract.docx"].text
    assert "500 000 руб." in docs["contract.docx"].text           # таблица docx
    assert "№ 17" in docs["cp1251.txt"].text                       # кодировка cp1251
    assert "1000 USD" in docs[str(Path("sub/invoice.pdf"))].text
    assert "Январь | 120" in docs[str(Path("sub/report.xlsx"))].text
    assert "Q2: launch" in docs["deck.pptx"].text
    assert "перенесена" in docs["archive.zip/inner/notes.md"].text  # zip распакован
    assert not any("__MACOSX" in p or "~$" in p for p in docs)       # мусор пропущен
    assert not docs["photo.png"].ok and "OCR" in docs["photo.png"].error


def test_broken_file_does_not_crash():
    [doc] = list(extract_bytes("bad.pdf", b"not a pdf"))
    assert not doc.ok and doc.error
