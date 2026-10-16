import io
import zipfile
from pathlib import Path

import pytest

@pytest.fixture()
def sample_folder(tmp_path: Path) -> Path:
    import docx
    import pymupdf as fitz
    import openpyxl
    from pptx import Presentation

    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)

    (root / "readme.txt").write_text("Проект «Альфа». Срок сдачи — 1 декабря 2026 года.", encoding="utf-8")
    (root / "cp1251.txt").write_bytes("Договор поставки № 17 от 03.03.2026.".encode("cp1251"))

    d = docx.Document()
    d.add_paragraph("Договор оказания услуг между ООО «Ромашка» и ИП Иванов.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Сумма", "500 000 руб."
    t.cell(1, 0).text, t.cell(1, 1).text = "Срок", "30 дней"
    d.save(root / "contract.docx")

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Invoice 42. Total due: 1000 USD.")
    pdf.save(root / "sub" / "invoice.pdf")

    wb = openpyxl.Workbook()
    wb.active.append(["Месяц", "Выручка"])
    wb.active.append(["Январь", 120])
    wb.save(root / "sub" / "report.xlsx")

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Roadmap"
    s.placeholders[1].text = "Q1: MVP. Q2: launch."
    prs.save(root / "deck.pptx")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner/notes.md", "# Заметки\nВстреча с заказчиком перенесена.")
        zf.writestr("__MACOSX/._junk", "x")
    (root / "archive.zip").write_bytes(buf.getvalue())

    (root / "photo.png").write_bytes(b"\x89PNG\r\n")
    (root / "~$lock.docx").write_bytes(b"junk")
    return root
