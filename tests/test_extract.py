from pathlib import Path

from docsum.extract import MAX_IMAGE_SIDE, extract_bytes, load_folder


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
    # изображения не отбрасываются, а ждут распознавания
    assert docs["photo.png"].pending_images and len(docs["photo.png"].images) == 1
    scan = docs[str(Path("sub/scan.pdf"))]
    assert scan.pending_images and [i.label for i in scan.images] == ["стр. 1"]
    assert not docs["broken.jpg"].ok and "повреждён" in docs["broken.jpg"].error


def test_mixed_pdf_keeps_text_pages_and_renders_scans():
    import io

    import pymupdf as fitz
    from conftest import make_text_image

    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Обычная текстовая страница с достаточным количеством текста.",
                               fontname="helv")
    buf = io.BytesIO()
    make_text_image("PAGE TWO").save(buf, "PNG")
    p2 = pdf.new_page()
    p2.insert_image(p2.rect, stream=buf.getvalue())
    pdf.new_page()                                   # пустая страница — пропускается
    [doc] = list(extract_bytes("mixed.pdf", pdf.tobytes()))

    assert "[стр. 1]" in doc.text
    assert [i.label for i in doc.images] == ["стр. 2"]
    assert isinstance(doc.parts[0], str)             # порядок страниц сохранён


def test_image_is_normalized_to_png():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (6000, 1500), (255, 0, 0, 128)).save(buf, "PNG")
    [doc] = list(extract_bytes("big.png", buf.getvalue()))
    with Image.open(io.BytesIO(doc.images[0].data)) as im:
        assert im.format == "PNG" and im.mode == "RGB" and max(im.size) == MAX_IMAGE_SIDE


def test_multipage_tiff_split_into_pages():
    import io

    from PIL import Image

    buf = io.BytesIO()
    frames = [Image.new("RGB", (100, 100), c) for c in ("white", "gray")]
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    [doc] = list(extract_bytes("scan.tiff", buf.getvalue()))
    assert [i.label for i in doc.images] == ["стр. 1", "стр. 2"]


def test_broken_file_does_not_crash():
    [doc] = list(extract_bytes("bad.pdf", b"not a pdf"))
    assert not doc.ok and doc.error
