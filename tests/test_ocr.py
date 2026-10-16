import io

import pytest
from conftest import make_text_image

from docsum import ocr as ocr_mod
from docsum.extract import Document, ImagePart
from docsum.ocr import OCRUnavailable, TesseractOCR, recognize, skip_images

has_tesseract = ocr_mod._find_tesseract() is not None
needs_tesseract = pytest.mark.skipif(not has_tesseract, reason="Tesseract не установлен")


def png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------- настоящий Tesseract --

@needs_tesseract
def test_reads_text_from_image():
    text, conf = TesseractOCR("eng").read(ImagePart("img", png(make_text_image("INVOICE 42 TOTAL 1000"))))
    assert "INVOICE" in text and "1000" in text
    assert conf > 60


@needs_tesseract
@pytest.mark.parametrize("angle", [90, 270])
def test_rotated_scan_is_straightened(angle):
    ocr = TesseractOCR("eng")
    im = make_text_image("QUARTERLY REPORT FOR THE BOARD", size=(1400, 300)).rotate(angle, expand=True)
    text, _ = ocr.read(ImagePart("img", png(im)))
    assert "QUARTERLY" in text and "BOARD" in text


@needs_tesseract
def test_short_rotated_text_without_osd_is_found_by_trying_angles(monkeypatch):
    ocr = TesseractOCR("eng")
    monkeypatch.setattr(ocr, "_rotation", lambda im: None)     # OSD «не справился»
    im = make_text_image("ACT OF ACCEPTANCE SIGNED", size=(1200, 200)).rotate(90, expand=True)
    text, conf = ocr.read(ImagePart("img", png(im)))
    assert "ACCEPTANCE" in text and conf > 60


@needs_tesseract
def test_missing_language_falls_back_to_available():
    assert TesseractOCR("zzz+eng").lang == "eng"
    with pytest.raises(OCRUnavailable, match="нет языков"):
        TesseractOCR("zzz")


def test_unavailable_tesseract_gives_install_hint(monkeypatch):
    monkeypatch.setattr(ocr_mod, "_find_tesseract", lambda: None)
    with pytest.raises(OCRUnavailable, match="brew install tesseract"):
        TesseractOCR()


# ------------------------------------------------- сборка текста (без Tesseract)

class FakeOCR:
    name = "fake-ocr"

    def __init__(self, answers):
        self.answers = answers           # label -> (text, conf) | Exception

    def read(self, image):
        a = self.answers[image.label]
        if isinstance(a, Exception):
            raise a
        return a


def test_recognize_keeps_page_order_and_marks_low_confidence():
    doc = Document("mixed.pdf", "pdf", parts=[
        "[стр. 1]\nТекстовая страница",
        ImagePart("стр. 2", b""),
        ImagePart("стр. 3", b""),        # фото без текста — пропускается
        ImagePart("стр. 4", b""),
    ])
    recognize([doc], FakeOCR({
        "стр. 2": ("Акт приёмки работ подписан сторонами", 91.0),
        "стр. 3": ("~ ,", 20.0),
        "стр. 4": ("Плохо различимый текст скана", 30.0),
    }), workers=1)

    assert doc.ok
    t = doc.text
    assert t.index("Текстовая страница") < t.index("Акт приёмки") < t.index("Плохо различимый")
    assert "[стр. 2 — распознано OCR]" in t
    assert "низкое качество — 30%" in t
    assert "стр. 3" not in t
    assert doc.meta["ocr_pages"] == 2 and doc.meta["ocr_confidence"] == 60


def test_recognize_errors():
    failed = Document("a.png", "png", parts=[ImagePart("изображение", b"")])
    empty = Document("b.png", "png", parts=[ImagePart("изображение", b"")])
    recognize([failed], FakeOCR({"изображение": RuntimeError("boom")}), workers=1)
    recognize([empty], FakeOCR({"изображение": ("", 0.0)}), workers=1)
    assert failed.error == "не удалось распознать изображение"
    assert empty.error == "на изображении не найден текст"


def test_skip_images_keeps_text_of_mixed_pdf():
    mixed = Document("m.pdf", "pdf", text="текст", parts=["текст", ImagePart("стр. 2", b"")])
    only_img = Document("p.png", "png", parts=[ImagePart("изображение", b"")])
    skip_images([mixed, only_img], "OCR выключен")
    assert mixed.ok and mixed.meta["images_skipped"] == 1
    assert only_img.error == "OCR выключен"


@pytest.mark.skipif(not has_tesseract, reason="Tesseract не установлен")
def test_russian_scan():
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont

    ocr_mod.TesseractOCR("eng")  # настраивает путь к tesseract
    if "rus" not in pytesseract.get_languages(config=""):
        pytest.skip("нет русского языкового пакета Tesseract")
    font_path = next((p for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                  "/Library/Fonts/Arial Unicode.ttf", "C:/Windows/Fonts/arial.ttf"]
                      if __import__("os").path.exists(p)), None)
    if font_path is None:
        pytest.skip("нет шрифта с кириллицей")
    im = Image.new("RGB", (1200, 200), "white")
    ImageDraw.Draw(im).text((30, 60), "Договор поставки на сумму 250 000 рублей",
                            fill="black", font=ImageFont.truetype(font_path, 48))
    text, conf = TesseractOCR("rus+eng").read(ImagePart("img", png(im)))
    assert "Договор" in text and "рублей" in text and conf > 60
