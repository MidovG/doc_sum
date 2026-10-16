"""Распознавание текста на изображениях и страницах-сканах PDF через Tesseract OCR."""

from __future__ import annotations

import io
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .extract import Document, ImagePart, normalize

log = logging.getLogger(__name__)

LOW_CONFIDENCE = 60      # % — ниже пробуем развернуть изображение и распознать ещё раз
WARN_CONFIDENCE = 45     # % — ниже помечаем текст для LLM как ненадёжный
MIN_WORDS = 2            # меньше осмысленных слов — считаем, что текста на изображении нет

INSTALL_HINT = (
    "macOS: `brew install tesseract tesseract-lang`; "
    "Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-rus`; "
    "Windows: установщик https://github.com/UB-Mannheim/tesseract/wiki (отметить Russian)"
)
WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


class OCRUnavailable(RuntimeError):
    pass


class TesseractOCR:
    def __init__(self, lang: str = "rus+eng"):
        try:
            import pytesseract
        except ImportError as e:  # pragma: no cover
            raise OCRUnavailable("не установлен пакет pytesseract: pip install pytesseract") from e
        self.tess = pytesseract

        cmd = _find_tesseract()
        if cmd is None:
            raise OCRUnavailable(f"Tesseract не установлен. {INSTALL_HINT}")
        pytesseract.pytesseract.tesseract_cmd = cmd

        installed = set(pytesseract.get_languages(config=""))
        wanted = [x for x in lang.split("+") if x]
        have = [x for x in wanted if x in installed]
        missing = [x for x in wanted if x not in installed]
        if not have:
            raise OCRUnavailable(f"в Tesseract нет языков {lang} (установлены: "
                                 f"{', '.join(sorted(installed))}). {INSTALL_HINT}")
        if missing:
            log.warning("Tesseract: нет языковых пакетов %s — распознаю только на %s. %s",
                        "+".join(missing), "+".join(have), INSTALL_HINT)
        self.lang = "+".join(have)
        self.can_rotate = "osd" in installed
        self.version = str(pytesseract.get_tesseract_version())

    @property
    def name(self) -> str:
        return f"Tesseract {self.version} ({self.lang})"

    # ------------------------------------------------------------------ #
    def read(self, image: ImagePart) -> tuple[str, float]:
        """Возвращает (текст, средняя уверенность 0–100)."""
        from PIL import Image

        with Image.open(io.BytesIO(image.data)) as im:
            prepared = _preprocess(im)
        text, conf = self._ocr(prepared)
        if conf >= LOW_CONFIDENCE:
            return text, conf
        # Низкая уверенность — возможно, скан повёрнут. Сначала пробуем угол из OSD Tesseract,
        # но он ошибается (особенно на кириллице и коротком тексте), поэтому если результат
        # не стал хорошим — перебираем остальные повороты и берём лучший по уверенности.
        osd = self._rotation(prepared)
        angles = ([osd] if osd else []) + [a for a in (90, 270, 180) if a != osd]
        for angle in angles:
            text2, conf2 = self._ocr(prepared.rotate(angle, expand=True))  # PIL: против часовой, как в OSD
            log.debug("%s: поворот %d°, уверенность %.0f → %.0f", image.label, angle, conf, conf2)
            if conf2 > conf:
                text, conf = text2, conf2
            if conf >= 85:
                break
        return text, conf

    def _ocr(self, im) -> tuple[str, float]:
        """Распознаёт и собирает строки из слов, считая среднюю уверенность по словам."""
        d = self.tess.image_to_data(im, lang=self.lang, config="--oem 1 --psm 3",
                                    output_type=self.tess.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(d["text"]):
            conf = float(d["conf"][i])
            if conf < 0 or not word.strip():
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(conf)
        out, prev_block = [], None
        for (block, par, _), words in lines.items():
            if prev_block is not None and (block, par) != prev_block:
                out.append("")                      # пустая строка между абзацами
            out.append(" ".join(words))
            prev_block = (block, par)
        mean = sum(confs) / len(confs) if confs else 0.0
        return "\n".join(out), mean

    def _rotation(self, im) -> int | None:
        """Угол поворота по OSD; None — определить не удалось (нет osd или мало текста)."""
        if not self.can_rotate:
            return None
        try:
            osd = self.tess.image_to_osd(im, output_type=self.tess.Output.DICT)
            return int(osd.get("rotate", 0))
        except self.tess.TesseractError:
            return None


def _meaningful_words(text: str) -> int:
    """Слова хотя бы из двух букв/цифр: отсекает шум вроде «~ , |» на фотографиях."""
    return sum(1 for w in text.split() if sum(c.isalnum() for c in w) >= 2)


def _find_tesseract() -> str | None:
    for cand in [os.getenv("TESSERACT_CMD"), shutil.which("tesseract"), *WINDOWS_PATHS]:
        if cand and Path(cand).exists():
            return cand
    return None


def _preprocess(im):
    """Оттенки серого, автоконтраст, увеличение мелких изображений (Tesseract любит ~300 dpi)."""
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(im).convert("L")
    if max(im.size) < 1500:
        k = 1500 / max(im.size)
        im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    return ImageOps.autocontrast(im, cutoff=1)


# --------------------------------------------------------------------------- #

def recognize(docs: list[Document], ocr: TesseractOCR, workers: int = 2, progress=None) -> None:
    """Распознаёт все изображения во всех документах и пересобирает их текст по порядку."""
    jobs = [(d, p) for d in docs if d.pending_images for p in d.images]
    if not jobs:
        return

    def run(job):
        doc, part = job
        try:
            return ocr.read(part)
        except Exception as e:  # noqa: BLE001 — одна плохая картинка не ломает прогон
            log.warning("OCR не удался: %s (%s): %s", doc.path, part.label, e)
            return e

    if workers > 1:
        # параллелим по изображениям, поэтому внутренние потоки Tesseract (OpenMP) отключаем,
        # иначе процессы делят ядра между собой и всё работает медленнее
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    results: dict[int, tuple[str, float] | Exception] = {}
    with ThreadPoolExecutor(max(1, workers)) as ex:   # tesseract — отдельный процесс, GIL не мешает
        for n, ((_, part), res) in enumerate(zip(jobs, ex.map(run, jobs)), 1):
            results[id(part)] = res
            if progress:
                progress(n, len(jobs))

    for doc in {id(d): d for d, _ in jobs}.values():
        chunks, recognized, failed, confs = [], 0, 0, []
        own_text = any(isinstance(p, str) and p.strip() for p in doc.parts)
        for part in doc.parts:                        # исходный порядок страниц сохраняется
            if isinstance(part, str):
                chunks.append(part)
                continue
            res = results[id(part)]
            if isinstance(res, Exception):
                failed += 1
                chunks.append(f"[{part.label}: не удалось распознать]")
                continue
            text, conf = res
            text = normalize(text)
            if _meaningful_words(text) < MIN_WORDS:
                continue                              # фото/рисунок без текста, шум OCR
            recognized += 1
            confs.append(conf)
            note = f", низкое качество — {conf:.0f}%, возможны ошибки" if conf < WARN_CONFIDENCE else ""
            chunks.append(f"[{part.label} — распознано OCR{note}]\n{text}")
        doc.text = normalize("\n\n".join(chunks))
        doc.parts = [doc.text]
        doc.meta.update(chars=len(doc.text), ocr=ocr.name, ocr_pages=recognized,
                        ocr_confidence=round(sum(confs) / len(confs)) if confs else None)
        if not recognized and not own_text:
            doc.error = "не удалось распознать изображение" if failed else "на изображении не найден текст"


def skip_images(docs: list[Document], reason: str) -> None:
    """OCR недоступен: документы только из изображений уходят в «Не обработаны»."""
    for d in docs:
        if not d.pending_images:
            continue
        if d.text.strip():   # смешанный PDF: текстовые страницы есть, сканы пропускаем
            d.meta["images_skipped"] = len(d.images)
        else:
            d.error = reason
        d.parts = [p for p in d.parts if isinstance(p, str)]
