"""Командная строка: python -m docsum <папка или ссылка на Google Drive> [опции]."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .extract import load_folder
from .ocr import OCRUnavailable, TesseractOCR, recognize, skip_images
from .llm import DEFAULT_MODELS, PROVIDERS, make_llm
from .sources import resolve_source
from .summarize import Result, Summarizer

log = logging.getLogger("docsum")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="docsum",
        description="Общее саммари по всем документам в папке (локальной или Google Drive).",
    )
    p.add_argument("source", help="путь к папке/файлу/ZIP или ссылка на публичную папку Google Drive")
    p.add_argument("-o", "--output", default="summary.md", help="куда сохранить отчёт (по умолчанию summary.md)")
    p.add_argument("--provider", choices=sorted(PROVIDERS), default=None,
                   help="LLM-провайдер (по умолчанию из DOCSUM_PROVIDER или openai)")
    p.add_argument("--model", help="модель; по умолчанию: " +
                   ", ".join(f"{k}={v}" for k, v in DEFAULT_MODELS.items()))
    p.add_argument("--base-url", help="URL OpenAI-совместимого API или Ollama")
    p.add_argument("--api-key", help="API-ключ (лучше через переменную окружения)")
    p.add_argument("--lang", default="русский", help="язык саммари (по умолчанию русский)")
    p.add_argument("--chunk-chars", type=int, default=None,
                   help="размер фрагмента документа, символов (по умолчанию 12000; для Ollama — по --num-ctx)")
    p.add_argument("--context-chars", type=int, default=None,
                   help="символов в одном запросе при объединении (по умолчанию 40000; для Ollama — по --num-ctx)")
    p.add_argument("--num-ctx", type=int, default=None,
                   help="Ollama: размер контекста модели в токенах (по умолчанию 16384 или OLLAMA_NUM_CTX)")
    p.add_argument("--doc-words", type=int, default=250, help="лимит слов на саммари одного документа")
    p.add_argument("--workers", type=int, default=None,
                   help="параллельных запросов к LLM (по умолчанию 4; для Ollama — 1)")
    p.add_argument("--ocr-lang", default="rus+eng",
                   help="языки Tesseract OCR для изображений и сканов (по умолчанию rus+eng)")
    p.add_argument("--no-ocr", action="store_true", help="не распознавать изображения и сканы PDF")
    p.add_argument("--max-tokens", type=int, default=2000, help="лимит токенов ответа модели")
    p.add_argument("--cache-dir", default=".docsum_cache", help="кэш загрузок и ответов LLM")
    p.add_argument("--no-cache", action="store_true", help="не кэшировать ответы LLM")
    p.add_argument("--json", dest="json_out", help="дополнительно сохранить результат в JSON")
    p.add_argument("--extract-only", action="store_true",
                   help="только извлечь текст и показать статистику, без LLM")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"docsum {__version__}")
    return p


CHARS_PER_TOKEN = 2.5   # консервативно для кириллицы (у латиницы ~4)
PROMPT_OVERHEAD = 600   # системный промпт + инструкции, токенов


def sizes_for(provider: str, num_ctx: int, args) -> tuple[int, int, int]:
    """Размер фрагмента, объём на объединение и число потоков.

    Для облачных моделей контекст большой — берём фиксированные значения.
    Для Ollama считаем от num_ctx: вход + ответ модели должны влезть в контекст,
    иначе Ollama обрежет промпт.
    """
    if provider == "ollama":
        budget = max(1000, num_ctx - args.max_tokens - PROMPT_OVERHEAD)
        auto_context = int(budget * CHARS_PER_TOKEN)
        auto_chunk = min(12_000, auto_context)
        auto_workers = 1  # Ollama по умолчанию обрабатывает запросы к модели по очереди
    else:
        auto_context, auto_chunk, auto_workers = 40_000, 12_000, 4
    return (args.chunk_chars or auto_chunk,
            args.context_chars or auto_context,
            args.workers or auto_workers)


def render_markdown(res: Result, source: str, model: str) -> str:
    lines = [
        "# Саммари содержимого папки",
        "",
        f"_Источник: {source} · модель: {model} · {datetime.now():%Y-%m-%d %H:%M}_",
        f"_Обработано документов: {sum(1 for d in res.documents if not d.error)}"
        f" · пропущено: {len(res.skipped) + sum(1 for d in res.documents if d.error)}_",
        "",
        res.overall.strip(),
        "",
        "---",
        "",
        "# Саммари по документам",
        "",
    ]
    for d in res.documents:
        lines.append(f"## {d.doc.path}")
        chars = f"{d.doc.meta.get('chars', 0):,}".replace(",", " ")
        extra = ""
        if d.doc.meta.get("ocr"):
            conf = d.doc.meta.get("ocr_confidence")
            extra = (f" · OCR: {d.doc.meta['ocr_pages']} изобр."
                     + (f", уверенность {conf}%" if conf is not None else ""))
        if d.doc.meta.get("images_skipped"):
            extra += f" · страниц-сканов пропущено: {d.doc.meta['images_skipped']}"
        lines.append(f"_{d.doc.kind} · {chars} символов · фрагментов: {d.chunks}{extra}_")
        lines.append("")
        lines.append(f"> Ошибка: {d.error}" if d.error else d.summary.strip())
        lines.append("")
    if res.skipped:
        lines += ["---", "", "## Не обработаны", ""]
        lines += [f"- `{d.path}` — {d.error}" for d in res.skipped]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows / при перенаправлении
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    if not args.verbose:  # приглушаем INFO-логи HTTP-клиентов SDK
        for h in logging.getLogger().handlers:
            h.addFilter(lambda r: r.name.startswith("docsum") or r.levelno >= logging.WARNING)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    provider = args.provider or os.getenv("DOCSUM_PROVIDER", "openai")
    model = args.model or os.getenv("DOCSUM_MODEL")
    cache = Path(args.cache_dir)

    t0 = time.monotonic()
    try:
        root = resolve_source(args.source, cache)
    except Exception as e:  # noqa: BLE001
        log.error("%s", e)
        return 2

    docs = load_folder(root)
    pending = [d for d in docs if d.pending_images]
    for d in docs:
        if d.error:
            log.warning("Пропуск %s: %s", d.path, d.error)

    if args.extract_only:
        for d in docs:
            if d.ok or d.pending_images:
                imgs = f"\tизображений: {len(d.images)}" if d.images else ""
                print(f"{d.path}\t{d.kind}\t{len(d.text)}{imgs}")
        return 0
    if not any(d.ok or d.pending_images for d in docs):
        log.error("В папке не найдено документов с текстом")
        return 1

    num_ctx = args.num_ctx or int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    llm = make_llm(provider, model, base_url=args.base_url, api_key=args.api_key,
                   max_tokens=args.max_tokens, num_ctx=num_ctx if provider == "ollama" else None,
                   cache_dir=None if args.no_cache else cache / "llm")
    if hasattr(llm, "check"):
        try:
            llm.check()
        except RuntimeError as e:
            log.error("%s", e)
            return 2

    chunk_chars, context_chars, workers = sizes_for(provider, num_ctx, args)

    # --- изображения и страницы-сканы: Tesseract OCR до саммари ---
    if pending:
        n_img = sum(len(d.images) for d in pending)
        ocr, reason = None, "OCR выключен (--no-ocr)"
        if not args.no_ocr:
            try:
                ocr = TesseractOCR(args.ocr_lang)
            except OCRUnavailable as e:
                reason = f"нужен OCR — {e}"
        if ocr is None:
            log.warning("Изображений: %d — пропускаю: %s", n_img, reason)
            skip_images(docs, reason)
        else:
            log.info("OCR: %d изображений в %d файлах · %s", n_img, len(pending), ocr.name)
            recognize(docs, ocr, workers=os.cpu_count() or 2,
                      progress=lambda i, n: log.info("  OCR %d/%d", i, n))
            for d in pending:
                if d.error:
                    log.warning("Пропуск %s: %s", d.path, d.error)

    ok = [d for d in docs if d.ok]
    log.info("Файлов: %d, с текстом: %d, всего символов: %s",
             len(docs), len(ok), f"{sum(len(d.text) for d in ok):,}".replace(",", " "))
    if not ok:
        log.error("В папке не найдено документов с текстом")
        return 1
    log.info("LLM: %s / %s · фрагмент %d симв. · объединение %d симв. · потоков %d%s",
             llm.name, llm.model, chunk_chars, context_chars, workers,
             f" · num_ctx {num_ctx}" if provider == "ollama" else "")

    summarizer = Summarizer(llm, chunk_chars=chunk_chars, context_chars=context_chars,
                            doc_words=args.doc_words, workers=workers, lang=args.lang)
    done = [0]

    def progress(s):
        done[0] += 1
        log.info("[%d/%d] %s%s", done[0], len(ok), s.doc.path, f" — ошибка: {s.error}" if s.error else "")

    try:
        res = summarizer.run(docs, progress=progress)
    except Exception as e:  # noqa: BLE001
        log.error("%s", e)
        return 1

    out = Path(args.output)
    out.write_text(render_markdown(res, args.source, f"{llm.name}/{llm.model}"), encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "source": args.source,
            "model": f"{llm.name}/{llm.model}",
            "overall": res.overall,
            "documents": [{"path": d.doc.path, "kind": d.doc.kind, "chars": len(d.doc.text),
                           "chunks": d.chunks, "summary": d.summary, "error": d.error,
                           "ocr": d.doc.meta.get("ocr"),
                           "ocr_confidence": d.doc.meta.get("ocr_confidence")}
                          for d in res.documents],
            "skipped": [{"path": d.path, "reason": d.error} for d in res.skipped],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Готово за %.1fс → %s", time.monotonic() - t0, out.resolve())
    print("\n" + res.overall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
