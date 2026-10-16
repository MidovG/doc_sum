"""Командная строка: python -m docsum <папка или ссылка на Google Drive> [опции]."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .extract import load_folder
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
    p.add_argument("--chunk-chars", type=int, default=12_000, help="размер фрагмента документа, символов")
    p.add_argument("--context-chars", type=int, default=40_000,
                   help="сколько символов саммари отправлять в один запрос при объединении")
    p.add_argument("--doc-words", type=int, default=250, help="лимит слов на саммари одного документа")
    p.add_argument("--workers", type=int, default=4, help="параллельных запросов к LLM")
    p.add_argument("--max-tokens", type=int, default=2000, help="лимит токенов ответа модели")
    p.add_argument("--cache-dir", default=".docsum_cache", help="кэш загрузок и ответов LLM")
    p.add_argument("--no-cache", action="store_true", help="не кэшировать ответы LLM")
    p.add_argument("--json", dest="json_out", help="дополнительно сохранить результат в JSON")
    p.add_argument("--extract-only", action="store_true",
                   help="только извлечь текст и показать статистику, без LLM")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"docsum {__version__}")
    return p


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
        lines.append(f"_{d.doc.kind} · {chars} символов · фрагментов: {d.chunks}_")
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

    import os
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
    ok = [d for d in docs if d.ok]
    log.info("Файлов: %d, с текстом: %d, всего символов: %s",
             len(docs), len(ok), f"{sum(len(d.text) for d in ok):,}".replace(",", " "))
    for d in docs:
        if not d.ok:
            log.warning("Пропуск %s: %s", d.path, d.error)

    if args.extract_only:
        for d in ok:
            print(f"{d.path}\t{d.kind}\t{len(d.text)}")
        return 0
    if not ok:
        log.error("В папке не найдено документов с текстом")
        return 1

    llm = make_llm(provider, model, base_url=args.base_url, api_key=args.api_key,
                   max_tokens=args.max_tokens,
                   cache_dir=None if args.no_cache else cache / "llm")
    log.info("LLM: %s / %s", llm.name, llm.model)

    summarizer = Summarizer(llm, chunk_chars=args.chunk_chars, context_chars=args.context_chars,
                            doc_words=args.doc_words, workers=args.workers, lang=args.lang)
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
                           "chunks": d.chunks, "summary": d.summary, "error": d.error}
                          for d in res.documents],
            "skipped": [{"path": d.path, "reason": d.error} for d in res.skipped],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("Готово за %.1fс → %s", time.monotonic() - t0, out.resolve())
    print("\n" + res.overall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
