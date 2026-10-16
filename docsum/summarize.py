"""Иерархическое (map-reduce) саммари: фрагменты → документы → вся папка."""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .extract import Document
from .llm import LLM

log = logging.getLogger(__name__)

SYSTEM = (
    "Ты — аналитик, который готовит точные выжимки из документов. "
    "Опирайся только на предоставленный текст, ничего не додумывай. "
    "Сохраняй конкретику: имена, организации, даты, суммы, сроки, номера, обязательства, "
    "требования, решения. Если чего-то в тексте нет — не упоминай это. "
    "Пиши на языке: {lang}. Формат — Markdown."
)

CHUNK_PROMPT = """Это фрагмент {part} из {total} документа «{name}».
Выпиши всё существенное из этого фрагмента в виде сжатого списка фактов (5–15 пунктов).

<<<
{text}
>>>"""

DOC_PROMPT = """Составь саммари документа «{name}».
Структура:
1. **О чём документ** — 1–3 предложения (тип документа, назначение).
2. **Ключевые сведения** — список главных фактов, цифр, дат, участников, условий.
3. **Важное/необычное** — риски, противоречия, сроки, открытые вопросы (если есть).
Объём — не больше {words} слов.

<<<
{text}
>>>"""

DOC_FROM_PARTS_PROMPT = """Ниже — выжимки по частям документа «{name}» (по порядку).
Объедини их в единое саммари документа без повторов.
Структура:
1. **О чём документ** — 1–3 предложения.
2. **Ключевые сведения** — главные факты, цифры, даты, участники, условия.
3. **Важное/необычное** — риски, противоречия, сроки, открытые вопросы (если есть).
Объём — не больше {words} слов.

<<<
{text}
>>>"""

GROUP_PROMPT = """Ниже — саммари группы документов из одной папки.
Сожми их в общую выжимку по группе, сохранив названия документов, ключевые факты и связи между ними.

<<<
{text}
>>>"""

FINAL_PROMPT = """В папке {n_docs} документ(ов). Ниже — саммари каждого из них.
Подготовь ОБЩЕЕ саммари по всему содержимому папки.

Структура ответа:
## Общая картина
Что это за набор документов, к какой задаче/делу/проекту он относится, как документы связаны между собой (3–6 предложений).
## Ключевые темы и выводы
Главные выводы по всей совокупности документов (списком).
## Важные факты
Участники, организации, даты, суммы, сроки, требования — сгруппируй по смыслу, а не по файлам. Указывай в скобках, из какого документа факт.
## Противоречия, риски, пробелы
Расхождения между документами, недостающая информация, на что обратить внимание. Если ничего нет — так и напиши.
## Состав папки
По одной строке на документ: «имя файла — что это».

<<<
{text}
>>>"""


# --------------------------------------------------------------------------- #

def split_text(text: str, max_chars: int, overlap: int = 300) -> list[str]:
    """Режет текст на куски ≤ max_chars по границам абзацев/предложений, с перекрытием."""
    if len(text) <= max_chars:
        return [text]
    pieces = re.split(r"(\n\s*\n)", text)
    units: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            units.append(p)
        else:  # огромный абзац — по предложениям, а затем жёстко
            for s in re.split(r"(?<=[.!?…])\s+", p):
                while len(s) > max_chars:
                    units.append(s[:max_chars])
                    s = s[max_chars:]
                units.append(s + " ")

    chunks, cur = [], ""
    for u in units:
        if len(cur) + len(u) > max_chars and cur.strip():
            chunks.append(cur.strip())
            cur = cur[-overlap:] if overlap else ""
        cur += u
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


@dataclass
class DocSummary:
    doc: Document
    summary: str = ""
    chunks: int = 0
    error: str | None = None


@dataclass
class Result:
    overall: str
    documents: list[DocSummary] = field(default_factory=list)
    skipped: list[Document] = field(default_factory=list)


class Summarizer:
    def __init__(self, llm: LLM, chunk_chars: int = 12_000, context_chars: int = 40_000,
                 doc_words: int = 250, workers: int = 4, lang: str = "русский"):
        self.llm = llm
        self.chunk_chars = chunk_chars
        self.context_chars = context_chars   # сколько текста отправляем в один запрос на reduce
        self.doc_words = doc_words
        self.workers = max(1, workers)
        self.system = SYSTEM.format(lang=lang)
        # пулы вложенные (документы × фрагменты), поэтому общий лимит одновременных запросов
        self._slots = threading.BoundedSemaphore(self.workers)

    def _ask(self, prompt: str) -> str:
        with self._slots:
            return self.llm.complete(self.system, prompt)

    # --- reduce с рекурсией, если не влезает в контекст -------------------- #
    def _reduce(self, items: list[str], final_prompt: str, fmt: dict) -> str:
        joined = "\n\n---\n\n".join(items)
        if len(joined) <= self.context_chars or len(items) == 1:
            return self._ask(final_prompt.format(text=joined, **fmt))
        groups, cur, size = [], [], 0
        for it in items:
            if cur and size + len(it) > self.context_chars:
                groups.append(cur)
                cur, size = [], 0
            cur.append(it)
            size += len(it)
        groups.append(cur)
        log.info("Промежуточное сжатие: %d элементов → %d групп", len(items), len(groups))
        with ThreadPoolExecutor(self.workers) as ex:
            merged = list(ex.map(
                lambda g: self._ask(GROUP_PROMPT.format(text="\n\n---\n\n".join(g))), groups))
        return self._reduce(merged, final_prompt, fmt)

    def summarize_document(self, doc: Document) -> DocSummary:
        chunks = split_text(doc.text, self.chunk_chars)
        res = DocSummary(doc, chunks=len(chunks))
        try:
            if len(chunks) == 1:
                res.summary = self._ask(DOC_PROMPT.format(name=doc.path, text=chunks[0], words=self.doc_words))
            else:
                log.info("%s: %d фрагментов", doc.path, len(chunks))
                with ThreadPoolExecutor(self.workers) as ex:
                    parts = list(ex.map(
                        lambda ic: self._ask(CHUNK_PROMPT.format(
                            name=doc.path, part=ic[0] + 1, total=len(chunks), text=ic[1])),
                        enumerate(chunks)))
                parts = [f"[часть {i + 1}]\n{p}" for i, p in enumerate(parts)]
                res.summary = self._reduce(parts, DOC_FROM_PARTS_PROMPT,
                                           {"name": doc.path, "words": self.doc_words})
        except Exception as e:  # noqa: BLE001
            log.error("Ошибка саммари %s: %s", doc.path, e)
            res.error = str(e)
        return res

    def run(self, docs: list[Document], progress=None) -> Result:
        good = [d for d in docs if d.ok]
        skipped = [d for d in docs if not d.ok]
        if not good:
            raise RuntimeError("Не найдено ни одного документа с текстом")

        summaries: list[DocSummary] = []
        with ThreadPoolExecutor(self.workers) as ex:
            for s in ex.map(self.summarize_document, good):
                summaries.append(s)
                if progress:
                    progress(s)

        ok = [s for s in summaries if not s.error]
        if not ok:
            raise RuntimeError("Не удалось получить саммари ни по одному документу")
        items = [f"### Документ: {s.doc.path}\n{s.summary}" for s in ok]
        overall = self._reduce(items, FINAL_PROMPT, {"n_docs": len(ok)})
        return Result(overall=overall, documents=summaries, skipped=skipped)
