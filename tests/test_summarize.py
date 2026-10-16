from docsum.extract import Document, load_folder
from docsum.llm import LLM
from docsum.summarize import Summarizer, split_text


def test_split_text_respects_limit_and_keeps_content():
    text = "\n\n".join(f"Абзац {i}. " + "слово " * 50 for i in range(40))
    chunks = split_text(text, 1000, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 1100 for c in chunks)
    for i in range(40):
        assert any(f"Абзац {i}." in c for c in chunks)


def test_split_text_handles_giant_paragraph():
    chunks = split_text("а" * 5000, 1000, overlap=0)
    assert all(len(c) <= 1000 for c in chunks) and sum(map(len, chunks)) >= 5000


# ------------------------------------------------------------------ pipeline --

class RecordingLLM(LLM):
    name = "rec"

    def __init__(self):
        super().__init__("rec")
        self.calls: list[str] = []

    def _complete(self, system, user):
        self.calls.append(user)
        return f"саммари #{len(self.calls)}"



def test_map_reduce_for_long_document():
    llm = RecordingLLM()
    long_doc = Document("big.txt", "txt", text="\n\n".join("Пункт договора. " * 30 for _ in range(30)))
    res = Summarizer(llm, chunk_chars=2000, context_chars=10**6, workers=2).run([long_doc])

    chunk_calls = [c for c in llm.calls if c.startswith("Это фрагмент")]
    assert len(chunk_calls) == res.documents[0].chunks > 1
    assert any("Объедини их" in c for c in llm.calls)      # reduce по документу
    assert llm.calls[-1].startswith("В папке 1")             # финальное саммари


def test_hierarchical_reduce_when_context_is_small(sample_folder):
    llm = RecordingLLM()
    docs = load_folder(sample_folder)
    Summarizer(llm, context_chars=30, workers=3).run(docs)
    assert any(c.startswith("Ниже — саммари группы") for c in llm.calls)
