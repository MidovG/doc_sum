from docsum.llm import LLM, make_llm


def test_llm_cache(tmp_path):
    llm = make_llm("fake", cache_dir=tmp_path)
    a = llm.complete("s", "<<<Один. Два.>>>")
    assert list(tmp_path.iterdir())
    assert llm.complete("s", "<<<Один. Два.>>>") == a


def test_retry_on_transient_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)

    class Flaky(LLM):
        n = 0

        def _complete(self, system, user):
            Flaky.n += 1
            if Flaky.n < 3:
                raise ConnectionError("boom")
            return "ok"

    assert Flaky("m").complete("s", "u") == "ok" and Flaky.n == 3
