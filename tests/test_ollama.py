import json
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from docsum.cli import main, sizes_for
from docsum.llm import Ollama, _ollama_base


@pytest.fixture()
def fake_ollama():
    """Мини-сервер с эндпоинтами Ollama /api/tags и /api/chat; запоминает запросы."""
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"models": [{"name": "qwen2.5:7b", "model": "qwen2.5:7b"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(req)
            if req["model"] != "qwen2.5:7b":
                return self._send({"error": f"model '{req['model']}' not found"}, 404)
            self._send({"model": req["model"], "done": True, "prompt_eval_count": 100,
                        "message": {"role": "assistant",
                                    "content": "<think>размышления</think>- факт из документа"}})

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def test_num_ctx_is_sent_and_think_stripped(fake_ollama):
    url, requests = fake_ollama
    llm = Ollama("qwen2.5:7b", base_url=url, num_ctx=8192, max_tokens=500)
    assert llm.complete("sys", "user") == "- факт из документа"
    opts = requests[0]["options"]
    assert opts["num_ctx"] == 8192 and opts["num_predict"] == 500
    assert requests[0]["stream"] is False


def test_check_reports_missing_model_and_dead_server(fake_ollama):
    url, _ = fake_ollama
    Ollama("qwen2.5:7b", base_url=url).check()  # не падает
    with pytest.raises(RuntimeError, match="ollama pull qwen3:8b"):
        Ollama("qwen3:8b", base_url=url).check()
    with pytest.raises(RuntimeError, match="Ollama недоступна"):
        Ollama("qwen2.5:7b", base_url="http://127.0.0.1:9").check()


def test_unknown_model_is_not_retried(fake_ollama, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("404 не должен повторяться"))
    url, requests = fake_ollama
    with pytest.raises(Exception):
        Ollama("missing", base_url=url).complete("s", "u")
    assert len(requests) == 1


@pytest.mark.parametrize("raw, expected", [
    ("http://localhost:11434", "http://localhost:11434"),
    ("http://localhost:11434/v1/", "http://localhost:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("192.168.1.10:11434", "http://192.168.1.10:11434"),
])
def test_ollama_base_url_normalization(raw, expected):
    assert _ollama_base(raw) == expected


def test_sizes_follow_num_ctx():
    args = Namespace(max_tokens=2000, chunk_chars=None, context_chars=None, workers=None)
    chunk, context, workers = sizes_for("ollama", 8192, args)
    assert workers == 1
    assert context / 2.5 + 2000 + 600 <= 8192      # вход + ответ влезают в контекст
    assert chunk <= context
    big = sizes_for("ollama", 32768, args)
    assert big[1] > context                         # больше контекст — больше объём
    assert sizes_for("openai", 8192, args) == (12_000, 40_000, 4)
    args.chunk_chars, args.workers = 3000, 3        # явные значения пользователя важнее
    assert sizes_for("ollama", 8192, args)[::2] == (3000, 3)


def test_cli_with_ollama(sample_folder, tmp_path, fake_ollama):
    url, requests = fake_ollama
    out = tmp_path / "summary.md"
    code = main([str(sample_folder), "--provider", "ollama", "--model", "qwen2.5:7b",
                 "--base-url", url, "--num-ctx", "8192", "-o", str(out), "--no-cache"])
    assert code == 0
    assert "факт из документа" in out.read_text(encoding="utf-8")
    assert requests and all(r["options"]["num_ctx"] == 8192 for r in requests)


def test_cli_ollama_missing_model(sample_folder, tmp_path, fake_ollama):
    url, requests = fake_ollama
    code = main([str(sample_folder), "--provider", "ollama", "--model", "llama3",
                 "--base-url", url, "-o", str(tmp_path / "s.md")])
    assert code == 2 and not requests              # упали до первого запроса к модели
