#!/usr/bin/env python3
"""End-to-end tests for the Kipp OpenAI Completions server.

Requires a built server binary and the converted model artifact, so this
suite runs through `make test-server` rather than the model-free `make test`.
The binary and model are selected with KIPP_SERVER_BINARY and
KIPP_SERVER_MODEL; the backend defaults to metal and can be overridden with
KIPP_SERVER_BACKEND.
"""

from __future__ import annotations

import http.client
import json
import os
import concurrent.futures
import pathlib
import signal
import socket
import subprocess
import time
import unittest
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = pathlib.Path(
    os.environ.get("KIPP_SERVER_BINARY", ROOT / "build" / "kipp-server-metal")
)
MODEL = pathlib.Path(
    os.environ.get(
        "KIPP_SERVER_MODEL",
        ROOT / "models" / "qwen3-4b-base" / "kipp-qwen3-4b-base-bf16.gguf",
    )
)
BACKEND = os.environ.get("KIPP_SERVER_BACKEND", "metal")
MODEL_ID = os.environ.get("KIPP_SERVER_MODEL_ID", MODEL.parent.name)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class ServerTests(unittest.TestCase):
    process: subprocess.Popen
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        if not BINARY.is_file():
            raise unittest.SkipTest(f"server binary {BINARY} is not built")
        if not MODEL.is_file():
            raise unittest.SkipTest(f"model artifact {MODEL} is missing")
        port = free_port()
        cls.base = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen(
            [
                str(BINARY),
                "--model",
                str(MODEL),
                "--backend",
                BACKEND,
                "--port",
                str(port),
                "--idle-timeout",
                "5",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError("server exited during startup")
            try:
                if cls.get("/healthz")[0] == 200:
                    return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("server did not become healthy in time")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process.poll() is None:
            cls.process.send_signal(signal.SIGTERM)
            cls.process.wait(timeout=30)

    @classmethod
    def get(cls, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(cls.base + path, timeout=300) as reply:
                return reply.status, json.load(reply)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.load(error)
            finally:
                error.close()

    @classmethod
    def raw_get(cls, path: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(cls.base + path, timeout=300) as reply:
                return reply.status, reply.read().decode()
        except urllib.error.HTTPError as error:
            try:
                return error.code, error.read().decode()
            finally:
                error.close()

    @classmethod
    def post(cls, path: str, body: bytes | dict) -> tuple[int, dict]:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = urllib.request.Request(
            cls.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                return reply.status, json.load(reply)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.load(error)
            finally:
                error.close()

    def completion(self, **fields) -> tuple[int, dict]:
        return self.post("/v1/completions", fields)

    def chat(self, **fields) -> tuple[int, dict]:
        return self.post("/v1/chat/completions", fields)

    def metric_values(self) -> dict[str, int]:
        import re

        status, text = self.raw_get("/metrics")
        self.assertEqual(status, 200)
        return {
            match.group(1): int(match.group(2))
            for match in re.finditer(r"^(kipp_\w+) (\d+)$", text, re.M)
        }

    def test_healthz(self) -> None:
        status, body = self.get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_models(self) -> None:
        status, body = self.get("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], MODEL_ID)

    def test_greedy_completion_schema(self) -> None:
        status, body = self.completion(
            model=MODEL_ID,
            prompt="The capital of France is",
            max_tokens=4,
            temperature=0,
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], MODEL_ID)
        self.assertEqual(body["object"], "text_completion")
        choice = body["choices"][0]
        self.assertIsInstance(choice["text"], str)
        self.assertTrue(choice["text"])
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["finish_reason"], "length")
        usage = body["usage"]
        self.assertEqual(usage["prompt_tokens"], 5)
        self.assertEqual(usage["completion_tokens"], 4)
        self.assertEqual(usage["total_tokens"], 9)

    def test_deterministic_seed(self) -> None:
        request = {
            "prompt": "Once upon a time",
            "max_tokens": 8,
            "temperature": 0.9,
            "top_p": 0.95,
            "seed": 123,
        }
        first = self.completion(**request)
        second = self.completion(**request)
        self.assertEqual(first[0], 200)
        self.assertEqual(
            first[1]["choices"][0]["text"], second[1]["choices"][0]["text"]
        )

    def test_stop_string(self) -> None:
        status, body = self.completion(
            prompt="Count: one, two, three,",
            max_tokens=24,
            temperature=0,
            stop=[" six"],
        )
        self.assertEqual(status, 200)
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("six", choice["text"])
        self.assertIn("five", choice["text"])

    def test_unsupported_field(self) -> None:
        status, body = self.completion(prompt="hi", nonsense_field=1)
        self.assertEqual(status, 400)
        self.assertIn("nonsense_field", body["error"]["message"])

    def test_sampling_fields_accepted(self) -> None:
        status, body = self.completion(
            prompt="The capital of France is",
            max_tokens=4,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            frequency_penalty=0.5,
            presence_penalty=0.2,
            repetition_penalty=1.1,
            logit_bias={"0": -100},
            seed=7,
        )
        self.assertEqual(status, 200)
        self.assertIn("text", body["choices"][0])

    def test_sampling_validation(self) -> None:
        for field, value, needle in [
            ("min_p", 2, "min_p"),
            ("top_k", -1, "top_k"),
            ("frequency_penalty", 5, "frequency_penalty"),
            ("repetition_penalty", 0, "repetition_penalty"),
            ("logit_bias", {"999999999": 1}, "token ids"),
        ]:
            status, body = self.completion(prompt="hi", **{field: value})
            self.assertEqual(status, 400, field)
            self.assertIn(needle, body["error"]["message"], field)

    def test_logprobs_legacy(self) -> None:
        status, body = self.completion(
            prompt="The capital of France is",
            max_tokens=4,
            temperature=0,
            logprobs=3,
        )
        self.assertEqual(status, 200)
        choice = body["choices"][0]
        lp = choice["logprobs"]
        count = body["usage"]["completion_tokens"]
        self.assertEqual(len(lp["tokens"]), count)
        self.assertEqual(len(lp["token_logprobs"]), count)
        self.assertEqual(len(lp["top_logprobs"]), count)
        self.assertEqual(len(lp["text_offset"]), count)
        # Greedy decoding: the chosen token is the argmax, so its logprob is
        # the largest, and every logprob is the log of a probability (<= 0).
        for token_lp, alternatives in zip(
            lp["token_logprobs"], lp["top_logprobs"]
        ):
            self.assertLessEqual(token_lp, 1e-4)
            self.assertLessEqual(len(alternatives), 3)
            for alt_lp in alternatives.values():
                self.assertLessEqual(alt_lp, token_lp + 1e-4)
        # text_offset is non-decreasing and matches the reconstructed text.
        offsets = lp["text_offset"]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual("".join(lp["tokens"]), choice["text"])

    def test_logprobs_chat(self) -> None:
        status, body = self.chat(
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=6,
            temperature=0,
            logprobs=True,
            top_logprobs=2,
        )
        if status == 400 and "base checkpoints" in body["error"]["message"]:
            return  # base checkpoint: no chat template, documented elsewhere
        self.assertEqual(status, 200)
        content = body["choices"][0]["logprobs"]["content"]
        self.assertEqual(len(content), body["usage"]["completion_tokens"])
        for entry in content:
            self.assertIn("token", entry)
            self.assertIsInstance(entry["bytes"], list)
            self.assertLessEqual(entry["logprob"], 1e-4)
            self.assertLessEqual(len(entry["top_logprobs"]), 2)
            for alt in entry["top_logprobs"]:
                self.assertLessEqual(alt["logprob"], entry["logprob"] + 1e-4)

    def test_logprobs_validation(self) -> None:
        status, body = self.completion(prompt="hi", max_tokens=2, logprobs=6)
        self.assertEqual(status, 400)
        self.assertIn("logprobs", body["error"]["message"])
        status, body = self.chat(
            messages=[{"role": "user", "content": "hi"}],
            top_logprobs=21,
            logprobs=True,
        )
        if not (status == 400 and "base checkpoints" in body["error"]["message"]):
            self.assertEqual(status, 400)
            self.assertIn("top_logprobs", body["error"]["message"])
        status, body = self.chat(
            messages=[{"role": "user", "content": "hi"}],
            top_logprobs=2,
        )
        if not (status == 400 and "base checkpoints" in body["error"]["message"]):
            self.assertEqual(status, 400)
            self.assertIn("requires logprobs", body["error"]["message"])

    def test_timings(self) -> None:
        status, body = self.completion(
            prompt="The capital of France is",
            max_tokens=5,
            temperature=0,
        )
        self.assertEqual(status, 200)
        timings = body["timings"]
        self.assertEqual(timings["predicted_n"], body["usage"]["completion_tokens"])
        self.assertEqual(timings["prompt_n"], body["usage"]["prompt_tokens"])
        self.assertGreaterEqual(timings["predicted_ms"], 0.0)
        self.assertGreaterEqual(timings["prompt_ms"], 0.0)

    def test_stream_include_usage(self) -> None:
        _, chunks, saw_done = self.stream_completion(
            prompt="The capital of France is",
            max_tokens=5,
            temperature=0,
            stream_options={"include_usage": True},
        )
        self.assertTrue(saw_done)
        usage_chunks = [c for c in chunks if c.get("usage") is not None]
        self.assertEqual(len(usage_chunks), 1)
        usage_chunk = usage_chunks[-1]
        self.assertEqual(usage_chunk["choices"], [])
        self.assertEqual(usage_chunk["usage"]["completion_tokens"], 5)
        self.assertEqual(usage_chunk["timings"]["predicted_n"], 5)

    def test_streaming_logprobs(self) -> None:
        _, chunks, saw_done = self.stream_completion(
            prompt="The capital of France is",
            max_tokens=5,
            temperature=0,
            logprobs=2,
        )
        self.assertTrue(saw_done)
        seen = []
        for chunk in chunks:
            for choice in chunk["choices"]:
                lp = choice.get("logprobs")
                if lp:
                    seen.extend(lp["tokens"])
        self.assertEqual(len(seen), 5)

    def test_metrics(self) -> None:
        # A completion should move the counters.
        self.completion(prompt="Hi", max_tokens=3, temperature=0)
        status, raw = self.raw_get("/metrics")
        self.assertEqual(status, 200)
        for metric in (
            "kipp_requests_total",
            "kipp_prompt_tokens_total",
            "kipp_generation_tokens_total",
            "kipp_requests_running",
            "kipp_queue_wait_seconds_sum",
            "kipp_queue_wait_seconds_count",
            "kipp_ttft_seconds_sum",
            "kipp_ttft_seconds_count",
            "kipp_decode_seconds_sum",
        ):
            self.assertIn(metric, raw)
        line = [
            row for row in raw.splitlines()
            if row.startswith("kipp_requests_total ")
        ][0]
        self.assertGreaterEqual(int(line.split()[1]), 1)

        def sample(name: str) -> float:
            row = [
                r for r in raw.splitlines() if r.startswith(name + " ")
            ][0]
            return float(row.split()[1])

        self.assertGreaterEqual(sample("kipp_ttft_seconds_count"), 1)
        self.assertGreater(sample("kipp_ttft_seconds_sum"), 0.0)
        self.assertGreaterEqual(sample("kipp_queue_wait_seconds_count"), 1)

    def test_chat_completion(self) -> None:
        status, body = self.chat(
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=8,
            temperature=0,
        )
        if status == 400 and "base checkpoints" in body["error"]["message"]:
            # A base checkpoint has no chat template; that rejection is the
            # correct behavior and this assertion documents it.
            return
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        message = body["choices"][0]["message"]
        self.assertEqual(message["role"], "assistant")
        self.assertIsInstance(message["content"], str)

    def test_invalid_json(self) -> None:
        status, body = self.post("/v1/completions", b"{oops")
        self.assertEqual(status, 400)
        self.assertIn("valid JSON", body["error"]["message"])

    def test_missing_prompt(self) -> None:
        status, body = self.completion(max_tokens=2)
        self.assertEqual(status, 400)
        self.assertIn("prompt is required", body["error"]["message"])

    def stream_completion(self, **fields) -> tuple[str, list[dict], bool]:
        """POST with stream=true; returns (content_type, chunks, saw_done)."""
        fields["stream"] = True
        connection = http.client.HTTPConnection(
            self.base.removeprefix("http://"), timeout=300
        )
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(fields),
            headers={"Content-Type": "application/json"},
        )
        reply = connection.getresponse()
        content_type = reply.getheader("Content-Type", "")
        raw = reply.read().decode("utf-8")
        connection.close()
        chunks = []
        saw_done = False
        for frame in raw.split("\n\n"):
            if not frame.startswith("data: "):
                continue
            payload = frame[len("data: "):]
            if payload == "[DONE]":
                saw_done = True
                continue
            chunks.append(json.loads(payload))
        return content_type, chunks, saw_done

    def test_streaming(self) -> None:
        content_type, chunks, saw_done = self.stream_completion(
            prompt="The capital of France is", max_tokens=6, temperature=0
        )
        self.assertIn("text/event-stream", content_type)
        self.assertTrue(saw_done)
        self.assertGreaterEqual(len(chunks), 2)
        text = "".join(c["choices"][0]["text"] for c in chunks)
        self.assertIn("Paris", text)
        self.assertIsNone(chunks[0]["choices"][0]["finish_reason"])
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")

    def test_streaming_holds_back_stop_text(self) -> None:
        _, chunks, saw_done = self.stream_completion(
            prompt="Count: one, two, three,",
            max_tokens=24,
            temperature=0,
            stop=[" six"],
        )
        self.assertTrue(saw_done)
        text = "".join(c["choices"][0]["text"] for c in chunks)
        self.assertNotIn("six", text)
        self.assertIn("five", text)
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_prefix_reuse_consistency(self) -> None:
        base = {"max_tokens": 6, "temperature": 0}
        first = self.completion(prompt="Kipp is a small engine that", **base)
        self.completion(
            prompt="Kipp is a small engine that runs one pinned model on",
            **base,
        )
        third = self.completion(prompt="Kipp is a small engine that", **base)
        self.assertEqual(first[0], 200)
        self.assertEqual(
            first[1]["choices"][0]["text"], third[1]["choices"][0]["text"]
        )

    def test_multiple_choices_greedy(self) -> None:
        single = self.completion(
            prompt="The capital of France is", max_tokens=4, temperature=0
        )
        status, body = self.completion(
            prompt="The capital of France is", max_tokens=4, temperature=0,
            n=3,
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["choices"]), 3)
        for index, choice in enumerate(body["choices"]):
            self.assertEqual(choice["index"], index)
            self.assertEqual(
                choice["text"], single[1]["choices"][0]["text"]
            )
        self.assertEqual(body["usage"]["completion_tokens"], 12)

    def test_choice_limit(self) -> None:
        status, body = self.completion(prompt="hi", n=9)
        self.assertEqual(status, 400)
        self.assertIn("between 1 and 8", body["error"]["message"])

    def test_choice_boundary_accepted(self) -> None:
        status, body = self.completion(
            prompt="hi", n=8, max_tokens=2, temperature=0
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(body["choices"]), 8)

    def test_stop_boundary(self) -> None:
        four = ["alpha", "beta", "gamma", "delta"]
        status, _ = self.completion(
            prompt="hi", max_tokens=2, temperature=0, stop=four
        )
        self.assertEqual(status, 200)
        status, body = self.completion(
            prompt="hi", max_tokens=2, temperature=0, stop=four + ["epsilon"]
        )
        self.assertEqual(status, 400)
        self.assertIn("stop", body["error"]["message"])

    def test_logit_bias_boundary(self) -> None:
        bias = {str(token): 1.0 for token in range(64)}
        status, _ = self.completion(
            prompt="hi", max_tokens=2, temperature=0, logit_bias=bias
        )
        self.assertEqual(status, 200)
        bias[str(64)] = 1.0
        status, body = self.completion(
            prompt="hi", max_tokens=2, temperature=0, logit_bias=bias
        )
        self.assertEqual(status, 400)
        self.assertIn("logit_bias", body["error"]["message"])

    def test_logit_bias_forces_token(self) -> None:
        # +100 on <|endoftext|> dominates every greedy step, so the first
        # sampled token is the stop token: empty text, finish "stop".
        status, body = self.completion(
            prompt="The capital of France is",
            max_tokens=8,
            temperature=0,
            logit_bias={"151643": 100.0},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["text"], "")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_streaming_multiple_choices(self) -> None:
        _, chunks, saw_done = self.stream_completion(
            prompt="The capital of France is", max_tokens=4, temperature=0,
            n=2,
        )
        self.assertTrue(saw_done)
        texts = {0: "", 1: ""}
        finishes = {}
        for chunk in chunks:
            choice = chunk["choices"][0]
            texts[choice["index"]] += choice["text"]
            if choice["finish_reason"] is not None:
                finishes[choice["index"]] = choice["finish_reason"]
        self.assertEqual(texts[0], texts[1])
        self.assertIn("Paris", texts[0])
        self.assertEqual(finishes, {0: "length", 1: "length"})

    def test_context_limit(self) -> None:
        status, body = self.completion(prompt="hi", max_tokens=1_000_000)
        self.assertEqual(status, 400)
        self.assertIn("max_tokens", body["error"]["message"])

    def test_concurrent_requests_batch_together(self) -> None:
        request = {
            "prompt": "The capital of France is",
            "max_tokens": 6,
            "temperature": 0,
        }
        serial = self.completion(**request)
        self.assertEqual(serial[0], 200)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(self.completion, **request) for _ in range(3)
            ]
            results = [f.result(timeout=300) for f in futures]
        for status, body in results:
            self.assertEqual(status, 200)
            self.assertEqual(
                body["choices"][0]["text"], serial[1]["choices"][0]["text"]
            )

    def test_more_than_eight_concurrent_decode(self) -> None:
        # 12 concurrent n=1 requests must all decode together now that the
        # generation cap matches the 32-item batch limit (it was 8).
        request = {
            "prompt": "The capital of France is",
            "max_tokens": 48,
            "temperature": 0,
        }
        serial = self.completion(**request)
        self.assertEqual(serial[0], 200)
        running_high = {"value": 0}
        stop_polling = {"value": False}

        def poll_running() -> None:
            import re

            while not stop_polling["value"]:
                try:
                    _, text = self.raw_get("/metrics")
                    match = re.search(
                        r"^kipp_requests_running (\d+)$", text, re.M
                    )
                    if match:
                        running_high["value"] = max(
                            running_high["value"], int(match.group(1))
                        )
                except OSError:
                    pass
                time.sleep(0.05)

        import threading

        poller = threading.Thread(target=poll_running)
        poller.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=12
            ) as pool:
                futures = [
                    pool.submit(self.completion, **request)
                    for _ in range(12)
                ]
                results = [f.result(timeout=300) for f in futures]
        finally:
            stop_polling["value"] = True
            poller.join(timeout=10)
        for status, body in results:
            self.assertEqual(status, 200)
            self.assertEqual(
                body["choices"][0]["text"], serial[1]["choices"][0]["text"]
            )
        self.assertGreaterEqual(running_high["value"], 9)

    def test_idle_partial_header_is_reaped(self) -> None:
        # A slow-loris client that sends part of a request line and stalls
        # must be disconnected once the idle timeout (5s here) elapses.
        host, port = self.base.replace("http://", "").split(":")
        with socket.create_connection((host, int(port)), timeout=20) as raw:
            raw.sendall(b"GET /healthz HTT")
            raw.settimeout(20)
            deadline = time.monotonic() + 15.0
            closed = False
            while time.monotonic() < deadline:
                try:
                    if raw.recv(64) == b"":
                        closed = True
                        break
                except socket.timeout:
                    break
            self.assertTrue(closed, "server kept the stalled socket open")

    def test_idle_silent_socket_is_reaped(self) -> None:
        host, port = self.base.replace("http://", "").split(":")
        with socket.create_connection((host, int(port)), timeout=20) as raw:
            raw.settimeout(20)
            deadline = time.monotonic() + 15.0
            closed = False
            while time.monotonic() < deadline:
                try:
                    if raw.recv(64) == b"":
                        closed = True
                        break
                except socket.timeout:
                    break
            self.assertTrue(closed, "server kept the silent socket open")

    def test_concurrent_mixed_prompts(self) -> None:
        prompts = [
            "The capital of France is",
            "Water is composed of hydrogen and",
            "Once upon a time",
        ]
        singles = [
            self.completion(prompt=p, max_tokens=5, temperature=0)
            for p in prompts
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    self.completion, prompt=p, max_tokens=5, temperature=0
                )
                for p in prompts
            ]
            results = [f.result(timeout=300) for f in futures]
        for single, (status, body) in zip(singles, results):
            self.assertEqual(status, 200)
            self.assertEqual(
                body["choices"][0]["text"],
                single[1]["choices"][0]["text"],
            )

    def test_disconnect_mid_stream_recovers(self) -> None:
        connection = http.client.HTTPConnection(
            self.base.removeprefix("http://"), timeout=300
        )
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(
                {
                    "prompt": "Tell me a very long story about",
                    "max_tokens": 200,
                    "temperature": 0,
                    "stream": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        reply = connection.getresponse()
        reply.read(64)  # receive a little, then vanish
        connection.close()
        status, body = self.completion(
            prompt="The capital of France is", max_tokens=3, temperature=0
        )
        self.assertEqual(status, 200)
        self.assertIn("Paris", body["choices"][0]["text"])

    def test_unknown_path_and_method(self) -> None:
        status, _ = self.get("/v2/whatever")
        self.assertEqual(status, 404)
        request = urllib.request.Request(
            self.base + "/healthz", method="DELETE"
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                status = reply.status
        except urllib.error.HTTPError as error:
            try:
                status = error.code
            finally:
                error.close()
        self.assertEqual(status, 405)

    # ------------------------------------------------ pooled prefix sharing

    POOL_PROMPT = (
        "the quick brown fox jumps over the lazy dog and " * 9
    )  # 100+ tokens: several whole 32-token blocks publish at release

    def test_pool_cross_request_reuse(self) -> None:
        first = self.completion(
            prompt=self.POOL_PROMPT, max_tokens=8, temperature=0
        )
        self.assertEqual(first[0], 200)
        before = self.metric_values()
        second = self.completion(
            prompt=self.POOL_PROMPT, max_tokens=8, temperature=0
        )
        self.assertEqual(second[0], 200)
        after = self.metric_values()
        self.assertEqual(
            first[1]["choices"][0]["text"], second[1]["choices"][0]["text"]
        )
        self.assertGreater(
            after["kipp_prefix_tokens_reused_total"],
            before["kipp_prefix_tokens_reused_total"],
        )
        self.assertGreater(
            after["kipp_kv_pool_reused_blocks_total"],
            before["kipp_kv_pool_reused_blocks_total"],
        )

    def test_pool_multi_choice(self) -> None:
        single = self.completion(
            prompt=self.POOL_PROMPT, max_tokens=6, temperature=0
        )
        status, body = self.completion(
            prompt=self.POOL_PROMPT, max_tokens=6, temperature=0, n=4
        )
        self.assertEqual(status, 200)
        texts = {choice["text"] for choice in body["choices"]}
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts.pop(), single[1]["choices"][0]["text"])

    def test_pool_disconnect_releases(self) -> None:
        idle_free = self.metric_values()["kipp_kv_pool_blocks_free"]
        connection = http.client.HTTPConnection(
            self.base.removeprefix("http://"), timeout=300
        )
        connection.request(
            "POST",
            "/v1/completions",
            body=json.dumps(
                {
                    "prompt": self.POOL_PROMPT,
                    "max_tokens": 200,
                    "temperature": 0,
                    "stream": True,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        reply = connection.getresponse()
        reply.read(64)  # receive a little, then vanish
        connection.close()
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if (
                self.metric_values()["kipp_kv_pool_blocks_free"]
                >= idle_free
            ):
                break
            time.sleep(0.5)
        self.assertGreaterEqual(
            self.metric_values()["kipp_kv_pool_blocks_free"], idle_free
        )
        before = self.metric_values()["kipp_kv_pool_reused_blocks_total"]
        status, body = self.completion(
            prompt=self.POOL_PROMPT, max_tokens=4, temperature=0
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["choices"][0]["text"])
        self.assertGreater(
            self.metric_values()["kipp_kv_pool_reused_blocks_total"], before
        )


class TinyPoolTests(unittest.TestCase):
    """Pool pressure: admission is delayed, never corrupted.

    A second server with a pool of only a few blocks forces concurrent
    requests to queue behind the block reservation. Every request must
    complete with its isolated-run output; delayed admission shows up as
    latency, never as a 5xx or wrong text.
    """

    process: subprocess.Popen
    base: str

    # One 4B block is 4.5 MiB of KV; 18 MiB = 4 blocks, enough for one
    # ~40-token request (2 blocks) plus one more reserving behind it.
    POOL_MIB = 18
    PROMPTS = [
        "The capital of France is famous for its museums and",
        "Water is composed of hydrogen and oxygen, which means",
        "Once upon a time in a village by the sea there",
        "The theory of relativity fundamentally changed how we",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        if not BINARY.is_file():
            raise unittest.SkipTest(f"server binary {BINARY} is not built")
        if not MODEL.is_file():
            raise unittest.SkipTest(f"model artifact {MODEL} is missing")
        port = free_port()
        cls.base = f"http://127.0.0.1:{port}"
        cls.process = subprocess.Popen(
            [
                str(BINARY),
                "--model",
                str(MODEL),
                "--backend",
                BACKEND,
                "--port",
                str(port),
                "--kv-pool-mib",
                str(cls.POOL_MIB),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError("tiny-pool server exited during startup")
            try:
                with urllib.request.urlopen(
                    cls.base + "/healthz", timeout=2
                ):
                    return
            except OSError:
                time.sleep(0.25)
        raise RuntimeError("tiny-pool server did not become healthy")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process.poll() is None:
            cls.process.send_signal(signal.SIGTERM)
            cls.process.wait(timeout=30)

    def completion(self, **fields) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base + "/v1/completions",
            data=json.dumps(fields).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                return reply.status, json.load(reply)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.load(error)
            finally:
                error.close()

    def test_pressure_delays_but_never_corrupts(self) -> None:
        isolated = []
        for prompt in self.PROMPTS:
            status, body = self.completion(
                prompt=prompt, max_tokens=8, temperature=0
            )
            self.assertEqual(status, 200)
            isolated.append(body["choices"][0]["text"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    self.completion,
                    prompt=prompt,
                    max_tokens=8,
                    temperature=0,
                )
                for prompt in self.PROMPTS
            ]
            results = [f.result(timeout=300) for f in futures]
        for expected, (status, body) in zip(isolated, results):
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["text"], expected)


if __name__ == "__main__":
    unittest.main()
