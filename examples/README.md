# Kipp Examples

These examples assume the default converted checkpoint:

```text
models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf
```

Choose another registered checkpoint with `CHECKPOINT=...` when running the
model tooling.

## Command-line generation

Build and run the CPU reference CLI:

```bash
make cpu
build/kipp --backend cpu \
  --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf \
  --prompt "The capital of France is" --decode 16
```

On Apple Silicon, replace `make cpu` with `make metal` and use
`build/kipp-metal --backend metal`.

## Interactive chat (instruct checkpoints)

`--chat` runs a multi-turn REPL through the native Qwen3 ChatML renderer,
reusing the session KV cache across turns (only each turn's new bytes are
evaluated). Type `exit` or press Ctrl-D to quit:

```bash
build/kipp-metal --backend metal \
  --model models/qwen3-4b-instruct-2507/kipp-qwen3-4b-instruct-2507-bf16.gguf \
  --chat --system "You are terse." --temperature 0.7
```

`--no-think` suppresses reasoning on the hybrid instruct checkpoints
(Qwen3-X); the 2507 variants ignore it. `--ctx N` bounds the session KV
(default 8192 tokens), and `--decode N` bounds each turn's reply.

## HTTP server

Start the server on loopback:

```bash
make server
build/kipp-server --backend cpu --port 8080 \
  --model models/qwen3-4b-base/kipp-qwen3-4b-base-bf16.gguf
```

Send a completion request:

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  --data @examples/http/completion.json
```

For a registered instruct checkpoint, send a chat completion:

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data @examples/http/chat-completion.json
```

The chat request uses SSE streaming. Add `-N` to `curl` to print each event as
it arrives. A streamed text completion with a trailing usage chunk:

```bash
curl -N --fail-with-body http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  --data @examples/http/completion-stream.json
```

Top-3 alternatives per generated token (legacy `logprobs` shape):

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  --data @examples/http/completion-logprobs.json
```

Inspect health and Prometheus metrics:

```bash
curl --fail-with-body http://127.0.0.1:8080/healthz
curl --fail-with-body http://127.0.0.1:8080/metrics
```

The server binds only to `127.0.0.1`. Backend selection is explicit and never
falls back to CPU.
