# Security Policy

## Threat model

Kipp's server is a **hand-written C11 HTTP server with its own HTTP and JSON
parsers**, which is worth stating plainly rather than burying: memory-safety
bugs in it are exploitable, and the parsers are the most exposed surface in the
project.

It is therefore built for a specific deployment shape:

- **Loopback only.** The server binds `127.0.0.1` and there is no option to
  bind a public interface.
- **No authentication and no TLS.** Anything that can reach the port can use
  the model.
- **One trust domain.** Requests are not isolated from each other beyond the
  KV cache's own correctness boundaries.

Putting Kipp directly on a network, or in front of untrusted clients, is
outside what the project supports. If you need that, terminate TLS and
authenticate in a reverse proxy in front of it, and keep the Kipp port
loopback-bound.

Model weights are a second trust boundary. The loader validates a checkpoint
against a compiled-in registry and rejects unknown architectures, tensor
names, shapes, dtypes, offsets, and alignment before inference — a GGUF
outside the registry is a rejection, not an extension point. That validation is
fuzz-tested (`test_gguf_reject_fuzz`), but it is a parser reading an
attacker-controllable file, so treat converting or loading an untrusted
checkpoint as you would any other untrusted input.

## Supported versions

Fixes land on `main` and ship in the next tagged release. Only the latest
release is supported; there are no backports.

## Reporting a vulnerability

Please report privately rather than opening a public issue:

1. Use GitHub's [private vulnerability
   reporting](https://github.com/Polished-Snow/Kipp/security/advisories/new)
   for this repository, or
2. contact the maintainers through the links in the README if that is
   unavailable.

Useful things to include: the affected version or commit, the backend and
platform, a request or input that reproduces it, and what you observed
(crash, hang, out-of-bounds read or write, unexpected memory growth). A
sanitizer trace from `make test-sanitize` is ideal but not required.

Expect an acknowledgement within a week. Since this is a small project, please
allow reasonable time for a fix before public disclosure, and tell us if you
have a disclosure deadline so we can plan around it.

## What already runs against this surface

- `make test-sanitize` builds the suite with AddressSanitizer and
  UndefinedBehaviorSanitizer and runs it in CI on every push.
- The JSON and HTTP parsers live in their own translation units with
  deterministic fuzz corpora (`test_json_parse_fuzz`, `test_http_header_fuzz`),
  which run under the sanitizers.
- The GGUF loader has a rejection fuzzer (`test_gguf_reject_fuzz`) plus
  malformed-file tests.
- The server suite (`make test-server`) covers oversized and malformed
  requests, unsupported fields, disconnects mid-stream, and cache-pressure
  admission.

There is no continuous fuzzing (OSS-Fuzz or a libFuzzer corpus) yet; the fuzz
tests are fixed-seed and run as unit tests. That is a known gap.
