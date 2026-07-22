# Releasing Kipp

The release pattern is the one v0.0.1 and v0.0.2 established: a release
branch, a single `Release vX.Y.Z` commit, a GitHub PR merged into `main`
with a merge commit, and an annotated tag on that merge. Everything below
must be green before the branch is cut.

## Pre-release checklist

1. **Gates, recorded.** On the release candidate commit, on real hardware:
   - `make test` (hermetic units, docs-check, tooling tests, paper-check)
     and `make test-sanitize` (ASan/UBSan).
   - CPU model gates via the built binary: `--model`, `--phase2-model`,
     `--multilogit`, `--paged-cpu`, `--pooled-cpu`, `--fault-reference`.
   - Metal gates: `--metal-operators`, `--phase3-metal`, `--paged-metal`,
     `--multilogit-metal`, `--pooled-metal`.
   - CUDA gates on an ephemeral cloud GPU:
     `KIPP_VERDA_CONFIRM_COST=yes KIPP_VERDA_SSH_KEY_ID=... \
      tools/ops/verda_cuda_gate.sh 2>&1 | tee cuda-gate-$(date +%Y%m%d).log`
     then `python3 tools/ops/collect_cuda_gates.py <log>` and commit the
     resulting `bench/results/cuda-*-gates.json`.
   - State plainly which backends were exercised on which hardware.
2. **Server suite**: `make test-server` (40+ tests) against a freshly built
   server binary.
3. **Benchmarks current**: any result file whose code path changed this
   release is re-measured under the steady-state protocol
   (`bench/README.md`), committed with `dirty: false`, and
   `make paper-data && make paper-check` passes.
4. **CHANGELOG**: retitle `## Unreleased` to `## vX.Y.Z (YYYY-MM-DD)`.
5. **Version bumps**: `KIPP_VERSION` in `src/kipp.h`; `version:` in
   `CITATION.cff`.
6. **Docs**: ROADMAP status block describes the release; drift check
   `grep -rn "not yet wired" docs src *.md` returns only intended lines;
   `make docs && make docs-check` green.
7. **Paper**: if any paper input changed, `cd paper && tectonic main.tex`
   and commit the rebuilt PDF.

## Release mechanics

```bash
git checkout -b release/vX.Y.Z          # from the validated branch tip
# apply items 4-5 above in one commit:
git commit -m "Release vX.Y.Z: <one-line summary>"
git push -u origin release/vX.Y.Z
gh pr create --base main --title "Release vX.Y.Z" --body "<summary + gate results>"
gh pr merge --merge                      # merge commit, matching PRs #1/#2
git checkout main && git pull
git tag -a vX.Y.Z -m "Kipp vX.Y.Z: <summary>"
git push origin vX.Y.Z
```

## Post-merge

- CI on `main` runs `make test`, which includes `paper-check`: every
  `EXPECTED_INPUTS` result JSON (see `tools/paper_data.py`) must have come
  over in the merge, or CI fails. Do not regenerate or prune
  `bench/results/` in the release commit.
- Verify the tag points at the merge commit and the GitHub release page
  (if drafted) links the CHANGELOG section.
