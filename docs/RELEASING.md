# Releasing Kipp

The release pattern established by v0.0.1–v0.0.3 is a release
branch, a single `Release vX.Y.Z` commit, a GitHub PR merged into `main`
with a merge commit, and an annotated tag on that merge. Everything below
must be green before the branch is cut.

## Pre-release checklist

1. **Gates, recorded.** On the release candidate commit, on real hardware:
   - `make test` (hermetic units, docs drift, and tooling tests) and
     `make test-sanitize` (ASan/UBSan).
   - CPU model gates via the built binary: `--model`, `--phase2-model`,
     `--multilogit`, `--paged-cpu`, `--pooled-cpu`, `--fault-reference`.
   - Metal gates: `--metal-operators`, `--phase3-metal`, `--paged-metal`,
     `--multilogit-metal`, `--pooled-metal`, `--longctx-metal`.
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
   (`bench/README.md`) and committed with `dirty: false`.
4. **CHANGELOG**: retitle `## Unreleased` to `## vX.Y.Z (YYYY-MM-DD)`.
5. **Version bump**: update `KIPP_VERSION` in `src/kipp.h`.
6. **Docs**: ROADMAP status block describes the release; drift check
   `grep -rn "not yet wired" docs src *.md` returns only intended lines;
   `make docs && make docs-check` green.

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

- CI on `main` runs `make test`; verify the native, tooling, and docs-drift
  gates pass on the merge commit.
- Do not regenerate or prune `bench/results/` in the release commit.
- Verify the tag points at the merge commit and the GitHub release page
  (if drafted) links the CHANGELOG section.
