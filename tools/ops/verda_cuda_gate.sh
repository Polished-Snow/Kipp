#!/bin/sh
#
# Provision a disposable Verda GPU, run Kipp's CUDA correctness gates, and
# delete both the VM and its OS volume. This command creates billable cloud
# resources. It refuses to start unless KIPP_VERDA_CONFIRM_COST=yes is set.
#
# Usage:
#   KIPP_VERDA_CONFIRM_COST=yes KIPP_VERDA_SSH_KEY_ID=... \
#     tools/ops/verda_cuda_gate.sh [checkpoint-id ...]
#
# Environment:
#   KIPP_VERDA_INSTANCE_TYPE  default: 1A100.22V
#   KIPP_VERDA_LOCATION       default: FIN-01
#   KIPP_VERDA_IMAGE          default: ubuntu-24.04-cuda-12.8-open
#   KIPP_VERDA_VOLUME_GB      default: 400
#   KIPP_VERDA_SSH_KEY        default: ~/.ssh/verda_h100
#   KIPP_KEEP_VM=1            skip teardown for manual debugging

set -eu

if [ "${KIPP_VERDA_CONFIRM_COST:-no}" != "yes" ]; then
    echo "refusing to create a billable VM; set KIPP_VERDA_CONFIRM_COST=yes" >&2
    exit 2
fi

for command in verda ssh rsync python3 mktemp; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "missing required command: $command" >&2
        exit 2
    }
done

CHECKPOINTS=${*:-"qwen3-4b-base qwen3-0.6b-base qwen3-4b-instruct-2507 qwen3-8b-base"}
SSH_KEY=${KIPP_VERDA_SSH_KEY:-$HOME/.ssh/verda_h100}
SSH_KEY_ID=${KIPP_VERDA_SSH_KEY_ID:?set KIPP_VERDA_SSH_KEY_ID}
INSTANCE_TYPE=${KIPP_VERDA_INSTANCE_TYPE:-1A100.22V}
LOCATION=${KIPP_VERDA_LOCATION:-FIN-01}
IMAGE=${KIPP_VERDA_IMAGE:-ubuntu-24.04-cuda-12.8-open}
VOLUME_GB=${KIPP_VERDA_VOLUME_GB:-400}
HOST=kipp-cuda-gate-$$
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
KNOWN_HOSTS=$(mktemp "${TMPDIR:-/tmp}/kipp-verda-known-hosts.XXXXXX")
REMOTE=root
VM_ID=
VOLUME_ID=
IP=

log() {
    printf '\n=== %s ===\n' "$1"
}

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [ -z "$VM_ID" ]; then
        VM_ID=$(verda vm list --agent -o json 2>/dev/null |
            python3 -c '
import json, sys
name = sys.argv[1]
match = next((item["id"] for item in json.load(sys.stdin)
              if item.get("hostname") == name), "")
print(match)
' "$HOST" 2>/dev/null || true)
    fi
    if [ -z "$VOLUME_ID" ]; then
        VOLUME_ID=$(verda volume list --agent -o json 2>/dev/null |
            python3 -c '
import json, sys
name = sys.argv[1] + "-os"
match = next((item["id"] for item in json.load(sys.stdin)
              if item.get("name") == name), "")
print(match)
' "$HOST" 2>/dev/null || true)
    fi
    if [ "${KIPP_KEEP_VM:-0}" = "1" ] && [ -n "$VM_ID" ]; then
        echo "KIPP_KEEP_VM=1; leaving $HOST ($VM_ID, $IP) running" >&2
    else
        if [ -n "$VM_ID" ]; then
            log "tearing down $HOST ($VM_ID)"
            verda vm delete "$VM_ID" --yes --agent >/dev/null 2>&1 || true
        fi
        if [ -n "$VOLUME_ID" ]; then
            # VM deletion can leave a briefly detached paid OS volume.
            attempts=0
            while [ "$attempts" -lt 6 ]; do
                if ! verda volume list --agent -o json 2>/dev/null |
                    python3 -c '
import json, sys
target = sys.argv[1]
raise SystemExit(0 if any(item.get("id") == target
                          for item in json.load(sys.stdin)) else 1)
' "$VOLUME_ID"; then
                    break
                fi
                if verda volume delete "$VOLUME_ID" --yes --agent \
                    >/dev/null 2>&1; then
                    break
                fi
                attempts=$((attempts + 1))
                sleep 5
            done
        fi
    fi
    rm -f "$KNOWN_HOSTS"
    exit "$status"
}
trap cleanup EXIT INT TERM

remote() {
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
        -o UserKnownHostsFile="$KNOWN_HOSTS" -o ConnectTimeout=20 \
        "$REMOTE@$IP" "$@"
}

log "provisioning $INSTANCE_TYPE in $LOCATION ($IMAGE, ${VOLUME_GB}GB)"
CREATED=$(verda vm create --kind gpu --instance-type "$INSTANCE_TYPE" \
    --location "$LOCATION" --os "$IMAGE" --os-volume-size "$VOLUME_GB" \
    --hostname "$HOST" --ssh-key "$SSH_KEY_ID" --contract pay_as_go \
    --wait --wait-timeout 10m --agent -o json)
VM_ID=$(printf '%s' "$CREATED" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
VOLUME_ID=$(printf '%s' "$CREATED" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["os_volume_id"])')
IP=$(printf '%s' "$CREATED" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["ip"] or "")')
if [ -z "$IP" ]; then
    echo "Verda returned no IP for $VM_ID" >&2
    exit 1
fi
echo "instance $HOST ($VM_ID) at $IP"

log "waiting for ssh"
until remote true 2>/dev/null; do
    sleep 5
done

log "syncing repository and pinned vectors"
rsync -az \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS" \
    --exclude=models/ --exclude=build/ --exclude=.git/ \
    --exclude=tools/.venv/ --exclude=__pycache__ \
    "$ROOT/" "$REMOTE@$IP:/root/kipp/"
remote "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1"
remote "apt-get install -y python3-dev >/dev/null 2>&1 || true"

log "building CUDA targets"
remote "cd /root/kipp && make build/kipp_test_cuda NVCC=/usr/local/cuda/bin/nvcc"

for id in $CHECKPOINTS; do
    log "gating $id"
    case "$id" in
        qwen3-32b) VEC_FLAGS="--device cuda --dtype bfloat16" ;;
        *) VEC_FLAGS="" ;;
    esac
    remote "cd /root/kipp && export PATH=/root/.local/bin:\$PATH && \
        tools/download_model.sh --checkpoint $id && \
        uv run --project tools --python 3.12 \
            python tools/convert_to_gguf.py --checkpoint $id && \
        ([ -f tests/test-vectors/$id/manifest.json ] || \
            uv run --project tools --python 3.12 \
                python tools/generate_test_vectors.py \
                --checkpoint $id $VEC_FLAGS) && \
        G=models/$id/kipp-$id-bf16.gguf && \
        V=tests/test-vectors/$id && \
        build/kipp_test_cuda --model \$G \$V && \
        build/kipp_test_cuda --phase4-cuda \$G \$V"
done

log "operator tests"
remote "cd /root/kipp && build/kipp_test_cuda --cuda-operators"

log "done"
