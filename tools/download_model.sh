#!/bin/sh
#
# Download and verify artifacts for Kipp's selected model family.
# The source, revision, checksums, and destination will be fixed after model
# selection; this script intentionally performs no download yet.

set -eu

echo "No Kipp target model has been selected; nothing to download." >&2
exit 1
