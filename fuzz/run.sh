#!/bin/sh
# Usage: fuzz/run.sh mcap|cdr|video [libFuzzer options]
# Requires atheris; lz4 and zstandard reach compressed MCAP chunks.
# New inputs go to $FUZZ_WORK/<target>/corpus, crashes to $FUZZ_WORK/<target>/.
set -e
fuzz=$(cd "$(dirname "$0")" && pwd)
target=$1
shift
work=${FUZZ_WORK:-${TMPDIR:-/tmp}/rosbag2_dataset_fuzz}/$target
mkdir -p "$work/corpus"
PYTHONPATH="$fuzz/..${PYTHONPATH:+:$PYTHONPATH}" exec "${PYTHON:-python3}" \
  "$fuzz/fuzz_$target.py" -artifact_prefix="$work/" -rss_limit_mb=2048 -timeout=10 "$@" \
  "$work/corpus" "$fuzz/corpus/$target"
