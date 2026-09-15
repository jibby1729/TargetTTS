#!/usr/bin/env bash
# Train a configurable KenLM model on cleaned WAXAL Twi text.
#
# Does not download data. Expects:
#   /home/ubuntu/kenlm/build/bin/lmplz
#   data/kenlm/waxal_aka_train_val.txt
#
# Usage (from repo root):
#   bash scripts/train_twi_kenlm.sh
#   TEXT=data/kenlm/waxal_aka_train.txt \
#   ARPA=data/kenlm/waxal_aka_train_5gram.arpa \
#   BINARY=data/kenlm/waxal_aka_train_5gram.klm \
#     bash scripts/train_twi_kenlm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LMPLZ="${LMPLZ:-/home/ubuntu/kenlm/build/bin/lmplz}"
BUILD_BINARY="${BUILD_BINARY:-/home/ubuntu/kenlm/build/bin/build_binary}"
TEXT="${TEXT:-${ROOT}/data/kenlm/waxal_aka_train_val.txt}"
ARPA="${ARPA:-${ROOT}/data/kenlm/waxal_aka_5gram.arpa}"
BINARY="${BINARY:-}"
ORDER="${ORDER:-5}"

if [[ ! -x "${LMPLZ}" ]]; then
  echo "error: lmplz not found or not executable: ${LMPLZ}" >&2
  echo "Build KenLM first (see /home/ubuntu/kenlm/build/bin/lmplz)." >&2
  exit 1
fi

if [[ ! -f "${TEXT}" ]]; then
  echo "error: training text not found: ${TEXT}" >&2
  echo "Run: python3 -m scripts.prepare_twi_lm_text" >&2
  exit 1
fi

mkdir -p "$(dirname "${ARPA}")"
if [[ -n "${BINARY}" ]]; then
  if [[ ! -x "${BUILD_BINARY}" ]]; then
    echo "error: build_binary not found or not executable: ${BUILD_BINARY}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${BINARY}")"
fi

echo "lmplz: ${LMPLZ}"
echo "text:  ${TEXT}"
echo "arpa:  ${ARPA}"
echo "order: ${ORDER}"

"${LMPLZ}" -o "${ORDER}" --discount_fallback \
  --text "${TEXT}" \
  --arpa "${ARPA}"

if [[ -n "${BINARY}" ]]; then
  echo "binary: ${BINARY}"
  "${BUILD_BINARY}" probing "${ARPA}" "${BINARY}"
fi
