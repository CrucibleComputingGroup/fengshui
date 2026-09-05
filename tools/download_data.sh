#!/usr/bin/env bash
# Fengshui AE — fetch the precomputed Timeloop database + side databases from
# Zenodo. Reviewers run this once (or -v mount / gunzip a local copy instead).
#
# Default destination is the repo's src/ directory, because that is where the
# notebook and every script look for the database:
#   notebooks/reproduce_all.ipynb -> HAVE_DB = exists(<scripts>/../unified_database.csv)
#   src/scripts/*                 -> default --database ../unified_database.csv
# i.e. the DB must end up at <repo>/src/unified_database.csv. Pass a different
# directory as $1 only if you know what you are doing (the notebook will then
# report the DB as ABSENT and skip every database-backed cell).
set -euo pipefail

ZENODO_RECORD="${ZENODO_RECORD:-21524898}"          # published record
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/src}"
BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"
mkdir -p "$DEST"

fetch() {  # $1 = filename as published on Zenodo
  echo "[download] $1"
  curl -fL --retry 3 -C - "${BASE}/$1?download=1" -o "$DEST/$1"
}

verify() {  # $@ = files to check; others in SHA256SUMS are absent by design
  # The published SHA256SUMS lists 6 files, including fengshui-code.tar.gz and
  # fengshui-analysis.tar.gz, which this script does not fetch. Checking the
  # whole file would therefore always fail (and abort under `set -e`), so we
  # extract the lines for the files we actually downloaded and check only those.
  if [ ! -f "$DEST/SHA256SUMS" ]; then
    echo "[warn] no SHA256SUMS — skipping integrity check for: $*"
    return 0
  fi
  local tmp lines f
  tmp="$(mktemp)"
  for f in "$@"; do
    # SHA256SUMS may list bare ("name"), path-prefixed ("./name") or binary-mode
    # ("*name") entries -- match all three, else a corrupt download passes silently.
    lines="$(awk -v f="$f" '{n=$NF; sub(/^\*/,"",n); sub(/^\.\//,"",n); if (n==f) print}' "$DEST/SHA256SUMS")"
    if [ -n "$lines" ]; then
      printf '%s\n' "$lines" >> "$tmp"
    else
      echo "[warn] $f not listed in SHA256SUMS — cannot verify it"
    fi
  done
  if [ -s "$tmp" ]; then
    echo "[verify] sha256: $*"
    ( cd "$DEST" && sha256sum -c "$tmp" )   # non-zero here aborts (set -e): mismatch
  fi
  rm -f "$tmp"
}

# checksums first so we can verify each download
fetch SHA256SUMS || echo "[warn] SHA256SUMS not on record yet"

# Main database: ships gzip-compressed (~271 MB -> 4.6 GiB). Verify the .gz
# BEFORE decompressing — gunzip removes the .gz, after which it can no longer
# be checked against SHA256SUMS.
fetch unified_database.csv.gz
verify unified_database.csv.gz
echo "[extract] unified_database.csv.gz -> $DEST/unified_database.csv"
gunzip -f "$DEST/unified_database.csv.gz"

# Side databases (small). None of these is needed for the main results
# (Figures 8--10 and Table 3) reproduced by notebooks/reproduce_all.ipynb:
#   vit_pim_database.csv    read ONLY by src/scripts/generate_paper_fig14.py
#                           (ViT case study); copy it to src/timeloop_experiments/
#                           if you want to run that script.
#   pim_database.csv        provenance only (xlsx->csv source for the PIM rows
#                           already baked into unified_database.csv); nothing in
#                           the reproduction path reads it.
#   flashattn_database.csv  standalone FlashAttention/FuseMax sweep artifact;
#                           not wired into the perf model or any figure script.
SIDE_DBS="flashattn_database.csv pim_database.csv vit_pim_database.csv"
for f in $SIDE_DBS; do
  fetch "$f"
done
verify $SIDE_DBS

echo "[done] database at $DEST/unified_database.csv (side DBs in $DEST)"
