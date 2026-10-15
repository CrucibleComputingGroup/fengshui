#!/usr/bin/env bash
# Fengshui — fetch the precomputed Timeloop database from Zenodo. Run this once
# (or gunzip a local copy of unified_database.csv.gz into src/ instead).
#
# Default destination is the repo's src/ directory, because that is where the
# notebook and every script look for the database:
#   notebooks/reproduce_all.ipynb -> HAVE_DB = exists(<scripts>/../unified_database.csv)
#   src/scripts/*                 -> --database ../unified_database.csv (default or passed)
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
  # The published SHA256SUMS lists 6 files, five of which this script does not
  # fetch. Checking the whole file would therefore always fail (and abort under
  # `set -e`), so we extract the lines for the files we actually downloaded and
  # check only those.
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

# The record's side databases (flashattn_database.csv, pim_database.csv,
# vit_pim_database.csv) are not needed for Figures 8 and 9: the PIM rows are
# already in unified_database.csv.

echo "[done] database at $DEST/unified_database.csv"
