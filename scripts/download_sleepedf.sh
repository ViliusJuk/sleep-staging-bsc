#!/usr/bin/env bash
set -euo pipefail

BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUTDIR="${OUTDIR:-data/raw}"
IFS=' ' read -r -a IDS <<< "${IDS:-4001 4002 4011 4012 4021 4022 4031 4032 4051 4052 4061 4062 4071 4072 4081 4082 4091 4092 4101 4102}"

mkdir -p "$OUTDIR"

have(){ command -v "$1" >/dev/null 2>&1; }
get(){ # $1=url -> prints body to stdout
  if have curl; then curl -fsSL "$1"; else wget -q -O - "$1"; fi
}
dl(){  # $1=url $2=out
  echo "  -> $2"
  if have wget; then wget -c -O "$2" "$1"; else curl -fSL -o "$2" "$1"; fi
}

for id in "${IDS[@]}"; do
  subj="SC${id}"
  echo "[INFO] $subj"

  # PSG E0
  psg="${subj}E0-PSG.edf"
  if [[ ! -f "$OUTDIR/$psg" ]]; then
    dl "$BASE/$psg" "$OUTDIR/$psg" || echo "  [WARN] PSG missing: $psg"
  else
    echo "  PSG exists: $OUTDIR/$psg"
  fi

  # HYP: iš HTML paieškos
  index_html="$(get "$BASE/")"
  hyp_file="$(echo "$index_html" | grep -oE "${subj}E[A-Z]-Hypnogram\.edf" | head -n1)"

  if [[ -z "${hyp_file:-}" ]]; then
    # kartais būna ir be papildomos raidės (retai)
    hyp_file="$(echo "$index_html" | grep -oE "${subj}-Hypnogram\.edf" | head -n1 || true)"
  fi

  if [[ -n "${hyp_file:-}" ]]; then
    if [[ ! -f "$OUTDIR/$hyp_file" ]]; then
      dl "$BASE/$hyp_file" "$OUTDIR/$hyp_file"
    else
      echo "  Hypnogram exists: $OUTDIR/$hyp_file"
    fi
  else
    echo "  [WARN] No hypnogram entry in index for $subj"
  fi

  sleep 1
done

echo "[OK] Done → $OUTDIR"

