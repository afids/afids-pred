#!/usr/bin/env bash
# Re-run the afids-pred analysis pipeline from the data in this repository.
#
#   ./reproduce.sh                 run all stages below, in order
#   ./reproduce.sh figdata figures run only the named stages (same order rules apply)
#
# Stages
#   analyses       cZI/fct/RN 10-model LOGO-CV screens (2.6_..._unified.py), held-out STN
#                  test set from the deployed model, AC/PC degeneracy check. (The older
#                  reproduce_subset_selection.py / reproduce_landmark_importance.py are superseded by
#                  `supplementary_analyses.py landmarks` and are not run.)
#   figdata        figure source data (figures/source/prepare_data.py)
#   supplementary  supplementary STN analyses (notebooks/supplementary_analyses.py);
#                  reads figures/source_data/predictions.csv, so it runs after figdata
#   figures        render figures (needs external template volumes, see docs/REPRODUCIBILITY.md;
#                  skipped with a message if they are missing) and verify them
#   notebooks      execute notebooks 2.1-2.6 headless into $NB_OUT (not run by default)
#
# Environment: PYTHON (default: python3), NB_OUT (default: ./executed_notebooks),
# AFIDS_FIG_EXTERNAL (directory holding the external figure inputs).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
NB_OUT="${NB_OUT:-$REPO/executed_notebooks}"
cd "$REPO"

if [ "$#" -gt 0 ]; then STAGES=("$@"); else STAGES=(analyses figdata supplementary figures); fi

log() { printf '\n==== %s [%s]\n' "$1" "$(date +%H:%M:%S)"; }

stage_analyses() {
  log "cZI / fct / RN screens (about 1.5-2 h)"
  "$PY" notebooks/2.6_Generalized_Target_Prediction_unified.py
  log "held-out STN test set (deployed model)"
  "$PY" notebooks/reproduce_test_set.py
  log "AC/PC degeneracy check"
  "$PY" notebooks/verify_acpc_degeneracy.py
}

stage_figdata() {
  log "figure source data"
  "$PY" figures/source/prepare_data.py
}

stage_supplementary() {
  log "supplementary analyses"
  # this script resolves its paths relative to notebooks/
  (cd notebooks && "$PY" supplementary_analyses.py)
}

stage_figures() {
  log "render figures"
  if "$PY" - <<'EOF'
import sys
sys.path.insert(0, "figures/source")
from drawing import EXT, OLD
need = [EXT / "Synthesized_FLASH25_in_MNI_v2_200um.nii.gz",
        OLD / "atlas_sources", OLD / "reference_review" / "geometry.npz",
        OLD / "reference_review" / "mni" / "mni_icbm152_nlin_asym_09b"]
missing = [str(p) for p in need if not p.exists()]
if missing:
    print("Skipping figure rendering; external inputs not found:\n  " + "\n  ".join(missing))
    print("The committed figures in figures/manuscript/ are kept. See docs/REPRODUCIBILITY.md.")
    sys.exit(1)
EOF
  then
    "$PY" figures/source/render_figures.py
  fi
  log "verify figures and figure source data"
  "$PY" figures/source/verify.py
}

stage_notebooks() {
  log "execute notebooks 2.1-2.6 into $NB_OUT"
  mkdir -p "$NB_OUT"
  for nb in notebooks/2.1_*.ipynb notebooks/2.2_*.ipynb notebooks/2.3_*.ipynb notebooks/2.4_*.ipynb \
            notebooks/2.5_*.ipynb notebooks/2.6_*.ipynb notebooks/2.x_*.ipynb; do
    "$PY" -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
      --output-dir "$NB_OUT" "$nb"
  done
}

for s in "${STAGES[@]}"; do
  case "$s" in
    analyses|figdata|supplementary|figures|notebooks) "stage_$s" ;;
    *) echo "unknown stage: $s" >&2; exit 2 ;;
  esac
done
log "done"
