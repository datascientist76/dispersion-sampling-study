# Structural Origin of Dispersion and Finite-Sample Representation

Follow-up to Paper 1 (dispersion dynamics, clipping sensitivity) and Paper 2
(Finite-Sample Representation Framework).

**Status:** Phase 0 (preparation). Protocol is a DRAFT until the tag
`protocol-frozen` exists. Nothing run before that tag counts as a primary result.

## Layout
- `protocol/`  dated protocol (export of Protocol v0.1) and its revisions
- `src/`       analysis and simulation code
- `configs/`   parameter files; the frozen primary config is committed before the primary run
- `results/`   small summary outputs only (large files go to Zenodo)

## Phase 0 checklist
- [ ] Paper 1 code and data deposited publicly (Zenodo DOI recorded here)
- [ ] Clipping check run on the real Paper 1 pipeline, results in `results/phase0_clipping.json`
- [ ] Literature search logged in the protocol (Appendix A)
- [ ] Protocol values in Section 12 filled in after the pilot

## Run
    python -m venv .venv
    .venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
    pip install -r requirements.txt
    python src/phase0_clipping_check.py --seeds 20 --out results/phase0_placeholder.json

Output from the placeholder simulator is NOT a Paper 1 result.
