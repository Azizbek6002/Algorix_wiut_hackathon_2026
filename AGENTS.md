# AGENTS.md

WIUT Hackathon 2026 — CV Track: traffic-event detection and accident anticipation from a fixed road camera. Read `HACKATHON_CONTEXT.md` (Russian) before working on the solution: it has the team's plan, intended `src/` architecture, chosen stack (YOLO11 + UCMCTrack Python), and known risks.

## Layout

- `wiut_cv_scripts/wiut_cv_scripts/` — the working code (double-nested from the zip); the sibling `wiut_cv_scripts/__MACOSX/` is zip junk, ignore.
- `video/` — empty; put sample videos here (organizers ship them as `samples/*.mp4` + `samples/camera.md`; there is no `samples/` dir yet).
- `wiut_cv_scripts/wiut_cv_scripts/solution.py` — the work file, currently a stub returning `[]`/`0.0`.
- `PLAN.md` — the implementation plan (deadline 27.09.2026 23:59). Read it after `HACKATHON_CONTEXT.md`.
- **Final submission package** (assembled in the last day): `solution.py` must sit at the repo ROOT next to unchanged `run_submission.py`/`evaluate.py`, plus `src/`, `weights/` (≤5 GB), `requirements.txt`, `predictions_samples.json`, `README.md`. Working copy stays in the nested dir until then.

## Hard constraints (miss any of these → scored 0)

- `run_submission.py` and `evaluate.py` are the organizers' harness/metric — never modify.
- `solution.py` is the only file defining the interface: `CLASSES`, `detect_events(video_path)`, `RiskEstimator.reset(meta)` + `.step(frame, t_sec)`. Keep names/signatures exactly.
- `CLASSES` holds the 14 official ids. You may REMOVE ids you never predict; never add.
- `RiskEstimator.step` is strictly causal: called for every frame in order, must not open the video, must not reuse Part A output.
- Time budget per video = 3× its duration for Part A + Part B together; a crash or over-budget → that video scored empty.
- Grading is offline (no internet): weights ≤ 5 GB must ship in the repo or be fetched by `weights/download.sh` before the run.
- Every test video must appear in `predictions.json`, even with `"events": []`.

## Local machine specifics

- Local laptop has an NVIDIA RTX 3070 Ti (8 GB) but the default PyPI torch is CPU-only. CUDA torch was installed to `C:\pylibs` (`--target`, because default pip site-packages hits Windows MAX_PATH) and wired via `...\site-packages\zz_cuda_torch.pth`. Torch is `2.14.0+cu126`.
  - If CUDA is ever missing again: `pip install --target C:\pylibs --no-deps "torch==2.14.0+cu126" "torchvision==0.29.0+cu126" --index-url https://download.pytorch.org/whl/cu126` then keep the `.pth` file.
  - Inference on the laptop runs on GPU (`TCV_DEVICE` defaults to `cuda:0`); never force `cpu` locally.
- CPU coolant/load: `TCV_CV_THREADS` (cv2 decode threads, default 4) and `TCV_OMP_THREADS` (default 4) cap CPU heat. On GPU local runs CPU is idle-ish; on the grading T4 it is a non-issue.
- Benchmark (C3905 slice, imgsz 800): GPU 45 ms/frame (~22 fps inference); CPU-only was ~580 ms/frame. Full 3 min video ≈ 4 min end-to-end on GPU at stride 3.

## Commands (run from the package dir)

```
pip install -r requirements.txt
python run_submission.py --videos samples --out predictions_samples.json --team <team>
python evaluate.py --pred predictions_samples.json --validate-only      # format check
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

- Ground truth you label yourself (e.g. `my_labels.json`) must match `examples/ground_truth.json` shape (`{video_name: {duration, fps, events}}`).
- `--no-risk`, `--risk-stride N`, `--time-factor F` exist for local dev only.

## Scoring quirks

- Score_A: macro F1 pooled over videos, per class × tIoU {0.3, 0.5, 0.7}. Classes = GT ∪ predictions, so predicting a class that never occurs in GT adds a 0-mean class.
- Score_B (accident only): chance-normalized AP (a flat/constant score scores 0) + alarm F1 (θ=0.5, W=10 s) + mTTA. `M = 0.7·A + 0.3·B`; `M = Score_A` if the test set has no accidents.
- Same-class segments in one video must not overlap (`clean_events` drops the later one); `end ≤ duration + 0.5`.
- Cross-class overlaps are ALLOWED and expected (e.g. `wrong_way` causing `accident` = two events) — do not suppress the second event.
- Simultaneous same-class events → report ONE segment covering both (annotators do the same).
- Class not predicted but present in GT → that class scores F1=0 and stays in the mean; predicted class absent from GT → F1=0 and is ADDED to the mean. Trim `CLASSES` to what you detect reliably (dev-set check: drop classes with F1 < ~0.2).

## Dev notes

- Hard-coding the scene layout (lanes, directions, stop line) from `camera.md` is allowed and expected — it ships together with the sample videos (`samples/camera.md`); obtain `samples/` first.
- **Scene geometry (no camera.md yet):** the four sample videos (C3896/C3897/C3902/C3905) are ONE fixed camera (same 3840×2160), calibrated camera-level into a single shared `scene_config.json` (package dir) — applies to every video, hidden test videos included; NO per-video configs. DEV config, NOT official. `src/scene/geometry.py` (was `src/geometry.py`) scales it to any frame size and exposes `get_lane/is_on_road/is_in_crosswalk/is_in_intersection/is_in_u_turn_zone/is_in_exclusion/crosses_stop_line/crosses_solid_line/get_traffic_light_state` (the last one always returns `"UNKNOWN"` — never fake a color). All spatial queries expect **bottom-center** points `((x1+x2)/2, y2)` in full-res video pixels. Only road + crosswalk are enabled (low confidence) as a starting estimate; lanes/stop/solid/intersection/u-turn/traffic-light are OFF until visually calibrated. `debug/scene_geometry.py` draws all enabled features + detections + bottom centers.
- **src/ packages (refactored, ref links):** flat `src/` is now split by responsibility — `src/pipeline/` (run_pipeline + VideoReader), `src/detection/` (YOLO wrapper), `src/tracking/` (trajectory/motion/interaction), `src/scene/` (scene + geometry), `src/events/` (temporal engine, rules, PHASE detectors, `EventManager`), `src/postprocessing/`, `src/risk/`, `src/config/settings.py`, `src/utils/`. Official interface unchanged: `solution.py` → `src.pipeline.run_pipeline` / `src.risk.RiskEstimator`. Legacy import paths (`from src.geometry import ...`) still work via aliases in `src/__init__.py`. The PHASE detectors (lane wrong_way, near_miss, illegal_turn, illegal_u_turn) are wired into the pool but behind `TCV_ENABLE_PHASE_DETECTORS=1`; the default pool is the legacy `src/events/rules.py` flag engine (output unchanged).
- **Calibration tool:** `debug/calibrate_scene.py` (web, http://127.0.0.1:8090). The Video selector is VIEW-ONLY (checking the shared geometry on different samples); Save draft → debug/scene_calib.json; Promote writes into the SHARED top-level features of the single scene_config.json. `debug/scene_similarity.py` is a warning gadget only (crude 64x36 backdrop correlation; low numbers mean "verify visually", not "different camera").
- Post-process event segments: merge fragments and drop sub-second blips before checking F1@0.7.
- Determinism is a hard rule: two runs on the same machine must produce the same output (fix seeds).
- `README.md` must state: install/run, how weights are obtained (`weights/download.sh`, run once with internet before offline eval), the approach, datasets + licences, fixed seeds, team members and roles.
- Repo is git-initialized (git status works, renames tracked); there is no CI, formatter, or lint config.