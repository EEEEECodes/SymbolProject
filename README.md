# Paired-Family Line Symbol Trainer

Train a conditional beta-VAE to add line work to a supplied base symbol and
export the results as resolution-independent SVG paths. Training examples are
organized into families: one hand-drawn base image plus 30-70 complete images
that build upon that base. The learned model is shared across families and does
not receive a family label, so it can be applied to a completely new base.

The normal interface is a private webpage served by Python on this computer.
It reads source images in place; it does not upload or copy them. The browser
shows validation overlays, automatic unseen-base audit results, training
progress, generated previews, and artifact links. Command-line wrappers remain
available for testing and recovery.

## Quick start on Windows

The project targets Python 3.12-3.14. The current local test run used Python
3.14 with CPU-only PyTorch; this checkout does not include an automated
multi-version or CUDA test matrix. Use an isolated virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Replace `3.14` with `3.12` or `3.13` when using one of those target versions.
Install the CUDA 12.6 PyTorch build, followed by the small remaining dependency
set:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Use the command from the [official PyTorch installer](https://docs.pytorch.org/get-started/locally/)
if the CUDA or Python version changes. For CPU-only operation:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Confirm the selected build. The second value is `True` when CUDA is usable:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Start the app:

```powershell
python app.py
```

It opens the local page automatically. Keep the terminal open and press
`Ctrl+C` there to stop it. Use `python app.py --no-browser` to open the printed
address yourself, or `python app.py --port 9000` to choose another local port.
The server binds only to `127.0.0.1`, has no public sharing mode, protects
actions with a per-launch token, and serves artifacts only from directories
selected during this session.

## Prepare the dataset

Every immediate child of the dataset directory is one family. Use this exact
layout:

```text
dataset/
  family-a/
    base.png
    deviations/
      deviation-001.png
      deviation-002.png
      ...
  family-b/
    base.webp
    deviations/
      scan-001.png
      nested-session/
        scan-002.png
  family-c/
    base.jpg
    deviations/
      ...
  family-d/
    base.png
    deviations/
      ...
```

Rules:

- A deviation is a complete final symbol containing a redrawn version of the
  family base plus mostly additional lines. It is not an added-line mask.
- Each family must contain exactly one supported file named `base.*` and one
  `deviations/` directory. Deviations are discovered recursively.
- Family directory names become stable IDs and must be unique without regard
  to letter case. Renaming a family changes the dataset fingerprint.
- Supported images are PNG, JPEG/JPG, BMP, TIFF/TIF, and WebP. Images found
  outside the defined locations are reported as unassigned and block training.
- At least four valid families and 20 unique usable deviations per family must
  remain after preprocessing and deduplication. Fewer than 30 in a family
  produces a warning; there is no maximum.

Do not put reports, generated images, or alternate base candidates inside the
dataset tree. The validator reports blank, corrupt, duplicate, unassigned, and
registration-failed files without modifying the originals.

## Extract a family from a full symbol sheet

`extract_sheet.py` is a command-line-only, two-phase workflow for turning one
scanned sheet into one paired family. It requires one sheet containing many
complete variations, one separate base image, a review directory outside the
dataset, and a stable family ID. It uses NumPy, Pillow, and the same
preprocessing and registration code as training; it does not upload images or
use the browser app.

First create review artifacts. The review directory must be new or empty:

```powershell
python extract_sheet.py analyze `
  --sheet .\scans\family-a-sheet.jpg `
  --base .\scans\family-a-base.jpg `
  --work-dir .\extraction-review\family-a
```

`analyze` writes `source-preview.png`, `contact-sheet.png`, one cleaned crop per
detected candidate under `review/`, and `manifest.json`. It never writes to the
dataset. A successful analysis reports `review_required`; that is expected.

### Mandatory Codex review

A fresh Codex session can resume from `manifest.json`; it does not need to run
analysis again. Before export, it must:

1. View the source sheet and base at original resolution and confirm the sheet
   represents exactly one base family.
2. View `source-preview.png`, `contact-sheet.png`, and every ambiguous crop.
3. Reject fragments, clipped drawings, page-rule remnants, merged neighbors,
   and drawings that do not contain the supplied base motif.
4. Correct any bad boxes in the manifest and confirm that every accepted box
   contains exactly one complete variation.
5. Keep the manifest, rejected crops, contact sheets, and export report under
   `extraction-review/`, never under `dataset/`.

The manifest keeps raw `candidates` for reference and an editable `exports`
list used by the exporter. Each export record has a unique `id`, one or more
`source_ids`, an exclusive-pixel box `[x0, y0, x1, y1]`, and one status:

- `accept`: export after all automatic safety checks pass.
- `reject`: deliberately omit the detection.
- `review`: keep it out of training until a later session resolves it.

Adjust a crop by editing its `bbox`. To split one detection, replace its export
record with two uniquely named records, separate boxes, and the same
`source_ids`. To merge detections, replace their records with one union box and
list all contributing `source_ids`. Do not edit the source hashes, processing
settings, `rules`, or raw `candidates`. When review is finished, set
`review.complete` to `true` and add a note; `--confirm-reviewed` is an explicit
command-line alternative for a review completed in the current session.

Export the reviewed records into a new family directory:

```powershell
python extract_sheet.py export `
  --analysis .\extraction-review\family-a\manifest.json `
  --dataset-root .\dataset `
  --family-id family-a `
  --confirm-reviewed
```

If `--family-id` is omitted, the base filename becomes a sanitized family ID.
Export refuses to overwrite a populated family. It cleans the base and accepted
crops to lossless black-on-white PNG, checks preprocessing, base registration,
no-op changes, and exact/near duplicates, then writes `export-report.json` next
to the manifest. A `shortfall` result and exit status `2` mean fewer than 20
usable deviations were exported; the valid files remain available for review.

Common adjustments when analysis is too fragmented or too merged:

```powershell
# Merge disconnected strokes more aggressively; use a smaller value if neighbors merge.
python extract_sheet.py analyze --sheet <sheet> --base <base> --work-dir <new-review-dir> --group-radius 8

# Retain faint ink or suppress more background noise.
python extract_sheet.py analyze --sheet <sheet> --base <base> --work-dir <new-review-dir> --threshold-offset 16
python extract_sheet.py analyze --sheet <sheet> --base <base> --work-dir <new-review-dir> --threshold-offset 4

# Keep very small intentional marks; increase this value for speckled scans.
python extract_sheet.py analyze --sheet <sheet> --base <base> --work-dir <new-review-dir> --minimum-ink-pixels 6
```

Ruled or bordered sheets are detected geometrically. Symbols touching a removed
rule, symbols clipped by the page edge, and suspiciously large or tiny regions
are routed to review. Do not accept a rule-touching crop merely to increase the
count: rescan it, adjust the box only if all genuine strokes remain, or omit it.

For the included first local extraction, use these exact Windows commands:

```powershell
python extract_sheet.py analyze `
  --sheet "C:\Users\Admin\Desktop\img8.jpg" `
  --base "C:\Users\Admin\Desktop\Layer 6.jpg" `
  --work-dir ".\extraction-review\layer-6"

python extract_sheet.py export `
  --analysis ".\extraction-review\layer-6\manifest.json" `
  --dataset-root ".\dataset" `
  --family-id "layer-6" `
  --confirm-reviewed
```

Export verifies this family by running every accepted image through the project
preprocessor. Full `python train.py validate` still requires a complete dataset
root with at least four valid families and 20 unique usable deviations in each;
a one-family root can therefore be extracted correctly while global validation
and training remain intentionally blocked.

## Browser workflow

### 1. Validate and inspect alignment

In **Dataset**, choose the dataset root and a new validation-report directory,
then select **Validate dataset**. Native Windows folder pickers are available;
an absolute path may also be entered manually.

The validator canonicalizes every base and deviation separately, aligns each
base to its hand-redrawn counterpart, and computes tolerant addition and
removal masks. Review the per-family counts and contact sheets carefully. Each
row shows an aspect-preserving source-base thumbnail, source-deviation
thumbnail, registered base, processed target, and colored add/remove overlay.
A pair is a semantic no-op only when the changed line length is below both 8
pixels and 8 percent.

Splits are deterministic from family IDs and content hashes, not absolute
paths. Each family receives at least three validation examples. Exact and near
duplicates are grouped so a leakage group cannot cross train and validation.

### 2. Train and run the unseen-base audit

In **Training**, choose a new run directory and select **Start training**. The
app first performs an automatic unseen-base audit:

- With four families, four temporary models each hold out one complete family.
- With five or more families, five deterministic folds are balanced by
  deviation count and hold out complete family groups.
- Fold models start from scratch and are scored on held-out reconstruction,
  prior-sample quality, base retention, diversity, and change-distribution
  agreement using fixed seeds.
- Completed fold weights are discarded. Assignments and metrics remain, and
  only an active fold has a resumable temporary checkpoint.

Each audit fold trains for `max(1, round(epochs * 0.20))` epochs, capped at the
configured epoch count, before the full final run. Four families therefore add
four shortened audit runs; five or more families add five. As an epoch-count
upper bound for normal multi-epoch runs, this is roughly 0.8x or 1.0x the
configured final-run epochs, respectively, although each fold sees fewer
families. Fixed-seed audit sampling adds further scoring time.

After the audit, the final model trains on every family and reports
family-macro-averaged validation metrics. Batches choose a family uniformly and
then choose a deviation, preventing a 70-example family from dominating a
30-example family. Training combines real aligned pairs, synthetic
partial-to-complete pairs, and identity examples.

Only one GPU-intensive job runs at a time. Progress, logs, loss charts,
base/add/remove/composed previews, audit metrics, and artifacts update in the
page. **Cancel** requests a cooperative stop and preserves resumable work.

### 3. Resume or initialize an expanded run

These controls have different purposes:

- **Resume checkpoint** continues an interrupted run in the same run directory.
  It requires the identical dataset fingerprint and restores optimizer,
  mixed-precision scaler, audit stage, and random-number state.
- **Initialize checkpoint** starts a new run from compatible learned model
  weights. Use it after adding families. Dataset references, calibration,
  optimizer state, random state, and audit results are rebuilt for the expanded
  dataset.

Do not select both. Adding a family is never an exact resume: retain all earlier
family directories, add the new family, choose a new run directory, and select
the previous `best.pt` under **Initialize checkpoint**.

### 4. Generate additions for a base

In **Generation**, select a paired-family `best.pt`, a required base image, and
a new output directory. The base may come from a training family or be entirely
new. Edit strength is percentile-calibrated: `0.35` requests approximately the
35th-percentile observed amount of changed line work rather than an arbitrary
pixel blend.

The decoder predicts addition and removal maps, composes them with the supplied
base, and vectorizes the complete result. No-op additions are rejected;
excessive removal or out-of-calibration changes are routed to `review/`.
Novelty is checked against training bases, deviations, and previously accepted
outputs. Sampling uses a bounded number of attempts, so a difficult request can
end with a documented shortfall while retaining every valid partial result.

Generating standalone base symbols is intentionally out of scope.

## Main controls and defaults

Every action writes its complete effective configuration to the selected
report, run, or output directory. Numeric bounds and cross-field rules are
validated before a job starts.

### Pairing, preprocessing, and model

| Control | Default | Purpose |
| --- | ---: | --- |
| Image size | `128` | Internal square raster size. |
| Margin | `12px` | Clear border after canonicalization. |
| Maximum source stroke width | `12px` | Separates line-like from solid source regions. |
| Minimum component size | `3px` | Removes isolated source noise. |
| Maximum input pixels | `40,000,000` | Rejects unexpectedly large images. |
| Filled policy | `outline` | Converts solid regions to unfilled boundary loops. |
| Validation fraction | `0.10` | Family-stratified final holdout, with at least three per family. |
| Registration angle | `+/-12 degrees` | Rotation search between base and target. |
| Registration translation | `+/-8px` | Translation search range. |
| Registration scale | `+/-12%` | Scale search range. |
| Match tolerance | `3px` | Tolerant base/add/remove correspondence. |
| Minimum overlap | `0.25` | Rejects implausible base-to-target registrations. |
| Latent dimension | `32` | Capacity of the learned addition distribution. |
| Base channels | `32` | Convolutional model width. |
| Minimum/maximum stroke widths | `1/6px` | Bounds exported uniform SVG strokes. |

### Training

| Control | Default | Purpose |
| --- | ---: | --- |
| Device | `auto` | CUDA when available, otherwise CPU. |
| Epochs / batch | `250 / 16` | Final-model training duration and batch size. |
| Learning rate / weight decay | `2e-4 / 1e-4` | AdamW settings. |
| Patience | `30` | Early-stopping interval. |
| Seed | `1337` | Stable splits, folds, sampling, and previews. |
| KL maximum / warmup | `1e-3 / 0.25` | Beta-VAE regularization schedule. |
| Real pair probability | `0.60` | Registered base-to-deviation examples. |
| Synthetic pair probability | `0.30` | Partial-to-complete examples for condition diversity. |
| Identity pair probability | `0.10` | Strength-zero preservation examples. |
| Delta loss weight | `0.50` | Supervises predicted add/remove regions. |
| Retention loss weight | `0.25` | Softly penalizes unnecessary base removal. |
| Audit samples | `32` | Fixed-seed outputs scored per held-out base. |
| Gradient clipping | `1.0` | Maximum gradient norm. |
| Workers | `0` | Safest deterministic Windows loader setting. |
| Deterministic / CUDA AMP | on / on | Repeatability and CUDA memory optimization. |
| Preview count/frequency | `8 / 10 epochs` | Fixed-seed training diagnostics. |

The three example probabilities must sum to `1.0`. The model never receives a
family ID. Joint geometric augmentation is applied to base and target together;
mild condition-only jitter simulates scanning and redraw variation.

### Generation, novelty, and quality

| Control | Default | Purpose |
| --- | ---: | --- |
| Count / sampling batch | `50 / 8` | Requested clearly novel results and decode batch. |
| Edit strength / temperature | `0.35 / 0.9` | Calibrated change amount and sampling spread. |
| Threshold | calibrated | Raster cutoff saved by training; optional override available. |
| Review cap | requested count | Maximum borderline results retained. |
| Attempt multiplier | `100` | Attempt limit equals multiplier times requested count. |
| Duplicate / review thresholds | `0.94 / 0.82` | Novelty routing boundaries. |
| Rotated/mirrored review | `0.90` | Symmetry-only matches sent to review. |
| Skeleton/mask/topology weights | `0.60/0.30/0.10` | Precise similarity components. |
| Alignment angle/translation/scale | `+/-6 degrees / +/-3px / +/-4%` | Novelty nuisance search. |
| Shortlist / precise finalists | `64 / 8` | Fast candidates and expensive comparisons. |
| Curve error / maximum ink | `0.75px / 0.35` | Vector fitting and filled-looking rejection. |
| Maximum components | `24` | Fragmentation limit. |
| Crowded-line limit/distance | `10% / 1.5x width` | Dense unrelated-line check. |
| Parallel bundle / solid diameter | `3 / 2.2x width` | Hatching and solid-looking checks. |
| No-op length/fraction | `8px / 8%` | Both must be low to classify a result as unchanged. |

Structural safety rules remain visible but cannot be disabled: four families,
20 usable deviations per family, supported SVG elements only, black strokes,
`fill="none"`, round caps and joins, and no embedded raster data.

## Artifacts and checkpoints

Validation report:

- `config.json` contains the effective validation configuration.
- JSON/CSV manifests contain family IDs, statuses, hashes, leakage groups,
  registration measurements, edit statistics, and deterministic splits.
- Paginated per-family contact sheets show sources, registration, and colored
  addition/removal overlays.

Training run:

- `config.json` and dataset manifests capture the complete reproducible input.
- Audit assignments and metrics describe every unseen-family fold.
- Metrics JSON/CSV and previews record final training progress.
- `best.pt` contains schema version 2 paired-model weights, conditional-prior
  weights, packed bases/targets, family associations, dataset fingerprint,
  registration and strength calibration, quality/novelty references, and audit
  provenance.
- `last.pt` additionally contains optimizer/scaler and random-number state for
  exact resume. An active audit fold may also have a temporary resumable
  checkpoint; completed temporary fold weights are removed.

Generation output:

- `config.json` and `manifest.json` record the input-base hash, request seed,
  requested strength, addition/removal measurements, audit provenance, nearest
  references, rejection counts, attempt limit, and any shortfall.
- `novel/*.svg` plus matching PNG previews are accepted results.
- `review/*.svg` plus matching previews require human inspection.

Older unpaired checkpoints are intentionally incompatible and produce a clear
instruction to retrain with the paired-family dataset. Checkpoints contain
reconstructible packed training masks; protect them when source imagery is
private.

## Command-line fallback

The webpage and CLI call the same importable `validate_dataset`, `train_model`,
and `generate_symbols` functions.

```powershell
# Inspect all options
python train.py --help
python train.py validate --help
python train.py train --help
python train.py generate --help

# Validate and start a fresh run
python train.py validate --data .\dataset --report .\validation
python train.py train --data .\dataset --run .\runs\paired --device auto

# Exact resume of the identical dataset/run
python train.py train --data .\dataset --run .\runs\paired --resume .\runs\paired\last.pt

# New audit/run initialized after adding families
python train.py train --data .\expanded-dataset --run .\runs\expanded --init-checkpoint .\runs\paired\best.pt

# A base image is always required for generation
python train.py generate --checkpoint .\runs\paired\best.pt --base .\new-base.png --out .\generated --count 50 --edit-strength 0.35
```

Generation exits with status `2` on a bounded shortfall and `130` after
cancellation while preserving valid artifacts and its manifest.

## Dependencies and tests

The webpage uses Python's standard library and plain HTML/CSS/JavaScript.
Runtime dependencies are PyTorch, NumPy, and Pillow. OpenCV, SciPy,
scikit-image, Shapely, resvg, Gradio, Node.js, a database, and a web framework
are not required; pure-Python fallbacks handle registration, skeletonization,
distance calculations, and SVG rerendering.

Install the development dependency and run the CPU suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The included automated suite is CPU-only. After installing a CUDA build, verify
`torch.cuda.is_available()` and run a deliberately short training job before a
long run; CUDA mixed precision is supported by the app but is not exercised by
the current tests.

## Upload the code to GitHub

The checkout already has `origin/main` configured. Nothing is pushed by the
app. Review the changes before staging:

```powershell
git status --short
git diff
```

Commit and push only project source:

```powershell
git add .gitignore README.md requirements.txt requirements-dev.txt train.py app.py web tests
git diff --cached --stat
git diff --cached
git commit -m "Add paired-family base extension trainer"
git push origin main
```

The default directory names in `.gitignore` cover `.venv/`, `dataset/`,
`runs/`, `validation/`, `generated/`, common artifact directories, and all
`*.pt`, `*.pth`, and `*.ckpt` files. Arbitrary names selected in the webpage,
such as `my-family-data/` or `experiment-7/`, are not automatically ignored.
Keep custom data and output folders outside the checkout, or add their exact
relative paths to the local-only `.git/info/exclude` file.

Before staging, check every sensitive path located inside the repository. Each
ignored path should print the matching rule; no output means that path is not
ignored:

```powershell
git check-ignore -v -- .\dataset .\runs\paired .\generated
```

The explicit `git add` command above stages only project source. Do not replace
it with `git add .` or force-add ignored content without reviewing the result:
datasets and self-contained checkpoints may expose training imagery or local
paths.

To publish deliberately reviewed examples, copy only those files to a tracked
`examples/` directory:

```powershell
New-Item -ItemType Directory -Force .\examples
Copy-Item .\generated\novel\symbol-0001.svg .\examples\
Copy-Item .\generated\novel\symbol-0001.png .\examples\
git add examples
git commit -m "Add reviewed generated examples"
git push origin main
```

Novelty screening is a technical similarity check, not proof of legal
originality; review published examples yourself.
