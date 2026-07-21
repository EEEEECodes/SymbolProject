# Line-Only SVG Symbol Generator

Train a conditional convolutional beta-VAE on black-on-white symbol images, then
sample new symbols as resolution-independent SVG paths. The generator supports
both fully random sampling and PNG-guided editing. Generated SVGs contain black,
unfilled strokes only.

The project is intended for roughly 100-2,000 unlabeled source images. It can run
with fewer, but training requires at least 20 unique symbols after preprocessing
and warns when fewer than 100 remain.

## Environment setup

Python 3.12 is recommended. PyTorch is deliberately not pinned in
`requirements.txt`, because its correct wheel depends on the operating system,
GPU, driver, and desired CUDA runtime.

1. Create and activate a Python 3.12 virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   On macOS or Linux, activate it with `source .venv/bin/activate`.

2. Install PyTorch using the command produced by the official
   [PyTorch Start Locally selector](https://pytorch.org/get-started/locally/).
   Choose the CUDA option supported by your NVIDIA setup, or CPU if CUDA is not
   available. If Torch is already installed, leave that installation in place.

3. Install the remaining dependencies:

   ```text
   python -m pip install -r requirements.txt
   ```

4. Check the environment:

   ```text
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python train.py --help
   ```

With `--device auto`, training uses CUDA when Torch reports that it is available
and otherwise falls back to CPU. The defaults are designed for a 4 GB NVIDIA GPU;
if memory is exhausted, lower the training batch size shown by
`python train.py train --help`.

## Dataset contract

Pass a directory containing complete raster symbols. Files are discovered
recursively, and folder names are not treated as labels. Supported formats are
PNG, JPEG/JPG, BMP, TIFF/TIF, and WebP.

Good inputs have dark symbol geometry on a white or transparent background.
Transparency is composited onto white. Each image is thresholded, cropped,
centered without changing its aspect ratio, and normalized to a 128x128 internal
canvas. Stroke-like regions become one-pixel centerlines. Genuinely solid regions
become unfilled exterior boundary loops, with separate boundary loops retained for
holes. The original files are never modified.

Use `--max-source-stroke-width` when the automatic stroke-versus-solid decision
needs adjustment. Use the same value for validation and training; it is recorded
in the checkpoint and reused during generation. Always inspect the validation
contact sheet before committing to a long training run.

## Validate and train

Start with a non-training validation pass:

```text
python train.py validate --data ./symbols --report ./validation
```

The report records accepted, rejected, corrupt, blank, and duplicate files and
includes before/after imagery for checking line conversion. Deduplication happens
on canonical processed line masks, before the deterministic 90/10 train/validation
split.

Train a new run:

```text
python train.py train --data ./symbols --run ./runs/symbols --device auto
```

Resume an interrupted run from its full-state checkpoint:

```text
python train.py train --data ./symbols --run ./runs/symbols --resume ./runs/symbols/last.pt
```

Training defaults to AdamW with a `2e-4` learning rate, batch size 16, 250 epochs,
patience 30, and seed 1337. CUDA training uses mixed precision. Self-supervised
conditioning examples are made by deleting or trimming line-graph segments and
mildly deforming retained geometry; empty conditions teach unconditional
generation.

A run directory contains readable configuration and metrics, dataset and
conversion manifests, fixed-seed previews, and these checkpoints:

- `best.pt`: CPU-loadable model weights plus all preprocessing, vectorization,
  quality, novelty, stroke-width, and embedded training-reference data required
  for generation without the original dataset.
- `last.pt`: the same generation data plus optimizer, epoch, gradient-scaler, and
  random-number-generator state used to resume training.

## Generate symbols

Generate 50 clearly novel symbols:

```text
python train.py generate --checkpoint ./runs/symbols/best.pt --out ./generated --count 50
```

Guide generation with a PNG:

```text
python train.py generate --checkpoint ./runs/symbols/best.pt --base ./base.png --edit-strength 0.35 --out ./extended --count 20
```

The base is processed through the same line pipeline as the training data. It is
a condition, not a layer pasted into the result: the model may extend, bend,
remove, or redraw its geometry. `--edit-strength` defaults to `0.35`; increasing
it asks for a looser relationship to the base. Base-guided no-ops are rejected,
and results that copy or disregard the base beyond calibrated bounds are routed
to review.

`--count N` requests N outputs classified as clearly novel. Borderline results
are extra and are capped at N by default. Sampling stops after at most
`100 * count` attempts. If it cannot produce the requested number, it preserves
all valid partial output and exits with a nonzero status.

The output directory contains:

- `novel/*.svg` for accepted symbols and corresponding white-background PNG
  previews.
- `review/*.svg` for borderline symbols and corresponding PNG previews.
- `manifest.json` with sampling seeds, nearest references, similarity components,
  base-change measurements, stroke widths, attempt totals, and rejection counts.

Each accepted SVG is re-rendered and quality checked. It contains only `<svg>`,
`<g>`, and `<path>` elements; paths have `fill="none"`, black stroke, a single
symbol-wide stroke width, and round joins and caps. Raster images, backgrounds,
fills, masks, patterns, and hatching are not emitted.

## Novelty policy and limitations

Novelty is measured from final re-rendered geometry against every embedded
training reference and previously accepted output. Exact canonical mask hashes
are rejected. Shortlisted candidates are aligned for small translation, rotation,
and scale differences and scored using skeleton proximity, rendered-mask overlap,
and graph structure.

- Similarity at or above `0.94` is treated as a duplicate and discarded.
- Similarity from `0.82` up to `0.94` is written to `review/`.
- Similarity below `0.82` is eligible for `novel/` after line-quality checks.
- Strong matches found only after mirroring or a 90-degree rotation go to review
  instead of being hard-rejected.

These checks provide measurable exact- and near-duplicate rejection; they cannot
prove semantic or legal originality. Review generated work before publication or
other consequential use. Checkpoints contain reconstructible low-resolution
representations of training symbols so novelty checks can work without the source
dataset; treat checkpoints as sensitive if the training images are sensitive.

## Tests

Run the focused test suite on CPU with:

```text
python -m pytest -q
```

CUDA smoke coverage is optional and runs only when a compatible CUDA-enabled
Torch installation is available.
