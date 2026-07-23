<style>
:root {
  --ink-navy: #071f3d;
  --deep-blue: #0b3768;
  --cobalt: #145da0;
  --river-blue: #3e86c6;
  --wash-blue: #eaf3fb;
  --paper: #fbfdff;
  --muted-ink: #42566d;
}

body {
  max-width: 980px;
  margin: 0 auto;
  padding: 2.5rem;
  color: var(--ink-navy);
  background: var(--paper);
  font-family: "Aptos", "Segoe UI", Arial, sans-serif;
  line-height: 1.62;
}

h1, h2, h3, h4 {
  color: var(--deep-blue);
  line-height: 1.2;
}

h2 {
  margin-top: 2.8rem;
  padding-bottom: 0.35rem;
  border-bottom: 3px solid var(--river-blue);
}

h3 {
  margin-top: 1.8rem;
}

blockquote {
  margin: 1.5rem 0;
  padding: 0.85rem 1.15rem;
  color: var(--deep-blue);
  background: var(--wash-blue);
  border-left: 5px solid var(--cobalt);
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0 1.6rem;
}

th {
  color: #ffffff;
  background: var(--deep-blue);
  text-align: left;
}

th, td {
  padding: 0.65rem 0.75rem;
  border: 1px solid #abc8e4;
  vertical-align: top;
}

tr:nth-child(even) td {
  background: #f1f7fc;
}

code {
  color: #083b73;
  background: #e4eff9;
}

.cover {
  padding: 3.2rem 2.7rem;
  color: #ffffff;
  background: linear-gradient(145deg, #06182f, #0b3768 62%, #145da0);
  border-radius: 18px;
  box-shadow: 0 18px 48px rgba(7, 31, 61, 0.2);
}

.cover h1, .cover h2, .cover p {
  color: #ffffff;
  border: 0;
}

.cover .kicker {
  margin: 0 0 0.7rem;
  color: #b9daf5;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

.cover .subtitle {
  max-width: 780px;
  color: #d9ecfb;
  font-size: 1.15rem;
}

.blank {
  display: inline-block;
  min-width: 13rem;
  border-bottom: 2px solid var(--cobalt);
}

.blue-note {
  padding: 1rem 1.2rem;
  background: var(--wash-blue);
  border: 1px solid #b8d2e9;
  border-radius: 10px;
}

@media print {
  body { max-width: none; padding: 0; }
  .cover { box-shadow: none; }
  h2, h3, table { break-after: avoid; }
  table { break-inside: avoid; }
}
</style>

<div class="cover">
  <p class="kicker">Practice-Based Art and Machine Learning Research Plan</p>
  <h1>Generative Line-Symbol Systems for Contemporary Ink Painting</h1>
  <p class="subtitle"><strong>A paired-family conditional β-VAE for expanding the artist’s search field of line shapes, additions, and structural combinations.</strong></p>
</div>

> **Project premise.** This research responds to the needs of artists who create large-scale contemporary ink paintings that combine the expressive force of Chinese ink with rigorous structural thinking. In this practice, line-based symbols are not decorative accessories: they are generative units from which larger pictorial structures, rhythms, tensions, and spatial systems can be built.

---

## 1. Project Summary

This project investigates whether a generative model can broaden an artist’s search field for new line-based symbols without replacing the artist’s judgment or flattening the material intelligence of ink painting. The proposed system learns from **paired symbol families**. Each family contains one hand-drawn base image and a set of complete deviations that redraw, extend, bend, subtract from, or otherwise build upon that base. The model is trained to learn the distribution of additions and limited removals that turns a base into a related but distinct symbol.

The practical need is straightforward. A large-scale ink painting may depend on many preliminary trials of line shapes, structural combinations, and symbol-to-symbol relationships. Manual invention is essential, but the search can become repetitive or remain too close to the artist’s established habits. A controlled generator can produce alternative structures quickly, enabling the artist to compare more possibilities, notice unexpected relationships, and return selected candidates to drawing, composition, and physical ink work.

The technical system is a shared conditional β-variational autoencoder (β-VAE). It receives only the base pixels and a requested edit strength; it never receives a categorical family label. A learned conditional prior models possible changes for a base, while a posterior learns from observed base-to-deviation pairs. The decoder predicts addition and removal likelihood maps, composes them with the supplied base, and exports the complete result as a constrained line-only SVG with matching PNG previews. This formulation is designed so that a completely new base can be supplied at generation time.

The research combines four forms of inquiry:

1. **Practice-based inquiry:** the artist generates, selects, rejects, redraws, recombines, and translates machine suggestions into ink studies and larger works.
2. **Technical experimentation:** model variants are compared through family-held-out tests, ablations, geometry checks, novelty analysis, and reproducible checkpoints.
3. **Visual analysis:** generated symbols are evaluated for structural coherence, line economy, relation to the base, surprise, and compositional potential.
4. **Reflective documentation:** decisions, failures, selection criteria, and the movement from digital symbol to physical painting are recorded as research evidence.

The primary contribution is therefore not a claim that the system independently makes artworks. It is a **co-creative symbol-search method**: a reproducible way to learn transformations from an artist’s own visual families, test transfer to unfamiliar bases, and place machine-generated proposals inside an accountable studio process.

### Intended outputs

- A curated paired-family image corpus.
- A local, privacy-conscious model trainer and generation interface.
- A trained conditional symbol-addition model and documented checkpoints.
- A structured evaluation of transfer to unseen base symbols.
- An archive of accepted, rejected, and review-routed SVG proposals.
- Studio studies and selected large-scale ink works developed from the symbol archive.
- A written analysis connecting technical behavior to artistic decision-making.

---

## 2. Research Motivation

### 2.1 Artistic need

The project begins from a visual practice in which symbols made primarily from lines act as structural seeds. A symbol may be repeated, enlarged, rotated, mirrored, joined to another symbol, interrupted, or distributed across a painting. Its value lies not only in what it represents but also in how it carries force: direction, density, balance, interval, pressure, and relation to empty space.

Searching for such symbols is a combinatorial problem. A base form can support many additions, yet the space of possible line placements rapidly exceeds what an artist can review manually. Conventional image generators are poorly matched to this task because they often prioritize pictorial appearance, texture, or semantic resemblance. They may also alter the base excessively, produce filled regions, or return raster images that are difficult to inspect as clean structures.

This research requires a narrower and more legible system. The generator should preserve a supplied base, propose bounded structural change, retain line topology, avoid accidental filled masses, and export geometry that can be enlarged without resolution loss. It should expose the relation between input and output so the artist can see what was added, what was removed, and why a result was accepted or routed for review.

### 2.2 Research gap

Existing sketch and vector-generation research shows that machines can learn compact representations of drawn structures. However, public sketch datasets usually organize drawings by object category, while this project’s data expresses a different relationship: one hand-drawn base is repeatedly reinterpreted through related complete symbols. The problem is not “generate another member of a named class.” It is “learn a reusable grammar of addition from several visual families and transfer that grammar to a new, unnamed base.”

The project also addresses a gap between technical novelty and studio usefulness. A symbol may be statistically dissimilar to the training set yet artistically inert. Conversely, a modest variation may unlock a productive painting. Technical filtering is necessary for duplicate detection and geometry safety, but it cannot replace situated artistic judgment. The evaluation must therefore combine computational measurements with human selection, reflective notes, and material trials.

### 2.3 Technical and methodological difficulty

The source images are hand drawn and may be scanned or photographed. A deviation contains a redrawn version of the base rather than a perfectly identical copy, so naïve pixel subtraction would mistake scanning shifts and hand variation for meaningful additions. The system must separately canonicalize base and target, register the base to each deviation, tolerate small stroke differences, and distinguish actual structural change from nuisance variation.

Generalization is also difficult because early evidence may come from only a small number of distinct bases, even when each base has many deviations. Treating every image as independent would inflate apparent performance. The appropriate unit of generalization is the **family**, not the individual image. Family-held-out evaluation is consequently central to the research design.

---

## 3. Research Questions

1. **Transfer:** To what extent can a conditional generative model trained on paired base→deviation families learn additions that remain structurally coherent when applied to a completely unseen base symbol?
2. **Control and preservation:** Which combination of registration, condition-aware latent modeling, addition/removal prediction, identity examples, and retention penalties best balances base preservation with meaningful structural variation?
3. **Technical and artistic value:** How closely do technical measures of quality, novelty, diversity, and retention correspond to artists’ judgments of structural coherence, surprise, usefulness, and potential for large-scale ink painting?
4. **Creative workflow:** Does an artist-in-the-loop generator broaden the range of considered line symbols, reduce repetitive search effort, and lead to formal possibilities that would be less likely to emerge through manual iteration alone?

### Working hypotheses

- A family-aware training and evaluation protocol will give a more conservative but more credible estimate of unseen-base performance than a random image-level split.
- A learned prior conditioned on base pixels and edit strength will yield more relevant variations than sampling from an unconditional standard normal distribution.
- Explicit addition/removal heads, identity examples, and a retention penalty will reduce destructive changes to the base.
- Technical novelty will be necessary but insufficient for artistic usefulness; the strongest evidence will come from combined metric, blind-review, and studio-trial results.

---

## 4. Research Context and Precedents

The study draws from generative sketch modeling, vector-graphics generation, few-shot visual concept learning, topology-aware image analysis, and practice-based research in computational creativity. These precedents establish useful techniques while also clarifying why a custom paired-family corpus and artist-centered evaluation are required.

| Reference / precedent | Relevance to this project | Research takeaway |
| --- | --- | --- |
| **Sketch-RNN — Ha and Eck, “A Neural Representation of Sketch Drawings”** | Demonstrates latent-variable modeling of human stroke sequences and controllable sampling of sketches. | A learned latent space can support variation, interpolation, and completion, but category-level public sketches do not encode this project’s base→deviation family relationship. |
| **DeepSVG — Carlier et al.** | Demonstrates learned representation and generation of scalable vector graphics. | SVG is an appropriate inspection and enlargement format. With a modest specialist corpus, raster learning followed by constrained vectorization is a lower-risk starting point than direct command-sequence generation. |
| **Conditional VAE — Sohn, Lee, and Yan** | Establishes conditional probabilistic modeling for diverse structured outputs. | Modeling `p(z | base, strength)` supports multiple valid deviations for the same base rather than collapsing to one deterministic answer. |
| **β-VAE — Higgins et al.** | Provides a controllable balance between reconstruction and latent regularization. | KL warmup and a small maximum β can preserve thin-line fidelity while maintaining a sampleable latent space. |
| **clDice — Shit et al.** | Introduces a centerline-sensitive topology measure for thin connected structures. | Pixel overlap alone is inadequate for line symbols; skeleton and connectivity-sensitive losses and metrics are required. |
| **Omniglot and Quick, Draw!** | Show the importance of human-drawn variation, standardized evaluation, and explicit train/validation/test protocols. | Public corpora can support external robustness comparisons, but they cannot replace artist-owned paired families because their classes, cultural contexts, and transformation logic differ. |
| **Practice-based and co-creative art research** | Treats making, selection, reflection, and material outcomes as knowledge-producing activities. | The artist’s annotations, rejected candidates, composition trials, and paintings are research data rather than post hoc illustrations of a technical result. |

The literature review will also address contemporary ink painting, Chinese ink materiality, symbol and sign systems, repetition and variation, computational creativity, and authorship. These art-historical and cultural sources should be selected with subject specialists so that “Chinese ink” is treated as a historically and materially specific field rather than a generic visual style.

---

## 5. Proposed System and Research Workflow

The proposed system is modular so that dataset assumptions, registration, learning, vectorization, and evaluation can be tested independently. The end-to-end workflow is:

1. **Create paired families.** For each family, produce one base image and 30–70 complete deviations that build upon it. Add new family directories as new bases are developed.
2. **Validate the corpus.** Detect missing or ambiguous bases, unsupported placements, corrupt files, blanks, duplicates, near duplicates, and families below the required minimum.
3. **Canonicalize line images.** Convert supported image formats to a consistent grayscale line representation, remove small noise, enforce a clear margin, and outline filled-looking regions when necessary.
4. **Register each pair.** Align the canonical base to each complete deviation with coarse-to-fine rotation, translation, and scale search. Reject implausible matches and save visual overlays for human inspection.
5. **Measure change.** Derive tolerant addition and removal masks. Map observed change to an empirical percentile so an edit strength such as `0.35` means approximately the 35th-percentile change in the corpus.
6. **Control leakage.** Deduplicate within families, assign exact and near duplicates to global leakage groups, and prevent any group from crossing training and validation boundaries.
7. **Audit unseen-base transfer.** Train temporary family-held-out models from scratch, score held-out families, preserve fold metrics, and discard completed fold weights.
8. **Train the final model.** Use every development family with family-balanced sampling, early stopping, fixed-seed previews, threshold calibration, and full-state resumable checkpoints.
9. **Generate for a required base.** Supply a trained checkpoint, a base image, edit strength, and count. Sample the learned conditional prior and export bounded-attempt SVG candidates.
10. **Review and translate.** Separate clearly novel candidates from review cases; compare add/remove overlays; select, redraw, combine, scale, and test candidates through physical ink studies.
11. **Document decisions.** Record seeds, configurations, rejections, nearest references, artist ratings, reflective notes, and the relationship between generated symbols and completed works.

### Conceptual workflow

> **Artist’s base** → **paired-family registration** → **learned change distribution** → **candidate line symbols** → **quality and novelty routing** → **artist selection and transformation** → **ink study / large-scale painting**

The artist remains responsible for the purpose, interpretation, selection, scale, composition, and material realization of the work. The model proposes possibilities inside a bounded visual grammar; it does not determine the artwork.

---

## 6. Study Scope

### 6.1 Primary user and use case

The primary user is an artist or artist-researcher working with a personally curated language of line symbols. The core use case is: **select a base, request a controlled degree of structural change, inspect multiple line-only candidates, and carry selected results into further drawing and ink practice.**

### 6.2 Core study features

- Fixed paired-family folder structure with one base and recursively discovered deviations.
- At least four valid families and 20 unique usable deviations per family; warnings below 30.
- Separate canonicalization and similarity registration for hand-redrawn base/target pairs.
- Tolerant addition/removal masks and semantic no-op rejection.
- One shared conditional β-VAE with no family ID input.
- Learned conditional prior and explicit addition/removal decoder heads.
- Family-balanced real, synthetic, and identity example sampling.
- Automatic unseen-family auditing before final training.
- Mandatory-base generation with calibrated edit strength.
- Line-only SVG export, rerender validation, topology checks, quality constraints, novelty filtering, and review routing.
- Local HTML interface with progress, logs, charts, previews, manifests, and artifact links.
- Complete configuration and provenance recording for every run.

### 6.3 Extensions if time and evidence permit

- Direct stylus/vector capture to preserve stroke order and pressure as additional variables.
- Artist ranking or pairwise preference learning for later reranking of valid candidates.
- Active learning that identifies which new base families would most improve transfer.
- Multi-scale composition tools for arranging several generated symbols on a larger virtual field.
- A searchable symbol atlas connecting generated candidates to paintings, notes, and rejected alternatives.
- Comparative studio workshops with additional ink artists.
- A later direct vector decoder if the expanded dataset supports command-level learning reliably.

### 6.4 Explicit exclusions

- Generating base symbols from nothing.
- Text-to-image or prompt-driven illustration generation.
- Simulating ink texture, brush loading, absorbency, or paper behavior.
- Claiming that an SVG is a finished artwork.
- Cloud hosting, public uploads, or automated publication of generated results.
- Treating novelty scores as proof of cultural, artistic, or legal originality.

---

## 7. Methodology

### 7.1 Research design

The study uses a mixed practice-based and experimental design. Technical experiments establish whether the method learns a transferable structural process. Studio experiments establish whether that process is meaningful in artistic practice. The two strands are analyzed together: a model is not considered successful merely because it minimizes a loss, and a compelling isolated image is not sufficient evidence of reliable generalization.

The unit of analysis is the **base family**. Individual deviations are repeated observations within a family, not independent examples of unseen-base generalization. All primary results will therefore report family-level scores and macro-averages across families.

### 7.2 Dataset plan

<div class="blue-note">
  <strong>Planned total corpus range:</strong> 1,000–6,000 source images<br>
  <strong>Confirmed total source images:</strong> <span class="blank">&nbsp;</span> images<br>
  <strong>Counting rule:</strong> total = all included base images + all included deviation images after the corpus is frozen for the study.
</div>

The pilot begins with approximately four base families and 50–70 deviations per base. The corpus will expand by adding new base families rather than only adding more deviations to existing families. Increasing the number and structural diversity of bases is more important for the central claim of unseen-base transfer than indefinitely enlarging a small number of families.

Each family follows this structure:

```text
dataset/
  family-id/
    base.png
    deviations/
      deviation-001.png
      deviation-002.png
      ...
```

Each deviation is a **complete final symbol**, not an overlay containing only the new lines. Supported sources may include PNG, JPEG, BMP, TIFF, and WebP. Family IDs are stable and case-insensitively unique.

#### Inclusion criteria

- The base and deviation are produced or licensed for this research.
- The symbol is primarily composed of dark line work on a light ground.
- The base remains visually present in the deviation, even when redrawn by hand.
- The deviation contains a meaningful structural change.
- The image can be canonicalized without losing its principal topology.
- Registration meets the declared minimum-overlap threshold.

#### Exclusion and reporting criteria

- Missing or multiple base candidates.
- Supported images outside the required family structure.
- Blank, corrupt, or implausibly large images.
- Exact duplicates and within-family near duplicates.
- Registration failures or severe cropping.
- Semantic no-ops whose changed line length is below both 8 pixels and 8% of base length.

Excluded items are retained in manifests with a status and reason; they are not silently deleted.

### 7.3 Data partitioning and leakage control

The main corpus will be divided at the family level before final reporting:

- **Development families:** used for model development, automatic family-held-out audits, ablations, and parameter selection.
- **Locked test families:** collected or designated before final evaluation and never used for training, checkpoint initialization, threshold tuning, or visual selection criteria.
- **Within-family validation:** 10% of usable deviations from each development family, with at least three validation deviations and at least one training deviation.

Exact and near duplicate targets are assigned to global leakage groups. A leakage group cannot cross a final train/validation boundary. Cross-family duplicates are excluded from unseen-family audit scoring because they would make a held-out family partially visible through another family. Splits are derived from family IDs, content hashes, the declared seed, and validation fraction—not absolute file paths—so relocating the corpus does not change the experiment.

For the pilot, four families produce four leave-one-family-out models. With five or more development families, five deterministic grouped folds are balanced by deviation count. After the model design is frozen, the locked test families provide a final prospective assessment on bases not involved in design decisions.

### 7.4 Preprocessing and registration

Base and target are canonicalized separately to a `128 × 128` internal canvas with a 12-pixel margin. Small components below 3 pixels are removed, source strokes wider than the declared limit are flagged or outlined, and source images above 40,000,000 pixels are rejected before expansion in memory.

The canonical base is registered to each deviation through a coarse-to-fine similarity search with default ranges of ±12° rotation, ±8 pixels translation, and ±12% scale. Matching uses a 3-pixel tolerance and requires at least 0.25 overlap. The overlap denominator remains the original base length so a transform cannot improve its score by clipping difficult pixels off the canvas.

After registration:

- **Addition mask:** target centerline not explained by the tolerant registered base.
- **Removal mask:** registered-base centerline not explained by the tolerant target.
- **Raw change ratio:** `(added length + 0.5 × removed length) / base length`.
- **Edit strength:** empirical percentile rank of the raw change ratio across the usable corpus.

Joint geometric augmentation is applied to base and target together. Mild condition-only line jitter simulates scan and redraw differences for non-identity examples. Identity pairs remain identical at strength zero.

### 7.5 Model

The principal model is a conditional β-VAE with latent dimension 32 and base channel width 32:

- Conditional prior: `p(z | base, strength)`
- Posterior: `q(z | base, target, strength)`
- Decoder outputs: addition likelihood `a`, removal likelihood `r`, and stroke-width estimate.
- Composed output: `target_hat = base × (1 − r) + (1 − base) × a`

No family ID is supplied to the network. This prevents the model from memorizing a categorical family code and makes base pixels the only structural condition.

The default objective is:

```text
L = weighted BCE
  + Dice
  + 0.5 × clDice
  + 0.1 × width Huber
  + β × KL(q || p)
  + 0.5 × delta loss
  + 0.25 × retention/removal penalty
```

The KL term warms to `1e-3` over the first 25% of training. The delta loss supervises addition and removal masks directly. The retention term penalizes predicted removal on portions of the base that should remain while allowing observed bends and redraws.

### 7.6 Training procedure

Each batch samples a family uniformly and then a deviation uniformly so a 70-deviation family does not dominate a 30-deviation family. The default example mix is:

- **60% real pairs:** registered base → complete deviation.
- **30% synthetic pairs:** partial condition → complete deviation, derived from all deviations to broaden condition shape.
- **10% identity pairs:** base → same base at strength zero.

Training uses AdamW, learning rate `2e-4`, weight decay `1e-4`, batch size 16, gradient clipping at 1.0, a maximum of 250 epochs, and early-stopping patience of 30. Deterministic loading and fixed seeds support repeatability. CUDA automatic mixed precision is enabled when available.

Before final training, temporary fold models start from scratch. They are scored on held-out reconstruction, conditional-prior sample quality, base retention, diversity, and change-distribution agreement using 32 fixed-seed samples per held-out base. Fold assignments and metrics are preserved; completed fold weights are discarded. Only an active interrupted fold retains a resumable checkpoint.

The final model is trained after auditing. Validation metrics are macro-averaged across families. The raster threshold and safe guided-change bounds are calibrated from validation data. `best.pt` is used for generation and initialization of a future expanded run; `last.pt` stores optimizer, random-number, data-loader, and mixed-precision state for exact resume.

### 7.7 Generation and studio selection

Generation requires a final checkpoint, a base image, and a new output directory. The base is preprocessed identically to training data. Samples are drawn from the learned conditional prior at the requested edit-strength percentile and temperature.

Each candidate passes through the following gates:

1. Decode addition/removal likelihood maps and compose with the base.
2. Threshold and skeletonize the result.
3. Trace paths and fit curves with a linear fallback.
4. Export a black, unfilled, round-cap, round-join SVG.
5. Rerender the SVG and reject topology-altering conversions.
6. Enforce ink density, component, crowding, parallel-bundle, and solid-diameter limits.
7. Reject semantic no-ops.
8. Compare novelty against all training bases, deviations, and previously accepted outputs.
9. Route excessive removal, out-of-calibration change, mirrored/rotated similarity, and borderline novelty to `review/`.
10. Preserve all accepted partial results if bounded attempts end before the requested novel count.

The artist then records one of the following actions for each reviewed candidate: reject, retain unchanged as a reference, redraw, combine with another symbol, alter scale/orientation, test in a small ink study, or advance to a larger composition.

### 7.8 Methodological test cases

| Sample or test type | Purpose | Expected failure mode / diagnostic value |
| --- | --- | --- |
| Clean base with moderate additions | Establish basic reconstruction and generation behavior. | Overly conservative no-op or literal memorization. |
| Translated, rotated, or scaled scan | Test registration invariance. | False additions caused by misalignment or clipping. |
| Hand-redrawn base with stroke variation | Test tolerance to authentic drawing differences. | Mistaking redraw texture for semantic structure. |
| Dense or highly connected deviation | Test topology and quality limits. | Filled-looking masses, broken junctions, or crowded bundles. |
| Deviation with legitimate removal or bend | Test the removal head and preservation penalty. | Either erasing the base unnecessarily or forbidding valid redraws. |
| Near-duplicate across two families | Test leakage grouping and audit exclusions. | Inflated unseen-family scores. |
| Completely new base | Test the central transfer claim. | Irrelevant additions, base destruction, or change outside calibrated range. |
| Identity request at strength zero | Test base preservation. | Unwanted edits or scan-jitter artifacts. |
| Increasing strength sequence | Test control monotonicity. | Change amount unrelated to requested percentile. |
| Accepted symbol translated into ink | Test studio relevance. | Structurally valid SVG that lacks material or compositional potential. |

### 7.9 Baselines and ablations

The principal model will be compared against:

- **No-edit baseline:** return the supplied base unchanged.
- **Nearest-family transfer baseline:** transfer the most similar observed addition pattern after registration.
- **Deterministic conditional autoencoder:** remove latent sampling to measure the contribution of probabilistic diversity.
- **Standard-prior ablation:** replace the learned conditional prior with `N(0, I)`.
- **No-synthetic ablation:** remove partial→complete examples.
- **No-identity ablation:** remove strength-zero preservation examples.
- **Single-output ablation:** predict only a complete image rather than addition/removal heads.
- **No-retention ablation:** remove the preservation penalty.
- **Image-balanced ablation:** sample deviations globally rather than sampling families uniformly.

All comparisons use identical family folds, fixed seeds where possible, the same attempt budgets, and the same review protocol.

### 7.10 Practice documentation

For each studio session, the researcher will record:

- model checkpoint, base hash, seed, strength, temperature, and thresholds;
- number of candidates generated, rejected, reviewed, and selected;
- reasons for selection or rejection;
- manual transformations made after generation;
- time spent reaching a shortlist compared with an unassisted session;
- photographs/scans of ink studies and larger works;
- reflective notes on surprise, resistance, authorship, repetition, and material translation.

This process creates a traceable chain from data and model behavior to artistic outcome.

---

## 8. Evaluation Plan

The project will use both technical and artist-centered evaluation. Technical measures determine whether the model behaves consistently and safely. Human and studio measures determine whether the behavior matters to the practice.

| Evaluation dimension | Metric or method | Why it matters |
| --- | --- | --- |
| Dataset integrity | Per-family accepted counts, duplicate rates, registration-failure rates, leakage-group audit | Establishes whether later results rest on a credible corpus. |
| Registration | Tolerant overlap, transform distribution, manually reviewed overlays | Distinguishes meaningful additions from scan/redraw nuisance. |
| Reconstruction | Dice, clDice, addition/removal accuracy, width error, macro-average by family | Tests whether observed transformations can be represented without losing line topology. |
| Conditional prior quality | Held-out prior-sample quality rate and reconstruction-to-prior gap | Tests whether generation works without seeing the target. |
| Base retention | Retained-base length and excessive-removal rate | Measures whether the system respects the supplied structural seed. |
| Strength control | Correlation and calibration between requested percentile and measured change | Determines whether edit strength has an interpretable effect. |
| Diversity | Pairwise skeleton distance, unique topology signatures, within-base coverage | Detects mode collapse and repetitive output. |
| Novelty | Nearest-reference similarity, exact/near duplicate rate, transformed-similarity review rate | Quantifies distance from known symbols while avoiding false claims of originality. |
| Vector quality | SVG schema pass rate, rerender topology agreement, curve error, component and density failures | Ensures candidates remain clean, scalable line structures. |
| Robustness | Results by scan quality, family size, line density, topology, and unseen-base difficulty | Identifies where the system fails rather than reporting only an average. |
| Artistic coherence | Blind 1–7 rating of balance, line economy, relation to base, and structural legibility | Tests whether valid outputs form convincing visual systems. |
| Artistic usefulness | Blind 1–7 rating of surprise, compositional potential, and willingness to develop | Measures value to practice rather than resemblance alone. |
| Workflow effect | Time to viable shortlist, number of distinct directions considered, unassisted-versus-assisted session comparison | Tests whether the system genuinely broadens search. |
| Material translation | Structured critique of small ink studies and selected large works | Tests the gap between clean digital geometry and the force/materiality of ink. |

### 8.1 Unseen-base protocol

For each held-out or locked test base, generate 32 fixed-seed audit samples for automatic metrics and a larger artist-review set across several predeclared strength levels. The main comparison uses the same seeds and attempt limits across model variants. Cross-family duplicates are excluded from audit scoring.

### 8.2 Human review protocol

Candidates will be randomized and shown without model-condition labels where possible. Reviewers will not be told whether an image comes from the main model, an ablation, a baseline, or a human deviation until after rating. The rubric will include:

- structural coherence;
- preservation and productive transformation of the base;
- distinctiveness without arbitrary noise;
- economy and force of line;
- potential for repetition, scaling, joining, or spatial composition;
- suitability as a starting point for contemporary ink work;
- overall willingness to develop the candidate further.

Ratings will be accompanied by short qualitative comments. When several reviewers participate, agreement will be reported; disagreement will be treated as meaningful evidence of plural artistic judgment rather than simply averaged away.

### 8.3 Analysis and reporting

- Report family-level results and family-macro averages, not only pooled image averages.
- Report median and distributional summaries for non-normal metrics.
- Use family bootstrap confidence intervals where the number of families permits.
- Publish failure categories and representative rejected examples alongside successes.
- Separate exploratory analyses from predeclared primary comparisons.
- Relate technical metrics to artist ratings through rank correlation, while avoiding the claim that one explains the other completely.
- Keep the locked test set closed until preprocessing, architecture, filtering rules, and rating rubric are frozen.

### 8.4 Success criteria

The research will be considered successful if it demonstrates all of the following:

1. measurable transfer to unseen bases above the no-edit and nearest-transfer baselines;
2. reliable base retention with controlled, non-trivial additions;
3. bounded geometry and novelty failures that are transparently routed rather than hidden;
4. reproducible generation and audit results from saved configurations and checkpoints;
5. artist-rated usefulness that is not explained only by similarity to training images;
6. documented studio outcomes in which generated proposals lead to decisions, structures, or paintings that the artist can critically account for.

---

## 9. Technical Architecture

The implementation is intentionally local and dependency-light so the research can be repeated without a hosted service or opaque external model.

- **Local interface:** vanilla HTML, CSS, and JavaScript served by Python’s standard library. It provides Dataset, Training, Generation, Results, and Environment sections.
- **Path selection:** native Windows file/folder pickers through `tkinter`, with manual absolute-path entry as fallback. Images remain in their source folders and are not uploaded or copied.
- **Typed backend API:** importable `validate_dataset`, `train_model`, and `generate_symbols` functions accept progress and cancellation callbacks. CLI commands call the same functions.
- **Dataset layer:** recursive family ingestion, status classification, canonical preprocessing, registration, addition/removal targets, content hashing, duplicate grouping, deterministic split assignment, manifests, and contact sheets.
- **Model layer:** PyTorch conditional β-VAE with base/strength condition encoder, learned prior, posterior, addition/removal heads, and stroke-width prediction.
- **Audit layer:** deterministic family folds, fresh fold models, internal training-family validation, fixed-seed held-out sampling, resumable active fold, and persistent metrics.
- **Generation layer:** conditional-prior sampling, base composition, skeleton tracing, curve fitting, line-only SVG serialization, pure-Python rerendering, and partial-result preservation.
- **Quality layer:** topology, component, density, crowding, bundle, solid-diameter, no-op, retention, and calibration checks.
- **Novelty layer:** packed reference masks, precomputed descriptors, vectorized 441-transform coarse search, and expensive comparison limited to the best eight transform/reference finalists by default.
- **Checkpoint layer:** atomic schema-versioned files containing only tensors and primitive values, safely loadable with `weights_only=True`; exact resume requires an identical dataset fingerprint.
- **Security layer:** loopback-only binding, no public share mode, per-launch session token for state-changing requests, restricted artifact roots, and one active GPU-intensive job.
- **Runtime dependencies:** PyTorch, NumPy, and Pillow. No Node.js, database, web framework, OpenCV, SciPy, scikit-image, Shapely, resvg, or Gradio is required.

### Default reproducible configuration

| Group | Principal defaults |
| --- | --- |
| Preprocessing | 128px image, 12px margin, 12px source-stroke limit, 3px minimum component |
| Registration | ±12° rotation, ±8px translation, ±12% scale, 3px tolerance, 0.25 minimum overlap |
| Model | latent dimension 32, base channels 32, 1–6px output stroke width |
| Training | 250 epochs, batch 16, AdamW `2e-4`, weight decay `1e-4`, patience 30, seed 1337 |
| Example mix | 0.60 real, 0.30 synthetic, 0.10 identity |
| Loss | KL max `1e-3`, delta 0.50, retention 0.25, gradient clip 1.0 |
| Generation | count 50, strength 0.35, temperature 0.9, batch 8, attempt multiplier 100 |
| Novelty | duplicate/review 0.94/0.82, transformed review 0.90, shortlist/finalists 64/8 |
| Quality | curve error 0.75px, maximum ink 0.35, maximum components 24, no-op 8px and 8% |

Every experiment saves its complete effective configuration; defaults are starting points, not hidden constants.

---

## 10. Work Plan

The schedule below is written as a twelve-month plan and can be compressed or extended to fit the programme calendar.

| Phase | Indicative timeline | Main tasks | Deliverable |
| --- | --- | --- | --- |
| Research framing and protocol | Month 1 | Finalize questions, inclusion rules, authorship/IP protocol, artist rubric, and locked-test policy. | Approved study protocol and corpus guide. |
| Pilot corpus and validation | Months 1–2 | Organize the initial four families; scan/photograph sources; validate registration and overlays; revise drawing instructions. | Pilot manifest, contact sheets, and registration report. |
| Corpus expansion | Months 2–5 | Add structurally varied base families with 30–70 deviations each; document provenance; freeze development and test designations. | Curated paired-family corpus; confirmed total entered on this plan. |
| Baseline model | Months 3–4 | Train the conditional model on the pilot; verify losses, checkpoints, calibrated strength, SVG output, and resume behavior. | Reproducible baseline checkpoint and technical report. |
| Main development experiments | Months 5–7 | Train on expanded development families; run automatic audit; perform ablations and failure analysis. | Fold metrics, ablation table, model-selection decision. |
| Locked unseen-base evaluation | Month 8 | Freeze configuration; evaluate locked test families; compute family-macro metrics and confidence intervals. | Final technical evaluation dataset and report. |
| Artist review study | Months 8–9 | Conduct blind candidate rating, assisted/unassisted search sessions, interviews or reflective annotation. | Ratings, coded comments, workflow measurements. |
| Studio translation | Months 9–10 | Redraw/recombine selected candidates; make small ink studies; develop selected large-scale compositions. | Documented studies and completed or in-progress artworks. |
| Synthesis and writing | Months 10–11 | Relate technical results, artist judgments, and material outcomes; write methods, limitations, and contribution chapters. | Full research draft and image archive. |
| Revision and dissemination | Month 12 | Audit reproducibility; redact sensitive paths/data; prepare exhibition/demo, paper, presentation, and source release. | Final report, presentation, demo, and dissemination package. |

### Decision gates

- **After pilot validation:** proceed only if registration overlays reliably distinguish redraw variation from additions.
- **After initial audit:** expand family diversity before tuning model capacity if held-out performance is unstable.
- **Before locked test:** freeze preprocessing, split logic, architecture, quality rules, and rating rubric.
- **Before public release:** review every dataset, checkpoint, generated example, and path for consent, copyright, and privacy.

---

## 11. Risks and Mitigations

- **Too few distinct bases:** many deviations from four bases may create strong reconstruction results but weak transfer evidence. **Mitigation:** prioritize new families, use whole-family audits, report uncertainty, and limit claims to the observed family diversity.
- **Uneven family sizes:** large families may dominate. **Mitigation:** sample families uniformly before deviations and macro-average evaluation across families.
- **Train/test leakage:** shared bases or near-duplicate deviations may inflate results. **Mitigation:** hash and group exact/near duplicates globally, prohibit leakage groups from crossing splits, and exclude cross-family duplicates from unseen-base scoring.
- **Registration errors:** hand redraws or scan transformations may be mislabeled as additions/removals. **Mitigation:** use tolerant similarity registration, save overlays, require minimum overlap, and manually audit low-overlap and high-removal cases.
- **Model destroys the base:** a flexible decoder may erase or bend defining lines. **Mitigation:** explicit addition/removal heads, identity training, retention loss, calibrated removal bounds, and review routing.
- **Mode collapse or repetitive symbols:** samples may differ numerically but not structurally. **Mitigation:** learned conditional prior, KL warmup, fixed-seed diversity measurement, temperature testing, topology signatures, and accepted-output novelty registration.
- **Novel but unusable output:** novelty scores may reward arbitrary geometry. **Mitigation:** combine topology and quality checks with blind artist ratings and studio tests; never equate metric novelty with artistic value.
- **SVG conversion changes topology:** curve fitting may join or delete lines. **Mitigation:** rerender every SVG, compare topology signatures, and fall back to linear paths when curves exceed error limits.
- **Compute cost:** automatic family folds substantially increase training time. **Mitigation:** use small pilot models, shortened audit epochs, CUDA mixed precision, early stopping, and resumable active-fold checkpoints.
- **Dataset or checkpoint disclosure:** self-contained checkpoints can reconstruct packed training masks and may record local paths. **Mitigation:** keep runs local, exclude them from Git, redact releases, and publish only deliberately selected examples or non-sensitive weights.
- **Copyright and consent:** artist-derived symbols may be identifiable or restricted. **Mitigation:** use artist-owned, public-domain, or explicitly licensed sources; maintain a rights ledger; separate internal research data from public examples.
- **Overclaiming authorship or automation:** the system may be presented as autonomously creative. **Mitigation:** document artist decisions and manual transformations, describe outputs as proposals, and report rejected as well as selected candidates.
- **Cultural flattening:** “Chinese ink” could be reduced to a visual effect detached from its histories and materials. **Mitigation:** situate the project in appropriate scholarship and lived practice, consult specialists, and frame the model as a line-structure tool used within—not a simulation of—the ink tradition.
- **Digital-to-material gap:** clean SVGs may fail when enlarged or painted with ink. **Mitigation:** introduce small physical studies early, document changes caused by brush, paper, water, scale, and gesture, and treat translation as a research stage rather than a final export step.
- **Researcher confirmation bias:** the artist may prefer outputs that support the intended argument. **Mitigation:** predeclare rating criteria, blind model labels, retain failure sets, involve additional reviewers where possible, and preserve an auditable decision trail.

---

## 12. Expected Contribution

The expected contribution is a documented bridge between a specialist visual practice and a transparent generative method. The project will show how line-symbol generation can be framed not as automatic image production but as structured variation around artist-supplied bases, evaluated through both unseen-family evidence and material studio consequences.

### Contribution to artistic practice

- A repeatable method for expanding the search field of line shapes and combinations.
- A visual archive that makes relations among base, addition, removal, and completed symbol inspectable.
- New compositional starting points for large-scale contemporary ink paintings.
- A critical account of where machine suggestion assists, distracts, surprises, or fails the artist.

### Contribution to research method

- A paired-family corpus model suited to hand-redrawn bases and complete deviations.
- A family-level evaluation protocol that avoids treating related images as independent evidence.
- A mixed technical/studio rubric connecting topology, novelty, retention, and diversity to artistic usefulness.
- A reproducible provenance chain from source image through checkpoint and generated SVG to ink study.

### Technical contribution

- A conditional β-VAE with learned base/strength prior and addition/removal composition.
- Pure-Python registration, geometry, SVG rerendering, and novelty fallbacks with minimal dependencies.
- Optimized descriptor search and bounded precise comparisons for novelty screening.
- Safe paired-family checkpoints supporting exact resume and expanded-dataset initialization.
- A private loopback webpage exposing the complete method without requiring a web framework or cloud service.

### Concrete deliverables

- Research protocol and paired-family dataset guide.
- Final curated corpus metadata and non-sensitive contact sheets.
- Local trainer/generator source code and test suite.
- Audit, ablation, and locked-test reports.
- Model and generation documentation.
- Curated symbol atlas with acceptance/rejection rationale.
- Ink studies and selected large-scale artwork documentation.
- Final written dissertation/report, presentation, and demonstration.

The strongest outcome would not be the largest number of generated symbols. It would be a defensible account of **how a controlled generative search changed what the artist was able to see, consider, reject, and ultimately make**.

---

## 13. Reference Links and Literature Development

### Technical references

- Ha, D., & Eck, D. **A Neural Representation of Sketch Drawings (Sketch-RNN).**  
  <https://arxiv.org/abs/1704.03477>
- Carlier, A., Danelljan, M., Alahi, A., & Timofte, R. **DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation.**  
  <https://proceedings.neurips.cc/paper/2020/hash/bcf9d6bd14a2095866ce8c950b702341-Abstract.html>
- Sohn, K., Lee, H., & Yan, X. **Learning Structured Output Representation Using Deep Conditional Generative Models.**  
  <https://proceedings.neurips.cc/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html>
- Higgins, I., et al. **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.**  
  <https://openreview.net/forum?id=Sy2fzU9gl>
- Shit, S., et al. **clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.**  
  <https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html>
- Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. **Human-level concept learning through probabilistic program induction (Omniglot).**  
  <https://www.science.org/doi/10.1126/science.aab3050>
- Google Creative Lab. **The Quick, Draw! Dataset.**  
  <https://github.com/googlecreativelab/quickdraw-dataset>

### Project documentation

- **Paired-Family Line Symbol Trainer: implementation, controls, checkpoints, and workflow.**  
  [README.md](./README.md)
- **Trainer and command-line API.**  
  [train.py](./train.py)
- **Local browser interface.**  
  [app.py](./app.py)

### Art-historical and practice-based literature to finalize

Before formal submission, this section should be expanded with sources selected in consultation with the artist and supervisor in the following areas:

- histories and theories of Chinese ink painting;
- contemporary ink practices and debates around scale, abstraction, and materiality;
- line, gesture, repetition, symbol, and structural composition;
- practice-based and studio-based research methodology;
- computational creativity, co-creation, and artist–AI authorship;
- ethics, cultural specificity, copyright, and dataset consent in generative art.

These sources should do more than provide background. They should shape the interpretation of the generated symbols, the artist-review rubric, and the claims the project is permitted to make.

---

## Research Record Checklist

- [ ] Enter the confirmed total image count after deduplication and corpus freeze.
- [ ] Record dataset and checkpoint rights/consent status.
- [ ] Freeze development and locked-test family lists.
- [ ] Predeclare primary metrics, baselines, ablations, and rating rubric.
- [ ] Save every effective configuration, seed, and software version.
- [ ] Preserve representative failures and rejected candidates.
- [ ] Link selected generated symbols to studio studies and final works.
- [ ] Complete the art-historical and practice-based bibliography.
- [ ] Redact private paths and training masks before public release.
- [ ] Distinguish technical novelty, artistic judgment, and legal originality in all reporting.
