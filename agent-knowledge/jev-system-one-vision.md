# Jev / System-One Vision — Agent Knowledge Note

**Status:** working technical synthesis, September 2026.  
**Purpose:** give another agent enough context to continue the Jev discussion and explore vision applications without needing the original ChatGPT conversation.

## 1. Core Jev idea

The useful mental model is not "a chatbot that answers faster." It is a **direct decision system over a semantic representation**.

Conventional autoregressive LLM:

```
input tokens
  -> transformer input processing / prefill
  -> KV cache
  -> token 1
  -> token 2
  -> ...
  -> token N
```

Jev-like decision system:

```
state/context
  -> semantic processing
  -> many parallel decisions / scores
  -> probabilities
  -> stop
```

The important distinction is that **input understanding still costs compute**. What disappears is the long serial autoregressive decoding path.

## 2. "Isn't this just outputting one token?"

That is a strong baseline and should always be compared against Jev.

A normal causal LLM can approximate a direct classifier by:

1. Prefill the input.
2. Read the next-token logits.
3. Restrict them to candidate answer tokens (A/B/C/D, yes/no, etc.).
4. Normalize those logits.
5. Stop without autoregressive decoding.

Conceptually:

```
prompt/context
   -> one causal-model forward pass
   -> next-token logits
   -> select candidate logits
   -> probabilities
   -> STOP
```

Therefore, **"one-token classification from an LLM" captures much of the latency advantage over ordinary generated answers.**

The potentially more interesting Jev property is not merely one-token output; it is **reuse of one encoded state across many independently specified decisions, evaluated in parallel**.

## 3. Input processing / "prefill"

Do not assume Jev removes prompt processing.

For a causal LLM, the input pass is called *prefill* because it also constructs a KV cache for later generation.

For a non-generative or encoder-style system there may technically be no "prefill" because there is no later decoder loop, but equivalent semantic computation remains:

```
tokens -> neural network -> contextual representation
```

A model cannot classify the contents of 5,000 tokens without processing those 5,000 tokens somehow.

"Near instant" can plausibly come from:
- no chain-of-thought generation,
- no serial token sampling,
- very small direct outputs,
- optimized serving,
- state reuse,
- parallel decision evaluation.

## 4. What is and is not publicly established about Jev

Public TypeSafe material describes:
- pretrained language models as the starting point,
- RLCD (Reinforcement Learning for Calibrated Decisions),
- direct probability/decision outputs,
- many questions evaluated in parallel against a state.

Architecture details such as exact backbone, attention structure, parameter count, causal vs bidirectional attention, and whether the production model literally uses classifier heads have not been publicly established in sufficient detail.

Treat claims about the exact hidden architecture as hypotheses, not facts.

Useful source starting points:
- https://typesafe.ai/blog/introducing-system-one-models-and-jev
- https://docs.typesafe.ai/
- https://systemonemodels.org/

## 5. Vision application: naive N-crop prototype

Suppose we want to find/label objects in an image.

Simple prototype:

```
image
  -> generate N crops
  -> stack N crops into a batch
  -> one batched vision-model invocation
  -> one embedding / score vector per crop
  -> objectness + open-vocabulary labels
```

Example:

```
[crop_1, crop_2, ..., crop_N]
          |
          v
      vision encoder
          |
          v
[N, embedding_dim]

candidate text labels
          |
          v
      text encoder
          |
          v
[M, embedding_dim]

scores = image_embeddings @ text_embeddings.T

=> [N, M] decisions
```

This has **no autoregressive generation**.

A practical backbone for this experiment is SigLIP / SigLIP2 or CLIP because arbitrary text labels can be supplied at runtime.

Example runtime labels:

```
[
  "bolt",
  "brake line",
  "hose",
  "control arm",
  "rust",
  "oil leak",
  "wiring",
  "bracket"
]
```

## 6. Important distinction: batching is not computation reuse

N crops in one GPU batch are parallel, but they still require approximately N crops' worth of visual encoding compute.

If overlapping crops contain the same pixels, the encoder repeatedly processes those pixels.

So:

```
crop 1 -> encoder
crop 2 -> encoder
crop 3 -> encoder
...
```

is useful as a prototype but is not the most Jev-like architecture.

## 7. Better vision architecture: encode the scene once

The stronger design is:

```
full image
    |
    v
vision encoder ONCE
    |
    v
dense spatial feature map
    |
    +--> region/query 1 --> semantic predicates
    +--> region/query 2 --> semantic predicates
    +--> region/query 3 --> semantic predicates
    ...
    +--> region/query N --> semantic predicates
```

Possible implementation mechanisms:
- ROIAlign / ROI pooling over dense image features,
- learned object queries,
- region tokens,
- cross-attention from region queries into shared image features,
- feature pyramids for multi-scale regions.

Then evaluate an arbitrary predicate bank:

```
N regions x M predicates -> N x M probability matrix
```

Example:

```
500 candidate regions
x
100 semantic predicates
=
50,000 direct decisions
```

from one expensive image encoding plus cheaper downstream scoring.

## 8. Do not limit predicates to object labels

The interesting part is that the runtime questions can be semantic predicates rather than fixed detector classes.

For every region:

- Is this a distinct object?
- What is it?
- Is it damaged?
- Is it corroded?
- Is it leaking?
- Is it movable?
- Can a robot grasp it?
- Is it dangerous to touch?
- Is it abnormal for this scene?
- Is it part of the braking system?
- Is a fastener missing?
- Does this correspond to the target in the previous frame?

Desired output is structured probability data, not prose:

```json
{
  "bbox": [448, 224, 672, 448],
  "objectness": 0.98,
  "labels": {
    "brake_line": 0.91,
    "hose": 0.06,
    "wiring": 0.03
  },
  "predicates": {
    "corroded": 0.81,
    "leaking": 0.08,
    "damaged": 0.14
  }
}
```

## 9. Existing vision ideas that already cover pieces of this

Do not present this whole concept as novel without checking prior work.

Relevant families:

**CLIP / SigLIP**
- joint image/text embeddings,
- open-vocabulary semantic scoring.

**DETR**
- encode an image and use many parallel object queries,
- direct set prediction rather than sliding-window classification.

**SAM**
- expensive shared image encoding,
- comparatively cheap downstream mask queries.

**Open-vocabulary object detection**
- object detection with dynamically supplied text labels.

**Region-based detectors / ROIAlign**
- reuse a shared convolutional/visual feature map for many candidate boxes.

The useful research direction is to combine these ideas with a Jev-like decision interface:
- shared state,
- many dynamically supplied questions,
- direct probabilities,
- calibration,
- no language decoding.

## 10. Main hypothesis to test

The central hypothesis is:

> The useful Jev-like property for vision is not "generate only one token." It is **encode a rich scene once, then cheaply evaluate many independently specified semantic questions against that shared representation.**

For vision, the target abstraction is therefore:

```
one scene representation
        +
N spatial queries
        x
M semantic predicates
        =
large direct decision matrix
```

rather than:

```
N crops treated as N unrelated images
```

## 11. Experimental roadmap

### v0 — batched crops
- Sliding or proposal-based crops.
- Batch all crops.
- SigLIP/CLIP image embeddings.
- Dynamic text labels.
- Direct similarity scores.
- Benchmark total throughput and latency.

### v1 — shared image features
- Encode full image once.
- Obtain dense spatial features.
- Pool N candidate ROIs.
- Score all ROIs against dynamic labels.
- Compare compute and latency against v0.

### v2 — predicate matrix
Move beyond labels:
- objectness,
- attributes,
- state,
- abnormalities,
- affordances,
- relationships.

Evaluate N regions x M predicates in parallel.

### v3 — probability calibration
Measure whether a reported probability corresponds to observed accuracy.

Possible tools:
- temperature scaling,
- isotonic calibration,
- learned calibration layers,
- task-specific post-training.

### v4 — video / robotics
Reuse scene/state representations over time:
- frame encode,
- object/state tracking,
- many cheap predicates per frame,
- temporal comparison queries.

## 12. Benchmarks that matter

Do not compare only against a chat model generating paragraphs.

Compare against strong low-latency baselines:

1. Standard VLM with normal generation.
2. Same VLM forced to one-token output.
3. CLIP/SigLIP batched crop classifier.
4. Shared-feature ROI model.
5. Open-vocabulary detector.
6. DETR-like query model.

Measure:
- first-result latency,
- total latency,
- images/sec,
- regions/sec,
- predicates/sec,
- FLOPs,
- GPU memory,
- accuracy,
- calibration error,
- scaling with N regions,
- scaling with M predicates,
- scaling with image resolution.

The key experiment is whether increasing N and M has **sublinear incremental latency** because expensive state encoding is shared.

## 13. Agent instruction

When continuing this research:

- distinguish **GPU batching** from **actual compute reuse**;
- distinguish **input encoding** from **autoregressive decoding**;
- compare every claimed Jev advantage against a one-token/logit-reading baseline;
- treat exact Jev architecture details as undisclosed unless new primary-source evidence appears;
- search prior work in DETR, SAM, CLIP/SigLIP, ROIAlign, open-vocabulary detection, region-language models, and calibrated classifiers before claiming novelty;
- prefer direct structured probability outputs over natural-language generation when designing the system;
- focus on the scaling behavior of **one shared state x many queries**.
