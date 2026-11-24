# SAE capacity comparison — teacher-model interpretability review

Reviewer: `opus_teacher_model_pass`. Scope: the ~30 auto-sampled features per width
(10 highest-frequency + 10 median-frequency + 10 rarest-alive) recorded in
`evaluations/interpretability/sae_{4096,8192,16384}_feature_stats.json`. Each feature
was hand-labelled into one of four categories from its top activating examples; full
per-feature justifications live in the sibling `sae_*_reviewed.json` files.

## Label distribution by width

| Label | 4096 | 8192 | 16384 |
|---|---|---|---|
| template_driven | 6 | 5 | 5 |
| domain_or_script_marker | 7 | 6 | 7 |
| coherent_semantic | **3** | **1** | **3** |
| uninterpretable_or_mixed | 14 | 18 | 15 |
| **total** | 30 | 30 | 30 |

Coherent-semantic share: 4096 = 3/30 (10%), 8192 = 1/30 (3%), 16384 = 3/30 (10%).

## What the categories actually contain

- **template_driven** is the single biggest source of *high-activation* features at every
  width. These fire at ~2–3 activation on a fixed sentence frame with an interchangeable
  slot, regardless of domain or meaning — e.g. the Gandhi "Summarize the established
  historical information about X that illuminates participant experience" (8192 feat 795),
  "A historian examines X through political context" (8192 feat 963 / high-freq 7210),
  and the Science "Compare a X and a Y with respect to mechanism…" (4096 feat 3036),
  "Given the observation that X, formulate a hypothesis…" (8192 feat 3938), and
  "How do a X and a Y differ in their underlying assumptions…" (16384 feat 10754). These
  are dataset-template artifacts, not concepts.
- **domain_or_script_marker** is the other large bucket, and is dominated by Hindi-script
  detectors (e.g. 4096 328/1065/1692/2982/3041; 8192 2728/2974/6271/7906; 16384
  306/1657/4105/6518) plus a few whole-domain detectors (DNA: 8192 2922, 16384 133/2468;
  EMOTIONS: 4096 64/836, 16384 15234). The Hindi features in particular key on Devanagari
  rather than any meaning.
- **uninterpretable_or_mixed** is inflated at every width by two structural artifacts: (a)
  the rarest-alive sampling bucket is mostly *dead single-token* features (≈10 per width)
  that fire once at trivial activation, and (b) each width has exactly one near-ubiquitous
  "always-on" unit firing on ~97% of tokens (4096 feat 3902, 8192 feat 972, 16384 feat
  4068). Neither is concept-specific.
- **coherent_semantic** is the smallest bucket everywhere and is essentially always a
  single-domain subtopic: emotion-regulation / emotional-subtext / familial context at
  4096 (978, 1740, 4037); genomics-sequencing methodology at 8192 (5558); classical
  mechanics, lagging-strand/Okazaki replication, and DNA alkylation repair at 16384 (6556,
  15175, 6531). None cleanly generalises a concept across *different* domains.

## Did wider dictionaries produce more genuinely interpretable features?

Not in this sample. The coherent-semantic count is flat-to-noisy across widths (3, 1, 3),
and 8192 actually looks the *worst* on this axis — most of its extra capacity in the
sampled band went to redundant Gandhi/Science templates, Hindi-script markers, and dead
singletons rather than new concepts. The honest read is that **widening the dictionary
mostly multiplied template and script/domain artifacts, not genuine concepts.**

The one qualitative difference worth noting is *quality, not quantity*: 16384's semantic
hits (classical mechanics, Okazaki-fragment processing, alkylation repair) are crisper,
narrower single-subtopic features than 4096's broader emotion-themed ones — consistent
with feature-splitting giving sharper but not more numerous monosemantic units at higher
width. But on the headline question, there is no evidence here that bigger dictionaries
yielded a higher fraction of genuinely interpretable features.

## Caveats

This is a ~30-feature-per-width *sample*, deliberately skewed toward frequency extremes
(the rarest-alive third is structurally near-dead), so the absolute "fraction semantic"
figures understate what a uniform random sample of the dictionary might show. The
cross-width *comparison* is the more reliable signal, since all three widths were sampled
by the same procedure. Labels rely on top-example inspection only (no activation-steering
or causal validation), so "coherent_semantic" here means "the top stimuli share real
meaning across templates," not a verified causal concept.
