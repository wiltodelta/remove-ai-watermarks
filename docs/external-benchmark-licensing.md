# External watermark benchmarks: ETI and W-Bench licensing verdict

Investigation completed 2026-09-07 against the primary sources, closing the
last open item of the original research map: whether an honest external
benchmark profile can be built on ETI or W-Bench. The verdict is split:
W-Bench is legally usable, ETI is license-blocked.

## ETI (NeurIPS 2024 "Erasing the Invisible")

Sources verified: the Hugging Face dataset
`furonghuang-lab/ETI_Competition_Data` at revision `d80024580b5a` and the
organizers' evaluation repository `erasinginvisible/eval-program`.

- **License: absent everywhere.** The dataset carries no license tag and no
  license section in its card; the evaluation repository has no LICENSE file.
  With no grant of rights, the corpus is unusable in this public repository,
  and downloading it for local evaluation builds results that cannot be
  published. The only unblock is written permission from the organizers -
  an external action that needs an explicit decision, not a default.
- **Composition (verified from the card):** 300 Beige-track images
  (Gaussian Shading on Stable Diffusion 2.1, StegaStamp on Flux.1) and 300
  Black-track images (JigMark, PRC, StableSignature, Trufo, plus
  Gaussian-Shading+JigMark and StableSignature+StegaStamp doubles), plus
  ~1,072 Beige and ~1,650 Black valid submissions with per-watermark
  detection results, image-quality metrics, and final scores.
- One earlier blocker is now stale: the attacked submission images ARE part
  of the release per the card note, not missing. The license blocker alone
  remains decisive.

## W-Bench / VINE (ICLR 2025)

Sources verified: the Hugging Face dataset `Shilin-LU/W-Bench` at revision
`c81f3924f0fb` and the VINE repository `Shilin-LU/VINE`.

- **Dataset license: MIT**, declared on the dataset card. The benchmark is
  legally usable.
- **Composition (verified from the card):** 10,000 instances sourced from
  COCO, Flickr, ShareGPT4V and similar datasets, eleven watermarking
  methods, evaluated across four editing profiles - regeneration, global
  editing, local editing, image-to-video - plus image distortions. An
  earlier internal note said "15k images"; the card says 10,000, and the
  card wins.
- **Code license: NTUITIVE Non-Commercial** (NTU technology transfer;
  commercial use requires contacting the authors). The VINE code cannot be
  vendored into this Apache-2.0-licensed project, and pipeline outputs derived
  from running it would inherit that restriction. What remains lawful and
  useful: consuming the MIT dataset directly with our own pinned oracles
  and benchmark kernel, adopting the attack taxonomy as protocol
  definitions reimplemented in our stack.

## Verdict for the map item

An external W-Bench profile is buildable: MIT dataset, our kernel, our
oracles, no VINE code. Whether to build it is a scoping decision, not a
legal one - the image cohorts already cover detection/removal/fidelity
questions with pinned local evidence, and W-Bench would add its
editing-profile attack axis (regeneration, image-to-video). The ETI profile
stays closed until the organizers grant a license; nothing in this
repository should download or derive from it meanwhile.

One protocol detail is worth keeping as a candidate, not a commitment: the
ETI scoring program runs submissions through median blur and JPEG
compression before decoding, so an attack only wins if the watermark stays
dead under ordinary handling - that filter separates fragile
adversarial-perturbation removals from structural ones. Our removal paths
are structural (fill, regeneration), so no driving question exists today;
if removal durability under platform re-encoding ever becomes one, the
follow-up is an hour-sized handling arm over removed-state artifacts in the
existing cohorts. Participant attack code was never published (the release
carries attacked images and metrics only), so there is no implementation to
learn from even where reading would be lawful.
