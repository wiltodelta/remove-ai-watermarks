# Research sweep, September 2026

A survey of every area the library works in, run on 2026-09-22 and 2026-09-23 to
find what changed since the topic pages were written and what earlier research
missed. Nine areas were covered: Google SynthID, other vendors' invisible
watermarks, the watermark attack literature, visible AI labels, metadata and
provenance standards, video and audio, AI image classifiers, law and regulation,
and fill or regeneration models with competing tools. Claims were checked
against primary sources (specifications, vendor documentation, legal texts,
arXiv pages); live captures from the maintainer's accounts settled several
questions directly.

This page is an index. Each finding lives in its canonical page.

## Where the findings went

| Area | Change | Page |
| --- | --- | --- |
| Law and regulation | Research purpose, no-liability notice, jurisdiction table (China, EU, India, US, South Korea), platform labeling behavior | [legal-and-safety.md](legal-and-safety.md) |
| SynthID adoption | 100B+ items, ElevenLabs and OpenAI audio (with OpenAI's verification API limits), Apple (announced), Vertex C2PA, Nano Banana 2 Lite and Gemini Omni Flash, optional visible mark, portal status | [synthid.md](synthid.md), [provider-oracles.md](provider-oracles.md) |
| Other vendors | Microsoft Foundry provenance for Azure-hosted Flux and MAI, Amazon Nova Reel, audio adopters | [watermarking-landscape.md](watermarking-landscape.md) |
| C2PA 2.3 and 2.4 | `c2pa.ai-disclosure` and ingredient `digitalSourceType` now read; parser claim corrected | [watermarking-landscape.md](watermarking-landscape.md) |
| Containers | GIF metadata stripped without re-encoding; TIFF and DNG refused; C2PA after large WAV and AVI data now scanned | [known-limitations.md](known-limitations.md) |
| TC260 in audio | WAV, MP3, OGG, Opus and FLAC label placements from the 2025-08 audio guide | [supported-signals.md](supported-signals.md) |
| Video labels | Sora discontinued; opening-only labels no longer filled over the whole clip; any video mark vetoed by another vendor's C2PA AI claim, Kling also by a non-Kling TC260 producer; TC260 video producers named | [supported-signals.md](supported-signals.md), [known-limitations.md](known-limitations.md) |
| Apple | Image Playground and Photos Clean Up credits attributed | [supported-signals.md](supported-signals.md) |
| New visible marks | `vidu` video mark and `wan` image mark, each calibrated on one real export | [supported-signals.md](supported-signals.md), [known-limitations.md](known-limitations.md) |
| Google Photos | AI edits reported as "Google Photos (AI edit)" rather than Gemini, with SynthID present: Google's checker found it in 4 of 4 Photos AI edits (Ask, eraser and two others) | [synthid.md](synthid.md), [supported-signals.md](supported-signals.md) |
| YouTube | Upload test: C2PA-driven "Made with AI" label, public streams without C2PA, Studio downloads re-signed by YouTube; re-encoded non-Google video no longer read as Google AI with SynthID | [supported-signals.md](supported-signals.md), [legal-and-safety.md](legal-and-safety.md), [module-internals.md](module-internals.md) |
| Audio | Generated audio from OpenAI, ElevenLabs and Microsoft passes through the video path unchanged | [known-limitations.md](known-limitations.md) |
| Attack literature | Fidelity band on open encoders now 30-45 dB; WmForger and WMCopier status | [synthid-removal-research.md](synthid-removal-research.md), [watermark-forgery-study.md](watermark-forgery-study.md) |
| Models | FLUX.2 klein has a strength-controlled pipeline in diffusers | [chroma1-engine-research.md](chroma1-engine-research.md) |
| Dependencies | TrustMark extra on every supported Python (`trustmark>=0.9.2`) | [installation.md](installation.md) |
| Live captures | Gemini Omni, Grok, Vidu, Wan, Apple, Google Photos, Nano Banana 2 Lite and Gemini Omni Flash through the API; Runway, Higgsfield and Dreamina | [watermarking-landscape.md](watermarking-landscape.md#live-captures-2026-09-23-and-2026-09-24) |

## Open items

- **Google Photos Reimagine manifests.** Magic Editor, and so Reimagine, is not
  offered in Google Photos on iOS, so no Reimagine file is captured. The
  question it was meant to answer is settled without one: Google's checker found
  SynthID in all four Photos AI edits tested on 2026-09-25, none of which
  records SynthID in its manifest.
- **Vidu and Wan marks on more samples.** Both are registered from one real
  export each (`vidu` video, `wan` image); their gates are provisional until a
  wider cohort is measured, and a Wan video label is not yet captured.
- **TC260 practice guides.** The audio placements from TC260-PG-202510A are
  now read (the guide PDFs load in a browser from
  `tc260.org.cn/tc260/sjzn/list.shtml`, not to scripted requests). The text-file
  guide (TC260-PG-20258A) covers documents this image and video tool does not
  process. The security-protection guide (TC260-PG-202511A) records a
  `SecurityData` JSON object, a digital signature over the label and optionally
  the content, in `ReservedCode1` and `ReservedCode2`. The library accepts those
  fields as label evidence but does not verify the signature.
- **Experiments.** Chroma restoration after regeneration (from the NeurIPS 2024
  "Erasing the Invisible" winning solution, arXiv:2508.21072), FLUX.2 klein and
  Z-Image-Turbo as global stages (with an Apple Silicon arm through mflux),
  linear probes on newer frozen backbones for `classify`, Resemble Perth and
  Meta PixelSeal as benchmark oracles, re-watermarking as a removal primitive
  (arXiv:2605.16796, MIT code), RAVEN view-synthesis removal
  (arXiv:2601.08832, no code released as of 2026-09-24), and Apache-2.0 video
  inpainting (VACE, ROSE) against the per-frame visible video fill.
  MiniMax-Remover (CC-BY-NC-4.0 weights) and ProPainter (S-Lab License,
  non-commercial) are excluded on license.
- **New C2PA signers without a vendor row.** The
  [C2PA conformance list](https://github.com/c2pa-org/conformance-public/blob/main/conforming-products/conforming-products-list.json)
  names generator certificates `Amazon Bedrock`, `Getty Images
  AI-Generated Image` and `AI-Modified Image` (and the iStock pair), `Jasper`,
  `RefaceAI`, and vivo `vivo Albums` and `JOVI Albums`. It also lists `Google
  Photos` for both Android and iOS, the signer the Photos rule keys on. A
  real sample per vendor is still needed to confirm the manifest shape before
  adding a row. Runway is done: its own models sign as `RUNWAY AI, INC.`
  (measured 2026-09-24) and the row covers the conformance-list spelling too.
- **Chinese phone galleries.** Huawei's help pages say the Xiaoyi photo-edit AI
  watermark can be switched off; Honor, OPPO and Xiaomi label behavior is not
  documented in any primary source found. vivo's Album setting is already
  covered.
- **Claims not supported by a primary source.** Kling embedding SynthID
  (repeated by remover blogs; Kuaishou is not in Google's adopter list),
  Hailuo pixel watermarks, and platforms stripping C2PA on upload. None is used
  by the library.
- **Google Photos small edits (deferred).** The "SynthID present" rule for
  Photos AI edits rests on four checked edits, none of them tiny. Still to
  measure on the phone app: a tiny eraser edit, a large one, a small "Help me
  edit" change, a restyle, and a crop-only control, each checked in Gemini. iPhone
  Mirroring does not show the Photos editing toolbar, and the web editor has no
  generative tool, so this needs edits made on the device.
- **Captures blocked by region or device.** Doubao, Jimeng, Samsung. Higgsfield,
  Runway and Dreamina were captured on paid plans on 2026-09-24.

## Decisions

- **Dreamina `AI` badge: not registered (2026-09-26).** Dreamina stamps a small
  boxed `AI` badge top-left on images and video, and calls it its visible AI
  watermark. It stays unregistered for three reasons. A generic boxed `AI` is the
  shape of the disclosure icon the EU Code of Practice defines, which this
  project does not template, so a template would remove any such label. The same
  generic shape would match other platforms' badges and plain `AI` text and
  misattribute them to Dreamina. And it adds nothing to attribution: Dreamina
  exports are identified by their C2PA (`ByteDance Media Transcode Service`
  signer with a `Dreamina/7.5.0` ingredient), and Dreamina's own "Remove watermarks" setting
  removes the badge. Revisit only if the badge becomes a distinctive brand mark.
