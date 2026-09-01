# OpenAI SynthID oracle ladders

> This page is a routing hub. The mixed 2026-08 campaign log was split by
> purpose on 2026-08-22. Numeric tables now live on the page for that
> purpose. Raw images stay gitignored under
> `.local-eval/synthid/prc-oklab-attack-2026-08-15/`.
>
> Seeds s1, s2, s3 are listed in
> [SynthID local detector research](synthid-detector-research.md).

| Page | Use it for |
| --- | --- |
| [SynthID local detector research](synthid-detector-research.md) | Hunt for a keyless local mark detector. Open. Closed routes on that page. |
| [AI-generated image classifiers](ai-generated-image-classifiers.md) | Primary metadata-free AI-generation task, Model 1 partial result, and frozen-checkpoint sweep. |
| [SynthID source classifiers](synthid-classifiers.md) | OpenAI/Gemini source finding and `pipeline_lattice` as google-lineage. |
| [SynthID mark removal research](synthid-removal-research.md) | Quiet-remover hunt. Closed except ~19-24 dB 16-32 scramble and blur sigma 7. |
| [Mechanism reference](synthid.md) | How SynthID works, provenance, robustness, regeneration. |
| [Chronological plan archive](synthid-detector-removal-plan.md) | Dated H-gates, corpora, and session notes in original order. |

## Where former sections went

| Former heading | Now |
| --- | --- |
| 2026-08-21 pairs, flats, L1 labels | [detector](synthid-detector-research.md) |
| Identity, token/layout, tomography, preprocess E1 | [detector](synthid-detector-research.md) (presentation gate) |
| L1 repair, L1 geometry, camera vs edit pair, E3 | [detector](synthid-detector-research.md) |
| CLIP-L photo vs AI, provider CLIP, union | [AI classifiers](ai-generated-image-classifiers.md), [source classifiers](synthid-classifiers.md) |
| 124-d three-class and binary AI | [source classifiers](synthid-classifiers.md) |
| `pipeline_lattice` re-check, Spaces census | [source classifiers](synthid-classifiers.md) |
| Attack / kill / flip / add / surround ladders | [removal](synthid-removal-research.md) |
| 16-32 titration, E2 scramble, Bayer and geometry | [removal](synthid-removal-research.md) |
| S4 provider split, M2 Imagen `addWatermark`, reverse-SynthID, Bypass | [detector](synthid-detector-research.md) |
| Photo `d'` budget 13.4 dB, 128-photo student, 16-32 residual without flat `G` | [detector](synthid-detector-research.md) |
| OKLab codeword replacement, add-context as presentation | [removal](synthid-removal-research.md) |
| Three-class OpenAI / Gemini / photo ask | [source classifiers](synthid-classifiers.md) |
| Wild hypotheses 2026-08-23 (sort, CDMA, bispectrum, PRNU, affine 16-32 NCC no lock) | [detector](synthid-detector-research.md) |
| Wild oracle 2026-08-23: misnamed polar-1632 is cartesian scramble replicate `not_detected`; band-transplant `not_detected`; palette64 `detected` | [removal](synthid-removal-research.md) |
| External literature 2026-08-23 (Gowal, AWPD, PRC, Zhao, UnMarker, CtrlRegen, MarkNull, reverse-SynthID) | [detector](synthid-detector-research.md), [removal](synthid-removal-research.md), [AI classifiers](ai-generated-image-classifiers.md) |
| Adjacent literature, not SynthID (Cox, HiDDeN, StegaStamp, Tree-Ring, Ojha CLIP, Corvi Fourier, DCCT CFA, PRNU PCE) | [detector](synthid-detector-research.md), [AI classifiers](ai-generated-image-classifiers.md), [removal](synthid-removal-research.md) |
| Image investigation and data hiding (LSB, UNIWARD, Baluja, HiNet, Gaussian Shading, ELA, JPEG ghosts, DIRE, CNNDetect) | [detector](synthid-detector-research.md), [AI classifiers](ai-generated-image-classifiers.md) |
| Waveforms in a picture (DFT, Fourier-Mellin, Zernike, chirps, Fresnel, DRPE, cyclostationary) | [detector](synthid-detector-research.md) |
| Remaining hypotheses 2026-08-23 (CFA, JPEG ghost, Paeth, face-gate, Baker, Poisson, nested LSB, angular scramble) | [detector](synthid-detector-research.md), [removal](synthid-removal-research.md), [AI classifiers](ai-generated-image-classifiers.md) |
| Non-local 2026-08-23 (HEIF/AV1 survive, print-scan sim 24.95 dB still detected, gpt-image-1.5/mini encoder-off, dated gpt-image-2 same stamp, DIRE DDIM texture) | [detector](synthid-detector-research.md), [removal](synthid-removal-research.md), [AI classifiers](ai-generated-image-classifiers.md) |
| Waveform shells 2026-08-23 (Mellin/Zernike/PCET/Fresnel no photo lock; Y scramble kills, chroma and 90° sector do not; radial-phase kills) | [detector](synthid-detector-research.md), [removal](synthid-removal-research.md) |
| Public decoder sweep 2026-08-23/24 (`Rinne414` fixed templates 0/400 on current OpenAI/Google images; injected template spoofs the public correlator, but 0.25-2x subtraction stays official `detected` at 43.13-25.72 dB) | [detector](synthid-detector-research.md), [removal](synthid-removal-research.md) |
| Public CNN sweep 2026-08-24 (`newideas99` retains 172/200 OpenAI but accepts 110/200 Google; a whole-frame gradient makes the ensemble effectively zero but stays official `detected` through ±8/255 / 34.30 dB) | [detector](synthid-detector-research.md), [source classifiers](synthid-classifiers.md), [removal](synthid-removal-research.md) |
| `reverse-SynthID` V4 repeat 2026-08-24 (77/200 Google and 76/200 OpenAI at the published cut; Round-06 manual Gemini verdicts are not published as per-image evidence) | [detector](synthid-detector-research.md), [source classifiers](synthid-classifiers.md), [removal](synthid-removal-research.md) |
| Broad GitHub sweep 2026-08-24 (133 unique primary repositories; REGRET and the vordme flat-field SVM fail the strict v7 transfer; literal ports and generic heuristics add no signal) | [detector](synthid-detector-research.md), [source classifiers](synthid-classifiers.md) |
| Additional public removal sweep 2026-08-24 (DeSynth stays official `detected`; froggeric and other ControlNet/diffusion projects corroborate lossy regeneration; PortOS closes high-fidelity phase/noise jamming) | [removal](synthid-removal-research.md) |
| Google verifier surfaces 2026-08-24 (retired Vertex `imageverification@001`; live but unlisted and inaccessible `synthid.googleapis.com/upload/v1:verifyContent`) | [detector](synthid-detector-research.md) |
| Metadata-free source hunt 2026-08-24 (1% cascades rejected; frozen multiscale fusion found 276/600 OpenAI-or-Google targets, 0/1,000 photographs and 1/25 TC260 on blind v4) | [source classifiers](synthid-classifiers.md) |
| Original-export source hybrids 2026-08-24 (frozen v7 215/400 exact; post-hoc v8 292/400 and per-codec v11 316/400 exact) | [source classifiers](synthid-classifiers.md) |
| Frozen general-detector transfer 2026-08-24 (Community Forensics and SPAI do not replace or safely fuse with Model 1 at the 1% calibration FPR; B-Free weights unavailable and license noncommercial) | [AI classifiers](ai-generated-image-classifiers.md#frozen-public-checkpoint-transfer-2026-08-24) |
| General-classifier GitHub sweep 2026-08-25 (2,006 unique repositories found; DDA and PGC expose complementary frozen signals, but DDA raises independent FPR and PGC's OpenAI gain confounds 24/24 Kodak scans; DGS-Net, SAFE, RINE, and Nonescape Mini rejected) | [AI classifiers](ai-generated-image-classifiers.md#general-ai-classifier-github-sweep-2026-08-25) |
