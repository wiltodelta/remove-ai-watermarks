# SynthID mark removal research

> Research archive for pixel-only SynthID removal, not diffusion
> regeneration. Not a statement of current product capability. Shipped
> invisible removal is lossy regeneration:
> [known limitations](known-limitations.md).
>
> Sister pages: [SynthID local detector](synthid-detector-research.md),
> [SynthID source classifiers](synthid-classifiers.md),
> [mechanism reference](synthid.md).

## Result

The quality-preserving OpenAI SynthID remover hunt closed 2026-08-20.
Bayer, VNG demosaic, upscale-then-Bayer, barrel distortion, scanline
jitter, and a 2 px shift closed 2026-08-22 on s1/s2 and 2026-08-23 on s3
and fish: they leave the official oracle `detected`.

Working residual kills on photographs cost about 19-26 dB:

- 16-32 px cartesian phase scramble (s1 24.6-24.8 dB, fish 23.2 dB, s2 19.0 dB)
- Fourier-angle scramble of the same annulus (s1 24.1 dB)
- Radial-phase scramble of the same annulus (s1 25.0 dB)
- Y-only 16-32 scramble (s1 24.6 dB); Cb/Cr-only do not kill
- File named polar-1632, actually cartesian (s1 25.6 dB)
- Replace 16-32 with a COCO photo's 16-32 (s1 25.2 dB)
- Gaussian blur sigma 7 (23.8 dB), holds 3/3 versus sigma 6
- Additive 16-32 jam only at a=24 / 18.6 dB, worse than scramble

Baker-map, 8-seam carve, Poisson, nested LSB, palette64, and ICC rewrite
do not kill at a better PSNR.

JPEG q5, noise sigma 16, grayscale, rot90, flip, 5°, downscale 0.20x,
median 7, posterize 4, VAE round-trip, and white pad to 40% linear stay
`detected`. Elastic warp is not a stable kill (s3 still `detected` at
22.0 dB).

The product remainder is diffusion regeneration (`qwen-zimage` /
`sdxl-zimage`), which does not decode and delete a payload. Defeating the
verifier does not restore forensic deniability; see
[synthid.md](synthid.md#23-removal-attacks-and-forensic-detectability).

Official `not_detected` on collage, photo-pad, and two-panel layouts is a
presentation gate, not residual damage. Those rungs are in
[detector research](synthid-detector-research.md).

## Closed quiet removers

| Attack | Close | Notes |
| --- | --- | --- |
| Quality-preserving photo remover | 2026-08-20 | Every residual `not_detected` that is not a collage is below usable quality |
| Additive in-band jam as a quiet remover | 2026-08-21 | Fish flips only at 18.6 dB; 4-8 px at the same PSNR stays `detected` |
| Bayer bilinear / VNG / upscale-Bayer | 2026-08-23 | s1/s2/s3 and fish still `detected`. s3 bilinear 37.1 dB, fish 34.7 dB. VNG on s2 is dirtier than scramble (18.5 vs 19.0 dB) and the mark remains |
| Barrel k1=0.06, scanline ±0.8 px, and shift 2 px | 2026-08-23 | s1/s2/s3/fish `detected` even at 14-20 dB barrel. Fish scanline 32.5 dB and shift 22.7 dB still `detected` |
| TrustMark-style micro-warp | 2026-08-21 | 0.25 px / 32 dB still 100% detect on TrustMark P; OpenAI elastic ~21 dB unreplicated |
| OKLab random-codeword replacement as a quiet wipe | 2026-08-15 | On four public Google-oracle positives, replacing the period-16 tile at 0.95 dropped the *local* lattice score below 0.173 at ~54 dB. Gemini pixel verify never accepted the candidates (`Connecting to Verify AI`). Local-score intervention, not an official wipe |
| PRC-style OKLab a/b noise plus spectral peak cleaning | 2026-08-15 | Independent high-pass OKLab noise at 32.6 dB moved the median local score by only `-0.0219`. Comb suppression moved one of four below threshold at 58 dB. Not a quiet official kill |
| Rinne414 fixed-template subtraction | 2026-08-24 | The pinned pre-May GPT-Image2 residual was resized onto a current verified OpenAI image and subtracted at 0.25x, 0.5x, 1x, and 2x. Official SynthID stayed `detected` at 43.13, 37.49, 31.60, and 25.72 dB after metadata stripping. Even the dirtiest rung is no better than the existing 24-25 dB annulus-phase scramble. The template can spoof its own public correlator but does not cancel the current mark |
| `newideas99` CNN gradient | 2026-08-24 | A 3x3 crop-grid objective spread the white-box perturbation over the full marked frame. The frozen ensemble fell from 0.99995 to 0.25956 at ±2/255 (45.92 dB) and to effectively zero at ±4/255 (40.30 dB). Official OpenAI SynthID stayed `detected` at ±1, ±2, ±4, and ±8/255, through 34.30 dB. The gradient attacks the surrogate's shortcut, not the production mark |
| `reverse-SynthID` V4 Round-06 | 2026-08-24 | The advertised `final`/`nuke` path is SD-VAE regeneration plus elastic and affine warps, resize squeeze, color change, residual FFT subtraction, and a JPEG/noise chain, with PSNR floors of only 14/11 dB. The repository claims 20 manual Gemini-app successes but contains no manifest, tally, or per-image verdicts. Its four bundled older cleaned pairs are 45.7-50.2 dB, yet the reproduced V4 score rises on three and is nearly unchanged on the fourth. No callable Google pixel oracle was available for an independent Round-06 verdict. This is an unverified lossy regeneration/distortion stack, not a quiet pattern cancellation |
| [`0xROOTPLS/DeSynth`](https://github.com/0xROOTPLS/DeSynth/tree/96db920731c2a3d04bf13163a5077b9a67706d1b) | 2026-08-24 | Qwen Image img2img at strength 0.25, followed by a Gaussian frequency split that restores the source high frequencies above sigma 1.95. The repository's public OpenAI original, default output, and edge-guided output all returned official `detected` in a current repeat after metadata stripping. The published `not found` claim does not reproduce against the current OpenAI oracle |
| [`froggeric/gemini-watermark-and-synthid-remover`](https://github.com/froggeric/gemini-watermark-and-synthid-remover/tree/5918384ce403968de0560cefd889e50eba0163bc) | 2026-08-24 | SDXL img2img with a documented manual Google-verifier ladder. The author reports 7/8 clears at strength 0.08 and 9/9, including a double mark, at strength 0.10 with five effective denoise steps and PSNR 29-41 dB. The exact nine before/after verdict artifacts are not tracked, so this is useful external regeneration corroboration, not an independently reproduced oracle result |
| [`atomantic/PortOS`](https://github.com/atomantic/PortOS/tree/b11a93e110262925c64a1b145a154ca87b340055) adversarial-jamming experiment | 2026-08-24 | Its own one-image manual OpenAI run found that quality-preserving phase noise, band noise, blur, and 0.70 resize squeeze stayed detected. Only visibly destructive phase perturbation cleared. A 0.85-0.90 resize caused repeated detector timeouts, which the repository correctly keeps separate from `not_detected`. This independently closes high-fidelity additive/phase jamming, but the source artifacts are not published |
| Generic regeneration claims | 2026-08-24 | `mertizci/noai-watermark`, `BovineOverlord/Loyal-Bear`, `obaskly/NeuralBleach`, and `tymongumienik/unwatermark` are SD/CtrlRegen/ControlNet redraws. Their algorithms are already covered by the product's regeneration remainder and the CtrlRegen literature. README verdicts without a frozen manifest do not establish a quieter or more reliable operating point |
| Add unmarked context (frame, pad, hstack, collage) | 2026-08-19 | Official `not_detected` is a presentation gate. White pad 40% linear stays `detected`; photo fill at the same geometry is `not_detected`. See detector identity/layout |
| `cebeuq/Synthid-Bypass` diffusion reconstruction | 2026-08-13 | 12 public pairs. Lattice and Gemini Verify go quiet on the *after* images because the image was redrawn, not because a payload was deleted. Faces and text are a quality claim for regeneration, not a local wipe |

Development oracle: the internal OpenAI provenance helper after AI-metadata
strip. Seeds s1, s2, s3 as in
[detector research](synthid-detector-research.md). Raw files:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/`.

## Band that actually carries the mark

Phase-randomize one octave at a time, preserve energy (E2, s1, replicated
p16_32 on s2 and s3):

| Octave (period px) | Verdict |
| --- | --- |
| 2-4 | detected |
| 4-8 | detected |
| 8-16 | detected |
| 16-32 | not_detected |
| 32-64 | detected |
| 64-128 | detected |

Destroying only 16-32 px periods silences the official decoder. The
lattice band 8-16 px does not. On a marked gray flat the same scramble is
51.1 dB `not_detected` because the band is almost empty. On a photograph
the band holds the scene (24.3 dB). Replacing native 16-32 with a
2x-pyramid prediction is 48.4 dB and still `detected`.

Fish additive jam, luma RMS `a`:

| Band | a | PSNR | Verdict |
| --- | ---: | ---: | --- |
| 16-32 | 0.06-4.0 | 51.2-33.6 | detected |
| 16-32 | 8 | 27.7 | detected |
| 16-32 | 16 | 21.9 | detected |
| 16-32 | 24 | 18.6 | not_detected |
| 4-8 | 16 | 21.9 | detected |
| 4-8 | 24 | 18.6 | detected |

Preregistered close `a*_photo < 0.53` failed (`a* > 16`).

## Bayer and geometry (2026-08-22)

Frozen one-pass batch. Lattice scores on these OpenAI sources are already
`indeterminate` and are not the label.

| Attack | s1 | s2 | s3 | fish |
| --- | --- | --- | --- | --- |
| source | detected | detected | detected | detected |
| Bayer bilinear | 32.7 detected | 28.4 detected | 37.1 detected | 34.7 detected |
| Bayer VNG | 25.0 detected | 18.5 detected | 26.4 detected | 25.4 detected |
| upscale 1.15 then Bayer | 33.9 detected | 29.8 detected | 38.4 detected | 36.0 detected |
| barrel k1=0.06 | 19.9 detected | 14.2 detected | 20.2 detected | 18.5 detected |
| scanline ±0.8 px | 31.5 detected | 26.7 detected | 35.7 detected | 32.5 detected |
| shift 2 px | 23.1 detected | 15.9 detected | 24.7 detected | 22.7 detected |
| 16-32 phase scramble | 24.6 not_detected | 19.0 not_detected | 25.2 not_detected | 23.2 not_detected |

Camera pipeline and mild geometry do not hit the decoder basis. A 2 px
shift, which kills `pipeline_lattice`, left s1/s2/s3/fish `detected`.
Scramble remains the residual kill on 4/4 sources.
`.local-eval/synthid/prc-oklab-attack-2026-08-15/bayer-geometry-2026-08-22/`.
Manifest `status` is `oracle_complete` (2026-08-22 22:07).

## Attack ladders (OpenAI oracle)

Pilot 2026-08-17 (n=3): JPEG q40, 0.5x, noise8+jpeg85, 0.35x+jpeg50 all
`detected` (s2 0.35x+jpeg50 at 20.1 dB still detected).

Kill and flip (s1), then replication 3/3:

| Rung | s1 | s2 | s3 |
| --- | --- | --- | --- |
| blur sigma 6 | detected | detected | detected |
| blur sigma 7 | not_detected | not_detected | not_detected |
| keep 0.50 linear | detected | detected | detected |
| keep 0.40 linear | not_detected | not_detected | not_detected |
| elastic ~24 dB | detected 24.6 | detected 17.9 | detected 26.1 |
| elastic ~21 dB | not_detected 20.8 | not_detected 14.8 | detected 22.0 |
| pad white 40% linear | detected | detected | detected |
| hstack with photo | not_detected | not_detected | not_detected |
| blend 25% marked | detected | detected | detected |
| inset native 1:1 on 2x photo | not_detected | not_detected | not_detected |
| JPEG q5 | detected | detected | detected |

Holds on 3/3: blur 6 vs 7, crop keep 0.50 vs 0.40, white pad 40%
`detected`, hstack `not_detected`, blend 25% `detected`, JPEG q5
`detected`. Elastic at ~21 dB does not hold. Crop keep-0.40 and hstack
are presentation or support, not a quiet residual wipe.

Surround: photo fill at the white-pad geometries is `not_detected` on
3/3; white pad at those sizes was `detected`. Unmarked photo strip flips
between 25% width (`detected`) and 35% (`not_detected`). Photo frame 15%
each side is `not_detected` on 3/3; 12% does not hold.

## Wild removers (oracle 2026-08-23)

Preregistered on s1, one pass, 25 s gap. Manifest:
`.local-eval/synthid/prc-oklab-attack-2026-08-15/wild-attacks-2026-08-23/oracle-manifest.json`.

| Attack | PSNR | Verdict | Note |
| --- | ---: | --- | --- |
| Replace 16-32 with a COCO photo's 16-32 | 25.2 | not_detected | Same annulus as cartesian scramble (24.6 dB). Substituting a camera band kills the decoder, 0.6 dB quieter than scramble on s1, not a new quality class |
| File named polar-1632 | 25.6 | not_detected | Misnamed. The raster randomizes cartesian FFT phase in the 16-32 annulus, same family as scramble (24.6 dB). It is a scramble replicate, not a `(r, θ)` polar test |
| 64-color median-cut palette | 35.6 | detected | Quieter than scramble and still marked. Posterize-4 already stayed `detected`; a smarter quantizer is not enough |
| PIL RGB to CMYK to RGB | inf | not submitted | No-op on this PNG |

The 16-32 kill is luma phase in that annulus. Cartesian, Fourier-angle,
and radial-phase all silence the decoder near 25 dB. A 90° sector of the
same ring does not (27.5 dB `detected`). Cb-only and Cr-only 16-32
scrambles stay `detected` at 44-45 dB. A foreign-scene transplant of the
same band also silences the decoder near 25 dB. Palette, Baker-map, seam
carve, Poisson, nested LSB, and ICC do not.

Follow-up 2026-08-23, s1, 25 s gap, `remaining-2026-08-23/oracle-manifest.json`:

| Attack | PSNR | Verdict |
| --- | ---: | --- |
| Fourier-angle 16-32 scramble | 24.1 | not_detected |
| Cartesian 16-32 scramble (replicate) | 24.8 | not_detected |
| Baker-map of the 16-32 band | 27.8 | detected |
| Poisson noise | 30.7 | detected |
| Nested LSB in blue | 55.9 | detected |
| Seam carve 8 | 27.7 | detected |
| ICC sRGB rewrite | inf | not submitted, no-op |

Non-local codecs, 2026-08-23, s1, 25 s gap:

| Attack | PSNR | Verdict |
| --- | ---: | --- |
| HEIF q80 | 46.3 | detected |
| HEIF q50 | 39.3 | detected |
| AV1 CRF 32 | 37.2 | detected |
| Print-scan simulation | 24.95 | detected |

Physical print-scan is still blocked unattended (Brother DCP-L2520DW idle and
accepting, no `scanimage`, no ImageCapture pyobjc). Face-gated scramble is
unnecessary: Haar on s1 put *more* 16-32 energy on faces. Generic 25 dB is
not the kill: this simulation stays `detected` at the PSNR where 16-32
phase scramble does not.

Waveform-shell splits, 2026-08-23, s1, 25 s gap,
`waveforms-shells-2026-08-23/oracle-manifest.json`:

| Attack | PSNR | Verdict |
| --- | ---: | --- |
| Y-only 16-32 scramble | 24.6 | not_detected |
| Cb-only 16-32 scramble | 45.0 | detected |
| Cr-only 16-32 scramble | 43.9 | detected |
| 90° Fourier sector of 16-32 | 27.5 | detected |
| Radial-phase-only 16-32 | 25.0 | not_detected |

## External literature (surveyed 2026-08-23)

Primary sources. Detector papers live in
[detector research](synthid-detector-research.md). Forensic stealth of
regeneration is already in [synthid.md](synthid.md#23-removal-attacks-and-forensic-detectability).

| Source | Attack | Against SynthID? | Map to this campaign |
| --- | --- | --- | --- |
| Zhao et al., [arXiv:2306.01953](https://arxiv.org/abs/2306.01953) (NeurIPS 2024) | Add noise, then denoise or regenerate (VAE / diffusion). Pixel-level invisible marks are provably removable. Semantic watermarks proposed as the alternative | Open post-hoc schemes, not production SynthID | This is the family our product uses (`qwen-zimage` / `sdxl-zimage`). Gowal trains SynthID-O against *weak* VAE regeneration. Our foreign-VAE round-trip at 22.3 dB stayed `detected`. Regeneration works when it redraws, not when it is a light codec |
| Liu et al., [arXiv:2410.05470](https://arxiv.org/abs/2410.05470) (CtrlRegen, ICLR 2025) | Controllable diffusion from clean noise, with a knob on how many noise steps to add | SOTA open watermarks | Same family. Goonatilake later finds CtrlRegen+ the *most* forensically detectable remover (AUROC 0.9999) |
| Kassis and Hengartner, [arXiv:2405.08363](https://arxiv.org/abs/2405.08363) (UnMarker, IEEE S&P 2025) | No decoder feedback. Two adversarial spectral optimizations. Breaks even some semantic watermarks (best remaining detection 43%) | Not production SynthID | Spectral disruption without an oracle is the honest analog of our 16-32 scramble, except UnMarker is optimized and we used a one-octave phase shuffle. Goonatilake: UnMarker TPR 98.28% at 0.1% FPR as a *forensic* leftover |
| Tallam et al., [arXiv:2505.08234](https://arxiv.org/abs/2505.08234) (SemanticRegen) | Partial, label-free regeneration of main objects | Tree-Ring, StegaStamp, StableSig, DWT/DCT. Not SynthID | Partial redraw. Our collage / photo-pad `not_detected` is a presentation gate, not this attack |
| Cao et al., [arXiv:2608.10166](https://arxiv.org/abs/2608.10166) (MarkNull, USENIX Security 2026) | On-manifold latent decorrelation via a public diffusion proxy. Claims 100% on 20 Imagen-3 Gemini-verify images. PSNR 25.36 dB, SSIM 0.80 | Small Gemini-verify set | Independent evidence that a no-box latent reconstruction can confuse Gemini. Does not meet this project's 40 dB / 0.99 SSIM release gate. Still generation, not a pixel-only wipe |
| Goonatilake and Ateniese, [arXiv:2605.09203](https://arxiv.org/abs/2605.09203) | Six removers all leave a forensic residue a ResNet-50 sees at >98% TPR @ 1% FPR | Applies to UnMarker, Zhao's WatermarkAttacker, CtrlRegen+ | Defeating a provider oracle is not deniability. This is the product remainder |
| An et al., [arXiv:2401.08573](https://arxiv.org/abs/2401.08573) (WAVES, ICML 2024) | 26 attacks on StegaStamp, Stable Signature, Tree-Ring. Regeneration, not JPEG, is the attack that matters. StegaStamp TPR@1%FPR 1.00 to 0.01; Tree-Ring 0.99 to 0.12 | Open watermarks | Protocol. Our blur-sigma-7 and 16-32 scramble are closer to WAVES "distortion" than to regeneration |
| Wen et al., [arXiv:2305.20030](https://arxiv.org/abs/2305.20030) (Tree-Ring) | In-generation Fourier pattern in initial noise | Not SynthID | Kill is DDIM inversion plus latent wipe, not a pixel scramble. Lin and Juarez [arXiv:2506.10502](https://arxiv.org/abs/2506.10502) (USENIX 2025) remove it from public knowledge |
| Fernandez et al., Stable Signature, ICCV 2023 | Fine-tuned VAE decoder | Not SynthID | Regeneration that replaces the decoder is in-family. Our foreign VAE at 22.3 dB did not kill OpenAI SynthID, which is the expected mismatch |
| Google to [The Verge, 2026-04-14](https://www.theverge.com/ai-artificial-intelligence/911579/google-synthid-ai-watermarking-system-reverse-engineered) | reverse-SynthID does not systematically remove SynthID | Google spokesperson on the spectral bypass | Agrees with our V4 audit and with Bypass: reconstruction or decoder confusion, not a payload delete |

Quiet pixel-only removal is not a published result on production
OpenAI or Google SynthID (still true in the 2026-09-22 literature sweep).
Against open encoders the fidelity band has moved: re-watermarking
([arXiv:2605.16796](https://arxiv.org/abs/2605.16796)), WmForger
([arXiv:2510.20468](https://arxiv.org/abs/2510.20468)) and Hide&Seek
([arXiv:2603.01067](https://arxiv.org/abs/2603.01067)) report roughly 30 dB,
and SPFM-Net and FDDWAN (arXiv:2607.27811, 2607.27800) claim 40-45 dB, but on
small images and with bit error rates that leave much of the payload intact.
None of these is a result on a production closed watermark.

## Product remainder for removal

Invisible removal in this package regenerates through `qwen-zimage` or
`sdxl-zimage`. It is lossy. There is no shipped pixel-only OpenAI or
Google SynthID wipe. Do not add Bayer or geometry as remover arms.
