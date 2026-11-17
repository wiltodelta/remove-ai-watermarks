# Non-SynthID video coverage

Source review: 2026-11-09.

This is the evidence map for the library's non-SynthID video work. It separates
metadata declarations, open-model message decoding, and deployed-provider
verification. Seedance/ByteDance and CapCut require their own provider-specific
evidence; surrogate VideoSeal results do not certify another provider's watermark.

## Structure-preserving regeneration research

Preserving packet timestamps and copied audio does not establish preservation
of object layout or motion. A VAE round trip also does not test learned
denoising under spatial conditioning. The follow-up research compares
unconditioned SDXL img2img with Canny-conditioned img2img, using identical
native geometry, padding, clean latent modes, scheduler, and noise. A separate
flow-transported-noise arm tests a temporal constraint. Shared seeds alone are
not evidence of temporal consistency, and flow-transported noise is not a
trained video diffusion model.

[`scripts/video_structure_probe.py`](../scripts/video_structure_probe.py)
provides research-only native-pixel diagnostics:

- Bidirectional edge distances, with missing edges reported explicitly.
- Mutual SIFT match displacement without alignment, together with match count
  and source convex-hull coverage. Sparse surviving matches cannot certify
  every object or newly hallucinated content.
- Source-relative optical-flow error on textured source pixels with consistent
  forward/backward flow. Candidate failures do not select the validity mask.
  Flat frames remain unverified; estimated flow is not ground-truth tracking.

Read these diagnostics alongside full-frame fidelity, worst-frame review, scene
cuts, faces, text, and native audio/video clocks. Short-window ablations do not
establish full-clip quality. Private designs, pinned models, decoded frames,
and measured results stay outside the repository. Model execution or improved
geometry is not a watermark-removal verdict: each delivered candidate still
requires a matched source-positive provider check.

[PoseRegen](https://arxiv.org/html/2610.22457v2) is a separate trained,
pose-conditioned video reconstruction approach. Its appendix includes a
source-conditioned joint-trajectory metric with video-level similarity
alignment; that aligned score does not establish preservation of absolute
native-pixel positions. The checked [project page](https://sites.google.com/view/poseregen)
exposed a demo but no usable code/checkpoint download. A released pose-generation
backbone alone does not supply PoseRegen's trained motion adapter. Keep this
architecture hypothesis separate from the executable Canny ablation.

The other executable follow-up families are
[BasicVSR](https://github.com/XPixelGroup/BasicSR), used as a temporally recurrent
restoration model, and [RIFE](https://github.com/hzwer/ECCV2022-RIFE), used to
replace frames with learned midpoint interpolation. Restoration needs both the
processed-input baseline and an identical resampling-only control; an apparent
improvement over one resize path is not proof that source fidelity was restored.
Native-input restoration and restoration after downsampling are separate arms.
Interpolation must check the actual source timestamps before assuming a temporal
midpoint. Retained original endpoints or alternating frames remain declared
limitations even if an eventual clip-level verifier returns negative.

These local ablations do not execute PS-SR, SeedVR2, FlashVSR+, PoseRegen, or a
temporally coupled UnMarker optimizer. They must not be reported as tests of
those architectures. A video-restoration backend, a spatially controlled image
denoiser, shared-noise sampling, and a jointly trained pose-conditioned video
model test different hypotheses.

## Coverage matrix

| Signal | Local identification | Pixel or audio removal evidence | Remaining requirement |
| --- | --- | --- | --- |
| Research VideoSeal 1.0, 256-bit message | Pinned explicit `video message` decoder with an independent reference; `videoseal` benchmark adapter; C2PA names remain separate hints | The shipped VAE profile crossed below the benchmark threshold on three of four standardized carriers; the fourth remained positive | This bounded comparison does not certify consistent removal or erasure of the message |
| Research PixelSeal video mode | Pinned explicit `video message` decoder with an independent reference; `pixelseal-video` research adapter | The default VAE output fell below threshold on all three source-positive, control-positive carriers; a fourth source failed calibration | This bounded comparison does not establish message erasure or a production-provider result |
| Deployed VideoSeal integrations | Registry names such as `com.aiwatermark.videoseal.1`, `me.reconize.videoseal.1`, and `be.immostory.videoseal.1` become metadata hints | Research weights are not evidence of compatibility with each deployment's payload and detector | A compatible positive carrier and verifier for that deployment |
| Deployed Meta Content Seal on video | The project Web adapter accepts video, but its recorded project evidence is image-only | Image removal certification does not certify deployed video | A provider-positive native video and an independently recorded video verdict |
| Amazon Nova Reel | Bedrock provenance produces an explicit `Amazon Nova Reel` policy hint | Metadata removal is separate from the proprietary pixel watermark | A video-capable Amazon watermark verdict; the documented image checker is insufficient |
| Microsoft InvisMark on video | Registered C2PA name becomes a hint; no compatible production video decoder is shipped | Existing image oracle results do not certify video | A native positive video and a matching Microsoft watermark result |
| Microsoft WavMark in video | Registered declaration becomes a hint; the public research WavMark decoder is separately available | `video audio --scheme wavmark` processes a source-positive compatible soundtrack and checks the final mux | Public research model compatibility does not certify the registered or deployed Microsoft scheme |
| Adobe FlowMark variant A | The registered `com.adobe.flowmark.A` name becomes a metadata hint | No compatible model, positive carrier, or pixel-removal certification is integrated | A compatible checkpoint or provider verifier and a controlled positive carrier |
| Published OpenVoice default raw WavMark framing | The existing WavMark route also checks fixed native blocks against the known default message | `video audio --scheme wavmark` applies the same source-positive and final-mux-negative gates | Custom messages and shifted/resampled framing are outside this receiver; a match does not establish provider identity |
| Archived Chatterbox default PerTh audio | Default marker source and checkpoint match; controlled native Nano outputs passed library and CLI checks | `video audio --scheme perth` passed final-mux verification, with original native-rate PerTh checks and untouched-stream packet/timing preservation | Inspected default checkpoint and tested Nano outputs only; no hosted PerTh v2 or provider-attribution claim |
| Public AudioSeal / Perth v1 audio inside a video | Optional pinned decoders through the explicit `audio` routes | `video audio` processes a selected track behind source-positive and final-mux-negative checks, preserving other encoded streams | Broaden held-out carriers; production-provider compatibility remains separate |
| Public SilentCipher-44k audio inside a video | Experimental repeated-symbol receiver through `--scheme silentcipher` | Controlled CLI and selected-track final-mux checks passed, with untouched-stream packet and timing verification | Broaden carriers and controls; deployed-provider compatibility remains separate |

The runtime registry is the source of truth for exact C2PA algorithms and
their `watermark` or `fingerprint` kind. This matrix does not expand automatic
removal to stock, copyright-protection, or other paid-asset marks. A registered
name alone establishes neither AI origin nor successful pixel decoding.
Video identification keeps declaration-only soft bindings out of its AI-origin
verdict. The metadata and watermark hints remain readable; independent AI
source types, generator provenance, and accepted visible marks keep their usual
effect.

## PixelSeal video protocol

The upstream [PixelSeal model card](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/videoseal/cards/pixelseal.yaml)
configures temporal chunking and propagation. The oracle uses that pinned
source and checkpoint with `is_video=True`. It retains the existing image
reader separately and delegates video logit aggregation to the same pinned
family implementation as VideoSeal.

The C2PA entry `com.aiwatermark.pixelseal.1` declares image media. The research
model's video mode therefore does not establish a deployed PixelSeal video
format or extend that registry entry's declared media types.

```bash
python scripts/videoseal_temporal_study.py --adapter pixelseal-video \
  --regenerate --output-dir /outside/repo/pixelseal-video
```

Start with the project's all-extras research stack. In an external research
environment, add `timm` and install the pinned `videoseal` source with
`--no-deps`, as required by
[the benchmark kernel](benchmark-kernel.md#perth-and-pixelseal-local-oracle-study).
The source revision and checkpoint are checked by the oracle; the ordinary PyPI
wheel lacks the pinned model card. Regeneration uses the diffusion stack.
The command creates a fresh output directory and makes no provider requests.
The default adapter remains `videoseal`. The study calls
`regenerate_video_candidate` directly; CLI routing and proprietary provider
verdicts are outside these measurements.

Each carrier has a saved clean source, an embedded marked source, a matching
codec control, a zero-noise VAE output, a default VAE output, and a default VAE
output of the clean negative. Decoding operates on hashed saved MP4 files.
Delivered-file PSNR, SSIM, and temporal residuals include resize and codec
effects. Below-threshold message recovery is not proof that all information
was erased. Audio disposition remains a separate boundary.

## PixelSeal saved-artifact evaluation

The 2026-11-09 run used the four standardized carriers from the VideoSeal
comparison, each with 64 RGB frames at 256 by 256 pixels and 12 fps. The shipped
regeneration profile delivered 512 by 512 pixels. The table reports average-logit
message bit accuracy against the fixed research message, with a detection
threshold of 0.9. Quality is measured against the decoded marked source.

| Carrier | Marked source | Codec control | Zero-noise VAE | Default VAE | Clean default VAE | Default PSNR / SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Moving gradient | 1.0000 | 0.9727 | 0.9375 | 0.7461 | 0.4883 | 33.89 dB / 0.9685 |
| Moving texture | 0.8164 | 0.9180 | 0.5977 | 0.5234 | 0.4570 | 24.83 dB / 0.8405 |
| Veo carrier | 1.0000 | 1.0000 | 0.9961 | 0.8125 | 0.4844 | 32.94 dB / 0.9639 |
| Sora carrier | 1.0000 | 1.0000 | 0.8750 | 0.5547 | 0.5117 | 27.05 dB / 0.8072 |

The moving-texture marked source is below threshold. Its positive codec-control
reading does not repair the failed source calibration, so that carrier cannot
support a removal claim. On the other three carriers, the source and codec
control remain positive while the default VAE output falls below threshold.
The Sora zero-noise output also crosses the threshold, so noise is not necessary
for that observed effect. All clean sources and regenerated clean negatives
remain negative. Veo and Sora name the public carrier content, not the watermark
being decoded or a provider watermark verdict.

Message bit accuracy remains above 0.5 on the eligible default outputs,
particularly Veo. Single-frame inspection of the Veo and Sora outputs
also shows changed fine detail. These measurements support a bounded detector
comparison, not complete erasure, preserved visual identity, or general success
across seeds and clips.

The local evidence directory is
`~/.cache/remove-ai-watermarks-dev/evaluations/pixelseal-video-2026-11-09-8919hgv8/`.
`cases.jsonl` records saved-artifact hashes, frame readings, quality, source
revision, and checkpoint pins. The checkpoint SHA-256 is
`0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd`;
the upstream revision is the model-card revision linked above. Generated media
remain outside the repository.

The independent `pixelseal-video` benchmark run completed all 28 manifest cases
without an oracle error. Its 24 readings of the study artifacts matched exactly;
all four foreign-message artifacts were negative against the target message.
Three foreign-message artifacts were positive against their own message, while
the texture failed that calibration too. A wrong-message negative establishes
message mismatch, not watermark absence.
`benchmark-verification.json` records manifest SHA-256
`5bf20e00665250fd006349eee6036632b74319de963b646651865e9ef5841ea3`
and result SHA-256
`9dbbdd78268e5069f2c6ccfc0d089bc6b9390268ff15fb279b47493d4e20f793`.

## Audit scope

The metadata regression replay covered every tracked public video in the
checkout, with full serialized reports unchanged after the classification fix.
The retained inventory and replay evidence live under the evaluation directory's
`validation/` folder. This run did not replay private historical inputs or
certify proprietary pixel-watermark removal. The controlled PixelSeal study
covers one target message, one foreign message, one regeneration seed, and the
four listed carriers. Soundtrack watermarks remain outside those measurements.

## Provider-verifier limits

[Amazon's Nova Reel service card](https://docs.aws.amazon.com/ai/responsible-ai/nova-reel/overview.html)
states that generated videos carry an invisible watermark. The documented
[Bedrock watermark checker](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-image-models.html#titan-image-models-watermark)
describes Titan Image and Nova Canvas with JPG/PNG input. That interface is not
evidence of a working Nova Reel video check. A quiet local metadata report is
not a negative pixel-watermark verdict.

The tracked Nova Reel fixture's metadata-only strip was checked against every
encoded video packet, including PTS, DTS, duration, and payload hash. They were
identical after stripping, and the local metadata report became unknown. This
fixture has no audio stream. The result establishes metadata-removal parity
without making a pixel-watermark or audio claim.

[Microsoft's Content Provenance Detection guide](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/how-to-provenance-detection)
documents image, audio, and video inputs and separate `Watermark` and `C2PA`
results. Media acceptance alone does not establish compatibility with a given
production generator or watermark framing. Keep each provider's source-positive
and control-positive checks separate from research-model observations.

[Adobe FlowMark's paper](https://arxiv.org/abs/2607.05261) describes a separate
mask-guided video watermark. The registry and paper establish the scheme's
identity, not a compatible local implementation. This audit found no model
release link in the inspected paper; it does not claim a global absence of code.

## Explicit known-message runtime

The experimental [`video message` commands](cli.md#public-model-video-messages-experimental)
and Python APIs expose pinned VideoSeal and PixelSeal decoders without the
training SDK. PixelSeal detector weights strictly match the verified JIT graph;
raw logits and decoded bits matched the official pinned implementation across
single frames, rectangular inputs, a temporal chunk boundary, and larger inputs.
The swapped graph is exposed only for decoding; its embedding network remains
VideoSeal and must not generate PixelSeal fixtures.

A caller supplies an independently known encoder message. The receiver uses
every frame, four aggregations, and complete windows including a context tail.
The fixed codec/VAE processing sequence checks the actual saved file before
publication and reports delivered quality. This criterion is stricter than the
historical average-logit benchmark above, so its outcomes must be evaluated
separately. Unknown-message detection and provider deployments remain outside
this route. Native-256 VAE tests also showed substantial texture loss on some
carriers; a decoder negative is not a fidelity guarantee.

### Known-message runtime integration check

The retained `audio-backend-feasibility-2026-11-10/packaged-video-v3`
evaluation exercised the public API on complete 13-frame excerpts from the four
controlled carriers. Lossless YUV encoding with no B frames made uniform test
streams; the first 13 decoded RGB frames matched each retained source exactly.
This is a short-clip integration check, separate from the full-length research
comparisons above. It makes no provider requests.

The actual automatic route chose these profiles at long side 256 and 12 fps.
The maximum covers every frame, all four aggregations, and full/context-tail
windows, with a threshold of 0.9:

| Public model | Carrier | Selected profile | Saved maximum bit accuracy | Delivered SSIM |
| --- | --- | --- | ---: | ---: |
| VideoSeal | Moving gradient | Codec | 0.8086 | 0.9850 |
| VideoSeal | Moving texture | Codec | 0.6250 | 0.9911 |
| VideoSeal | Veo carrier | Zero-noise VAE | 0.8281 | 0.9786 |
| VideoSeal | Sora carrier | Zero-noise VAE | 0.5742 | 0.7112 |
| PixelSeal | Moving gradient | Codec | 0.7617 | 0.9842 |
| PixelSeal | Moving texture | Skipped, source did not qualify | Not measured | Not measured |
| PixelSeal | Veo carrier | Zero-noise VAE | 0.7578 | 0.9775 |
| PixelSeal | Sora carrier | Zero-noise VAE | 0.5469 | 0.7138 |

Every processed clean control for a qualified carrier remained negative under
its selected profile. Each model also passed an independent foreign-message
positive, wrong-reference negative, and saved processed-output check. Clean
sources and their codec outputs also stayed negative under each foreign reference. The new
streaming float64 receiver matched the frozen research float32 receiver's
scores on these saved readouts. The final CLI was checked separately, without
the evaluation's hash-bound readout reuse. A video-only environment supported
explicit unverified codec processing without Torch; a `video-detect` environment
loaded both actual decoders without the training SDK, torchvision, or diffusion.

Codec alone still matched both real-source carriers. VAE therefore supplied
additional disruption there, but its Sora texture loss is material. A sampled
frame comparison showed smoothing of the dogs' fur; temporal residual ratios
also rose. These outputs do not establish visually transparent removal, long
video reliability, complete message erasure, or deployed-provider compatibility.
Audio tests separately checked copied PCM and AAC tracks, including reordered
stream indices, language/disposition metadata, offsets, and audio tails.

### High-resolution scale versus linear-bandwidth factorial

To test whether spatial downsampling or linear low-pass filtering accounts for
known-message VideoSeal suppression at native high resolution, the frozen local
study `carbon-video-bandwidth-and-768-b092199d` (`plan.json` SHA-256
`138c44cc2f25a08b6df77ac5b6a9e59a8a4108f236b2b59ef6b48208bcc3a3a6`) embedded
the deterministic 256-bit `VideoSeal 1.0` research message (`seed=7`,
`is_video=True`) into 36-frame silent native-rate prefixes of the public `veo`
(`1280x720 @ 24 fps`) and `sora` (`704x1280 @ 30 fps`) fixtures and inspected
every saved MP4 through `identify_video_message(..., scheme="videoseal")` across
whole-clip float64 aggregations, per-frame maxima, and all 25 sliding 12-frame
windows (`threshold = 0.90`):

| Carrier | Arm | Saved status | Max bit accuracy | Common-grid PSNR / gray SSIM |
| --- | --- | --- | ---: | ---: |
| Veo `1280x720 @ 24 fps` | `marked-source` / `codec-1280` | `matched` | `1.0000` / `1.0000` | `inf` / `38.76 dB` (`0.9930`) |
| Veo `1280x720 @ 24 fps` | `codec-768` / `linear-bw-768match` | `matched` | `1.0000` / `1.0000` | `37.29 dB` (`0.9887`) / `38.55 dB` (`0.9926`) |
| Veo `1280x720 @ 24 fps` | `codec-512` / `linear-bw-512match` | `matched` | `0.9844` / `1.0000` | `36.03 dB` (`0.9851`) / `38.00 dB` (`0.9914`) |
| Veo `1280x720 @ 24 fps` | `vae-1280` / `vae-768` / `vae-512` | `not_matched` | `0.8047` / `0.8398` / `0.7656` | `34.52` (`0.9805`) / `33.19` (`0.9778`) / `31.55 dB` (`0.9721`) |
| Sora `704x1280 @ 30 fps` | `marked-source` / `codec-1280` | `matched` | `1.0000` / `1.0000` | `inf` / `36.90 dB` (`0.9785`) |
| Sora `704x1280 @ 30 fps` | `codec-768` / `linear-bw-768match` | `matched` | `1.0000` / `1.0000` | `33.61 dB` (`0.9379`) / `32.68 dB` (`0.9143`) |
| Sora `704x1280 @ 30 fps` | `codec-512` / `linear-bw-512match` | `matched` | `1.0000` / `1.0000` | `29.29 dB` (`0.8241`) / `31.92 dB` (`0.8953`) |
| Sora `704x1280 @ 30 fps` | `vae-1280` / `vae-768` / `vae-512` | `not_matched` | `0.8086` / `0.6797` / `0.5664` | `25.10` (`0.6249`) / `23.23` (`0.5047`) / `22.14 dB` (`0.4630`) |

Clean sources (`0.5742-0.5781`), clean `vae-768` controls (`0.5391`), and
foreign-reference checks (`seed=19`, `0.5391-0.6055`) all remained `not_matched`
(`part_b_sliding25_rescore.json`).
Pure Lanczos downsampling plus H.264 (`codec-768`, `codec-512`) and native-grid
isotropic Gaussian blurs calibrated to the empirical `-3 dB` radial cutoffs of
`vae-768` and `vae-512` (`linear-bw-768match`, `linear-bw-512match`) left full
or near-full message recovery (`0.9844-1.0000`). Crossing below `0.90` required
the nonlinear `AutoencoderKL` latent bottleneck with shared perturbation
(`noise_std=0.15`), which succeeded even at native `1280` long side (`vae-1280`),
while residual correlation (`0.5664-0.8398`) remained above chance (`0.50`) and
dense fine fur texture on `sora` still degraded substantially across all VAE
scales. Because `vae-512` clears all 25 sliding windows with wide headroom below
the threshold (`0.7656` on `veo` and `0.5664` on `sora`, compared with `0.8398`
at `vae-768`), `DEFAULT_VIDEO_MESSAGE_LONG_SIDE` defaults to `512` rather than
`256`. These local research-model measurements do not transfer to deployed
Meta, ByteDance/CapCut, or Google SynthID verifiers.

### Single-encode motion-aware temporal stabilization

Per-frame spatial VAE regeneration introduces frame-to-frame temporal residual
elevation (`temporal_residual_ratio` `1.795` on `flow-001` and `1.279` on
`lite-003` at `768x432 @ 24 fps`). Applying stabilization after H.264 encoding
would incur a second `yuv420p` chroma-quantization pass (`~11.6 MSE`), so
`iter_stabilized_regenerated_frames` (`temporal_stabilize=True` on
`regenerate_video_candidate` and `remove_video_message`) runs in bounded
3-frame memory on raw decoded VAE frames before the single FFmpeg encode,
blending cycle-consistent Farnebäck-warped predecessor and successor frames
(`neighbor_weight=0.25`) gated on source motion-compensated residual
(`carbon-video-temporal-stabilization-b092199d/final-summary.json`):

| Cohort / Carrier | Arm | Unstabilized -> Stabilized common PSNR | Common gray SSIM | Temporal residual ratio | 25-window VideoSeal max bit accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Google `flow-001` (`192f`) | `vae-768-24fps` | `33.63 -> 33.76 dB` | `0.9794 -> 0.9803` | `1.795 -> 1.329` (`-26.0%`) | n/a (native Google cohort) |
| Google `lite-003` (`192f`) | `vae-768-24fps` | `25.78 -> 25.81 dB` | `0.8256 -> 0.8269` | `1.279 -> 1.139` (`-10.9%`) | n/a (native Google cohort) |
| Sora `704x1280 @ 30 fps` (`36f`) | `vae-768` | `23.23 -> 23.38 dB` | `0.5047 -> 0.5096` | `2.096 -> 1.452` (`-30.7%`) | `0.6797 -> 0.6602` (`not_matched`) |
| Sora `704x1280 @ 30 fps` (`36f`) | `vae-512` | `22.14 -> 22.26 dB` | `0.4630 -> 0.4663` | `1.956 -> 1.429` (`-26.9%`) | `0.5664 -> 0.5547` (`not_matched`) |
| Veo `1280x720 @ 24 fps` (`36f`) | `vae-768` | `33.19 -> 33.35 dB` | `0.9778 -> 0.9788` | `1.734 -> 1.327` (`-23.5%`) | `0.8398 -> 0.8242` (`not_matched`) |
| Veo `1280x720 @ 24 fps` (`36f`) | `vae-512` | `31.55 -> 31.70 dB` | `0.9721 -> 0.9732` | `1.828 -> 1.410` (`-22.9%`) | `0.7656 -> 0.7734` (`not_matched`) |

Single-encode stabilization reduced `temporal_residual_ratio` by `10.9-30.7%`
and slightly improved PSNR and SSIM on every carrier without re-strengthening
25-window `VideoSeal 1.0`. On `lite-003`, unaligned SIFT p95 displacement also
dropped (`47.07 -> 31.64 px`), while on `flow-001` smooth water-surface motion
shifted SIFT p95 modestly (`8.27 -> 11.89 px`). High-level `remove_video_invisible`,
`remove_video_all`, and `remove_video_batch` enable `temporal_stabilize=True` by
default (`--no-temporal-stabilize` disables it), while low-level
`regenerate_video_candidate` and `remove_video_message` keep `temporal_stabilize=False`
unless explicitly passed.
