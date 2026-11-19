# Video and audio methods and competitors

Research snapshot: 2026-11-11. This evidence review supports the
[execution roadmap](media-removal-roadmap.md), [provider verifier matrix](media-provider-verifiers.md),
and [real product corpus](media-sample-corpus.md). It is a bounded review of
official sites, public implementations, and primary papers, not a claim to have
exhausted the internet or independently verified every commercial service.

## Evidence rules

Keep advertised behavior, inspected implementation, reproducible execution,
and provider verification separate. A changed hash proves changed bytes.
Metadata removal proves no waveform or pixel watermark outcome. A general
AI-origin classifier is not a compatible watermark decoder. Public research
models do not establish compatibility with a vendor's current production mark.
Closed services' server implementations remain unknown unless demonstrated.

Independent Carbon, Fable 5.2, and Astra reviews informed the search. Their
claims were cross-checked against primary sources; model agreement is not
evidence. In particular, ShadowMarkDefense exists, Meta's identification page
accepts three media modalities, and ElevenLabs documents a separate watermark
API. Similar service names do not establish shared ownership or backends.

## Commercial and browser tools

The rows below describe the checked public surface. None qualifies as an
independently reproduced, universal video/audio remover in this review.

| Service | Checked scope or claim | What remains to establish |
| --- | --- | --- |
| [removesynthid.io](https://removesynthid.io/) | Advertises image, video, and audio processing | Hash-bound native inputs, matched controls, exact provider verdicts and fidelity; backend is closed |
| [SynthIDx video](https://synthidx.com/ai-video-watermark-remover) | Explicitly scopes the video path to visible marks and container metadata, with copied audio | The page explicitly excludes video SynthID removal; image claims must not be transferred to video |
| [removesynth.com](https://removesynth.com/) | Audio spectral-processing claims and video workflow | Browser DSP or a fallback is not proof of the remote backend or a specific production receiver outcome |
| [Remove AI Label](https://removeailabel.io/) | Format-specific AI-label/provenance offering | C2PA/container cleanup must remain separate from invisible audio/video efficacy |
| [SynthID Watermark Remover](https://synthidwatermarkremover.com/) | Checked uploader is image-only despite navigation wording | No working video/audio submission established in the checked page |
| [Erasy](https://erasy.co/), [Undetectr](https://undetectr.com/) | Audio attribution/distribution claims; Erasy links to Undetectr | Provider mark, metadata, content fingerprint and classifier claims need independent separation; no distribution-pass guarantee |
| [SunoMaster](https://sunomaster.com/) | Adjacent browser mastering/EQ/loudness product | Checked homepage does not establish a compatible watermark-removal result; mastering is not a receiver verdict |
| [PrivyClean](https://www.privyclean.app/) | Adjacent metadata/privacy product focused on photos/documents | Its checked scope does not establish invisible video/audio processing; metadata-only outcomes remain separate |
| MarkCut, Imaginoa, Media.io, HitPaw, VEED and general visible-watermark editors | Discovery cohort for visible-removal workflows | Advertising is not evidence of invisible video/audio compatibility; automatic third-party stock protection removal is outside project scope |

The commercial inventory is a comparison queue, not a recommendation to upload
private media. Future comparisons use explicitly authorized files and frozen
outputs, not website screenshots or a vendor's own classifier alone. Private
research retains checked pages and unresolved leads rather than promoting
uninspected marketing into implementation facts.

## Public implementations

| Implementation | Mechanism or checked boundary | Engineering implication |
| --- | --- | --- |
| [ShadowMarkDefense](https://github.com/shikanon/ShadowMarkDefense) | Documents metadata/SEI cleanup, FFmpeg and DCT transforms, external BasicVSR, RIFE and PS-SR adapters | Useful experiment orchestration; restoration/interpolation still needs independent efficacy and clock/geometry checks. Local RIFE/BasicVSR results are in the Seedance study |
| [wmr](https://github.com/froggeric/wmr), [noai-watermark](https://github.com/mertizci/noai-watermark) | Image regeneration/wrapper approaches in the inspected research | Image support does not establish a trained temporal video backend or audio processing |
| [watermarks-remover](https://github.com/guillaumemeyer/watermarks-remover) | Audio FFmpeg tempo/pitch/EQ/encoding path; video delegates image processing | Changes to voice, duration or frame independence require explicit measurement; no proprietary decoder certification inferred |
| [pagedMark](https://github.com/doofzoff/pagedMark) | Framewise VAE/shared-noise video approach in the inspected implementation | Copied audio remains unprocessed; shared image noise is not temporal-model training |
| [FlowDenoise](https://github.com/AIMZ-GFX/ComfyUI-FlowDenoise) | Temporal restoration candidate | A quality-restoration hypothesis, not established watermark removal |
| [Gemini visible-label remover](https://github.com/GargantuaX/gemini-watermark-remover), [GeminiWatermarkTool](https://github.com/allenk/GeminiWatermarkTool) | Visible-label workflows | Edited outputs are not pristine native Veo controls; retained files bearing these tool comments must be classified accordingly |
| [AI Audio Fingerprint Remover](https://github.com/geeknik/ai-audio-fingerprint-remover) | Archived project advertises metadata, spectral filters and statistical changes; example verification is a hash comparison | Its claimed Suno/OpenAI/ElevenLabs coverage is not established by changed hashes or heuristic spectral peaks |
| [NaSuno](https://github.com/AhmadHassan-BTed/NaSuno-SongHumanizer) | Public wrapper advertises notch filtering, masking, dispersion and metadata cleanup; high-throughput engine is explicitly private | Claimed universal carriers and neutralization are hypotheses, not provider evidence. Public wrapper availability does not imply availability of its private engine |
| [WatermarkAttacker](https://github.com/XuandongZhao/WatermarkAttacker), [CtrlRegen](https://github.com/yepengliu/CtrlRegen), [UnMarker](https://github.com/andrekassis/ai-watermark) | Image robustness/regeneration or optimization research | Establish temporal fidelity, compute and licenses separately before video integration; research-use restrictions can block production reuse |

Visible inpainting/cropping tools and forks are not independent evidence of an
invisible receiver result. General “humanization,” ultrasonic filtering and
“remove all fingerprints” claims need compatible positive controls before
they influence a public profile. Content-based matching may survive every
metadata edit and is a separate problem from an embedded watermark. External
competitor outputs (`ShadowMarkDefense`, `pagedMark`, `watermarks-remover`) can
be evaluated out-of-process against identical local receivers and perceptual
gates via `scripts/media_removal_benchmark.py --external-outputs-dir` without
importing non-permissive or unpinned code into the package.

## Open-source library and license audit (2026-11-18)

| Candidate library / checkpoint | Modality & role | Checked license | Integration disposition |
| --- | --- | --- | --- |
| `descript/dac_44khz` | 44.1 kHz RVQ neural audio codec | MIT (code + weights) | **Shipped** (`--method dac`, fixed 3 quantizers, `cpu`/`mps`/`cuda`, raw M/S stereo + `5.1`/`7.1` surround stems, overlap-window streaming, `--vendor` default) |
| `nvidia/bigvgan_v2_44khz_128band_512x` | 44.1 kHz 128-band phase-free studio neural vocoder | MIT (code + weights) | **Shipped** (`--method bigvgan`); pre-warped Mel bottleneck benchmarked and **held back** (`MR-STFT 1.63` vs `DAC 1.11`, retained `Perth` 16k, 100x slower on `mps`) |
| `hubertsiuzdak/snac_44khz` & `jishengpeng/WavTokenizer` | Multi-scale 44.1 kHz (`12 Hz` coarse tokens) & single-quantizer neural audio codecs | MIT (code + weights) | **Future candidates** (`DAC 44.1 kHz` won all removal + `MR-STFT 1.11` axes) |
| `facebook/encodec_24khz` & `facebook/encodec_48khz` | 24 kHz mono & 48 kHz native stereo neural audio codecs (`transformers`) | MIT (code + weights) | `24 kHz` M/S **shipped** (`--method encodec`, `cpu`/`mps`/`cuda`); `48 kHz` (`6 kbps`) **held back** (`MR-STFT 1.99` vs `DAC 1.11`, `12/24 kbps` retained `AudioSeal`) |
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` & `Wan2.2` | 3D causal video VAE + flow-matching diffusion | Apache-2.0 | **Shipped** (`wan21-*` with CPU `UMT5-XXL` prompt offload cutting accelerator memory to `~3.4 GiB`, `cuda` + explicit `mps`, 5-min 1080p streaming budget) |
| `madebyollin/taehv` (`TAEW2_1`, `TAEW2_2`) & `westlake-repl/LeanVAE` | Fast causal 3D video autoencoders aligned to Wan latents | MIT (code + weights) | **Phase 3 candidates** (`0.4 GiB` VRAM / 10-44x speedup) paired with few-step `Wan 2.1` |
| `microsoft/DCVC` (`DCVC-RT`, `DCVC-UF`) | Real-time chunk/P-frame neural video codecs (`100-450+ fps`) | MIT (code + weights) | **Phase 3 candidate** (requires QP sweep against local `VideoSeal`/`PixelSeal`/`ChunkySeal` + provider oracles) |
| `facebookresearch/videoseal` (`VideoSeal 1.0`, `PixelSeal`, `ChunkySeal`) | Public video watermark detectors (`256`-bit and `1024`-bit; upstream `VideoSeal 0.0` is `96`-bit) | MIT | `VideoSeal 1.0` (`256`-bit), `PixelSeal` (`256`-bit) & `ChunkySeal` (`1024`-bit) **shipped** |
| `Peterande/D-FINE`, `lyuwenyu/RT-DETR` & `facebookresearch/sam2` (`SAM 2.1 Tiny`) | Permissive real-time object detection & video mask propagation | Apache-2.0 | **Phase 2 candidates** for moving AI-badge localization & `video erase --track` |
| `allenk/GeminiWatermarkTool` | Calibrated reverse alpha unblending + optional FDnCNN post-filter | MIT | **Phase 2 reference** for self-calibrated multi-frame `YCbCr` reverse alpha unblending |
| `moyangkuo/AudioMarkBench` & `zhengyuan-jiang/VideoMarkBench` | Academic audio & video watermark robustness suites | MIT | **External evaluation reference** (keep out of wheel dependencies) |
| `sczhou/ProPainter`, `lixiaowen-xw/DiffuEraser`, `zibojia/MiniMax-Remover`, `TencentARC/VideoPainter`, `MCG-NKU/E2FGVI` | Video inpainting / object removal | **NTU S-Lab 1.0 / CC-BY-NC-4.0 / Research-only** | **Excluded from package** (`DiffuEraser` Stage 1 depends on NC `ProPainter`; replaced by multi-frame `YCbCr` unblending + bidirectional clean-plate flow + `MI-GAN`/`LaMa`) |
| `linkedlist771/SoraWatermarkCleaner` (`ultralytics` YOLOv11 + `E2FGVI-HQ`) | Moving Sora badge cleaner | **AGPL-3.0 + CC-BY-NC-4.0** | **Excluded from package** (transitively copyleft + non-commercial) |
| `Lightricks/LTX-Video` / `LTX-2` & `NVIDIA/Cosmos-Tokenizer` | Video DiT / causal video tokenizers | **OpenRAIL / ARR-capped Community / NVIDIA Open Model License** | **Excluded from default package** (weight restrictions & guardrail-circumvention termination clause) |
| `facebookresearch/demucs` (`htdemucs` weights), `HKUSTAudio/xcodec2`, `nvidia/audio-codec-44khz` | Stem separation & audio codecs | **MUSDB18-HQ NC / CC-BY-NC-4.0 / NSCLv1** | **Excluded from package** (official `htdemucs` weights are derived from non-commercial `MUSDB18-HQ`) |
| `stabilityai/stable-audio-open-1.0` (`AutoencoderOobleck`) | 44.1 kHz stereo continuous audio VAE | **Stability AI Community License** | **Excluded from package** (revenue/commercial restrictions) |
| `TimbreWatermark` / `DiffErase` mel subtree / `CtrlRegen` | Academic watermark attacks | **GPL-3.0 / CC-BY-NC-SA-4.0** | **Excluded from package** |

## Primary audio literature

| Work | Mechanism and artifact evidence | Limit that changes our experiment |
| --- | --- | --- |
| [AudioMarkBench](https://arxiv.org/html/2406.06979v2), [code](https://github.com/moyangkuo/AudioMarkBench) | Public evaluation of watermark schemes and signal/optimization attacks | Separate no-query transforms from detector-query or white-box experiments; a public-model benchmark does not certify proprietary compatibility |
| [RAW-Bench](https://github.com/SonyResearch/raw_bench), [errata](https://github.com/SonyResearch/raw_bench/blob/main/docs/2026-09-29-errata.md) | Reproducible audio attack benchmark with corrected codec handling | Incorrect DAC input rates and double MP3 encoding distorted earlier conclusions. Pin corrected source and compare rate/codec controls before attributing removal |
| [SoK audio watermarking](https://arxiv.org/html/2503.19176v2), [project](https://sokaudiowm.github.io/) | Broad scheme/attack comparison, including voice conversion and synthesis | Strong signal suppression can change speaker, words or timing; these are destructive comparators, not faithful default processing |
| [DiffErase](https://arxiv.org/html/2605.30614), [code](https://github.com/DiffErase/DiffErase) | The mel route includes actual noise addition and reverse diffusion, separately from its vocoder; public artifacts | Stock paths crop/downmix. The checked mel subtree is GPL-3.0, root wrapper and weight grants remain unresolved. Full-file adaptation and an actual diffusion-bypass control are prerequisites; do not infer a permissive weight license from code |
| [AudioSR code](https://github.com/haoheliu/versatile_audio_super_resolution/tree/d312fbab9f0e94087d9f2802d03cf184353cc805), [Basic weights](https://huggingface.co/haoheliu/audiosr_basic/tree/74a47f49061a1e788e968cc43ad45c0b6243f37d) | Conditional latent DDIM followed by VAE/vocoder; checked Basic model card declares Apache-2.0 and source declares MIT | Stock processing restores input lowpass mel/STFT, rescales amplitudes and uses mono helpers. Saving trims by container duration, not an exact native sample clock. Required legacy text dependencies have separate licenses despite being unused by audio inference. Pin the loader and all artifacts, audit full state, and distinguish lowpass-only, diffusion-bypass and full diffusion controls; speech weights have no checked equivalent grant |
| [Mimi weights](https://huggingface.co/kyutai/mimi/tree/89091b3e466eb6a9d11e537bf26b144f194978f7), [Transformers implementation](https://github.com/huggingface/transformers/tree/v5.15.1/src/transformers/models/mimi) | Semantic/acoustic neural codec with a checked 24 kHz, default 32-quantizer path; weights CC-BY-4.0, implementation Apache-2.0 | Full-track CPU experiments removed local Perth on independent speech generations, but a short case failed the frozen relative voice guard. Raw M/S retained stereo better than independent channels in the native cohort. Provider candidates remain unverified; internal source-rate conversion and official decoder-excess prefix selection must be explicit |
| [HarmonicAttack](https://arxiv.org/html/2511.21577v2), [linked artifact](https://anonymous.4open.science/r/harmonicaudio-B18A/) | Learned waveform/STFT suppression trained with paired data; artifact is explicitly linked | Artifact access returned 403 in this pass; license, usable checkpoints and completeness remain unverified. Query-free inference does not eliminate training-data cost |
| [Generative speech enhancement](https://arxiv.org/html/2609.29040v1) | Compares generative GenSE/UniSE, discriminative enhancement and vocoder baselines | Speech results cannot be transferred to music or current proprietary schemes. Preserve transcript, identity and clocks, and isolate frontend/vocoder effects |
| [FocalCodec v0.0.1](https://github.com/lucadellalib/focalcodec/tree/cad8bf81d70875f756770449c34909d8b0951642), [50 Hz weights](https://huggingface.co/lucadellalib/focalcodec_50hz/tree/d6d6d9524e52155c85193c2c3b8da1cf8842f019) | Apache-2.0 source and weight card; exact historical `v0.0.1` schema loads all 355 safetensors keys strictly without legacy remapping or unused `knn-vc` code, exposing continuous 6-layer WavLM feature-to-Vocos and binary-token reconstruction | Internal 16 kHz grid caps output at 8 kHz, and Vocos ISTFT randomizes carrier phase with ~`+2.38 ms` group delay. Continuous feature resynthesis with symmetric `(40, 40)` WavLM context passed complete paired WAV/MOV/MP4 Perth, ASR, speaker, and phase-invariant timing gates on `LONG03` and preserved `"wind"` on held-out `LONG04`, but Vocos STFT synthesis cross-reacted with Perth in a `LONG04` pause window even on the clean twin; keep explicit and saved-output gated |
| [SGMSE](https://github.com/sp-uhh/sgmse/tree/1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e), [checkpoint repository](https://huggingface.co/sp-uhh/speech-enhancement-sgmse/tree/b6485214b3662a7f90309f397cacf1384046783c) | MIT source and model card; complex-STFT diffusion provides a continuous alternative to discrete speech generation | Stock paths normalize amplitude and select a model clock; convenience inference contains CUDA-specific calls. Full-state loading, checkpoint lineage and native-grid forward/inverse handling remain unqualified; no new watermark efficacy result |
| [GenSE implementation](https://github.com/yaoxunji/gen-se/tree/bc14cd3b3ae3b131537bfa73501bd504985eecd5), [weights](https://huggingface.co/yaoxunji/gen-se) | Published semantic/acoustic generation pipeline and MIT-tagged model repository | The checked complete code tree has no license file. Stock inference keeps only the first channel and resamples to 16 kHz. Weight metadata does not establish a code grant or native stereo/clock fidelity |
| [UniSE implementation](https://github.com/alibaba/unified-audio/tree/c4004e217ddf7c514f72ea22f2d0fbf43f02ae90/QuarkAudio-UniSE), [weights](https://huggingface.co/QuarkAudio/QuarkAudio-UniSE) | Apache-2.0 source combines WavLM, autoregressive prediction and BiCodec | The checked weight card and official ModelScope mirror supply no explicit checkpoint grant. Its required [Spark-TTS BiCodec weights](https://huggingface.co/SparkAudio/Spark-TTS-0.5B/blob/642071559bfc6346c2359d19dcb6be3f9dd8a05d/README.md) declare CC BY-NC-SA 4.0, independently of Apache source headers. Default test configuration selects separation; stock inference uses independent blocks, repeated-tail padding and amplitude normalization, so count equality alone cannot qualify native timing |
| [UNIVERSE++](https://github.com/line/open-universe/tree/939dd3c12acaac9da5c45d241437169cca3e647a) | Apache-2.0 source and official weights; pinned full-track CPU speech pilot with strict raw/EMA mapping and matched controls | The default diffusion arm, auxiliary synthesis and fixed step-count family all retained local Perth. Several arms passed the frozen initial quality gates, so perceptual restoration is distinct from removal. No held-out generation or Google verdict was promoted |
| [Training-free fragility](https://arxiv.org/html/2608.16566), [code](https://github.com/i-618/audio-watermark-fragility) | Clean/marked paired probes characterize residual structure and guide signal processing | Requires aligned pairs, not one unknown file. Checked repository has only a subset of paper adapters; other schemes need external reproduction. One-sample misalignment can invalidate residual probes |
| [SweepMark](https://arxiv.org/html/2609.34003) | Learned stationary/nonstationary artifact suppression; v2 renames the earlier DeMark work | Count it as one method. No usable official code link located in the checked paper/search; training and artifact audit precede implementation |
| [NeuMark](https://arxiv.org/html/2609.25719), [code](https://github.com/goannan/NeuMark) | Tokenizer-latent watermark designed for codec robustness | Resistant reference cohort; not a removal backend. Weight and license audit remains necessary |
| [DeltaMark](https://arxiv.org/html/2609.25830), [code](https://github.com/Munich-Music-Labs/latent-audio-watermarking) | Latent audio watermarking with codec robustness | Tests the limits of codec cycling; compare equivalent encoding distortion rather than assuming all latent marks fail |
| [CRAW](https://arxiv.org/html/2609.03107), [code](https://github.com/DavidC1212/craw) | Codec/restoration-aware robust embedding and decoding | Another counterexample to universal codec-removal claims; cited artifact availability is not an executed reproduction |

AudioSeal, WavMark, SilentCipher and Perth provide useful controlled local
receivers already under study in this tree. Their outcomes remain scoped to
the pinned implementation and checkpoint. PESQ or a speech-only intelligibility
metric is insufficient for music, stereo or environmental fidelity.

## Primary video literature and candidate models

| Work or model | Evidence and intended use | Boundary |
| --- | --- | --- |
| [VideoMarkBench](https://arxiv.org/html/2505.21620v1), [code](https://github.com/zhengyuan-jiang/VideoMarkBench) | Public watermark evaluation including temporal aggregation and ordinary video transforms | The checked benchmark is not evidence that full diffusion/VAE regeneration has already succeeded. Aggregation and bit accuracy do not prove every-frame erasure |
| [VideoSeal](https://github.com/facebookresearch/videoseal) | Reproducible public video mark/receiver for controlled tests | Not proof of Meta production ContentSeal compatibility |
| [PoseRegen](https://arxiv.org/html/2610.22457v2), [project](https://sites.google.com/view/poseregen) | Pose-guided trained motion reconstruction | Canny/optical flow is not its trained adapter. No usable adapter checkpoint was established in the checked project page; human-pose results do not cover arbitrary scenes |
| [RAVEN](https://arxiv.org/html/2601.08832) | Novel-view image regeneration | Viewpoint/geometry changes conflict with position preservation; not a temporal-video implementation |
| [CogVideoX](https://github.com/THUDM/CogVideo) | Full native-geometry posterior-mode round trip completed after resource-bounded FP16 qualification | CapCut detected both the AV and silent-video outputs, with a positive silent codec control. Retain this arm as a measured structural baseline; no removal qualification on this export |
| [VidTok](https://github.com/microsoft/VidTok), [official checkpoints](https://huggingface.co/microsoft/VidTok) | Pinned v1.1 causal FSQ model loaded strictly; full native-grid delivery and matched codec control checked | Official replicated temporal padding was accounted for explicitly. CapCut detected both audiovisual and silent reconstructed deliveries with a positive silent control. This fixed arm failed removal qualification |
| [VACE](https://github.com/ali-vilab/VACE) | Pinned temporal video model with structural conditioning; actual native-grid conditioned prefix and full Wan VAE posterior-mode reconstruction executed | The conditioned prefix failed its frozen structure gate. CapCut detected the full posterior-mode delivery and matched codec control. Full posterior-sample decoding stopped at the declared memory limit; it has no delivered-file verdict |
| RIFE, BasicVSR, PS-SR, SeedVR2, FlashVSR | Interpolation/restoration mechanisms worth controlled comparisons | Restoration quality is not watermark efficacy. RIFE may retain source frames; output cadence/geometry can change |

BigVGAN supplies a reproducible vocoder baseline. Its local preflight exposed
hop-boundary length loss; subsequent complete mono paired tests preserved
sample counts but retained Perth and showed nonzero lags. The
[evidence log](media-removal-evidence-log.md) records the full codec/vocoder rate grid,
quality limits and the scoped EnCodec results. Rate-correct handling, exact
length, perceptual quality and compatible signal outcomes remain separate gates.

## Search coverage and unresolved work

An expansion pass added Cartesia, Stable Audio, Nova Sonic/Polly, Fish Audio,
Mureka, Murf, PlayHT, Flow Music, PixVerse, Tencent Hunyuan, Krea, Magnific,
Grok Imagine/voice, and LTX. Their primary sources and acquisition distinctions
are in the verifier/corpus appendices. Especially useful new calibration leads
are Cartesia's public watermark detector, xAI's declared AudioSeal voice path,
Stable Audio's API credentials, and Nova's account-mediated detection route.

[Soft bindings benchmark](https://arxiv.org/html/2610.04151), with
[released artifacts](https://github.com/Original-Pictures/soft-bindings-benchmark),
adds a useful evaluation reference: expected-payload decisions, source-level
statistics, fingerprints versus marks, and hardware/color-conversion effects.
Its openly evaluated schemes do not certify proprietary compatibility. Reuse
the measurement lessons rather than turning a fingerprint match or a blind
decoder's false positive into a watermark-removal claim.

The acquisition/verifier appendices cover Google, ByteDance/CapCut, ElevenLabs,
Resemble, OpenAI, Meta, Microsoft, Suno, Udio, Runway, Kling, Luma, Pika,
MiniMax/Hailuo, Vidu, Wan, Ovi, HeyGen, Synthesia and Adobe. A product front end
may host another manufacturer's model; native exports must record both.
“No compatible public receiver located in checked official sources” is an
access/evidence gap, not proof that a producer embeds no watermark.

The highest-value next evidence is a lineage-qualified real-output corpus,
whole-file vocoder and temporal-VAE measurements, and compatible provider
verdicts on frozen candidates. Additional commercial trials, inaccessible
artifacts and unpublished methods remain a comparison queue. Priorities,
falsification conditions and promotion gates are defined in the roadmap.
