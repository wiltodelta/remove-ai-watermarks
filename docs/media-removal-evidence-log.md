# Real video and audio removal evidence log

This archive preserves the chronological experiment cards and provider-oracle
tranches behind the [real video and audio removal roadmap](media-removal-roadmap.md).
Current shipped behavior, certified operating points, and active engineering
priorities live in [the roadmap](media-removal-roadmap.md),
[module internals](module-internals.md), [audio/video coverage](audio-video-removal.md),
and the tracked oracle manifests under [`data/evaluations/`](../data/evaluations/).

## Experiment cards

Reuse existing [native-clock AudioSeal qualification](audio-video-removal.md#native-clock-audioseal-compatibility)
for the published LTX marking stack. Its opt-in public adapter and model pack
are concrete reference evidence, distinct from hosted LTX Studio. Preserve
the validated 24/48 kHz receiver paths alongside 16 kHz; a resampling-only
negative can otherwise hide the mark. Existing musical-negative Perth
cross-reactions also mean a discovery positive on music cannot identify a
producer's deployed scheme.

### Initial full-file vocoder result

The pinned BigVGAN round trip was run on both full clean/marked Chatterbox
Nano pairs. Both marked sources, rate-only controls and reconstructed outputs
remained positive for the pinned local Perth receiver; clean twins stayed
negative. The files retained their native sample counts, but correlation found
nonzero reconstruction lags. This rejects promotion of this plain vocoder arm
for the tested cohort. It remains a useful frontend control for generative
enhancement/diffusion, with no claim about hosted Resemble, music, stereo, or
other proprietary schemes. Full private result manifests retain parameters,
hashes, saved-output readings and alignment measurements.

The ordinary AAC comparator retained the local mark, while the existing
EnCodec 6 kbps arm returned negative on both marked pairs with clean negatives,
unchanged sample counts and no measured global lag. AAC and EnCodec were not
matched for distortion, so this does not establish an efficacy/quality
advantage at equivalent fidelity. EnCodec remains a candidate for quality and
production qualification; the vocoder remains a control. Transcript, identity,
listening and final-mux checks are still separate gates.

A subsequent fixed exploratory rate grid retained the mark through AAC at
6, 12, 24 and 48 kbps on both marked recordings. EnCodec at 12 kbps gave mixed
outcomes, and 24 kbps remained positive on both. Clean controls stayed negative.
Together with the original 6 kbps arm, this motivates operator-specific
rate/fidelity testing rather than a claim that low nominal bitrate alone
explains the result. The grid reused observed pairs and is not a new untouched
holdout; sample-error and spectral metrics do not establish perceptual matching.

Every experiment freezes an input manifest, model/code/weight revisions,
parameters, seeds, preprocessing, output encoding, controls, measurement scope,
and falsification before any external verdict. The result manifest is append-only
and retains failures. Selection may use local quality checks; it must not tune
candidates adaptively against an external oracle.

### Audio tranche

The retained native-provider tranche established exact-file SynthID-positive
audio baselines. The strongest lineage is the publication-cleared Flow Veo
fixture: its decoded PCM matches the previously verified WAV, and two Gemini
slots explicitly identified SynthID on both the native WAV and its PCM16
control. Two retained Spaces recordings also have explicit Gemini SynthID
positives, although their containers have prior visual editing history. Their
WAVs match the retained containers' decoded PCM; this does not establish an
unedited chain back to generation. Reuse these historical verdicts only for
their exact upload hashes and verification surfaces.

The existing combined spectral/EnCodec arm remained explicitly positive on
the Veo recording. Responses to the other two combined candidates were
indeterminate, not qualified negatives. A direct detector portal error supplies
no audio verdict. The subsequent general-audio vocoder and codec comparison
used these actual source-positive recordings rather than assuming the local
Perth result transfers to SynthID. Full BigVGAN and codec-only EnCodec arms,
with their model-rate round-trip controls, preserved saved native rates,
channel counts and sample counts. EnCodec retained stronger source-relative
waveform and stereo diagnostics than independent-channel vocoding in this
cohort.

The first frozen native Veo audio pass subsequently produced an explicit
Google SynthID-negative for the saved BigVGAN L/R output, with an explicit
source-positive in the same Gemini account/pass. The EnCodec6 output remained
explicitly SynthID-positive there. Independent-channel vocoding changed stereo
coherence; a separate fixed M/S quality arm improved measured channel
correlation and side/mid power on Veo and the two retained speech recordings.
Once Chrome transport recovered, the frozen native Veo M/S file received
explicit SynthID-negative replies in separate configured accounts, with exact
source and resampling-only files positive in those same passes, whereas a
different native Veo Lite generation and an independent speech pass retained
SynthID after the same fixed BigVGAN M/S processing. Thus plain BigVGAN M/S has
a scoped success and real counterexamples, not a general removal qualification;
it remains explicit (`--method bigvgan`), excluded from automatic selection, and
cannot verify SynthID itself.

The licensed AudioSR Basic follow-up exercised conditional diffusion with strict
full-state loading and paired local Perth controls. Its official
low-frequency-restored output retained Perth, while removing source
low-frequency reinsertion degraded words and speaker consistency. MetricGAN+
and UNIVERSE++ likewise retained local Perth on matched speech pairs. On a
complete native long-speech comparison (`LONG03`), EnCodec 6 was negative over
the whole file but positive in individual windows; Mimi passed `LONG03` WAV and
AAC MP4 window scans on the marked twin but regularized a word (`"wind"` to
`"wooden"`) and crossed Perth after AAC (`0.6262`) on held-out `LONG04`.

Matching FocalCodec `50 Hz` safetensors against its historical `v0.0.1` source
schema (`cad8bf81d70875f756770449c34909d8b0951642`) with symmetric `(40, 40)`
WavLM boundary context and source-conditioned pause gating
(`resynthesize_conditioned` in `audio_focalcodec.py`) passed paired `LONG03`,
`LONG04`, and held-out `LONG05` local Perth/ASR/speaker/timing gates, and
returned `5/5` `not_detected` verdicts on the initial `gemini-web` cohort, but
remained `detected` (`0:00-0:10`) on the held-out 10-second
`google-gemini-omni-flash-api` track. By contrast, 3-codebook `44.1 kHz` `DAC`
Mid/Side (`--method dac`, `descript/dac_44khz`) returned `4/4` `not_detected`
verdicts across all source-positive native Google audio tracks (including
`google-gemini-omni-flash-api`, certified in
[`data/evaluations/audio-synthid-oracle.csv`](../data/evaluations/audio-synthid-oracle.csv))
and cleared `elevenlabs-api` across multi-window native MP3 exports while
preserving `22.05 kHz` stereo bandwidth. A follow-up 48 kHz stereo music
comparison across `AudioSeal` and `WavMark` carriers measured `DAC` (`44.1 kHz`,
`q=3`: both `verified=True`, `MR-STFT 1.11-1.12`, `Mel 4.06-4.19 dB`,
`Side/Mid drift 0.21-0.54 dB`) ahead of `facebook/encodec_48khz` (`6 kbps`: both
`verified=True`, `MR-STFT 1.99`, `Mel 4.15-4.26 dB`, `Side/Mid drift 2.40-2.95 dB`;
`12 kbps` and `24 kbps` retained `AudioSeal`) and far ahead of `24 kHz`
`encodec` (`6 kbps`: retained `AudioSeal`, `Mel 18.50-18.59 dB`), keeping `DAC`
as the studio neural-codec baseline entering Phase 1.

### Video tranche

On the 121-frame `854x480 @ 24 fps` Seedance forest cohort, zero-noise
posterior-mode temporal bottlenecks (`CogVideoX` FP16, `VidTok` v1.1 causal FSQ,
and zero-noise `AutoencoderKLWan`) preserved native frame clocks and stream-copied
audio packets, but all remained `detected` on CapCut alongside matched silent
codec controls, whereas extracted M4A audio was `not_detected`. Conditioned
`VACE` failed its frozen prefix structure gate. By contrast, CPU/MPS
guide-registered sigma-1 texture transfer (`seedance-experimental`) and NVIDIA
H100 CUDA 3D `Wan 2.1` video diffusion (`--profile wan21-seedance-experimental`,
`strength=0.12`) returned offline-verified CapCut `not_detected` receipts on
both `forest` and held-out `bottle` (`34.42 dB PSNR`, `TRR 1.0682` vs
`31.85 dB`, `TRR 1.3854`), while Dreamina and BytePlus enterprise API exports
cleared via stream-copy C2PA metadata stripping alone.

On the Google Veo visual cohort (`flow-001`, `lite-003`, 240-frame
`google-gemini-omni-flash-api`, and 96-frame `veo-3-1`), matched silent H.264
controls and native-grid zero-noise reconstructions remained `detected`. Raising
2D `sd-vae-ft-mse` (`noise_std=0.15`) from `512 @ 12 fps` to `768x432 @ 24 fps`
cleared `flow-001` (`not_detected`) and all 25 sliding `VideoSeal 1.0` windows
(with `temporal_stabilize=True` reducing `temporal_residual_ratio` by
`10.9-30.7%`), but remained `detected` on textured `lite-003`, keeping the 2D
spatial-VAE default pinned at `512 @ 12 fps`. On NVIDIA CUDA, 3D causal
`Wan 2.1` diffusion (`--profile wan21-synthid-experimental`, `strength=0.38`,
streaming `chunk_frames=193` / `overlap_frames=17`) returned `not_detected`
across all tested silent `1280x720 @ 24 fps` deliveries and combined
full-audiovisual (`Wan 2.1` + `DAC` AAC-192k) MP4 deliveries, certified on
publication-cleared carriers in
[`data/evaluations/video-wan-oracle.csv`](../data/evaluations/video-wan-oracle.csv).

Subsequent Stage 2/3 provider-oracle runs wired `elevenlabs-api`,
`cartesia-api`, `cartesia-web`, `resemble-api`, and `kling-web` into
`scripts/provider_oracles.py`: `meta-web` verified positive-to-negative video
removal on production `VideoSeal v4` (`Muse Image 1`); and authenticated
`kling-web` runs across native Kling MP4 and PNG exports proved that Kling's
`"Invisible watermark"` row on MP4 reads the H.264 `kling-ai` SEI tag (`Detected
-> Not detected` under lossless `remove_video_metadata` alone).

In the Phase 1-3 audio/video architecture pass (2026-11-18):
- **48 kHz Stereo Music & 16 kHz Speech Bottleneck Benchmark**: across paired 48 kHz
  stereo music (`AudioSeal`, `WavMark`) and 16 kHz speech (`Perth`) carriers,
  3-codebook `DAC 44.1 kHz` (`--method dac`, `cpu`/`mps`/`cuda`) achieved `verified=True`
  across all three schemes with the lowest spectral distortion (`MR-STFT = 1.1098`,
  `Mel L1 = 4.289 dB`, `Side/Mid drift = 0.208 dB` on 48 kHz music) and ~100x faster
  Apple Silicon `mps` runtime than `BigVGAN v2` (`plain-bigvgan` `MR-STFT = 1.3093` and
  pre-warped `bigvgan-mel` `MR-STFT = 1.6349` both retained `Perth` on 16 kHz speech;
  `encodec48` 6 kbps scored `MR-STFT = 1.9863`, `Side/Mid drift = 2.398 dB`). Tagged
  `5.1` and `7.1` surround stem decomposition (`FL/FR` M/S, `FC` mono, `<120 Hz`
  Butterworth LFE, surround M/S pairs) and 16/24-bit PCM sample-format preservation
  shipped on the neural audio path alongside `--vendor` cohort routing.
- **Self-Calibrated Multi-Frame `YCrCb` Reverse Alpha Unblending**: chaining stepwise
  `PyrLK + RANSAC` affine transforms across moving context rings to form median donor
  references (`estimate_clip_alpha_matte` + `unblend_ycbcr_roi` in `video_temporal.py`)
  achieved `Alpha MAE = 0.0010` and `36.98 dB` ROI PSNR (`+20.50 dB` over spatial Telea's
  `16.48 dB`) while leaving 100% of clean background pixels inside the detection box
  untouched. `dreamina` and `wan` visible video detectors (`10` registered video marks
  total), user-directed `video erase --region --frames`, 10-bit HDR (`yuv420p10le`,
  `bt2020nc`, `smpte2084`/`arib-std-b67`) preservation, fMP4 `emsg` + H.264/HEVC AI SEI +
  Matroska `application/c2pa` attachment scrubbing, blind `VideoSeal`/`PixelSeal`/`ChunkySeal`
  (`1024`-bit) presence detection (`video message detect`), and CPU prompt-offloaded
  `Wan 2.1` (`~3.4 GiB` accelerator footprint, `cuda` and explicit `mps`, up to 5-minute
  1080p streaming budget) all shipped with unit and integration coverage.


In the Wave 2 audio/video market-leadership pass (2026-11-20):
- **Pure-PyTorch `SNAC 44.1 kHz` (`--method snac`, `hubertsiuzdak/snac_44khz@873ebef9718b89660340c6f55a2b515e98cfa1d9`, MIT)**:
  across `q in {2, 3, 4}` multi-scale RVQ codebooks (`vq_strides = [8, 4, 2, 1]`),
  retaining all 4 codebooks (`q=4`) preserved `Perth` on 16 kHz speech via the
  finest `stride=1` residual, whereas fixing `FIXED_QUANTIZERS = 3` (`strides = 8, 4, 2`)
  cleared `AudioSeal`, `WavMark`, and `Perth` (`3/3` `not_detected`, `quality_passed=True`,
  `Side/Mid drift = 2.66 dB`) while running in `4.56 s` on Apple Silicon `mps` (`3.85x`
  faster than CPU `DAC`). `DAC` (`MR-STFT = 1.11`, `Mel = 4.06 dB`) remains the
  highest-fidelity `--vendor` default, while `--method snac` ships as the fast
  pure-PyTorch 44.1 kHz alternative across mono, raw M/S stereo, and `5.1`/`7.1`
  surround stems.
- **Wan 2.1 Zero-UMT5 Prompt Cache, 18-Mark Visible Video Suite & AV1 `OBU_METADATA` Scrubbing**:
  persisting the static neutral prompt embedding on disk skips downloading and loading
  the `22.7 GB` `UMT5-XXL` encoder on warm cache hits, unlocking `WAN21_AUTO_MAX_FRAME_PIXELS = 1920 * 1080`
  and `WAN21_MIN_CUDA_VRAM_BYTES = 8 GiB` (with `WAN21_COMPACT_CHUNK_FRAMES = 81` step-down,
  while keeping `--profile auto` off `mps`). Eight additional visible video detectors
  (`jimeng`, `jimeng_pill`, `qwen`, `yuanbao`, `baidu`, `runninghub`, `microsoft`,
  `generic_ai_label`) calibrated at `0/71` false positives across tracked `data/` MP4s
  expanded `VIDEO_VISIBLE_MARKS` to `18`, paired with spatially disjoint dual-mark
  co-selection, `--trim-bumper` outro trimming, `erase --track` pyramidal LK + Sobel/NCC
  tracking, single-pass `visible -> invisible` frame streaming, `.webm` video/audio
  support, and HDR-preserving AV1 `OBU_METADATA` -> `OBU_PADDING` lossless scrubbing
  across ISOBMFF `av01` and Matroska/WebM `V_AV1` (recomputing EBML `0xBF` `CRC-32`).
- **Head-to-Head Multi-Surface Competitor Benchmark (`scripts/media_removal_benchmark.py`)**:
  evaluated across a 10-case manifest spanning clean negative controls (`audioseal-mono-clean`,
  `videoseal-clean`), 48 kHz stereo music (`AudioSeal` under `auto`, `dac`, and `snac`),
  stereo M/S `WavMark`, 16 kHz `Perth` (`dac` and `snac`), `VideoSeal` video, and the
  4-layer audiovisual clip (`TC260 AIGC` metadata + visible `Kling` + `VideoSeal` pixels +
  `AudioSeal` audio):
  - `remove-ai-watermarks` (Wave 2): **`10/10 passed (100%)`** (`0` failed, byte-exact
    skip on both clean controls, `verified_removed=True` and `quality_passed=True` on all
    8 marked cases, `31.38 dB PSNR` / `0.9641 SSIM` on `video-all-4layer-marked`).
  - Open-source lossy transcode + resample baseline (`ShadowMarkDefense`-style `24 kHz`
    `AAC 96k` bounce + `H.264 CRF 28`): **`1/10 passed (10%)`**, `9/10 failed` (retained
    `AudioSeal`, `WavMark`, and `Perth`, left visible `Kling` + `TC260 AIGC` + `AudioSeal`
    on `video-all-4layer-marked`, and degraded both clean controls).
  - Open-source spectral bandstop + blur baseline (`watermarks-remover`-style `3-7 kHz`
    bandreject + `10 kHz` lowpass + `gblur=1.8` + `H.264 CRF 26`): **`1/10 passed (10%)`**,
    `9/10 failed` (retained `AudioSeal`, `WavMark`, and `Perth`, left visible `Kling` +
    `AudioSeal` on `video-all-4layer-marked`, and degraded both clean controls).

In Wave 6 (2026-11-21):
- **Apple Silicon `CoreMLExecutionProvider` vs `CPUExecutionProvider` ONNX Benchmark**:
  measured on `lama_fp32.onnx` (`512x512`) and `migan.onnx` (`512x512`): `MLProgram`
  failed plan construction on both models; `NeuralNetwork` partitioned `lama_fp32.onnx`
  into 585 subgraphs (`78.34 s` vs `81.10 s` CPU, `1.04x`, `MAE = 0.0378`, plus multi-minute
  compile overhead) and ran `migan.onnx` `2.8x` slower than CPU (`18.95 s` vs `6.78 s`,
  `0.36x`). `region_eraser.py` therefore remains pinned to `("CUDAExecutionProvider",
  "CPUExecutionProvider")` with batched CUDA ONNX inference (`fill_batch`).
- **Studio Mastering High-Frequency Crossover (`preserve_hf_hz`) on 48 kHz Stereo Music**:
  paired 8th-order zero-phase Butterworth crossover (`sosfiltfilt`) on `44.1 kHz` `DAC`
  with `AudioSeal` embedded on 48 kHz stereo music with bright `15.5-18.4 kHz` transients:
  - `DAC full` (default): `MR-STFT = 1.2001`, `HF_RMSE (>14 kHz) = 0.05788`, `AudioSeal = (0.2953, 0.3053)`.
  - `DAC + preserve_hf_hz=14000`: `MR-STFT = 0.7770` (`-35%`), `HF_RMSE = 0.00156` (`37x` lower),
    `AudioSeal = (0.2955, 0.3054)` (unchanged removal).
  - Kept as an explicit opt-in (`preserve_hf_hz`) so oracle-certified full-band `DAC`
    defaults for Google SynthID, ElevenLabs, and `SilentCipher-44k` remain untouched.
- **Official `synthid.com` Live Audio Oracle Checks (2026-11-22, signed-in Chrome direct route)**:
  - `SNAC 44.1 kHz` (`q=3`, `hubertsiuzdak/snac_44khz`): **`2/2` `not_detected`** (`SynthID was not detected`)
    on both source-positive Google audio tracks:
    - `higgsfield-veo-3-1-lite_snac_q3.wav` (`sha256:08457e6fa490a9cc394a9890588020afddbbdd5f5898b3841d213e2ce4e424a0`)
    - `google-gemini-omni-flash-api_snac_q3.wav` (`sha256:940a3774b09e8af5da847f9d287d45be263ed91621d79d671234721f3561ae02`)
  - `DAC 44.1 kHz` (`q=3`) + **`preserve_hf_hz=14000.0`**: **`1/1` `not_detected`** (`SynthID was not detected`)
    on `google-gemini-omni-flash-api_dac_hf14k.wav` (`sha256:70adefe96315c03f0e6ccc8cb19c25fb2c72a5c5b38fc4f0bc4a6dde0ac8c07d`),
    confirming no Google SynthID leakage above 14 kHz on the tested track.

In Wave 9 (2026-11-22, `v0.70.0`):
- **Seedance single-pass streaming + guide-resolution temporal stabilization + CIE Lab color anchoring**
  (`dreamina-seedance-2-mini.mp4`, `1280x720 @ 24 fps`, CPU/MPS `sd-vae-ft-mse`):
  - Eliminating intermediate lossy `guide.mp4` (`CRF 18`, `yuv420p`), applying
    `iter_stabilized_regenerated_frames` at guide resolution, and anchoring
    ultra-low-frequency CIE Lab illumination/chroma (`sigma >= 32 px`) improved
    PSNR from `31.04 dB` to **`33.81 dB`** (`+2.77 dB`), cut `Lab Delta-E` from
    `5.90` to **`4.06`** (`-31%`), and cut temporal residual ratio from `1.3723`
    to **`1.1506`** (`-60%` excess flicker) with sharpness ratio preserved (`0.711`).
- **2D SynthID default `temporal_stabilize=True` + CIE Lab color anchoring**
  (`higgsfield-veo-3-1-lite.mp4`, `512x288 @ 12 fps`):
  - Temporal residual ratio improved from `1.7709` to **`1.3929`** (`-49%` excess
    flicker), `PSNR` `24.53 -> 24.63 dB`, `Lab Delta-E` `10.09 -> 9.89`.
- **Audio quiet-stem gain-staging (`DAC` / `SNAC`) + native-48 kHz `AudioSeal` check**:
  - On `google-gemini-omni-flash-api.mp4` (`4s` quiet stereo stem, `source_peak=0.103`):
    `DAC 44.1 kHz` max Side/Mid drift dropped from `0.487 dB` to **`0.276 dB`**
    (`-43%`, `0.271 dB` with `preserve_hf_hz=14000`, `RMS err 0.00595 -> 0.00579`);
    `SNAC 44.1 kHz` max Side/Mid drift dropped from `0.829 dB` to **`0.554 dB`**
    (`0.533 dB` with `preserve_hf_hz=14000`, `-36%`).
  - On native-48 kHz `AudioSeal` (`audioseal-native48k-music-marked`, LTX-2 style
    stamped directly at 48 kHz without 16 kHz downsampling): before `mean_detect_prob=0.999974`
    (`frac=1.0`); after `DAC q=3` `0.0140` (`frac=0.0024`), and with `preserve_hf_hz=14000`
    `0.0139` (`frac=0.0008`, lower `RMS err 0.05277 -> 0.04990`).

In Wave 11 (2026-11-22, `v0.73.0`):
- **Live Google Gemini SynthID oracle checks (`gemini.google.com/u/1/app`, `2/2` `not_detected`)**:
  - `higgsfield-veo-3-1-lite_dac_hf14k_v071.wav`
    (`sha256:94215d0981e47778dc38db06aae9c4cbde2627e972fedeea691c15a72cf11f0b`,
    `DAC 44.1 kHz` `q=3` + `preserve_hf_hz=14000.0`, `quality_passed=True`):
    `"This audio file does not contain a SynthID watermark indicating it was made with Google AI."`
    Combined with the prior `synthid.com` check on `google-gemini-omni-flash-api_dac_hf14k.wav`,
    `DAC + preserve_hf_hz=14000.0` is now **`2/2` `not_detected`** on live Google SynthID
    oracles and promoted to the default for `DAC` on `>= 44.1 kHz` tracks.
  - `higgsfield-veo-3-1-lite_2d_synthid_v071_silent.mp4`
    (`sha256:2d557ef65283020babe853065675e001344bb610ad1eb4aaa6157f84c6799d62`,
    `v0.70.0+` 2D SynthID `temporal_stabilize=True` + CIE Lab low-frequency color anchoring):
    `"No SynthID watermark from Google AI was detected in this video, and no valid C2PA Content Credentials were found."`
- **Unified containers and pruned obsolete experimental audio backends**:
  - Removed obsolete experimental audio methods (`bigvgan`, `mimi`, `focalcodec`) and
    the terminated `silentcipher16` scheme; retained production `spectral`, `encodec`,
    `combined`, `dac`, and `snac`.
  - Unified all video pipelines (`seedance`, `wan21-synthid`, `wan21-seedance`, `synthid`,
    `visible`, `erase`, `video audio`) to accept all seven video containers (`.mp4`, `.mov`,
    `.m4v`, `.mkv`, `.webm`, `.avi`, `.flv`), unified all nine audio formats (`.wav`, `.flac`,
    `.mp3`, `.m4a`, `.ogg`, `.oga`, `.opus`, `.aac`, `.mka`) on both input and output, and
    removed artificial resolution/duration hard caps in `video_wan.py` (keeping explicit
    opt-in worker guardrails `max_duration_seconds` / `max_frame_pixels` defaulting to `None`).

In Wave 12 (2026-11-23, `v0.74.0`):
- **Live Google Gemini SynthID oracle check on `SNAC 44.1 kHz + preserve_hf_hz=14000.0`**
  (`gemini.google.com/u/1/app/e7c5c997ababf122` & `/u/1/app/a334f969227f34ff`):
  - `veo_snac_hf14k.wav` (`sha256:16aba611ab22c73df45995035169234a0d08effaad01b2cca69c13656edffe2c`,
    `hubertsiuzdak/snac_44khz@873ebef9718b89660340c6f55a2b515e98cfa1d9`, `q=3`, `preserve_hf_hz=14000.0`,
    `quality_passed=True` on `higgsfield-veo-3-1-lite.mp4`): **`not_detected`**
    (`"No SynthID watermark or valid C2PA Content Credentials were detected in this audio file, and there is no indication that it was made with Google AI..."`).
    Recorded in `data/evaluations/audio-synthid-oracle.csv`, and promoted `preserve_hf_hz=14000.0`
    to the default for `snac` on `>= 44.1 kHz` audio (matching `dac`).
- **Fast native-resolution guided 2D SynthID (`--native-guide`), pyramid-scaled optical flow, and `--wan-steps`**:
  - Scaling dense Farneback flow to `<= 512` long-side accelerated 720p/1080p motion estimation
    by `6.25x-14x`; disabling `AutoencoderKL` tiling at `<= 768` kept `512p` MPS VAE throughput
    at `0.87 s/frame` (`11x` faster than `1280p` tiled VAE at `9.6 s/frame`).
  - `native_guide=True` (`--native-guide`) runs `512p` 2D VAE regeneration and upscales cleanly
    to native `WxH @ native fps` with clean-guide unsharp and CIE Lab color anchoring, measuring
    **`32.482 dB` PSNR** on `veo/provider-original.mp4` (`1280x720@24fps`).
  - Exposed `--wan-steps` / `wan_steps` (`26.872 dB` PSNR at 6 steps, `3.3x` fewer DiT steps than
    the certified 20-step default).
- **23-mark visible video suite (`+5` marks + international `Qwen`) & selective GOP stream-copy**:
  - Added `pixverse`, `luma`, `heygen` (corner badge only), `sync_labs`, and `tiktok_ai`, plus
    the international 3-lobe pinwheel + `Qwen` arm inside `qwen`, verified across 8 native corpus
    positives, 23/23 synthetic gallery examples, and 0 false positives across 70+ negative clips
    (including HeyGen diagonal tiled preview and PixVerse clean paid exports).
  - Added clean-segment donor fast-forward in `iter_clean_video_frames` and verified H.264
    keyframe GOP stream-copy (`_try_selective_gop_clean_encode`) for untouched GOPs (`>= 30` frames)
    and keyframe-aligned `--trim-bumper` cuts.
- **Audio/video provenance & detect/strip parity fixes (`signal-discovery` 2,426-file audit)**:
  - Registered `PixVerse` and `Suno Inc` in `C2PA_AI_VENDORS`, `com.suno.fingerprint` ->
    `Suno Audio Fingerprint` soft-binding label, `"suno"` in `_audio_metadata_vendor`,
    `audio_generator` fallback in `_platform_from_video_metadata`, `HZ_XYKJ` Honor TC260 video
    code, and standalone ISOBMFF/XMP `IPTC_AI_MARKERS` (`source_type` + Meta/Canva `app_aigc`)
    in `get_ai_metadata()` for 100% `has_ai_metadata` <-> `get_ai_metadata` <-> `strip_and_verify` parity.

In Wave 13 (2026-11-24, `v0.75.0`):
- **Unknown-video (`mark="auto"`) visible-mark search & timestamp probing speedup (`19.4x - 588.8x`)**:
  - Packet-level PTS probing (`ffprobe -show_packets -show_entries packet=pts_time,flags`, falling back
    to `-show_frames` only on edit-list discard flags, missing PTS, or duplicate PTS) cut 10-clip
    timestamp probing from `7.796 s` to **`0.402 s` (`19.41x` faster)** with `0/18` timestamp mismatches.
  - Added 2x coarse-to-fine spatial pyramid refinement (`_refine_pyramid_peaks`) in full-frame
    `detect_sora_frame` (keeping `_match_template` at full resolution so low-threshold corner ROIs still
    observe drifting background texture peaks outside `±6 px`), shared per-frame `MORPH_TOPHAT` ROI
    caching (`prepared.feature_cache`) across `_detect_fixed_mark` and `detect_veo_frame`, up-front
    `contradicts_video_provenance` filtering with zero-detector timeline probing (`probe_empty_video_scan`)
    when all candidate marks are contradicted, scoped single-thread OpenCV execution
    (`cv2.setNumThreads(1)`) inside `_scan_video_detectors`, and two-stage stride-4 coarse-to-fine
    temporal scanning (`_should_promote_coarse_mark` requiring spatially overlapping coarse support for
    `min_stable_frames >= 12` marks):
    - 4-clip unknown `identify_video` benchmark (`veo/example.mp4`, `jimeng/example.mp4`,
      `pixverse/example.mp4`, `xai-grok-imagine-video.mp4`): CPU time dropped from `134.94 s` to
      **`9.29 s` (`14.5x`)** and wall time from `318.35 s` to **`9.18 s` (`34.7x`)** (`veo`:
      `127.33 s -> 3.55 s`, clean `xai-grok`: `70.66 s -> 0.12 s`, **`588.8x`**).
    - 300-frame 720p `sora/provider-original.mp4` unknown scan dropped from `137.5 s` to **`4.73 s`**
      (`300/300` frames matched), all `23/23` `visible/*/example.mp4` unknown scans passed in `55.81 s`
      (`2.43 s/clip` average), and `remove_video_visible(veo/example.mp4)` preserved exact output
      SHA-256 (`ea2ea00b74227741689d31886aa07e638071a853ca34bc4aabca6038fbf53d39`).
- **Single-pass multi-decoder audio inspection, `--low-memory` / `release_models()`, & zero-copy video pipes**:
  - `_scan_audio_multi` (`_WindowSegmentAccumulator`) fuses `identify_audio(..., scheme="auto" | "all")`
    into **1** `ffmpeg` decode pass (down from `5` on 48 kHz stereo) with shared 16 kHz polyphase
    resampling across `AudioSeal`, `WavMark`, and `Perth`, inline native `AudioSeal` + `OpenVoice`
    prefix checks, single-window segment reuse, and 64-frame `WavMark` batching: wall time dropped from
    `90.59 s` to **`5.41 s` (`16.75x`, CPU `18.40 s -> 9.76 s`)** with bit-for-bit identical
    `AudioProvenanceReport` outputs.
  - Added `raiw.release_models()`, `release_audio_models()`, `release_video_models()`,
    `empty_accelerator_cache()`, `--low-memory` on `audio/video invisible|all|batch`
    (`78.13 MiB -> 0.00 MiB` cached detector storage), automatic video runtime eviction before
    `include_audio_invisible=True` in `remove_video_all`, `cuda` + `mps` cache clearing across
    `video_wan`, `video_seedance`, and `video_invisible`, single-pass source decode + `float32` metric
    diffs (`94.92 MiB -> 47.52 MiB` at `1080p`) in `video_seedance`, and zero-copy `memoryview` pipe
    writes (`write_raw_frame_bytes`: `6075.1 KiB -> 0.63 KiB` peak Python allocation per `1080p` frame).

In Wave 14 (2026-11-26, `v0.76.0`):
- **Live Google Gemini SynthID oracle certification of native-resolution + native-FPS guided 2D `synthid`**
  (`gemini.google.com/u/1/app/78a7b2439b97f3e1` & `/u/1/app/279dc48d99564bf2`):
  - `gemini_omni_flash_synthid_2d_native.mp4` (`sha256:701d93cc378ffba1ef4ede332289b5516b96420d1578b5cc4ccdbc80f0f6e898`,
    derived from silent video stream of `data/fixtures/provenance/google-gemini-omni-flash-api.mp4`,
    `1280x720 @ 24.0 fps`, `240` frames, `vae=stabilityai/sd-vae-ft-mse`, `noise_std=0.15`, `long_side=512`,
    `psnr_db=28.8728`, `temporal_residual_ratio=1.2936`): **`not_detected`**
    (`"This video does not show signs of being made with Google AI... No reliable signals were detected indicating how the content was created"`).
  - `veo_3_1_lite_synthid_2d_native.mp4` (`sha256:81c40d5f31ddba379e8bdcadcfc818e42ee2df42023e0e79e6f373a5139c15d8`,
    derived from silent video stream of `data/fixtures/provenance/higgsfield-veo-3-1-lite.mp4`,
    `1280x720 @ 24.0 fps`, `192` frames, `vae=stabilityai/sd-vae-ft-mse`, `noise_std=0.15`, `long_side=512`,
    `psnr_db=24.9739`, `temporal_residual_ratio=1.4836`): **`not_detected`**
    (`"The video does not show signs of being made with Google AI... No reliable signals were detected indicating how the content was created"`).
  - Recorded both rows in `data/evaluations/video-synthid-oracle.csv`, promoted native resolution + native FPS
    (`DEFAULT_VIDEO_SYNTHID_FPS = None`) to the standard `synthid` default, updated
    `test_shipped_defaults_match_a_certified_manifest_row` to bind `(vae, noise_std, long_side, fps)` to the manifest,
    and removed the `--native-guide` / `--keep-fps` workaround flags.
- **CLI surface cleanup & default unification across image, video, audio, and MCP**:
  - Retired 19 redundant, workaround, or oracle-uncertified CLI flags (`--native-guide`, `--keep-fps`, `--wan-steps`,
    `--fidelity-anchor`/`--no-fidelity-anchor`, `--text-manifest`, `--long-side` and `--no-temporal-stabilize` on
    `video invisible|all|batch`, `--allow-positive`, `--allow-quality-override`, `--audio-allow-positive`,
    `--audio-allow-quality-override`, `--bandwidth`, `--all-streams`, `--target-stream`, `--inpaint-method`,
    `--humanize`, `--unsharp`), collapsed `video audio` track selection onto repeatable `--stream`, and unified
    `--backend auto`, `--scheme auto` / `--audio-scheme auto`, and `batch --mode all` across the CLI.
- **Visible video detector hardening (`detect_veo_frame` & `hailuo`)**:
  - Added `_VEO_TRANSITION_CONFIDENCE = 0.35` gate in `detect_veo_frame` so sub-transition background noise returns
    `FrameLocalization(frame_index, 0.0, None)` instead of uncalibrated candidate boxes.
  - Tightened `VISIBLE_MARK_POLICIES["hailuo"].anchor_iou` from `0.80` to `0.99` to reject `6 px` horizontal
    scene-texture drift on wide `272x32` boxes (clearing the `192/192` post-C2PA-strip `hailuo` false positive on
    `veo/provider-original.mp4` and `240/240` on clean ceramic-jar video `41b3e3bf...` while preserving `90/90` on
    `hailuo/example.mp4` and `124/124` on `hailuo/provider-original.mp4` including `720p`/`480p` CRF 28 re-encodes).


