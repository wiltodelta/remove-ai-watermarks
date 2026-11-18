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
preserving `22.05 kHz` stereo bandwidth.

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
