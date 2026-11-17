# Audio and video watermark removal

This is the implementation map for extending removal across video and audio.
Coverage requires a compatible detector, saved positive and negative controls,
and a measured output. A registered name, a research-model result, and a
production-provider verdict establish different things.

## Coverage and next implementation work

| Surface | Existing evidence or implementation | Next requirement |
| --- | --- | --- |
| Audio and video metadata | The generic metadata reader and stripper support the documented containers | Preserve stream identity for metadata-only changes and keep declarations separate from AI-origin claims |
| Registered visible video labels | The shared video registry, temporal selection, and visible fills | Extend from publication-cleared positive exports, with matched negatives and delivered-video quality checks |
| Video SynthID | The shipped regeneration profile has a hash-bound oracle evaluation | Keep each new provider, profile, and carrier behind its own positive-control gate |
| Research VideoSeal and PixelSeal video | Saved-artifact studies, benchmark adapters, and explicit known-message library/CLI routes | Broaden carriers and seeds; below-threshold message recovery is not complete erasure |
| Research AudioSeal and Perth v1 | Pinned local oracles and saved spectral-profile results; AudioSeal-gated library and CLI | Extend held-out carriers and runtime decoder families; public Perth v1 now has a pinned runtime |
| Public WavMark | Pinned runtime, benchmark adapter, and saved positive/negative removal controls | Extend held-out carriers; deployed-provider compatibility remains separate |
| Public SilentCipher-44k | Pinned experimental runtime, source-positive held-out controls, saved-output checks, and final-mux evaluation | Broaden negative controls and carriers; published 16 kHz artifacts have incompatible SDK byte framing; deployed-provider compatibility remains unverified |
| Experimental SilentCipher-16k terminated candidate | Pinned receiver, library/CLI/MCP routing, saved WAV/FLAC checks, and selected-track final-mux evidence | Broaden publication-cleared carriers; SDK bytes, the general 32-bit profile, and deployed-provider compatibility remain unqualified |
| Proprietary audio and video schemes | Provider policies, registry declarations, and the existing provider-oracle routes | Obtain a native positive carrier and a compatible verifier before recording removal success |

Research video details are in [non-SynthID video coverage](non-synthid-video.md).
The [audio provenance experiment](audio-provenance-experiment.md) and
[benchmark kernel](benchmark-kernel.md) describe the existing audio oracles.
The exact C2PA algorithms, media types, and watermark/fingerprint distinction
come from the packaged registry, not this table. A registry entry does not
automatically qualify a scheme for AI-watermark removal.

Compatibility between the public WavMark research model and the registered
`com.microsoft.wavmark.1` algorithm has not been established by this project.
The public model and registry refer to the same
[WavMark paper](https://arxiv.org/abs/2308.12770); that does not pin deployed
weights or framing.
Perth v1 results likewise do not certify Perth v2. Compatibility must be established for the actual weights and framing.

### Native audio metadata and provider boundaries

Exact Suno and Suno Studio creation comments are supported as unsigned
`audio_generator` declarations when ffprobe is installed. The parser requires
a valid complete timestamp and, outside Studio, a canonical UUID. Audio-only
ISOBMFF comments are blanked without moving media offsets. Other supported
audio containers use stream-copy removal, which drops container tags and
chapters, including ordinary ones. Attached cover images are allowed;
a comment on an audio/video clip does not attribute its video pixels.

Suno C2PA credentials, an embedded fingerprint declaration, and a waveform
watermark are separate signals. [Suno Credentials](https://suno.com/suno-credentials)
checks C2PA provenance and documents a file-upload API. Its negative result
cannot establish removal of a waveform watermark. Local signature integrity
also does not establish signer trust when the certificate is outside the
configured trust store.

Perth v1 can respond positively to constructed musical signals with no
embedded watermark. A fixed synthetic musical-negative experiment reproduced
this cross-reaction with the pinned model and unchanged threshold. A positive
Perth response on native music therefore does not identify the provider's
scheme. Discovery windows are not whole-song negatives, and the runtime's
source-positive/saved-output-negative gate remains specific to that public
model. Do not select a production-provider remover from this response alone.

### Native-clock AudioSeal compatibility

The published [LTX marking adapter](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/marking.md)
stamps channels independently without resampling their PCM. Its
[marking pack](https://huggingface.co/Lightricks/LTX-2-Marking/tree/718f58d124e0acf58da85752692d93b0bd54c38a)
advertises the same AudioSeal generator and detector SHA-256 digests as this
project's pinned public weights. This does not establish compatibility with
every hosted LTX export.

Controlled stereo PCM replay at native 24 and 48 kHz reproduced a missed-mark
case: the native reference decoder was positive while the old resampled 16 kHz
runtime was negative. For these validated source rates, AudioSeal inspection
now retains both receiver clocks,
including separate per-channel, downmix, whole-clip, and window results. JSON
`decoder_sample_rate` identifies each receiver. Removal publication requires
both clocks to be negative; a native-receiver failure remains unverified. The
16 kHz source path is checked once. Short tails reuse preceding samples;
constructed unmarked noise reproduced false positives when a tiny remainder
was decoded alone, including on the old 16 kHz receiver.
Matching the public model does not attribute
a file to LTX, and a negative does not certify complete payload erasure.

Automatic native-clock inspection is limited to the validated 24 and 48 kHz
profiles. A constructed unmarked 96 kHz chirp triggered the unrestricted native
receiver, so other native clocks are outside this calibrated coverage and
retain the existing resampled receiver. Do not extrapolate this qualification
to an arbitrary source rate or production deployment.

A separate AAC-compressed controlled pair remained inconclusive after combined
processing: the 16 kHz downmix receiver responded positively on both the marked
carrier and its processed unmarked control, while the native-channel checks
were negative. This does not distinguish a residual mark from a decoder
cross-reaction. A direct upstream-model replay reproduced the clean-control
response; independent-channel and batched codec processing agreed within
floating-point error. This localizes the observed false response to the model
on that processed control, without resolving the marked candidate. Default
publication rejects that candidate. PCM/FLAC success
must not be extrapolated to native AAC exports from this comparison.

### OpenVoice default raw WavMark framing

The published [OpenVoice conversion helper](https://github.com/myshell-ai/OpenVoice/blob/main/openvoice/api.py)
uses a raw WavMark message instead of the synchronized public-model wrapper.
Its [requirements](https://github.com/myshell-ai/OpenVoice/blob/main/requirements.txt)
pin `wavmark==0.0.3`; the default `.model.pkl` checkpoint matches this project's
pinned checkpoint SHA-256. This establishes the inspected helper's model
compatibility, not every hosted export.

The helper pads the default text to `b"default "` and writes two raw 32-bit
blocks, each spanning 16000 native samples, at offsets 0 and 32000. It does not
resample or add WavMark's 16-bit sync prefix. Controlled replay at 16, 22.05, and
44.1 kHz recovered both reference blocks exactly, while the previous synchronized
receiver was negative. The existing `wavmark` route now retains both receivers.
The native receiver checks each channel and the downmix, closes its bounded
48000-frame reader before inference, and labels readings
`framing="openvoice-default"` with their native `decoder_sample_rate`.

The decision retains both individual-fragment and aggregate correlations. Its
binomial tail is corrected for the three comparisons when both blocks exist,
but is not an empirically calibrated false-positive probability. Custom
messages, shifted/resampled framing, and disabled marking remain outside this
receiver. Neither a signature match nor its disappearance certifies provider
identity or complete payload erasure. Publication still requires every
requested receiver on the saved output to be negative.

All three source-qualified controlled carriers passed the saved-FLAC gate
through the default EnCodec profile. Sample geometry was retained, fidelity
measurements were saved, and no clipping was observed; this is not a perceptual
quality assessment of speech or music. The actual CLI identified the native
signature. A two-audio-track H.264/FLAC MKV passed final-mux verification,
including packet payload and timing preservation for the video and untouched
audio track. Constructed unmarked controls stayed quiet before processing,
and the paired clean carriers also stayed negative through the same codec
profile. Their forced runs are diagnostic controls, not successful removals.
The new fixed-prefix receiver also produced no candidates on the retained
audio snapshot, with unchanged hashes and metadata. This historical
check covers the fixed default blocks, not a whole-song watermark absence
claim. Existing synchronized decoding and checkpoint-pinning functions were
unchanged. Exact readings, controls, source snapshots, and hashes stay in the
external evaluation store. A separate local synthetic-speech carrier also
passed the default saved-file gate, and its processed clean control stayed
negative. Intrusive speech metrics were retained for embedding alone and for
both processed variants. The codec affected those metrics much more than
embedding in this single utterance; these measurements do not replace listening
or establish speech/music quality across a population. The explicit spectral
profile was also checked on the same three carriers and synthetic speech. All
marked candidates stayed positive and were rejected, while their paired
processed clean controls stayed negative. The speech candidate retained the
complete decoded default reference. These results do not justify substituting
the spectral profile for the existing default codec route.

### Published Perth v1 runtime

The published `resemble-perth` 1.0.1 wheel was replayed against the retained
Perth study artifacts in an isolated modern environment. All 84 confidence
scores and verdicts matched the pinned development-source readings exactly.
Its bundled checkpoint has the same verified SHA-256 as the research adapter.
The wheel declares no dependencies; its immediate import obstacle is
`pkg_resources`, supplied by `setuptools==80.9.0` in this experiment. The runtime now pins
the installed source tree, config, and checkpoint before importing the model.
`audio-perth` installs this backend; `--scheme perth` selects it. Complete
one-second windows, including a reused-context short tail, participate in the
saved-output gate. Default auto processing tries spectral then combined.
All four eligible retained carriers passed the actual saved-WAV gate with the
spectral profile, and their processed clean controls stayed negative. The two
other synthetic marked carriers failed the source-positive gate and were
excluded. The actual CLI passed the same gate, and a two-audio-track MP4
passed final-mux verification after selected-track processing, including packet
hashes and timing checks on untouched streams. Saved fidelity and speech ESTOI
measurements are retained with the evaluation. These are controlled public-model
results, not certification of a production Perth deployment.
Exact wheel/source snapshots, package metadata, environment versions, replay
script, and per-artifact readings are retained under the local evaluation
`audio-backend-feasibility-2026-11-10`.

### Archived Chatterbox default PerTh compatibility

The inspected [Chatterbox archive](https://github.com/resemble-ai/chatterbox/tree/ab55923c27757c90cfcc407baa57dc25a7d25217)
uses `PerthImplicitWatermarker` in its original TTS, Turbo/Nano, multilingual,
and voice-conversion paths. They pass the generated mono waveform at 24 kHz.
Its declared dependency points to PerTh Git; the inspected
[PerTh snapshot](https://github.com/resemble-ai/Perth/tree/ff1c8ac55a976971245cdd53c18d6131ca00d993)
declares package version 1.1.0 and ships the same implicit checkpoint and model
configuration as the runtime's pinned 1.0.1 wheel. Package version 1.1.0 does
not establish compatibility with the proprietary PerTh v2 algorithm.

The reference marker converts native input to its trained 32 kHz clock. The
runtime's existing 16 kHz path adds an intermediate conversion. Controlled
speech, chirp, and noise comparisons at 16, 24, 32, and 48 kHz produced slightly
different confidence scores but matching verdicts. Unmarked chirps responded
positively with both receivers; those pairs do not qualify removal claims.

A separate isolated environment ran the complete, unchanged Nano renderer on
CPU with the pinned
[Nano snapshot](https://huggingface.co/ResembleAI/chatterbox-nano/tree/71ccd1d0081b430592cea481f4307e764e07bc64).
The default voice generated independent utterances. Instrumentation saved the
waveform immediately before the default marker, delegated to the original
marker, and retained its actual output. Clean controls were negative and marked
outputs positive with both the library and the original native-rate reader.

The existing `audio invisible --scheme perth` path passed the saved-FLAC gate
through the library and actual CLI using the spectral profile. Paired processed
clean controls stayed negative. A selected-track H.264/FLAC MKV passed the
final-mux check, including exact packet and timing preservation for the video
and untouched audio track. The original PerTh reader independently checked
source and delivered audio, both whole-clip and context-completed one-second
windows. These native-rate readings are development evidence; the packaged
receiver retains its existing 16 kHz preprocessing. Native source was positive
and delivered audio negative. Sample
geometry, clipping, fidelity, and intrusive speech metrics against the
pre-marking waveform were recorded outside the repository with source,
environment, weights, and artifact hashes.

Derived PCM16 WAV, MP3, AAC/M4A, and Opus/Ogg controls from both utterances
also passed the same source-positive and saved-output-negative gate. Their
matched clean controls stayed negative, including independent whole-clip and
window checks by the original native-rate PerTh reader. These are controlled
codec transformations of generated waveforms, not additional hosted-provider
exports. An H.264 MP4 with AAC audio passed selected-track processing into MKV,
with native source-positive/delivered-negative readings and untouched-stream
packet/timing verification. Decoded sample counts and rates were preserved,
including the AAC and Opus controls. Intrusive codec metrics use the decoded
clean control of the same codec as reference; they are not population quality
estimates.

These results cover the inspected default checkpoint and tested Nano outputs.
A PerTh response alone does not identify Chatterbox, and below-threshold
presence confidence does not prove complete physical erasure. Other voices,
languages, forks, and hosted PerTh deployments require their own qualification.

### Experimental SilentCipher-44k receiver

`audio-silentcipher` and `--scheme silentcipher` select the pinned public Sony
44.1 kHz model. The released 1.0.5 SDK declares an obsolete Torch upper bound;
the runtime carries the MIT-licensed decoder architecture and STFT directly,
checks the pinned config and weights, and loads weights with `weights_only=True`.
The architecture and STFT matched the released implementation exactly in local
CPU and MPS comparisons. This establishes math compatibility, not production
provider detection or watermark erasure.

The project receiver accepts any valid five-byte payload after at least three
complete repeated messages and symbol consistency of at least 0.9. That threshold
is an experimental project criterion, not the SDK default. Every phase offset
from 0 through 2047 in steps of 10 must be checked before a negative. A valid
message at another phase cannot be hidden by a stronger invalid pattern.
Whole-clip checks and three-second windows, including a context-completed short
tail, participate in publication. Short sources and invalid neural values stay
unverified. Inputs retain their native rate until model resampling to 44.1 kHz.
CPU inference can be slow; CUDA or MPS is selected when available, with CPU
fallback for device failures. Numerical failures remain errors.

The SDK's generic decode failure is not used as evidence of absence. Exploratory
codec outputs could produce low-consistency random payloads, so loss of the
original fixed message alone does not qualify as removal. Independent controls
and saved-output evaluation are retained separately from threshold selection.
Reports set `decoder_sample_rate=44100` independently of the source rate.
A 16 kHz source is resampled for this receiver and does not select the
published 16 kHz model. Its negative cannot rule out that model or another
framing. Native SynthID audio remains a separate unresolved family.

The published 16 kHz artifacts are incompatible with the released SDK's
terminated byte framing. The config requires 15 payload symbols followed by a
terminator, while byte conversion emits four quaternary symbols per byte.
No whole-byte payload can satisfy that length. Unmodified SDK encoding with
the loaded 16 kHz checkpoint fails the length assertion for the tested byte
lengths. Changing the period to 17 alone also fails: payload digit 3 is shifted
to class 4, outside the checkpoint's four output classes. Reserving class 0
for the terminator leaves only three payload classes. A custom ternary subset
would not establish the paper's 32-bit framing or compatibility with existing
marks and is not exposed as native support.

The examined SDK 1.0.3, 1.0.4, and 1.0.5 implementations share this framing.
The official release archive and pinned Hugging Face 16 kHz model contain
byte-identical weights and config. These are artifact-specific incompatibility
findings, not evidence that every possible 16 kHz framing is impossible or
that a file lacks a watermark. A separately qualified framing with independent
positive and negative controls is needed before adding a receiver. The
[authors' paired 16 kHz demonstrations](https://interspeech2024.github.io/silentcipher/)
provide a separate research corpus; they do not establish compatibility with
the released byte API or the [paper's 32-bit profile](https://arxiv.org/html/2406.03822v1).
Primary sources:
[released SDK and model archive](https://github.com/sony/silentcipher/releases),
[pinned model artifacts](https://huggingface.co/sony/silentcipher/tree/a1c4d021905e0dc5b24be5f68db5fc4dba410ee1/16_khz/97561_iteration),
[length and class-index reports](https://github.com/sony/silentcipher/issues/2),
and [byte-framing report](https://github.com/sony/silentcipher/issues/3).

The frozen EnCodec-24 profile was then evaluated independently on three
synthetic carriers and two held-out system voices. Every marked source
recovered the exact expected five-byte message; a separate Moira source
recovered a different message. All five unmarked sources and their processed
controls were negative. All six marked saved outputs were negative on the
whole clip and every three-second/context-completed window, each with the
complete 205-phase search. Native frame counts and rates were preserved and
no output samples exceeded full scale. Speech ESTOI ranged from 0.968 to 0.975;
paired waveform and 10 ms RMS-envelope correlations found zero lag in the
measured speech outputs. This is metric evidence, not a listening assessment.
Noise carriers changed substantially, including negative waveform SNR for
white noise; successful decoding disruption does not guarantee acceptable
fidelity. The small controlled set does not establish a real-world
false-positive rate or erasure of all payload information.

The actual installed CLI passed a separate cropped-carrier check. The
selected-track video route also passed source-positive and actual final-mux
negative checks, with unchanged video and second-audio packet payloads and
timings. Runtime network/STFT parity, exact-control post-audit, hashes,
quality, content-lag measurements, and the CLI/video reports are retained under
`audio-backend-feasibility-2026-11-10`. The public model checks do not certify
native SynthID audio removal.

### Experimental SilentCipher-16k framing qualification

`_internal/silentcipher16.py` implements an experimental terminated pattern:
seven ternary payload symbols and one zero terminator, repeated every eight
spectrogram frames. Its four-class decoder and STFT match the pinned public
16 kHz checkpoint. The receiver requires three complete repetitions and
agreement at least 0.9, with a 103-position phase grid from 0 through 1023 in
steps of 10 before a negative. Its payload symbols are not bytes, and this
does not qualify the SDK byte API or the paper's general 32-bit profile.
The released config remains unchanged; the receiver framing is a separate
experimental interpretation of the model output.

A frozen controlled study used known messages on three system voices,
including uniform minimum and maximum ternary payloads. The encoder study
replaced only symbol packing in the released SDK, preserving its neural
encoder, normalization, and STFT. All messages were recovered exactly on the
saved sources and their three-second windows. The existing EnCodec-24 profile
produced saved outputs negative over the whole clip and every window, with
every phase checked. Paired clean sources and processed controls were also
negative. Sample counts and rates were preserved, outputs had no clipping,
speech ESTOI ranged from 0.970 to 0.976, and measured waveform lag was zero.
These are metric results, not a listening assessment or a population false-positive rate.

The same encoder did not recover the expected messages on exploratory noise
and tone carriers. Repetition alone, without the terminator constraint,
responded strongly on clean tones. A 32-bit raw-symbol round trip on speech
therefore does not establish a safe general receiver. The authors' paired
demonstrations and independent speech controls support continued work on the
terminated candidate, without identifying a deployed production provider.

`--scheme silentcipher16` selects this candidate in the library, CLI, MCP audio
tools, and selected-audio-track video path. `all` includes it and requires its
checks for a complete negative. Existing `--scheme silentcipher` continues to
select only the 44.1 kHz receiver. Final delivery qualification is recorded
separately from the controlled encoder study.

Actual library and CLI delivery then passed on mono WAV/FLAC and on a
16 kHz-marked source resampled to 48 kHz stereo. The final saved files were
negative on every channel and downmix, whole clip, and three-second window,
with the complete phase grid. The matched clean processing control remained
negative and retained a null removal verdict; without force it exited without
writing. These checks do not turn an unmarked control into a removal success.

Selected-second-audio-track CLI delivery passed for FLAC input in MKV and AAC
input in MP4 delivered as MOV. Both cases passed source-positive and
final-mux-negative checks for this receiver. Video and the other audio track kept
their packet payloads and timing. Independent decoded-frame checks preserved
the selected track's start, end, and sample count, including AAC priming and
padding already present in the decoded source. That source count is not
necessarily the pre-encoding WAV count. Delivered speech ESTOI was about 0.97,
with zero measured waveform lag and no clipping. This remains metric evidence,
not a listening assessment or production-provider qualification.

A retained-history check preserved source hashes and metadata verdicts. Its
separate bounded native16 calibration inspected only the first three seconds
of each input, per channel and downmix, without a terminated-pattern response.
A silent prefix remained unverified. This is not full-duration invisible
replay or evidence that a proprietary producer's watermark is absent.

### Alternate codec feasibility

A separate frozen Descript Audio Codec experiment used the published 44.1 kHz
model with all nine quantizers, independent channels, eight-second blocks,
one-second neighboring context, and exact native-frame trimming. Among the
source-qualified AudioSeal and Perth v1 cases, all ten marked saved outputs
remained positive, while their processed clean controls remained negative.
This profile therefore supplies no removal evidence and is not a runtime
method. The converted Hugging Face model card leaves its license field
unfilled, but a later audit confirmed the original publisher's explicit
[MIT grant for model weights](https://github.com/descriptinc/descript-audio-codec#weights).
That card metadata gap does not establish an absent upstream weight grant.
Numerical conversion parity with the original PTH is a separate, unverified
measurement. Model pins, input hashes,
saved outputs, loading checks, and quality measurements are retained under
`audio-backend-feasibility-2026-11-10/dac-feasibility`.
See the [DAC implementation documentation](https://huggingface.co/docs/transformers/model_doc/dac)
and [original code repository](https://github.com/descriptinc/descript-audio-codec).
No native provider requests were made in this experiment.

A later, separately frozen full-track comparison used complete short local
Chatterbox Nano generations and three, six or nine DAC codebooks. These were
different inputs and a different boundary recipe from the older blocked
experiment, not repeated verdicts for unchanged files. The three-codebook
outputs became negative through the public Perth reader, including whole-clip
and complete window checks, with positive source/rate controls and negative
clean controls. Six and nine codebooks remained positive on the initial case.
The selected three-codebook arm passed the initial quality gates but failed
the frozen speaker-similarity guard on an independent short generation, while
retaining exact recognized words, native counts and measured timing. A short
embedding failure does not certify a changed voice identity, and the passing
case does not establish general voice preservation. No SynthID result or
public DAC runtime method follows from this pilot. A later complete long-speech
case remained negative with three codebooks but failed exact transcript
agreement on its marked output. The failure was retained without changing the
frozen threshold or processing the next held-out generation.

## Audio implementation sequence

The `audio identify`, `audio metadata`, `audio invisible`, `audio all`, and
`audio batch` commands implement the optional audio runtime. `audio` provides
streaming spectral processing, `audio-detect` adds the pinned open AudioSeal
verifier, and `audio-wavmark` adds the public WavMark verifier. Select
`--scheme auto`, `audioseal`, `wavmark`, `perth`, `silentcipher`,
`silentcipher16`, or `all`; `--scheme auto` scans installed production-ready
decoders (`audioseal`, `wavmark`, `perth`) and adapts candidate profiles to the
detected scheme(s), whereas a missing explicitly requested decoder prevents a
complete negative. `audio-codec` adds pinned EnCodec resynthesis (using stereo
Mid/Side coding on 2-channel audio to preserve inter-channel spatial image).
Default `--method auto` tries spectral then combined for AudioSeal and Perth v1,
neural then combined for WavMark and both experimental SilentCipher receivers,
and combined when multiple schemes require joint removal. Explicit `spectral`,
`encodec`, or `combined` fixes one profile. The default publishes `.wav`,
`.flac`, `.mp3`, `.m4a`, `.ogg`, or `.opus` only after a positive source and
fully negative saved output (re-decoded from the delivered file, including post-encode
lossy verification and Opus 48 kHz rate adaptation). Explicit force can write an
unverified or resistant candidate, with that outcome preserved in the report.
Native channels, rate (48 kHz for Opus), and sample-frame count (within codec
priming bounds on lossy containers) are checked. Fidelity is measured on the
delivered file. Whole-clip verification uses segmented `<= 30 s` windows on long
inputs; window-only discovery negatives cannot certify the whole clip.

Explicit `--method bigvgan` adds the pinned experimental full-track mel-vocoder
through `audio-vocoder`. It uses raw mid/side for stereo and the original rational
sample grid; mono/stereo, 30-second and sample-value bounds apply. CPU/MPS/CUDA
are explicitly selected through `--vocoder-device`, without fallback or chunking.
It is excluded from auto selection and uses the same local source/saved-output
decoder gates. See the [research roadmap](media-removal-roadmap.md) for separately
recorded provider observations; those are not a built-in SynthID verifier.

Explicit `--method mimi` uses the separate experimental pinned CPU FP32 codec
with its default 32 quantizers from `audio-codec`. Its full-track mono/stereo
route shares the native rational sample grid and bitexact WAV writer. The model
owns decoder-excess prefix selection; no external padding, normalization,
alignment or clipping is added. Auto selection excludes it, and native output
rate does not undo the internal 24 kHz bandwidth limit. Local independent Perth
examples became negative, but one short example failed the frozen relative
voice guard; provider and general fidelity qualification remain separate.

On a later complete long native generation, EnCodec 6 was negative over the
whole file but remained positive in individual windows. The detector therefore
reported a positive overall result. Mimi passed complete full-file and window
verification on the marked file, with exact recognized words, both-reference
speaker consistency and full-duration timing. Its processed clean twin failed
exact transcript agreement, so general paired qualification still failed.
The exact marked Mimi file also passed verified public WAV and selected-track
MOV delivery. A separate AAC MP4 experiment passed fresh quality and complete
Perth checks on the actual final decoded track, preserving native count/start,
priming/tail interpretation and untouched stream packets. These are scoped
local Perth results on one native Chatterbox Nano file. They establish neither
hosted Resemble compatibility nor Google SynthID removal; short speaker
embeddings and ASR agreement are measured guards, not identity or listening
certificates.

Explicit `--method focalcodec` uses the separate experimental pinned CPU FP32
continuous 6-layer WavLM feature-to-Vocos resynthesizer from `audio-codec`
(`lucadellalib/focalcodec_50hz@d6d6d9524e52155c85193c2c3b8da1cf8842f019`,
matching upstream `v0.0.1` `cad8bf81d70875f756770449c34909d8b0951642` under
Apache-2.0). It operates on an internal 16 kHz grid (`8 kHz` ceiling) with
symmetric `(40, 40)` WavLM boundary context, raw mid/side stereo, exact native
sample counts, and bitexact WAV serialization, while bypassing binary spherical
quantization. Synthetic impulse and bandlimited burst probes showed that Vocos
ISTFT synthesis introduces a small consistent group delay (around `+2.38 ms`)
while randomizing carrier waveform phase. Measured with a calibrated
phase-invariant log-mel onset-flux and modulation-envelope timing observer,
FocalCodec passed complete paired clean and marked checks on that first long
native Chatterbox Nano generation across WAV, selected-track PCM MOV, and lossy
AAC MP4 (negative full-file and quarter-second Perth windows, exact words,
speaker cosine `>= 0.98` against both source references, bounded loudness/RMS
change, zero clipping, full-duration timing within `10 ms`, and untouched
non-target container packets). On a second held-out native speech generation,
FocalCodec preserved an atypical phrase (`"a wind shelf"`) where Mimi
regularized `"wind"` to `"wooden"` and failed Perth after AAC MP4 encoding,
whereas FocalCodec's Vocos STFT synthesis cross-reacted with Perth in a pause
interval (`[4.0s, 5.0s]`), including on the unwatermarked clean twin. Both
explicit experimental codecs therefore remain excluded from `auto` and gated by
mandatory saved-output verification. Both `--method focalcodec` and
`--method dac` process tracks up to `30 s` in a single pass and stream tracks
longer than `30 s` across `20 s` windows with `2 s` raised-cosine crossfades
while preserving the exact native sample-frame count.

### Local exploratory study, 2026-11-09

The fixed 1,000-1,500 Hz zero-phase notch was compared with MP3 controls,
broader filtering, additive noise, and EnCodec resynthesis on synthetic carriers
and system-voice speech. Matched clean and processed-clean controls accompanied
the saved-file readings. AudioSeal calibrated on all six carriers; Perth v1
calibrated on four, while two failed the source-positive gate and were excluded.
The narrow notch made five of six eligible AudioSeal files and all four eligible
Perth files negative. The AudioSeal tone stack remained positive. All eligible
speech outputs were negative in whole-clip and per-second checks, with ESTOI
between 0.953 and 0.966 relative to the decoded marked speech. MP3 alone retained
both marks in every eligible case. Wider filtering was not consistently better.

These are exploratory results on three voices and synthetic signals, not held-out
music or production-provider certification. The actual streaming library was
replayed on every clean and marked carrier, and the real CLI both published a
verified speech output and rejected the resistant tone without replacing an
existing output. Model pins, saved artifacts, per-window evidence, and fidelity
live outside the source tree in the local evaluation store under
`audio-removal-2026-11-09-2j0c5egx`. A local Perth v1 experiment does not add a
production Perth or SynthID compatibility claim.

The separate public WavMark study retained five eligible carriers and excluded
white noise after its source-positive check failed. All five retained the mark
after the narrow notch, low-pass filtering, or MP3. EnCodec at each frozen
bandwidth made all five negative; processed clean controls stayed negative.
The runtime scanner replayed all 78 saved artifacts, including complete
one-second windows and final valid offsets, without changing their verdicts.
These results use the study's original EnCodec implementation. The optional
runtime uses Transformers and must pass its own saved-output checks.

The actual runtime subsequently passed source-positive and saved-output-negative
checks on two held-out system voices, Karen and Moira. Their spectral outputs
retained exact frame counts across the streaming boundary; speech ESTOI was
0.966 and 0.965 against the decoded marked source. The runtime's Transformers
codec also made the retained WavMark Samantha carrier negative, with ESTOI
0.973. Its clean control remained negative. A separate synthetic stereo music
carrier failed clean-negative calibration and is excluded from removal evidence.
That carrier still exercised native 44.1 kHz stereo geometry and neural chunk
boundaries. The saved seam metrics establish measured differences, not inaudible
or universally acceptable music quality.

For AudioSeal, a separate frozen comparison on a tone stack and one speech
carrier found that combined filtering and neural resynthesis passed whole-clip
and window checks where neural resynthesis alone could retain a positive
window. This supports a bounded fallback, not a universal success claim.

### Native SynthID audio check

A frozen four-file package contains an extracted Veo audio source, its PCM16
control, a clean system voice, and a combined-processing candidate. The personal
and Lorsey Gemini slots detected SynthID in both source controls and the candidate,
while the clean voice was negative. Thus this candidate did not remove that native
mark. Veedma's control request stalled; Veryfi's control response was ambiguous.
Their remaining candidate rows were left unsubmitted. The anonymous detector
requested sign-in; the existing authenticated Chrome route returned a service
error on the source. These are indeterminate results, not negatives, and remain
separate in the evaluation store. Local AudioSeal, WavMark, or Perth v1
negatives never substitute for a native SynthID verdict.

### Conditioned FocalCodec pause qualification and unified removal benchmark

Unconditioned continuous `FocalCodec` (`lucadellalib/focalcodec_50hz@d6d6d9524e52155c85193c2c3b8da1cf8842f019`)
passed paired WAV, PCM MOV, and AAC 192 kbps MP4 checks on `LONG03`, and
preserved an atypical lexical phrase (`"brass compass"`) on `LONG04` where Mimi
regularized a word (`"brass"` to `"glass"`), but Vocos ISTFT phase randomization
cross-reacted with magnitude-domain Perth v1 in quiet pause windows on `LONG04`
(including on its unwatermarked twin). Adding smooth source-activity pause
conditioning (`resynthesize_conditioned` in `_internal/audio_focalcodec.py`:
`20 ms` RMS envelope, `-50 dBFS` quiet floor to `-42 dBFS` active speech, `80 ms`
holdover, and `40 ms` raised-cosine transitions) eliminated pause false
positives while keeping active speech fully resynthesized. Across `LONG03`,
`LONG04`, and a fresh untouched held-out `LONG05` speech pair
(`carbon-long05-focalcodec-holdout-b092199d/final-summary.json`), conditioned
`FocalCodec` passed all **18/18** saved-file checks (`WAV`, `PCM MOV`, and
`AAC 192 kbps MP4` on both clean and Perth-marked pairs across all three
generations; `LONG05` marked Perth confidence dropped from `0.988-0.999` to
`0.056-0.130`, with exact `whisper-large-v3` transcript preservation).

To prevent regressions across schemes, durations, and container muxes,
[`scripts/media_removal_benchmark.py`](../scripts/media_removal_benchmark.py)
synthesizes deterministic clean and watermarked fixtures (or reads an external
JSONL manifest), executes the real `remove_audio_invisible`,
`remove_video_audio`, `remove_video_audio_all_streams`, `remove_video_message`,
and 4-layer `remove_video_all` pipelines, and enforces both saved-output
negative verification and model-free `quality_passed` gates
(`_internal/audio_quality.audio_quality_passed` and video PSNR/SSIM/TRR gates).
In the expanded `--builtin-suite --strict` run
(`carbon-media-removal-benchmark-v2-b092199d/report.json`, manifest SHA-256
`6ec1fe651e8e5ac0e6e202161ffc61b0b9ec43e3039832d4d75709554df2215c`), all
**18/18** cases passed (`AudioSeal` mono, `scheme="auto"` discovery, lossy `.m4a`
and `.mp3` delivery, stereo, 31-second segmented `whole_segments` verification,
3-stream `MP4 AAC 192 kbps` selected-track delivery, and multi-track
`video_audio_all_streams`; `WavMark` mono and stereo Mid/Side `encodec` at
`SNR 16.22 dB`; `Perth v1` via `auto` -> `combined` and explicit `focalcodec`;
36-frame `VideoSeal 1.0` via `auto` -> `vae` at `512x288`, `PSNR 32.46 dB`,
`SSIM 0.9701`; and unified 4-layer `video_all_4layer` cleaning TC260 `AIGC`
metadata, visible Kling corner mark, `VideoSeal 1.0` 256-bit pixel mark, and
`AudioSeal` AAC audio in one pipeline).

### Google Gemini SynthID audio and native-resolution video oracle tranches (2026-11-14 to 2026-11-16)

Following the local benchmark qualification, frozen candidate and control batches
were checked against Google Gemini's built-in `@synthid` verification tool
(`gemini-web`) across an initial cohort of three native Google audio tracks (Flow Veo
`veo-flow`, a retained Veo Lite generation `veo-lite`, and a second retained
native Google audio track `spaces-01`, where plain `BigVGAN M/S` had previously
remained `detected` on both `veo-lite` and `spaces-01`) and two Veo 3.1 video
carriers (`flow-001` and textured `lite-003`), followed by a held-out Stage 5
multi-window and full-audiovisual evaluation on `google-gemini-omni-flash-api`
(`1280x720 @ 24 fps`, `240` frames / `10.005 s`) and native `veo-3-1`
(`1280x720 @ 24 fps`, `96` frames / `4.010 s`):

- **Audio `SynthID`**: Matched 16 kHz resampling controls remained `detected` and
  clean speech controls remained `not_detected`. Conditioned `FocalCodec`
  (`--method focalcodec`) returned **5/5** explicit `not_detected` verdicts on
  the initial cohort (`veo-flow` bitexact WAV, 192 kbps AAC-decoded WAV, and
  muxed MP4, `veo-lite`, and `spaces-01`), but **remained `detected`** (`0:00` to
  `0:10`) on the held-out 10-second `google-gemini-omni-flash-api` track. By
  contrast, full-bandwidth 44.1 kHz 3-codebook Descript Audio Codec (`--method dac`,
  `descript/dac_44khz@c1bc521685adf9cfe247bc39a5ca58917eda1ac4`, raw M/S stereo
  with stem RMS preservation) returned explicit `not_detected` verdicts across
  all **4/4** source-positive native audio tracks (`veo-flow` bitexact WAV,
  192 kbps AAC-decoded WAV, and muxed MP4; `veo-lite`; `spaces-01` after quota
  reset; and held-out `google-gemini-omni-flash-api`, whose matched source WAV
  control returned `detected`) while preserving full `22.05 kHz` stereo bandwidth.
  On held-out `veo-3-1` isolated audio WAVs, Gemini returned a generic Google-AI
  origin sentence on the source WAV and stalled in `Connecting toVerify AI` on
  the `DAC` WAV; both were recorded verbatim as `indeterminate` without retry.
- **Video and full audiovisual `SynthID`**: Matched silent H.264 controls on
  `flow-001`, `lite-003`, 240-frame `google-gemini-omni-flash-api`, and 96-frame
  `veo-3-1` all remained `detected`. Upscaling 2D `sd-vae-ft-mse` (`noise_std=0.15`)
  to `768x432 @ 24 fps` cleared `flow-001` (`not_detected`) but failed on textured
  `lite-003` (`detected` with temporal stabilization; unstabilized row `refused`
  at the video quota limit), leaving the shipped 2D VAE default pinned at
  `512 @ 16 fps`. On an NVIDIA H100 CUDA GPU, 3D causal diffusion via
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (`--profile wan21-synthid-experimental`,
  `strength=0.38`, using `chunk_frames=193` and `overlap_frames=17` raised-cosine
  blending for clips exceeding 193 frames) returned explicit `not_detected`
  verdicts across all **6/6** silent native `1280x720 @ 24 fps` video deliveries
  (`35.14 dB PSNR`, `0.9844 SSIM` on `flow-001`; `28.31 dB PSNR`, `0.9032 SSIM`
  on `lite-003`; `27.27 dB PSNR`, `TRR 0.9301` across two 3D causal windows on
  240-frame `google-gemini-omni-flash-api`; and `35.00 dB PSNR`, `TRR 0.9035` on
  96-frame `veo-3-1`) and **2/2** combined full-audiovisual MP4 deliveries
  (`Wan 2.1` 720p video + `DAC` 44.1 kHz 192 kbps AAC audio on both
  `google-gemini-omni-flash-api` and `veo-3-1`). These remain exact-file oracle
  observations rather than a universal per-file guarantee.

### Protocol for extending coverage

The existing public routes provide availability checks, saved-output gates,
selected-track video handling, and real CLI coverage. Apply this protocol to
additional signal families and carriers.

1. Freeze clean and marked carriers and record their saved-file detector
   readings. Reject a removal claim when the original positive or its matched
   negative fails calibration.
2. Compare codec controls, spectral processing, controlled noise, and neural
   resynthesis. Process clean negatives too. Record detector scores, recovered
   messages where available, file hashes, exact sample counts, clipping, and
   quality against the decoded source.
3. Validate selected profiles on held-out speech and music. Keep exploratory
   parameter selection separate from certification. Evaluate channel behavior,
   silence, short inputs, and long-file chunk boundaries.
4. Integrate each additional optional backend with explicit availability checks. Reports
   distinguish detected, not detected by a named compatible decoder, and
   unverified. Missing models or a quiet metadata report are not clean verdicts.
5. Exercise each additional audio scheme through the explicit selected-track
   video path. Preserve every non-target stream and verify start times, sample
   counts, duration, and synchronization.
6. Exercise the real library and CLI on saved outputs, including atomic-write
   failures and minimal installations. Update the public API, command guide,
   installation extras, and published agent skill with those contracts.

## Boundaries

The project targets provenance on content users generated or edited themselves.
Automatic removal of stock, paid-asset, copyright-protection, and regulator
disclosure marks remains outside [the project boundary](legal-and-safety.md).
Filters and resynthesis can alter speech timbre, music, or fine visual detail;
quality and synchronization must accompany a detector result.

By default, visual video paths copy audio unchanged unless `--audio-invisible`
(`include_audio_invisible=True` on `remove_video_all` / `remove_video_batch`) is
enabled. `video audio` processes one selected track by default, or every
watermarked audio stream when `--all-streams` (`remove_video_audio_all_streams`)
is passed; it saves float PCM in MKV/MOV or lossy AAC in MP4 (target 192 kbps),
and copies all non-target streams unchanged. Publication checks untouched
streams' encoded packet hashes, timestamps, and audio priming/tail trimming,
each cleaned audio track's continuous timeline, sample-frame count and offset,
and fresh decoder readings after the final mux. Single-track `video audio`
without `--all-streams` does not certify or remove marks on uninspected sibling
audio tracks. Unsupported muxes and discontinuous source audio are rejected
without padding or truncating samples. Fidelity and detector readings cover the
actual decoded final output. Uncompressed audio can substantially increase
output size.
