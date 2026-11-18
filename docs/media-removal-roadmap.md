# Real video and audio removal roadmap

Research snapshot: 2026-11-11. This is an execution plan, not a promise that
every provider's watermark is removable. It covers content the user owns or
has permission to process. Text watermarks and third-party stock protection
are outside this work.

The evidence appendices are [methods and competitors](media-removal-research.md),
[provider verifiers](media-provider-verifiers.md), and
[real product samples](media-sample-corpus.md). Existing implementations and
qualification records remain in [audio/video coverage](audio-video-removal.md),
[non-SynthID video](non-synthid-video.md), and the
[Seedance study](seedance-video-study.md).

## Outcome and acceptance contract

The active goal is a working, reproducible removal pipeline for real video
and audio signals, with compatible verification and measured fidelity.
Research reports, completed provider batches and rejected hypotheses are
milestones within that goal. They do not satisfy its completion condition.
Continue through alternative mechanisms when an experiment remains positive;
record authorization or resource boundaries without declaring removal achieved.

Deliver a reproducible path from a native product export to an output with a
named, independently checked provenance outcome and measured fidelity. A
metadata-negative result, a generic AI-classifier score, and a watermark-negative
result are different measurements. An unavailable detector remains unverified.

Keep four outcomes separate: metadata, visible labels, the video signal, and
each audio track's signal. A provider's combined audiovisual verdict does not
identify which stream caused it. Copied audio is not processed audio.

Promotion requires all of the following:

1. Bind the original and delivered bytes to hashes, actual model/export path,
   source lineage, permissions, implementation revision, and parameters.
2. Establish a positive source and a positive matched processing/codec control
   on the same compatible verifier. Known-clean negative controls must remain
   negative. A negative matched positive control disqualifies that comparison.
3. Verify the saved output, then the selected track extracted from the final
   delivered container. Record incomplete coverage and other surviving tracks.
4. Preserve native geometry and clocks; validate duration, sample count,
   channel layout, priming/delay, sync, and untouched streams separately.
5. Measure fidelity against both the decoded source and a matched codec/front-end
   control. Inspect words, speaker identity, stereo, faces, text, object shape,
   occlusion, color, texture, and motion where applicable.
6. Demonstrate the result on independent held-out generations. Frames and
   alternative exports from one generation are not independent observations.

The agreed positive-control/negative-candidate criterion can qualify a saved
provider-specific example. It does not establish universal erasure, every-frame
erasure, human authorship, or a distribution-platform reach benefit.

## Priority order

Priority reflects current evidence, information gained, reproducibility,
fidelity risk, and implementation cost. It is engineering judgment, not a
measured probability of success. An artifact, license, or access blocker moves
an experiment behind the work that resolves it.

| Priority | Hypothesis or prerequisite | Why now | First bounded check | Stop or revise when |
| --- | --- | --- | --- | --- |
| P0 | Native sample lineage and compatible verifiers prevent false successes | Existing retained files include edited exports and unsigned provider hints | Build the corpus manifest and qualify positive/clean/codec controls | Origin, permission, signal compatibility, or full-file coverage is unknown |
| P0 | Exact length is insufficient without delay and final-mux checks | BigVGAN interface preflight exposed hop-boundary loss; copied audio leaves its signal untouched | Measure full-track delay, channel coherence, words, and extracted final-track identity | Length is restored by padding while content is shifted, truncated, or selected incorrectly |
| P1 audio control | A mel-vocoder baseline isolates the contribution of later generative processing | Full local speech pilot retained Perth in both marked outputs; plain BigVGAN is not qualified removal | Reuse the frozen full-file baseline when evaluating enhancement/diffusion; expand to music only with qualified native positives | Do not promote a vocoder-only profile without a new efficacy/fidelity advantage |
| P1 video control | Continuous temporal reconstruction supplies a structure baseline for stronger candidates | Full posterior-mode CogVideoX round trip preserved clocks but remained positive in CapCut, including without audio | Reuse the qualified delivery/measurement path as the continuous baseline for a distinct mechanism | Do not promote the tested zero-noise arm as removal; require new evidence for other strengths or models |
| P1 diagnostic | Paired structural probes can explain failures before adding more models | Training-free audio-fragility work has public probes; controlled local pairs exist | Probe aligned clean/marked pairs with public encoders, using a held-out split | Pair alignment is wrong, the method needs unavailable clean originals, or findings fail on held-out carriers |
| P1 video | Discrete temporal quantization removes residuals that a continuous VAE retains | The fixed full VidTok FSQ arm and matched control remained CapCut-positive | Require a distinct, justified mechanism before another frozen comparison; reuse the failed arm as a structural baseline | Improvement disappears after matching codec distortion or retaining geometry |
| P2 audio | Generative speech enhancement improves difficult speech cases | GenSE/UniSE checkpoints exist; published speech results differ from ordinary denoisers | Speech-only comparison with vocoder/front-end controls and transcript checks | Words, voice, timing, or quality deteriorate; do not extend speech claims to music |
| P2 audio | Diffusion or learned artifact suppression handles codec-resistant cases | DiffErase, SweepMark, and HarmonicAttack offer different mechanisms | First audit full-file handling, license, weights, and training assumptions | Only a cropped/mono excerpt is processed, artifacts cannot be reproduced, or use is not licensed |
| P2 video | Joint video reconstruction with spatial controls improves the efficacy/fidelity frontier | The fixed conditioned VACE prefix failed its structural-quality gate; full Wan VAE mode retained the signal | Require a new conditioning design that passes the frozen prefix gate before full-clip or provider work | Native shape/trajectory fidelity is lost or resource cost yields no advantage |
| P2 robustness | A method must also face codec-trained latent marks | NeuMark, DeltaMark, and CRAW challenge the assumption that codec cycling always works | Add reproducible resistant public schemes as adversarial reference cohorts | A claimed general backend works only on older waveform marks |
| P3 diagnostic | Temporal replacement parity and aggregation explain the isolated RIFE result | Current positive/negative outcomes vary by scene and branch | Offline public-decoder aggregation/parity controls before a new finite provider pass | Results depend on retained source frames or an unqualified control; do not promote a universal profile |
| P3 research | Temporally coupled spectral optimization may reduce regeneration damage | UnMarker motivates a different mechanism | Independent, licensed local research with temporal and perceptual constraints | Flicker, geometry damage, license restrictions, or decoder-specific overfitting dominate |
| P3 research | Viewpoint/voice replacement can suppress a signal | RAVEN and VC/TTS studies motivate it | Treat as a distinct destructive comparator | The user's required position, identity, voice, words, or timing changes |

BasicVSR restoration, independent SDXL+Canny, and flow-transported image noise
already have local evidence. Repeating unchanged arms is lower value than
testing a new mechanism. BasicVSR is recurrent; the flow-noise experiment was a
short-window ablation, not a completed trained video model. Do not infer hidden
detector logits, fingerprinting, or signal placement from these outcomes.

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

The retained native-provider tranche now has three exact-file SynthID-positive
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
cohort. Vocoder correlation lags had weak peaks and cannot establish temporal
drift; waveform error alone also cannot establish perceived quality. Listening,
semantic fidelity and final-mux checks remain open.

A fixed first-provider probe uses the strongest Veo lineage to separate
resampling, low/high EnCodec bandwidth and BigVGAN effects, alongside exact
source-positive and historical surface-negative controls. That negative voice
has a different carrier format and is not a fully matched clean twin. The other
generations and all locally reconstructed files remain retained for independent
qualification. New outputs require their own compatible provider verification
before any removal claim; preparation and local metrics do not supply one.

The first frozen native Veo audio pass subsequently produced an explicit
Google SynthID-negative for the saved BigVGAN L/R output, with an explicit
source-positive in the same Gemini account/pass. The EnCodec6 output remained
explicitly SynthID-positive there. This establishes one scoped source/output
observation, not a qualified general pipeline. Other accounts returned stopped,
generic-origin, contradictory or no-tool responses; keep those outcomes and
unsubmitted rows rather than converting them to clean results. The specialist
SynthID portal's additional Terms were inspected but not accepted for this
development pass, and no file was sent there.

Independent-channel vocoding changed stereo coherence. A separate fixed M/S
quality arm substantially improved measured channel correlation and side/mid
power on Veo and the two retained speech recordings. Its new file hashes have
no watermark verdict yet. Local English ASR recognized the same text on both
speech sources and their earlier L/R/EnCodec reconstructions; that is automated
agreement, not human transcripts, proof of speaker identity, or a listening
assessment. Empty Veo recognition does not establish absence of speech.

Selected-track delivery was also checked separately: float PCM in MOV matched
the verified L/R waveform exactly, and AAC in MP4 preserved its native count
and start without measured global lag. Both retained the original encoded
video packets and clocks. Newly serialized extracted WAVs have different file
hashes and still require their own compatible verdicts; none of this establishes
removal of the untouched video's provenance.

For stronger native-export replication, the publication-cleared Omni API and
Gemini app originals in `data/fixtures/provenance/` are preferable to relying
only on the previously edited Spaces containers. Their capture records identify
the API model and app family separately; do not invent an exact app model
version. First establish a compatible positive on each extracted audio track.
Their audio clocks also extend beyond the video clocks, which subsequent mux
checks must account for. The retained capture descriptions do not establish
whether the generation requests supplied input audio, so independent audio
generation lineage remains a separate qualification requirement.

The rational resampling arm retains the exact native half-open sample grid:
ceil the forward model count, remove only explicit model-context padding, then
select inverse samples whose timestamps precede the original endpoint. This
avoids the observed one-sample loss on fractional-duration Omni exports.
Both float MOV and AAC MP4 delivery checks preserved those native counts and
clocks, including the audio endpoint beyond the video endpoint. Float delivery
preserved selected-track PCM exactly; AAC changed it and needs its own verdict.
Source-relative ASR agreement and WavLM speaker-embedding comparisons on the
speech cohort are diagnostics, not human transcription or speaker identity
certification.

The internal pinned BigVGAN implementation now reproduces the official CPU
and MPS paths exactly on the measured inputs, including the retained full-track
rational M/S candidate. It keeps optional imports lazy and verifies model/config
hashes before strict weights-only loading. Numerical parity and sample-clock
tests alone do not qualify watermark removal for this new M/S profile. Once
Chrome transport recovered, the frozen native Veo M/S file received explicit
SynthID-negative replies in separate configured accounts, with exact source
and resampling-only files positive in those same passes. A different native
Veo Lite generation retained SynthID after the same fixed M/S processing;
its source and rate-only control were also positive. Another generation's
candidate reply named a Google-AI watermark without explicitly naming SynthID
and remains conservatively indeterminate. Thus the fixed profile has a scoped
success and a real counterexample, not a general removal qualification.
The subsequent independent speech pass also retained SynthID in the fixed M/S
candidate, with its source and resampling-only control positive. A later source
request received an explicit audio-modality verification quota refusal. All
Google uploads stopped at that point, without switching accounts or repeating
previously checked baselines. The raw public-API WAV and delivered AAC remain
unsubmitted. Preserve the quota response separately from model efficacy and
resume external qualification only when service availability is established.
Generic answers remain indeterminate. The initial public API reproduced the
candidate's PCM exactly but used a different WAV serialization; that historical
file has no transferred verdict. The new profile's deterministic bitexact WAV
writer subsequently reproduced the complete frozen provider-checked file
byte-for-byte in a direct API run. Its exact hash and compatible Gemini surface
allow reuse of those retained verdicts without another upload. This is one
fixed source/device/output example, not a guarantee for new files or encodings.
The explicit experimental profile is exposed in
audio and selected-video-track processing with the existing local decoder
gates; it is excluded from automatic selection and cannot verify SynthID itself.

Start with retained source-positive local-generator speech and its actual clean
twin, plus a separate native-provider cohort. Keep public encoder compatibility
and hosted production compatibility in different rows. Add music and ambience
before claiming those modalities.

The licensed AudioSR Basic follow-up exercised genuine conditional diffusion
with strict full-state loading and paired local Perth controls. Its official
low-frequency-restored output retained Perth. The variant before both source
low-frequency reinsertions fell below the receiver threshold, while the actual
VAE/vocoder bypass remained positive. All clean controls were negative. That
causal observation did not qualify faithful removal: automated words and speaker
embedding consistency degraded, and amplitude changes alone did not explain it.
The fixed source-latent partial-noise comparison then used matching forward and
reverse scheduler endpoints, a posterior-mode zero-noise control, and recorded
affine scale inversion only. No tested operating point passed both removal and
the preregistered word/voice guards. The independent generation stayed held out.
These are local public-model findings, not Google or hosted-provider verdicts.

The discriminative MetricGAN+ default mask retained Perth on the same local
cohort. Its source-phase synthesis was validated, without a removal advantage.
The actual library EnCodec 6 route and separately pinned Mimi default codec
both removed local Perth from independent speech generations, with matching
recognized text, native counts and measured global lag. One generation passed
the frozen relative speaker-consistency guard and final selected-video-track
checks; another failed that guard. Preserve that distinction rather than
equating a decoder negative with general fidelity qualification. The Mimi
backend is therefore explicit and experimental, never an automatic fallback.
The separately pinned UNIVERSE++ diffusion enhancer was then tested on the
initial matched speech pair, using audited tensor-only checkpoint conversion,
strict inference-state loading, source-ordered EMA and the unchanged trained
mel frontend. Default synthesis, auxiliary conditioner synthesis and a frozen
step-count family all retained local Perth with source-positive and
rate-control-positive evidence. Several variants passed all frozen initial
quality checks; one marginal short-clip speaker score failed in others. No
variant jointly passed efficacy and fidelity, so none advanced to the
independent generation. Do not infer removal from the enhancement quality or
the reduction of a still-positive detector score.
A later complete native long-speech comparison exposed the importance of window
coverage. EnCodec 6 was negative over the whole file but positive in individual
windows, so its removal result failed. Mimi was negative over the full file
and every tested window, and the marked file passed the unchanged words,
speaker-consistency, gain and full-duration timing guards. Its processed clean
twin failed exact transcript agreement, however, so broad paired qualification
still failed. A fixed DAC three-codebook arm also remained negative locally
but failed exact transcript agreement on the marked file. No failed threshold
was relaxed, and the next held-out generation remained unprocessed.

The exact qualifying marked Mimi file was reproduced through the public
source-positive-gated WAV and selected-track MOV routes. The actual MOV track
matched its frozen PCM exactly. A separate fixed AAC MP4 delivery then passed
fresh transcript, speaker-consistency and complete timing checks on the actual
decoded final track, with complete negative Perth scans. Native sample count,
start time, encoder priming and terminal discard were checked; video and the
other audio track retained their packet payloads and rational timing. This is
one-file local Perth evidence on native Chatterbox Nano speech, not hosted
Resemble compatibility, Google SynthID removal or general Mimi qualification.
The public selected-track operation also accepts explicit MP4 output with
selected-track AAC. A mux-only public-operation check using the frozen processed
waveform reproduced the separately measured MP4 exactly; it adds delivery
integration evidence, not an independent model result.

A subsequent fixed receiver-offset audit used the same saved LONG03 files and
an independently encoded processed-clean AAC control. Whole-file and every
one-second window on a quarter-second grid, including the exact terminal
window, remained negative for the marked Mimi WAV and final AAC. Native marked
speech remained positive, and all clean controls were negative on that grid.
This extends finite window-origin coverage; it does not cover every possible
offset, change the public receiver policy, or overturn the failed paired ASR
gate. The actual final AAC remains its own measured waveform.

The Mimi native stereo comparison favors raw mid/side over independent channels
for channel coherence, without demonstrating a voice advantage. Short speaker
windows without a qualified context length remain exploratory. New native
provider candidates and SRC24 controls are frozen for later verification;
no Google requests follow the explicit quota refusal.

The public Mimi route was also exercised on a complete native stereo video.
The selected float-PCM MOV track reproduced the previously frozen M/S waveform
bit for bit, retained its source sample count and start, and preserved the
untouched video packets. A separate AAC MP4 delivery retained the same decoded
count and video packets after honoring encoder priming and terminal discard
metadata. This checks the actual container boundary, not removal or a provider
verdict; the compressed waveform has different bytes and needs its own check.

Matching FocalCodec `50 Hz` safetensors against its exact historical `v0.0.1`
source schema (`cad8bf81d70875f756770449c34909d8b0951642`) resolved every tensor
name strictly across all 355 state keys without legacy remapping or unused
conversion dependencies. Synthetic clock probes showed that WavLM (`RF=400`,
`hop=320`) with symmetric `(40, 40)` boundary context and Vocos ISTFT synthesis
produces a consistent `+2.38 ms` group delay while randomizing carrier phase,
whereas Mimi's mid-utterance `"A"` to `"The"` substitution on the clean `LONG03`
twin arose inside its discrete RVQ/Transformer bottleneck rather than boundary
padding. Under a calibrated 20-case phase-invariant log-mel onset-flux and
envelope-velocity timing observer, continuous 6-layer WavLM feature-to-Vocos
resynthesis (`--method focalcodec`, `16 kHz` internal grid, `8 kHz` ceiling)
passed every paired clean and marked `LONG03` gate across standalone WAV,
selected-track PCM MOV, and `192 kbps` AAC MP4 (all full-file and quarter-second
Perth windows negative, exact 43/43 words on both twins, speaker cosine
`0.9842-0.9880 >= 0.98`, RMS `-0.39/-0.40 dB`, LUFS `-0.4/-0.5`, zero clipping,
mean lag `2.31-2.42 ms` within `10 ms`, and untouched video/audio packet hashes).
On held-out native speech `LONG04`, FocalCodec preserved exact 41/41 words
(retaining `"a wind shelf"` where Mimi regularized `"wind"` to `"wooden"` across
WAV and MP4 and crossed Perth after AAC at `0.6262`), speaker cosine
`0.9921-0.9945`, and active-window timing (`1.81-1.91 ms`), but Vocos STFT
synthesis cross-reacted with Perth's magnitude detector during pause window
`[4.0s, 5.0s]` on both twins (`0.4302` clean WAV, `0.5074` clean AAC false
positive, `0.5345` marked WAV).

Source-conditioned pause gating (`resynthesize_conditioned` in
`audio_focalcodec.py`, measuring 16 kHz 20 ms cosine-tapered RMS activity
$45\text{ dB}$ below the utterance's 95th-percentile active speech floor with a
one-frame dilation margin and matched Vocos `+2.38 ms` pause delay) resolved
that Vocos silence cross-reaction across both `LONG03` and `LONG04` and passed a
fresh independent held-out native speech generation (`LONG05`, 42 words) across
standalone WAV, selected-track PCM MOV, and `192 kbps` AAC MP4 (every full-file
and quarter-second Perth window negative on clean and marked twins, exact word
preservation, speaker cosine above `0.98`, zero clipping, and untouched
video/audio packet hashes). Conditioned `FocalCodec` candidates (`--method focalcodec`) and 3-codebook
`44.1 kHz` `DAC` Mid/Side candidates (`--method dac`) were subsequently checked
on `gemini-web`: matched 16 kHz resampling controls stayed `detected` and clean
controls stayed `not_detected`, while conditioned `FocalCodec` returned `5/5`
`not_detected` across Flow Veo (WAV, AAC-decoded WAV, and MP4) plus both hard
native counterexamples where plain `BigVGAN M/S` had remained `detected`, and
`44.1 kHz` `DAC` (`n_quantizers=3`) returned `2/2` `not_detected` on Flow Veo
and the Veo Lite counterexample while preserving full `22.05 kHz` bandwidth
(with one remaining row halted on an audio modality quota refusal). Both
profiles remain explicit (`--method focalcodec`, `--method dac`) rather than
`auto` because no local SynthID audio decoder exists.

The first fixed arms are identity/decode-save, resample-only, matched lossy
codec, existing EnCodec, and BigVGAN. Keep the exact model rate and the source
rate explicit. Handle hop-boundary padding explicitly, preserve all channels,
and measure delay rather than hiding it with arbitrary tail padding or
unbounded alignment. A full-file scan includes quiet intervals and the tail.

Measure clipping, loudness, source-relative error, intelligibility/transcript
errors, speaker identity, perceptual quality, and stereo coherence. Speech
metrics alone do not qualify music. Blind listening complements metrics.
Evaluate the final selected mux track and prove replacement of the intended
track while retaining non-target streams. An alternative retained original
track is reported as surviving provenance, not ignored.

### Video tranche

The initial full native-geometry FP32/MPS temporal-VAE attempt completed its
encode stage but was stopped during decoding after its measured physical
footprint consumed most of the host's memory. It produced no delivered video,
so it supplies no quality or watermark-removal result. This is a resource
qualification failure, not a rejection of temporal reconstruction in general.
Subsequent runs first bound a full-geometry prefix, record progress and memory
at synchronized temporal batches, preserve causal caches, and save the latent
before decoding. Precision and spatial tiling changes define new candidate arms.

The bounded FP16 follow-up completed the entire forest clip with native
geometry, all 121 frame timestamps and video duration unchanged. Its final
audio packet payloads, clocks, codec configuration and decoded waveform matched
the original exactly. All frames and transitions received structural/fidelity
diagnostics. Source-relative gray SSIM was about 0.938, versus 0.974 for the
matched delivery-codec control. Selected native-detail plates show remaining
texture/color differences. This establishes a full-delivery candidate, not
watermark removal or universal preservation of object shape/trajectories;
broader scene qualification remains open.

The authorized frozen CapCut pass on 2026-11-12 settled all six files once.
The metadata-only source, AV codec control, temporal-VAE AV candidate, silent
codec control and silent temporal-VAE candidate were all detected. The
source audio extracted without re-encoding into M4A was not detected. The fully
captured silent-video
control/candidate comparison therefore rejects this particular zero-noise
temporal bottleneck as a qualified remover on the forest export. Copied audio
cannot solely explain the candidate's positive result, since its silent version
also remained positive. The audio negative is not audio-removal evidence and
does not establish absence of another scheme or a false negative.

Request/file-hash/response bindings were complete for the three later video
rows and the audio row. The first source and AV-codec rows retain typed
responses and UI evidence but have incomplete request capture, so they are
excluded from complete receipt certification. No missing receipt was invented
and no file was reuploaded. Empty signal-detail arrays do not identify the
underlying watermark algorithm. The released Seedance reconstruction remains
a separate, previously qualified baseline.

Use independent owned generations with both smooth and textured content,
occlusion, motion, scene cuts, faces, and text/UI. Historical baseline clips
remain regression examples, not the complete test set.

The discrete comparison uses the official VidTok v1.1 causal FSQ checkpoint.
Its direct encoder/decoder interfaces include replicated temporal prefix
padding; the engine's own reconstruction method removes that prefix. Local
synthetic localization probes, both untiled and using the official temporal
chunk/cache path, confirmed this mapping. The native-geometry prefix decoded,
and the full-clip encoder produced a saved quantized latent whose prefix codes
and indices exactly match the short run. This is inference validation, not a
watermark-removal result. The initial direct-decode length assertion failed
before the padding audit; preserve that failure rather than relabeling it a
passed preflight.

The short prefix underestimated full decoding cost because middle chunks use
additional lookahead. Preserve the verified full latent before decoding and
budget from measured middle-chunk work, not just the initial chunk. On the
shared Mac, concurrent audio inference also incurred swapping. Serialize the
large local experiments and retain interrupted-run disposition separately
from signal efficacy. The subsequent full decoder completed from the saved
latent. Delivered-frame geometry and timestamps and copied audio packets/
clocks/configuration/PCM matched the source. Structure/motion diagnostics
covered all frames and transitions. Detail plates and source-relative metrics show stronger texture
and structure changes than the continuous baseline. Those diagnostics do not
establish semantic object/trajectory preservation. The subsequently authorized
frozen audiovisual, silent-video and matched-control files were each submitted
once to CapCut. All three remained detected. Complete captured upload hashes,
pending jobs and settled responses passed offline receipt replay, including
rejection of wrong hashes, repeated uploads and wrong jobs. This fixed discrete
arm failed the removal criterion on this forest export; copied audio cannot
explain the silent candidate's positive. This is not a rejection of every
discrete or temporally coupled method.

A separate fixed guide-registration A/B preserved the native source/output
grid and clocks. It resized only the decoded reconstruction guide, estimated
coarse temporally smoothed source-to-guide flow, then combined registered
low-frequency guide content with the same source high-frequency residual as
the matched unregistered arm. This inherits source pixels and is not a pure
regeneration experiment. Registration modestly improved measured structure and
motion error against that paired baseline, without establishing object-shape
or trajectory ground truth. In the same frozen CapCut pass, the codec-matched
source remained detected and both registered and unregistered outputs were
not detected. Captured upload hashes, jobs and final responses passed offline
replay and binding mutations. This qualifies the two exact files on one source
and that surface; independent generations, stronger detail preservation and
audio-watermark qualification remain open.

An independent owned bottle clip was then processed once through the unchanged
released profile. Its exact label-cleaned source had a retained CapCut-positive
receipt; that derivative is not a pristine native export and is private, without
publication clearance. Both the new reconstructed delivery and its fixed
codec-only control were negative. Preserve this result alongside the earlier
raw-codec-negative/cleaned-positive contradiction. It does not isolate a
regeneration benefit or supply a codec-control-positive replication. No codec
parameters were retuned after these verdicts. Native frame clocks and copied
audio packets passed checks, without an audio-watermark removal claim.

The first temporal-VAE arm is deterministic posterior mode without added
latent noise. Pad and crop without rescaling the native grid, retain the source
frame clock, and encode all comparison arms identically. Respect the model's
temporal length/cache rules. A separate fixed-noise arm requires its own design
and must not be described as trained temporal denoising.

The pinned Wan VAE full-clip posterior-mode arm preserved all native frames
and rational presentation times after CPU staging of decoder outputs. Its
delivery and the matched additional codec pass both remained detected by
CapCut, with upload hashes and settled jobs checked independently. Fine
texture was visibly smoothed despite good structural similarity. This fixed
arm failed removal qualification; structural quality alone is insufficient.
The separately frozen posterior sample used the model's own standard
deviation and fixed noise, without a strength sweep. Its complete decode
attempt stopped on the declared MPS memory limit after one frame. That frame
matched the preregistered prefix exactly, but no complete sample delivery or
provider verdict exists. Allocated, unwritten array positions are not frames.
The conditioned VACE prefix also failed its frozen structural-quality gate,
so it was not expanded or sent to a provider checker.

A separate Google video-only cohort isolates visual signals from audio. On
an owned native export, the unchanged published spatial-VAE profile completed
with its declared reduced resolution and cadence. Full delivered-frame
measurement and worst-frame inspection showed changed paper folds, leaf
contours and water texture relative to the matched codec control. A separately
frozen full-resolution, full-cadence float32 arm stopped at its declared wall
budget without a complete delivery. Neither this failure nor the completed
reduced-resolution reconstruction establishes a Google visual-watermark
verdict by itself. Subsequent compatible checks are recorded below;
an audio quota refusal does not supply evidence about silent video.

A distinct float16 native-grid arm subsequently passed the preregistered
three-frame comparison against float32 and completed the whole clip within
its own fixed resource budget. Every delivered frame and rational timestamp
was checked, with matching native geometry and cadence, no audio, and a
matched codec control. Worst-frame inspection retained the boat's broad
position while showing changes to paper, water and ground texture; estimated
motion residuals also increased relative to the codec control. This is a
better measured detail baseline than the reduced-resolution arm on this
source, not proof of identical shapes, trajectories or watermark removal.
An additional common-grid diagnostic compared every matching presentation
time at source resolution, spatially upsampling only the smaller deliveries
with fixed Lanczos interpolation and applying no temporal interpolation or
alignment. It confirmed the native arm's lower reconstruction error on that
shared grid; this is separate from directly comparing metrics across scales.
The exact saved native-grid file subsequently remained SynthID-positive in
Gemini, as did both matched codec controls. The reduced-grid VAE file received
no-reliable-signals responses in some accounts, retained as indeterminate under
the frozen protocol. Another configured account explicitly reported no Google
SynthID watermark on those exact same bytes. Its positive source and codec
controls were measured on the compatible Gemini surface in another account;
that account difference remains part of the evidence rather than being hidden.

The unchanged fixed reduced-grid profile was also applied to independent owned
Flow and Veo Lite generations before feedback. The Flow source and matched
codec control remained visual-SynthID-positive, while its exact reconstruction
received an explicit no-watermark-found response in the same account and pass.
That qualifies the saved Flow example under the agreed detector criterion.
Its measured texture and motion differences remain separate limitations; this
result does not establish universal removal or every-frame signal absence.
The independent Veo Lite reconstruction subsequently also received an explicit
negative SynthID-watermark response, while its source and matched codec control
remained positive on that same Gemini account and surface. The complete frozen
replication pass therefore meets the agreed saved-example detector criterion
on both independent generations. Both deliveries retain the reduced profile's
resolution/cadence limits, and the textured Veo Lite scene shows substantially
larger detail changes than its codec control. No audio was present in these
uploads, so these visual results cannot certify audio removal.

A separately frozen cadence follow-up kept the spatial VAE profile and shared
noise field fixed while retaining every source presentation time on the Flow
case. The complete reduced-grid delivery and its matched codec control passed
frame-count, rational-PTS, silent-track and metadata checks. Gemini refused the
new candidate because its video quota was exhausted; the new control remains
unsubmitted. No watermark verdict transfers from the historical half-cadence
file. Both Google modality holds remain distinct, and no account rotation was
used after the video refusal.

An exploratory source-annotated boat-region probe also compared corresponding
features at matching presentation times. The reconstruction retained fewer
source matches than the codec control, and the remaining inlier matches had
larger displacement. Source-region coverage fell further when the boat was
partly occluded. These regions include contextual pixels, and affine inliers
are estimates rather than object segmentation or trajectory ground truth.
Report both lost support and displacement; matching the canvas dimensions
alone does not establish preserved object position or shape.

A subsequent local bandwidth-and-scale study (`carbon-video-bandwidth-and-768-b092199d`,
plan SHA-256 `138c44cc2f25a08b6df77ac5b6a9e59a8a4108f236b2b59ef6b48208bcc3a3a6`)
completed full 192-frame native-cadence (`24/1 fps`) intermediate `768x432`
codec and spatial-VAE (`noise_std=0.15`, `seed=0`) arms on both owned Flow
(`flow-001`) and Veo Lite (`lite-003`) exports, the missing `512x288` `24 fps`
pair on `lite-003`, and native-grid (`1280x720`) isotropic Gaussian low-pass
controls (`linear-bw-512match` and `linear-bw-768match`) calibrated to match
the empirical 2D radial luminance `-3 dB` spatial cutoffs of the upsampled VAE
outputs (`0.167-0.170` and `0.230-0.270 cyc/px`, using $\sigma \approx 0.78-0.80\text{ px}$
and `0.54-0.59 px`). Every generated MP4 verified exact 192 frames, zero audio
tracks, zero AI metadata, and exact rational PTS. Raising the native-cadence
VAE grid from `512x288` to `768x432` improved common-grid RGB PSNR (`32.26` to
`33.63 dB` on `flow-001`; `24.31` to `25.78 dB` on `lite-003`), gray/RGB SSIM
(`0.9753/0.9593` to `0.9794/0.9644`; `0.7780/0.7621` to `0.8256/0.8089`),
Laplacian detail ratio (`0.344` to `0.617`; `0.124` to `0.318`), mutual SIFT
matches (`25.7` to `30.4`; `56.9` to `92.2`), unaligned SIFT p95 displacement
(`41.70` to `8.27 px`; `66.77` to `47.07 px`), Canny source-edge p95 distance
(`9.50` to `4.39 px`; `15.58` to `9.88 px`), and source-masked Farneback flow
p95 error (`2.081` to `1.423 px`; `0.575` to `0.419 px`). Comparing against the
`-3 dB`-matched linear Gaussian controls (`38.65 dB / 0.9928` on `flow-001` and
`35.44 dB / 0.9795` with `1203.3` SIFT matches on `lite-003`) isolates the
remaining detail and temporal-residual change (`1.795` and `1.279` vs `0.970`
and `0.998`) to the per-frame VAE bottleneck and latent perturbation rather
than spatial cutoff alone. Under the active Gemini video quota hold, zero
provider requests were sent; all candidate and control hashes are frozen in
`unsubmitted-gemini-manifest.json` (`74e07e43467bf69f477f79c3959caaadcb05a680bd8a3bd518b7b5431e3dd35d`).
On compatible high-resolution `VideoSeal 1.0` 36-frame prefixes (`1280x720 @ 24 fps`
and `704x1280 @ 30 fps`), every downsampled codec arm (`codec-768`, `codec-512`)
and `-3 dB`-matched linear blur arm (`linear-bw-768match`, `linear-bw-512match`)
remained `matched` (`0.9844-1.0000`), whereas all three spatial-VAE scales
(`vae-1280`, `vae-768`, `vae-512`) crossed below `0.90` (`0.7656-0.8398` on the
`720p` carrier, `0.5664-0.8008` on the `1280p` carrier) with negative clean and
foreign-message controls (`0.5391-0.6055`), confirming that VideoSeal disruption
comes from the nonlinear VAE bottleneck and perturbation rather than linear
spatial bandwidth truncation. Rescoring those saved 36-frame deliveries under a
stride-1 12-frame sliding-window scan (25 windows per clip) kept every codec and
linear-bandwidth control `matched` and every spatial-VAE arm `not_matched`.

Adding a single-encode 3-frame streaming motion-compensated temporal blend
(`temporal_stabilize=True` in `video_messages.py`, backward/forward Farneback
optical flow with forward-backward consistency masking and adaptive per-pixel
photometric confidence before the single H.264 encode) reduced
`temporal_residual_ratio` by `10.9-30.7%` and slightly improved common-grid RGB
PSNR and SSIM across `flow-001` (`768x432 @ 24 fps`), `lite-003`
(`768x432 @ 24 fps`), and both 36-frame `VideoSeal 1.0` prefixes (`512` and
`768` long sides), while keeping all 25 sliding `VideoSeal` windows
`not_matched`. Subsequent `gemini-web` video checks showed that matched H.264
controls stayed `detected` and 2D `sd-vae-ft-mse` at `768x432 @ 24 fps` cleared
`flow-001` (`not_detected`) but remained `detected` on textured `lite-003`
(keeping the 2D default at `512 @ 12 fps`), whereas NVIDIA H100 CUDA 3D
`Wan 2.1` video diffusion (`--profile wan21-synthid-experimental`, `strength=0.38`)
returned `4/4` `not_detected` verdicts across `flow-001` and `lite-003` at full
native `1280x720 @ 24 fps`, and `--profile wan21-seedance-experimental`
(`strength=0.12`) returned offline-verified CapCut `not_detected` receipts on
both the Seedance `forest` clip and the held-out `bottle` clip (`PSNR 34.42 dB`,
`TRR 1.0682` vs `seedance-experimental` `31.85 dB`, `TRR 1.3854`), while
zero-noise `AutoencoderKLWan` (`s=0.0`) and matched CRF 18 controls stayed
`detected` and Dreamina / BytePlus enterprise API exports cleared via stream-copy
C2PA metadata stripping alone. Extending `--method dac` and `--method focalcodec`
with raised-cosine overlap-window streaming (`> 30 s` via `20 s` windows with
`2 s` crossfade) and `wan21-*` with `4k+1` 3D causal window crossfades
(`chunk_frames=193`, `overlap_frames=17`) plus `--profile auto` routing enabled
Stage 5 held-out verification on 240-frame (`10.005 s`)
`google-gemini-omni-flash-api` and 96-frame (`4.010 s`) `veo-3-1`: `--method dac`
completed `4/4` `not_detected` source-positive native Google audio tracks
(including `spaces-01`, `veo-flow` AAC-192k MP4 mux, and `omni-flash-api` where
`FocalCodec` remained `detected`), and 3D `Wan 2.1` completed `6/6` silent 720p
video negatives plus `2/2` full-audiovisual (`Wan 2.1` + `DAC` AAC-192k) MP4
negatives against positive matched controls. Live isolated
Playwright checks on first-party public verifiers completed on `microsoft-web`
(both clean audio and video controls settled as `Inconclusive` / `not_detected`)
and `meta-web` (all six clean, open-source `AudioSeal`/`VideoSeal`-marked, and
removed audio/video files settled as `No AI signatures from Meta were found`,
confirming that Meta's public verifier checks proprietary production Content
Seal payloads rather than open-source reference keys), while `openai-web`
returned its generic error (`indeterminate`) on short isolated audio uploads and
notes a `10-60 s` clip range in its live FAQ. Specialist third-party portals
(`synthid.com`, `cartesia.ai/watermark-detector`, `ai-generated-detect.capcut.com`)
gate uploads behind additional legal Terms and were inspected read-only without
uploading files. Follow-up Stage 2/3 verification wired `elevenlabs-api`,
`cartesia-api`, `cartesia-web`, `resemble-api`, and `kling-web` into `scripts/provider_oracles.py`:
live `elevenlabs-api` checks on `34.9 s` (`Eleven Multilingual v2`) and `61.8 s`
(`eleven_v3_dpo`) native MP3s proved ElevenLabs's waveform watermark survives
C2PA stripping and `--method spectral` (`is_detected: true`) while clearing to
`is_detected: false` under `encodec`, `combined`, `focalcodec`, and multi-window
`dac` (driving `adaptive_auto_profiles` to route ElevenLabs and Google SynthID
metadata directly to `dac`/`encodec` rather than `spectral`); live `meta-web` checks on an H.264 MP4 carrying production `Muse Image 1`
`VideoSeal v4` (`detected` -> `not_detected` after `remove_video_invisible`)
established positive-to-negative video coverage; authorized authenticated-Chrome
`kling-web` checks across 9 native Kling exports and ablations
(`provider-original.mp4`, `Kling 3.0 Turbo`, `Kling O3 Standard` with AAC audio,
and `Kling Image v3`) proved Kling's `"Invisible watermark"` detector row on MP4
video reads the H.264 `kling-ai` SEI tag rather than a pixel/audio watermark
(`3/3` MP4 originals `Detected -> Not detected` under lossless
`remove_video_metadata` alone, and `Kling Image v3` cleared under lossless
`remove_ai_metadata`); and `media_removal_benchmark.py`
added phase-invariant multi-resolution STFT and 64-band log-Mel perceptual
distances alongside native `44.1 kHz` stereo multi-window `dac` verification.

Score all delivered frames and worst intervals, including temporal chunk
boundaries and occlusions. Report color separately from gray SSIM. SIFT needs
coverage/match counts; estimated flow is not ground truth. Add object/face/text
review and source-conditioned trajectory checks without silently aligning away
the displacement the user cares about. Preserve the original color/timing
interpretation instead of forcing a new color tag or zero start timestamp.

For audiovisual localization, prepare native source, silent-video, extracted
audio, video-only processed, audio-only processed, and combined delivered arms
only when the checker supports the question and each derivative has a matched
control. A result on a segment cannot certify an entire longer file.

## Verification ladder

1. Local format, metadata/C2PA and visible-label inspection.
2. Compatible pinned local decoders, calibrated on marked and known-clean inputs.
3. Provider-specific watermark API when available and accessible, otherwise
   the provider's documented web surface. Separate mixed classifier outputs.
4. Fingerprint/source-record lookup when documented, recorded independently.
5. Full saved-artifact and final-container verification, plus fidelity review.

Reuse a verdict only for the exact submitted hash, verifier surface, and
measurement scope. New frozen candidates take precedence over repeating
unchanged controls. Errors, quota refusals, unsupported formats, missing fields,
and unavailable accounts remain explicit states. Follow the current provider
terms; do not use accounts or routes to evade verification limits.

The [verifier matrix](media-provider-verifiers.md) records access and payment
gaps. A published endpoint is not a successful live integration. New media
transfers require the applicable authorization; a prepared batch must be
concrete and hash-bound before that boundary is reached. No new login or paid
generation/check is implied by this plan.

## Delivery stages and goals

| Stage | Reviewable deliverable | Exit gate |
| --- | --- | --- |
| 0: evidence | Competitor/method ledger, verifier matrix, acquisition map, ranked hypotheses | Sources distinguish claims, inspected code, execution, and unknowns |
| 1: corpus and measurement | Read-only native-export inventory, grouped sampler, immutable experiment manifests | No edited file promoted to pristine native; no generation leakage between tuning and holdout; hash and permission checks |
| 2: first mechanisms | Full-track codec/vocoder comparisons and full-clip temporal-VAE comparisons | Positive controls, distortion-aware comparisons, whole-output measurements, final-mux checks, quality review; failures retained |
| 3: external evidence | Frozen candidate/control packages with exact upload hashes | Authorized provider verdicts; unavailable routes recorded without inventing a negative |
| 4: library integration | Only qualified experimental profiles, saved-output gates and complete integration catalog | Meaningful regressions, full serialized maintain gate, docs/Agent Skill/interface parity, actual delivery checks |
| 5: expansion | Held-out providers, versions, export paths, and hard latent-reference schemes | No extrapolation across untested product/version/format cells |

The active goal is practical, reproducible signal removal on real owned video
and audio, with preserved content and qualified public integration. Stages are
milestones, never independent completion criteria. Completion requires matched
source-positive and saved-output-negative evidence on compatible verification
surfaces, independent generations, measured video structure/motion/detail and
audio words/voice/sample clocks, and final-container verification with the
required implementation checks. A failed arm can close that hypothesis while
the goal stays open. No result extends to untested schemes or provider cells.
Commits, publication, paid operations, account creation, and correspondence are
separate actions.
