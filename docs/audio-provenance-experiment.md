# Audio provenance experiment (local AudioSeal)

Development-only study, run on 2026-09-06 from `scripts/audioseal_experiment.py`.
It answers the first open item of the original watermark-landscape research:
does a video remain provenance-positive through its untouched audio track
after a visual-only cleaning pass? AudioSeal (Meta, MIT license) is used as a
local matched oracle: the same package embeds the watermark and reads it back,
so no provider API is contacted and no external upload happens.

This page records the design, the pinned inputs, and the measured outcome. It
is not a detector-quality claim: carriers are synthetic non-speech signals,
and AudioSeal is not any provider's production audio watermark.

## Design

The experiment separates three questions and never merges them:

1. `bitstream_identity`: are the cleaned video's audio packets byte-identical
   to the source video's audio packets?
2. `cleaning_invariance`: does the detector verdict on the decoded audio
   change across the visual cleaning pass?
3. `audio_attack`: how does the watermark respond to audio-path processing
   only, with no video involved?

Stages:

- Three deterministic 8-second 16 kHz mono synthetic carriers (AM tone stack,
  box-smoothed "pinkish" noise, white noise), a fixed 16-bit message, clean
  and marked WAVs. Watermark SNR 17.9-34.6 dB.
- Three system-voice speech carriers rendered by the local macOS `say` tool
  (Samantha, Daniel, Milena; no network, byte-deterministic per voice, pinned
  by voice, text digest, and macOS version), each with clean and marked rows
  and per-second interval readings on both the marked audio and a variant
  whose seconds two through four were overwritten with 10 dB-SNR noise while
  every other sample stays bit-identical.
- Audio attacks on the marked tone-stack WAV: AAC-LC 128k at 16 kHz and 48
  kHz, MP3 128k and 64k, 44.1 kHz resample, 8 kHz lowpass, additive noise at
  10 dB SNR, 0.5 gain.
- Three video arms muxing the same marked WAV under a stamped Sora-like mark
  (24 frames, 12 fps, 840x480): PCM in MOV (bitstream-preserving), AAC 128k at
  16 kHz, AAC 128k at 48 kHz. Each arm runs the production
  `remove_video_visible` path, which transcodes video and stream-copies
  audio. Audio packets are extracted container-independently and hashed before
  and after cleaning.

Weights are pinned to Hugging Face revision
`3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba` and verified against pinned SHA-256
digests before loading; the loader mirrors `AudioSeal.parse_model` with the
checkpoint bytes pinned and now lives in the shared `scripts/audioseal_oracle.py`.
Dependency versions at run time: audioseal 0.2.0, torch 2.13.0, Python
3.12.13, ffmpeg 9.0.1, macOS 26.6.2 for the system voices. The executed code is
bound by hash: experiment script
`sha256:cc1f8360403ca1e4f04d72717b4b18ff84ca0718fa01bc1a8316dc189359b5b9`,
oracle
`sha256:d765e31b4c439fda82536b72bb602b55dac666d13e954809b42f48acfad1d2e8`,
case rows
`sha256:99b56a68fdb7789f4adc9f321eadf86277354a383270fe6d689b2c87eba83883`.
Within one fixed environment two independent runs produced byte-identical
case files (verified twice, before and after the oracle refactor). A
full-extras `uv sync` between two recorded runs changed the cleaned-video
H.264 packet bytes - every detector reading, every audio packet digest, and
all source artifacts stayed identical, only the re-encoded video stream
moved - so cross-environment comparisons must bind on the audio evidence and
readings, not on cleaned-container hashes.

## Measured outcome, 2026-09-06

| Stage | Reading |
| --- | --- |
| Clean carriers (matched negatives) | mean detect 0.003-0.008, frac above threshold 0 |
| Marked carriers | mean detect >= 0.9997, bit accuracy 0.94-1.0 |
| All video arms, source audio | mean detect >= 0.9999, bit accuracy 1.0 |
| All video arms, after visual cleaning | identical readings; audio packets byte-identical |
| MP3 128k/64k, AAC 128k (16 and 48 kHz), 44.1 kHz resample, 8 kHz lowpass, 0.5 gain | mean detect >= 0.9999, bit accuracy 1.0 |
| Raw ADTS AAC decode | mean detect 0.990, bit accuracy 0.69 |
| Additive noise, 10 dB SNR | mean detect 0.30, bit accuracy 0.5 |

The thesis holds in this model: `remove_video_visible` leaves the audio
bitstream untouched in every arm, so the audio watermark verdict is invariant
across visual cleaning, and a cleaned video stays provenance-positive through
its audio track. The result is scoped to AudioSeal as the mark; it proves the
copy semantics of the cleaning path, not that any specific provider embeds a
comparable audio mark.

The speech carriers close the out-of-domain gap the synthetic trio exposed:
all three system voices decoded the full 16-bit message (the pinkish synthetic
carrier had flipped one bit), with watermark SNR of 27.9-32.2 dB and clean
speech far below the decision rule. Per-second interval readings localize the
signal in time: overwriting seconds two through four with 10 dB noise killed
exactly those two windows (0.19 falling to 0.0009 mean probability) while the
untouched windows kept their original readings to the sixth decimal. The
final window of a clip is a short remainder (0.1 s here) and its readings are
accordingly less stable - visible in the tables, never dropped.

Three findings qualify the naive readings:

- **Codecs are not the attack.** At these rates every codec and resample round
  trip preserved both detection and the full message, on synthetic and speech
  carriers alike. Full-clip additive noise at 10 dB SNR killed every speech
  detection while the white and pinkish synthetic carriers survived it: speech
  has quiet moments where the noise dominates, so the effective removal lever
  on audio is loud additive noise, and its power depends on the carrier. A
  removal pipeline that wants an audio path clean must not rely on
  transcoding alone.
- **The decode path is part of the measurement.** Decoding raw ADTS keeps the
  encoder's 1024-sample priming head in the stream, which alone moved the
  verdict from 0.9999/full message to 0.990/partial message; a gapless-aware
  container (MP4/M4A edit list, or MP3 gapless info) decodes sample-identical
  audio and restores the reading. The experiment carries the raw-ADTS arm
  (`aac_128k_adts_raw`) explicitly so the artifact stays a named measurement
  instead of contaminating codec rows.
- **Voice names are not carrier identity.** macOS 26.6.2 renders `Alex` and
  `Samantha` byte-identically, so a naive three-voice set was one carrier
  counted twice. The shipped script verifies distinct content hashes across
  configured voices and refuses an aliased set; diversity claims rest on the
  hash, never the name.

Detector cost on this host (Apple silicon, CPU): cold first call 176-465 ms
across runs, warm median 158-160 ms for 8-second clips. These are diagnostic
host timings, not a certified profile.

An earlier iteration derived per-carrier seeds from Python's `hash()`, which is
salted per process; it produced different carrier bytes on every run. The
shipped script derives seeds from `zlib.crc32`, and the two recorded final runs
are byte-identical.

## Follow-ups

- The speech and interval layers are complete; the remaining audio-side
  upgrade is folding per-second windows into the benchmark schema itself, if a
  case-level question ever needs them as rows. The forgery/remover-forensics
  and VideoSeal temporal items remain separate open directions from the
  original research map.
