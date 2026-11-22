# Removal examples across popular formats and invisible watermarks

This guide collects end-to-end examples across the most popular image, video,
audio, and text formats, focusing on files that carry **invisible watermarks**
in addition to visible logos and metadata.

Every example uses the same unified command:

```bash
remove-ai-watermarks all <input> -o <output>
```

`all` inspects each file first:
- on **images**, it removes visible marks, runs invisible regeneration whenever
  an invisible watermark or AI-provider provenance is present, and strips AI
  metadata;
- on **video**, it removes visible marks, automatically trims confirmed trailing
  AI outro bumpers, runs `profile="auto"` invisible regeneration whenever AI
  provenance or a visible AI mark is present (routing ByteDance/Seedance clips
  to `wan21-seedance` on BF16 CUDA or 10-bit-capable `seedance` on CPU/MPS, and
  Google/other clips to `wan21-synthid` or `synthid`), cleans watermarked or
  provenance-identified AI audio tracks, and strips all container/SEI/AV1 OBU/C2PA
  metadata;
- on **audio**, it detects local waveform marks (`AudioSeal`, `WavMark`, `Perth`,
  `SilentCipher`) or proprietary AI-audio provenance (Google SynthID, ElevenLabs,
  Cartesia, OpenAI), resynthesizes with quality verification, and strips metadata;
- on **clean files** with no AI signals, pixels and audio samples are left untouched.

---

## 1. Images with invisible watermarks (`PNG`, `JPG`, `WebP`)

### OpenAI ChatGPT (`PNG` - SynthID + C2PA)

```bash
remove-ai-watermarks identify data/fixtures/provenance/chatgpt-1.png
remove-ai-watermarks all data/fixtures/provenance/chatgpt-1.png -o chatgpt_clean.png
remove-ai-watermarks identify chatgpt_clean.png
```

Before/after comparison verified with OpenAI's official verifier (`qwen-zimage`):
- Source: [`ChatGPT Image May 30, 2026, 10_31_08 AM.png`](../data/synthid/originals/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM.png)
- Cleaned: [`ChatGPT Image May 30, 2026, 10_31_08 AM_full_clean.png`](images/qwen-zimage/ChatGPT/ChatGPT%20Image%20May%2030,%202026,%2010_31_08%20AM_full_clean.png)

### Google Gemini (`PNG` - visible sparkle + SynthID + C2PA)

```bash
remove-ai-watermarks all data/fixtures/provenance/google-nano-banana.png -o gemini_clean.png
```

Before/after comparison verified with Google's official verifier (`qwen-zimage`):
- Source: [`Gemini_Generated_Image_633uuy633uuy633u.png`](../data/synthid/originals/Gemini_Generated_Image_633uuy633uuy633u.png)
- Cleaned: [`Gemini_Generated_Image_633uuy633uuy633u_full_clean.png`](images/qwen-zimage/Gemini/Gemini_Generated_Image_633uuy633uuy633u_full_clean.png)

### Microsoft Copilot / Designer (`PNG` - Microsoft InvisMark + C2PA)

```bash
remove-ai-watermarks identify data/fixtures/provenance/microsoft-invismark.png
# Platform: Microsoft (Copilot / Designer) | C2PA + InvisMark
remove-ai-watermarks all data/fixtures/provenance/microsoft-invismark.png -o microsoft_clean.png
```

With `--pipeline auto`, Microsoft assets route automatically to `chroma-zimage`
(commercial-safe Apache-2.0) or `qwen-zimage`.

### Adobe Firefly (`PNG` - Adobe TrustMark-P + C2PA)

```bash
remove-ai-watermarks identify data/fixtures/provenance/adobe-trustmark-p.png
remove-ai-watermarks all data/fixtures/provenance/adobe-trustmark-p.png -o firefly_clean.png
```

`identify` decodes the embedded `TrustMark` payload locally, `all` strips C2PA
and scrubs the invisible watermark, and a follow-up `identify` confirms neither
`TrustMark` nor C2PA remains.

### Stable Diffusion XL (`PNG` - open DWT-DCT invisible watermark)

```bash
remove-ai-watermarks identify data/fixtures/provenance/sdxl-dwt-dct.png
# Platform: Stable Diffusion XL (open DWT-DCT watermark)
remove-ai-watermarks all data/fixtures/provenance/sdxl-dwt-dct.png -o sdxl_clean.png
```

### ByteDance Dreamina (`JPG`) and Recraft v3 (`WebP`)

```bash
# Lossless JPEG DCT preservation when only metadata/C2PA is stripped, or full auto clean:
remove-ai-watermarks all data/fixtures/provenance/dreamina-seedream-5-pro.jpg -o dreamina_clean.jpg

# WebP with C2PA / XMP:
remove-ai-watermarks all data/fixtures/provenance/recraft-v3.webp -o recraft_clean.webp
```

---

## 2. Video with visible + invisible watermarks (`MP4`, `MOV`, `WebM`, `MKV`, 10-bit HEVC)

### Google Veo (`MP4` - visible diamond + SynthID video + SynthID audio + C2PA)

```bash
remove-ai-watermarks identify data/fixtures/provenance/google-gemini-omni-app.mp4
remove-ai-watermarks all data/fixtures/provenance/google-gemini-omni-app.mp4 -o veo_clean.mp4
```

One command removes the visible `Veo` corner diamond, runs `wan21-synthid` on
BF16 CUDA (or `synthid` on CPU/MPS), resynthesizes the Google SynthID audio track
with `DAC 44.1 kHz`, and strips all C2PA/container metadata.

### ByteDance Seedance (`MP4` - 10-bit HEVC/HDR + TC260 + invisible Seedance)

```bash
remove-ai-watermarks all data/fixtures/provenance/dreamina-seedance-2-mini.mp4 -o seedance_clean.mp4
```

`all` (and `video all --invisible`) automatically detects ByteDance/Seedance
provenance and routes to `wan21-seedance` on BF16 CUDA or `seedance` on CPU/MPS.
Native **10-bit HEVC (`yuv420p10le`) and BT.709/BT.2020 HDR color tags** are
preserved without downgrading to 8-bit H.264. You can also select a profile
explicitly:

```bash
remove-ai-watermarks video invisible clip.mp4 -o clean.mp4 --profile seedance
remove-ai-watermarks video invisible clip.mp4 -o clean.mp4 --profile wan21-seedance
```

### Meta `VideoSeal` / `PixelSeal` (`MP4`, `MOV`, `MKV`, `WebM`, VFR & non-zero PTS)

```bash
remove-ai-watermarks video message marked.mp4 -o clean.mp4 --scheme videoseal
```

Decodes the embedded message before cleaning, cleans with minimal-distortion
re-encoding or VAE fallback, and verifies the saved output is decoder-negative
(including VFR and non-zero start-PTS clips).

---

## 3. Audio with invisible watermarks (`WAV`, `MP3`, `M4A`, `FLAC`, `OGG`, `Opus`)

### Google SynthID audio (`WAV` / `MP3` / `M4A`)

```bash
# Automatic provenance-triggered DAC 44.1 kHz resynthesis:
remove-ai-watermarks all veo_audio.wav -o veo_audio_clean.wav

# Explicit SNAC 44.1 kHz (verified 2/2 not_detected on synthid.com):
remove-ai-watermarks audio invisible veo_audio.wav -o veo_audio_snac.wav --method snac --allow-unverified

# Studio mastering high-frequency crossover (>14 kHz transients preserved; verified not_detected on synthid.com):
python -c "from remove_ai_watermarks import remove_audio_invisible; remove_audio_invisible('veo_audio.wav', 'clean.wav', method='dac', allow_unverified=True, preserve_hf_hz=14000)"
```

### `AudioSeal`, `WavMark`, `Resemble Perth`, and `SilentCipher` (`WAV`, `MP3`, `FLAC`)

```bash
remove-ai-watermarks audio identify song.wav
remove-ai-watermarks all song.wav -o song_clean.wav
remove-ai-watermarks audio identify song_clean.wav
```

For all locally decodable audio watermarks, `all` detects the active scheme(s),
picks the lowest-distortion profile (`spectral`, `dac`, `encodec`, or `combined`),
verifies the saved output is decoder-negative, and enforces objective audio
fidelity gates.

### ElevenLabs, Cartesia Sonic, and OpenAI audio (`MP3` / `WAV`)

```bash
# Automatic when pre-strip metadata identifies the provider:
remove-ai-watermarks all speech.mp3 -o speech_clean.mp3

# Explicit cohort when metadata was already stripped:
remove-ai-watermarks all speech.mp3 -o speech_clean.mp3 --vendor elevenlabs
remove-ai-watermarks all speech.mp3 -o speech_clean.mp3 --vendor cartesia
```

---

## 4. Text with invisible provenance (`Markdown`, `HTML`, `TXT`)

```bash
# Deterministic C2PATXT, HTML/ASCII armor, VSRMark, and Unicode steganography:
remove-ai-watermarks all article.md -o article_clean.md

# Include span-protected local prose regeneration for statistical token watermarks:
remove-ai-watermarks text all article.md -o article_clean.md --invisible
```

---

## 5. Interactive browser demo

All core image, video, and audio examples above are also available as one-click
presets in the live [Hugging Face Space](https://huggingface.co/spaces/wiltodelta/remove-ai-watermarks).
