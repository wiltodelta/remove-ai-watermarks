# MCP server

`remove-ai-watermarks mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io/)
server on stdin and stdout, so an agent client can identify and clean the
user's own images, video, and audio without a shell. It is a thin wrapper: each tool
runs one command of this CLI in a child process and returns that command's exit
code, output path and text.

The subcommand ships since 0.46.0. The server uses only the Python standard
library, so the default package runs it. The commands behind a tool need the
same extras they need on the command line
([CLI guide](cli.md#command-dependency-map)). A missing extra comes back as a
tool error carrying the CLI's install hint.

## Client configuration

Most clients take a command and arguments. With [uv](https://docs.astral.sh/uv/):

```json
{
  "mcpServers": {
    "remove-ai-watermarks": {
      "command": "uvx",
      "args": ["remove-ai-watermarks", "mcp"]
    }
  }
}
```

To make visible-mark removal and pixel detectors available, name the extras:

```json
{
  "mcpServers": {
    "remove-ai-watermarks": {
      "command": "uvx",
      "args": ["--from", "remove-ai-watermarks[visible,video]", "remove-ai-watermarks", "mcp"]
    }
  }
}
```

An existing installation can be started directly as `remove-ai-watermarks mcp`.
Paths in tool calls are read and written on the machine that runs the server.

The registry launcher selects Python 3.12 and the complete public backend set
`all,text-draft`. For a custom client configuration, use:

```json
{
  "mcpServers": {
    "remove-ai-watermarks": {
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "remove-ai-watermarks[all,text-draft]", "remove-ai-watermarks", "mcp"]
    }
  }
}
```

Model weights still download on first use. NVIDIA CUDA is required for invisible
image regeneration. The transport itself remains standard-library-only; launching
the bare package is supported for metadata operations and returns install hints
when an optional backend is absent.

## Tools

| Tool | Runs | Writes |
| --- | --- | --- |
| `identify` | `identify --json` for an image, `video identify --json` for a video suffix (`check_audio: true` adds `--audio`), `audio identify --json` for an audio suffix, or `text identify --json` for a text suffix; `check_visible: false` adds `--no-visible` (or `--no-invisible` for audio unless `check_audio` overrides it) | Nothing |
| `remove` | `metadata --remove`, `visible` or `all` for an image; `video metadata --remove`, `video visible` or `video all` for a video; `audio metadata --remove` or `audio all --invisible` for audio; `text metadata --remove` or `text all` for text (or top-level `all --preset` when `preset` is set) | One output file, `<name>_clean` beside the source unless `output` is given |
| `batch` | `batch --mode` for images or mixed directories (`media: "all"`), with optional `--preset`; `video batch --mode` for video (`media: "video"`); or `audio batch --mode` for audio (`media: "audio"`) | A separate output directory, `<dir>_clean` unless `output_dir` is given |
| `audio_identify` | `audio identify --json`; `check_invisible: false` skips models | Nothing |
| `audio_invisible` | `audio invisible --json`, with explicit `stream`, `vendor`, `allow_unverified`, `force`, and `verify` options | `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.oga`, `.opus`, `.aac`, or `.mka`, `<name>_processed.wav` by default |
| `video_message_identify` | `video message identify --json`, required `reference`, optional `scheme` | Nothing |
| `video_message_remove` | `video message remove --json`, same reference and scheme, profile and verification options | MP4, MOV, or M4V, `<name>_message_processed.mp4` by default |
| `video_audio` | `video audio --json`, with optional `stream` (omitted cleans all watermarked tracks) and the same audio options | MKV/MOV with PCM or MP4 with AAC, `<name>_audio_processed.mkv` by default |

Audio tools ship since 0.63.0. `stream` is a nonnegative audio-track ordinal (`video_audio` inspects all tracks and cleans every detected watermarked track while stream-copying unmarked tracks when `stream` is omitted, or targets one track when `stream` is set).
`audio_invisible` and `video_audio` accept `method: "auto"` (default), `"spectral"`, `"encodec"`,
`"combined"`, `"dac"`, or `"snac"`.
`audio_invisible` requires a compatible decoder extra such as `audio-detect` for default source-positive and
saved-output-negative verification. A `vendor` other than `"auto"` (`google`, `elevenlabs`, `suno`, `udio`, `openai`, `cartesia`, `meta`, `resemble`, `microsoft`) names a known source cohort for stripped files, selects a vendor-matched profile sequence, and implies `allow_unverified`. `allow_unverified: true` (for example with `method: "dac"`, or alongside `verify: false`) permits unverified processing with the `audio` extra while keeping the audio SNR/RMS/window integrity gate active (`force: true` also overrides final decoder and quality checks). Audio tools require ffmpeg and
ffprobe. `scheme: "auto"` (default; scans installed non-experimental decoders `audioseal`, `wavmark`, `perth` without failing on missing optional extras), `"audioseal"`, `"wavmark"`, `"perth"`, `"silentcipher"`, or `"all"` selects
compatible public-model checks. WavMark needs `audio-wavmark`, Perth v1 needs `audio-perth`, and SilentCipher-44k needs
`audio-silentcipher`; `all` needs every decoder extra. Auto tries spectral then combined for AudioSeal and Perth v1, neural then
combined for WavMark and SilentCipher-44k, and combined for `all`. Neural profiles need `audio-codec`.
Explicit `method: "dac"` uses the FP32 44.1 kHz 3-codebook DAC backend from
`audio-codec` (defaulting `preserve_hf_hz=14000.0` on `>= 44.1 kHz` tracks) with
raw mid/side stereo, stem RMS preservation, and single-pass / overlap-window
streaming. Explicit `method: "snac"` uses the 44.1 kHz multi-scale SNAC backend
from `audio-codec`. `vocoder_device` (`"auto"` default, `"cpu"`, `"mps"`, or
`"cuda"`) applies to DAC, SNAC, EnCodec, and `combined`. Bandwidth applies only
to EnCodec. `dac` is selected by `auto` when pre-strip metadata identifies
ElevenLabs or Google SynthID, or `vendor` names a cohort. Neither DAC nor SNAC
provides a built-in provider verifier; both are exposed by legacy tools and the
shared operation catalog.

`mode` is `metadata` (the default), `visible` or `all`. For an image, `all`
adds the invisible-watermark scrub where the installed extras and a CUDA GPU
allow it; for audio, `all` runs `audio all --invisible` across compatible public
decoders; for text, `all` runs `text all`; for video, `video all` removes visible
marks and metadata only (use `cli_video_all` when opt-in video or audio invisible
regeneration is needed). The server never
writes over an input: an `output` equal to the source, or an `output_dir` equal
to the input directory, is refused before anything runs.

Exit codes keep their CLI meaning. `0` and `2` are results, not tool errors, and
`2` means no supported signal was found, or a `batch` directory held nothing to
process, which is unknown rather than clean. An
input that is not a readable image or video exits `1`, so `identify` on it is a
tool error even though the command still printed a report.
Click also exits `2` on a usage error such as a missing file; its `Usage:` line on
stderr marks that case as a tool error. Every other nonzero exit code is returned
with `isError: true` and the command's stderr.

## Complete operation tools

Alongside the convenience tools above, the server advertises every non-server
CLI leaf as `cli_<command>`, replacing spaces and hyphens with underscores.
Examples: `cli_video_invisible`, `cli_video_all`, `cli_text_identify`,
`cli_text_invisible`, `cli_erase`, and `cli_classify_source`.
Their typed parameters come directly from the installed Click command tree,
including Seedance profiles, video quality, repeated regions, and fidelity options.
`api_draft_text` proposes OCR manifest lines for operator verification. Proposals
are never verified truth.

The catalog is shared with ComfyUI and the Space. CLI additions automatically
reach those interfaces instead of relying on copied command lists. Each operation
runs in an isolated child with the installed extras, preserves its input, and
returns the real exit code and report. Credentials travel on stdin and are redacted
from logs and returned diagnostics. User output paths must differ from the source,
including symlinks to it. Batch output directories must differ from the input.

## Scope

The server carries the project's intended-use boundary in its `instructions`:
it is for content the user generated or edited themselves, and not for stock,
agency, marketplace or other marks that protect a third party's asset. See
[legal and safety](legal-and-safety.md).

## Protocol

The server answers the initialize handshake for protocol versions 2025-11-25,
2025-06-18, 2025-03-26 and 2024-11-05, and lists its tools with input schemas
and annotations: `identify` is read-only, and `remove` and `batch` write new
files without destroying inputs. It does not implement `server/discover`. A
client that probes it first, such as the official Python SDK's `Client` in its
default mode, receives method-not-found and falls back to the handshake.
`tests/test_mcp_server.py` runs that client against the real subcommand.

## Registry listing

[`server.json`](../server.json) describes the PyPI package for the
[official MCP registry](https://registry.modelcontextprotocol.io/) under
`io.github.wiltodelta/remove-ai-watermarks`. The registry verifies ownership by
the `mcp-name:` marker in `README.md`, which becomes the PyPI description, so it
can only verify a release whose README carries that marker.
[`glama.json`](../glama.json) names the maintainer for Glama. Each release is
published automatically; the workflow is described in
[release and distribution](release-and-distribution.md#mcp-registry).

The video-message tools require an independent hexadecimal encoder message: 64
digits for VideoSeal and PixelSeal, 256 for ChunkySeal. They compare known targets
under public VideoSeal/PixelSeal/ChunkySeal weights, without classifying AI origin
or unknown marks. The blind presence check without a reference is available as
`cli_video_message_detect`. See the [CLI contract](cli.md#public-model-video-messages-experimental)
for fixed profiles, optional extras, timeline limits, quality reports, and force semantics.
