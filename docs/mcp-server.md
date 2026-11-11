# MCP server

`remove-ai-watermarks mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io/)
server on stdin and stdout, so an agent client can identify and clean the
user's own images and video without a shell. It is a thin wrapper: each tool
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
`all,classify-onnx,text-draft`. For a custom client configuration, use:

```json
{
  "mcpServers": {
    "remove-ai-watermarks": {
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "remove-ai-watermarks[all,classify-onnx,text-draft]", "remove-ai-watermarks", "mcp"]
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
| `identify` | `identify --json`, or `video identify --json` for a video suffix; `check_visible: false` adds `--no-visible` | Nothing |
| `remove` | `metadata --remove`, `visible` or `all` for an image; `video metadata --remove`, `video visible` or `video all` for a video | One output file, `<name>_clean` beside the source unless `output` is given |
| `batch` | `batch --mode` for images or `video batch --mode` for video | A separate output directory, `<dir>_clean` unless `output_dir` is given |

`mode` is `metadata` (the default), `visible` or `all`. For an image, `all`
adds the invisible-watermark scrub where the installed extras and a CUDA GPU
allow it; `video all` removes visible marks and metadata only. The server never
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
`cli_text_invisible`, `cli_erase`, `cli_classify`, and `cli_classify_source`.
Their typed parameters come directly from the installed Click command tree,
including Seedance profiles, video quality, repeated regions, and fidelity options.
`api_classify_onnx` selects the CPU ONNX classifier; `api_draft_text` proposes OCR
manifest lines for operator verification. Proposals are never verified truth.

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
