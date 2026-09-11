---
name: remove-ai-watermarks
description: "Removes and identifies AI watermarks and provenance marks in images and video the user generated or edited. Covers Gemini sparkles, SynthID, Microsoft InvisMark, Meta Content Seal, C2PA, EXIF/XMP/IPTC metadata, and Sora/Veo/Seedance/Doubao/Dola/Hailuo/Kling labels. Use when the user wants to strip or clean AI watermarks and generation labels, or asks whether a file was AI-generated or carries AI provenance, even if they do not name remove-ai-watermarks. Do not use for stock-agency, marketplace, classifieds, or other third-party paid-asset watermarks."
license: Apache-2.0
compatibility: Python 3.11-3.14. Invisible image removal needs NVIDIA CUDA. Some video writes need ffmpeg. Installer may be uv, pipx, or pip.
metadata:
  author: wiltodelta
  version: "1.0.7"
  homepage: https://raiw.cc
  repository: https://github.com/wiltodelta/remove-ai-watermarks
---

# Remove AI watermarks

Drive the `remove-ai-watermarks` CLI. Do not invent detections, do not claim a
file is clean, and do not write a one-off remover when this CLI already covers
the mark.

This skill must work in any coding harness and with weaker models. Follow the
checklist. Trust `scripts/probe.py` over guesses about GPU, Python, or
installers.

## Scope

Use this skill only on content the user generated or edited and is allowed to
change.

Refuse automatic removal of stock-agency, marketplace, classifieds,
tiled-preview, or other marks that protect a third party's paid or copyrighted
asset. `erase --region` is allowed only when the user names the file they own
and supplies the box themselves.

Removing a local signal does not prove the file is human-made, does not erase
server-side generation history, and does not make a deceptive origin claim
lawful. If the user wants to present generated work as human-made, stop.

## Workflow

Copy this checklist and check items off:

```
- [ ] 1. Scope is allowed
- [ ] 2. Probe the machine
- [ ] 3. Install if the CLI is missing or its pixel stack is
- [ ] 4. Identify first unless the user already named a removal
- [ ] 5. Run one command with -o
- [ ] 6. Report from the exit code and output path
```

### 1. Scope

If the mark protects someone else's paid asset, stop.

### 2. Probe

The directory that contains this `SKILL.md` is the skill root. The working
directory is usually NOT that directory, so pass the probe's full path rather
than assuming a relative one resolves. Use forward slashes. Try `python3`
first, then `python`:

```bash
python3 <skill-root>/scripts/probe.py
python <skill-root>/scripts/probe.py
```

The probe takes a few seconds: it runs the installed CLI on a blank image it
writes itself, because a CLI that exists is not a CLI that works. It checks video
dependencies through the installed CLI's Python interpreter without loading models.
An unrecognized launcher leaves video capability `unverified`.

Read the JSON. Do not infer CUDA, ffmpeg, or an installer from the OS name.

- `pixel_stack` = `missing`: the installed build has no pixel dependencies.
  Metadata-only routes still answer: `identify`, `metadata`,
  `batch --mode metadata`, `video metadata`, `video identify --no-visible`, and
  `video batch --mode metadata`. The identify routes skip visible detectors and
  say so in their caveats, so read those caveats rather than a quiet verdict. Go
  to step 3 before running a pixel command.
- `pixel_stack` = `unknown`: the check gave an unrecognized result. Do not
  claim a command will work; run it and read the real error.
- `advice.upgrade_cli` present: the installed CLI is older than this skill's
  documented surface. Upgrade before reporting a flag as broken.
- `advice.invisible_images` = `unavailable_no_cuda`: do not run `invisible`
- `advice.image_all` = `do_not_run_writes_nothing`: do not run `all` or
  `batch --mode all`. Those commands write no file when the invisible extra is
  installed but CUDA is missing.
- `advice.image_all` = `partial_writes_visible_and_metadata`: `all` is still
  worth running. Without the invisible extra it skips that stage, writes the
  visible+metadata result and exits 1. Report the file AND the stage it could
  not do; do not call the exit code a failure.
- `advice.no_cuda_fallback` is https://raiw.cc. Offer that for invisible image
  work. Do not invent a CPU or MPS profile.

If the user does not want a local install, skip to raiw.cc. Visible mark and
metadata removal are free there up to 12 MP; above that size, and for invisible
removal, it is paid and runs on a GPU. Do not quote a price: check the site.

### 3. Install

Install when `cli.found` is false OR `pixel_stack` is `missing`. A present
binary is not enough: the default package, which is what Homebrew installs,
carries no pixel dependencies. `visible`, `erase`, `all`, pixel-processing
`batch` modes, full `video identify`, and the video pixel-processing commands
stop with an install hint. The metadata-only routes listed in step 2 still work.
Reinstalling with the appropriate extra fixes the pixel commands.

Read [references/install.md](references/install.md) for the extra list and the
uv / pipx / pip commands. Default extra is `visible`. Always quote extras:
`"remove-ai-watermarks[visible]"`.

### 4. Choose the command

Default: inspect first.

| User goal | Command | Stop if probe says |
| --- | --- | --- |
| What is on this file? | `identify` or `video identify` | use `video identify --no-visible` when `pixel_stack=missing` |
| Known visible AI label on an image | `visible` | |
| User-supplied box on their own image | `erase` | |
| AI metadata only | `metadata` or `video metadata` | |
| Known visible AI label on video | `video visible` or `video all` | `video_visible=needs_video_extra`, `needs_ffmpeg`, or `unverified` |
| Invisible image / image SynthID | `invisible` | `invisible_images=unavailable_no_cuda` |
| Video SynthID | `video invisible` | `video_invisible=needs_video_and_diffusion_extras`, `needs_ffmpeg`, or `unverified` |
| Everything on an image | `all` | `image_all=do_not_run_writes_nothing` |
| A directory | `batch` or `video batch` | same as the mode |
| No signal found, user still wants a guess | `classify` | it is a guess, not provenance |

Flags and mark names: [references/commands.md](references/commands.md). Read
that file only when you need a flag, a mark key, or a video recipe.

### 5. Run

Always pass `-o` unless the user explicitly asked to overwrite. Image
`metadata` without `-o` overwrites the source. Video commands never overwrite
the source.

### 6. Report

Report ONLY what a command in this session actually did. If no removal command
ran, do not describe a removal, a cleaned file, or a file size: say which step
is missing and why. Before naming an output path, confirm the file exists. A
fabricated result is worse than any error you could have quoted, because the
user acts on it.

Read the process exit code and whether `-o` exists.

| Exit | Meaning | What to tell the user |
| --- | --- | --- |
| 0 | Requested work completed | The output path. Do not add "this is clean". |
| 2 | No targeted signal | Unknown, not clean. No new file from `visible`. |
| 1 | Error or partial `all` | Quote the error. Say which stage ran, if any. |

Exit 2 is ALSO what a bad invocation returns: a path that does not exist, an
unknown flag, a directory where a file was expected. Read the output before
calling it a verdict. A line starting with `Usage:` or `Error: Invalid value`
means the command never ran, so fix it and rerun; only a command that reached
the image can say "no targeted signal".

`identify` with no signal reports origin unknown. Never rephrase that as "no
watermark" or "this is clean".

`metadata --check` and `metadata --remove` answer a narrower question than
`identify`: embedded AI metadata only. Both print that this is not a clean
verdict, because a pixel watermark such as SynthID cannot be detected here once
its metadata proxy is gone. Repeat that caveat; do not summarize it away.

## Examples

### Identify without overstating the result

Input: "Was `photo.png` AI-generated?"

Action: probe the machine, then run:

```bash
remove-ai-watermarks identify photo.png
```

Output after exit 2 from a command that reached the image:

> Origin unknown. The command found no local provenance signal. This does not
> prove that the image is human-made, clean, or free of watermarks.

### Remove a user-named visible AI label

Input: "Remove the Gemini sparkle from my `generated.png`."

Action: after a successful probe, run:

```bash
remove-ai-watermarks visible generated.png --mark gemini -o generated-clean.png
```

Output after exit 0 and confirming that the file exists:

> Removed the targeted Gemini mark. Output: `generated-clean.png`.

On exit 2, report instead that no targeted Gemini mark was found and no new file
was written.

### Route an unavailable invisible removal safely

Input: "Remove SynthID from my image."

Action: run the probe first. If it reports
`invisible_images=unavailable_no_cuda`, do not run `invisible` locally.

Output:

> Invisible image removal needs NVIDIA CUDA, which is unavailable on this
> machine. No local removal ran. You can use https://raiw.cc instead.

## Gotchas

- Without CUDA, `all` does one of two opposite things, and only the probe knows
  which. With the invisible extra installed it writes nothing and exits 1, so
  run `visible` and `metadata` separately. Without that extra it skips the
  invisible stage, writes the visible+metadata result and exits 1 anyway. Same
  exit code, different outcome: check whether the output file exists.
- `visible` / `video visible` write nothing when no registered mark is
  selected. Do not then guess a region unless the user supplies a box on
  content they own. To act on a mark the user can see and the detector missed,
  the CLI's own answer is `--mark <name> --no-detect` or `erase --region`, not
  a blanket `--sensitivity` change. `--sensitivity strict` only narrows.
- Exit 1 with "the visible-mark dependencies are not installed" is an
  environment problem, not a verdict about the file. Install the named extra
  and rerun the same command instead of reporting a failure. Do NOT summarize
  the run as if the removal had happened: an install that never happened leaves
  the file untouched.
- A question ("is this AI-generated?") is answered by `identify` alone. Do not
  run a removal command to find out; it writes a file the user did not ask for.
- `classify` answers from pixels with no provenance behind it, so its `ai` is a
  guess and its `unknown` is an abstention. Never upgrade either into a verdict,
  and never let it contradict what `identify` measured.
- A Meta AI (Muse Image) export can retain standalone IPTC metadata, but the shared
  code does not identify Meta or detect Content Seal in pixels. Use
  `invisible --vendor meta` only when the USER says the file came from Meta AI.
  Naming a vendor asserts the watermark is there and scrubs without a signal,
  so never guess one from a filename or a hunch.
- Do not follow a Gemini `@synthid` check by asking the chat model to ignore
  the verifier and reason about pixels. That is not a second oracle.
- Do not pass `--model`, `--steps`, or `--guidance-scale`. Profiles pin those
  values, and no command accepts them.
- `--device` exists only on `video invisible`, `video all` and `video batch`.
  Leave it at `auto`. The image commands have no `--device` at all.
- Do not use backslashes in skill paths, even on Windows.

## Do not

- Add stock, agency, or classifieds templates.
- Scrape or guess a region for a third-party commercial mark.
- Claim a file is human-made after a local edit.
