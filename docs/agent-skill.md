# Agent skill

The published Agent Skill lives in
[`skills/remove-ai-watermarks/`](../skills/remove-ai-watermarks/). It teaches
coding agents to drive the `remove-ai-watermarks` CLI on content the user
generated or edited. It is not a fifth release surface: it ships with this
repository and updates when the skill files change.

## Install

Any agent that speaks the [Agent Skills](https://agentskills.io/specification)
format:

```bash
npx skills add wiltodelta/remove-ai-watermarks
```

Claude Code, from this repository as a marketplace:

```text
/plugin marketplace add wiltodelta/remove-ai-watermarks
/plugin install remove-ai-watermarks@remove-ai-watermarks
```

SkillsMP indexes public GitHub `SKILL.md` files and does not need a separate
upload. ClawHub and the Claude community marketplace are explicit publishes;
see [Release and distribution](release-and-distribution.md#agent-skill).

## Where the skill can actually be listed

Most skill catalogs are install registries, not discovery engines. A listing
without search traffic does not bring people to the CLI or to
[raiw.cc](https://raiw.cc). Prefer channels that a stranger can find, and keep
the intended-use boundary in every public description. Official review may
still reject a watermark-removal listing; that is a risk, not a reason to
widen the scope.

Do these in order after the skill is on the default branch.

1. **GitHub default branch.** Unlocks `npx skills add wiltodelta/remove-ai-watermarks`.
   Cursor, OpenCode, Pi, Codex, and Copilot have no separate skill store; they
   consume this repo or a copied `SKILL.md`.
2. **SkillsMP.** Aggregates public GitHub `SKILL.md` files. No submit form. A
   search for `remove-ai-watermarks` now returns this skill. Neighboring
   `watermark` hits are PDF-add and detection skills, not removal.
3. **skills.sh.** No submit form. A row appears on the leaderboard only after
   people run `npx skills add`. The listing is live and has recorded real
   installs; the README install line remains the seed, not a registration.
4. **Awesome-list PRs.** Hand-curated GitHub lists are how people browse skills
   today. One accurate PR per list, with the scope sentence intact.
   - `ComposioHQ/awesome-claude-skills`: submitted 2026-09-05 as PR 1841
     (Creative & Media); no maturity bar in its contribution rules.
   - `heilcheng/awesome-agent-skills`: submitted 2026-09-09 as PR 483 under
     Community Skills > Media. The skill's three concrete input/output examples
     satisfy the list's working-example requirement.
   - `VoltAgent/awesome-agent-skills`: DEFERRED. Its CONTRIBUTING rejects
     brand-new skills ("give your skill time to mature and gain users before
     submitting"). The skill now has real installs and shipped releases, but
     adoption remains too small to make the list's maturity claim comfortably.
   - `VoltAgent/awesome-openclaw-skills`: DEFERRED on adoption only. Entries
     are indexed via clawskills.sh, and the ClawHub prerequisite in step 6 is
     now met. The list still requires proven real-world usage. Re-attempt once
     the ClawHub listing shows organic installs.
5. **Claude community marketplace.** Form at
   [platform.claude.com/plugins/submit](https://platform.claude.com/plugins/submit)
   (individuals) or the claude.ai directory form (Team/Enterprise). Validate
   with `claude plugin validate` first. SUBMITTED 2026-09-07 for Claude Code
   only (strict validation passed; Apache 2.0; description carries the
   scope boundary; platform selection limited to Claude Code because Cowork
   was not tested). Pending review; the team may reach out by email. The
   official Anthropic marketplace is
   curated and has no application.
6. **ClawHub.** `clawhub login` then `clawhub skill publish` from
   `skills/remove-ai-watermarks/`. LISTED 2026-09-09: versions 1.0.5 and 1.0.6,
   submitted by the `clawhub` job in `distribute.yml` during the 0.39.0 and
   0.39.1 releases, are public. A green job proves only a submission, so read
   the catalog itself:
   `clawhub search remove-ai-watermarks` returns the row and
   `https://clawhub.ai/api/v1/skills/remove-ai-watermarks` reports
   `latestVersion.version` 1.0.6, while its `/versions` endpoint returns those
   two published versions. The earlier 1.0.3 and 1.0.4 submissions never became
   public and do not appear in that version list, so they were bypassed rather
   than recovered. The issue that tracked
   them, openclaw/clawhub#3643, was closed by the reporter on 2026-09-09 once
   1.0.5 went through; nothing upstream was confirmed fixed, and it is the
   fifth report of this shape (#3624, #3466, #3351, #3284), so treat a stalled
   submission as recurring and file a fresh one rather than assuming the
   pipeline is repaired. Release-time publishing is automated (see
   [release-and-distribution.md](release-and-distribution.md)), and the job
   fires only on a published GitHub Release: a skill version bumped after a
   release, as 1.0.7 was, reaches ClawHub with the next one. Do not publish by
   hand to close that gap. The listing records the license as MIT-0 while
   `SKILL.md` declares Apache-2.0, and that is not a listing error to repair:
   MIT-0 is a platform-wide constant in ClawHub's own client
   (`dist/schema/license.js`, and its version schema accepts only `"MIT-0"`
   or null), so every skill published there is offered under MIT No
   Attribution. Publishing to ClawHub means offering this skill on those
   terms in addition to the repository's Apache-2.0; do not try to edit the
   field, and do not read the mismatch as drift. The public catalog
   already has an add-watermark skill; this is the removal counterpart, framed
   as the user's own content. ClawHub disallows deception, impersonation, and
   fake-engagement install loops.
7. **OpenAI ChatGPT / Codex plugin directory.** Skills-only plugins are a
   supported submission type at
   [platform.openai.com/plugins](https://platform.openai.com/plugins). This is
   the largest consumer surface that actually reviews listings. It needs Apps
   Management write access, a verified Platform identity, public website /
   support / privacy / terms URLs, starter prompts, and five positive plus three
   negative test cases. Claude marketplace approval does not transfer. Contact
   your OpenAI partner before submitting: this skill's core value requires local
   execution, arbitrary file access, and optional hardware access, which the
   official guide routes through that product-specific review. Convert with
   [Submit your Claude Code plugin to OpenAI](https://developers.openai.com/plugins/guides/submit-claude-plugin).
8. **AgenticSkills.** SUBMITTED 2026-09-09 as review issue 174 under AI/ML
   Development. The submission carries the user-owned-content boundary and
   links to the repository rather than hosting a separate copy.
9. **AI Skills Catalog.** SUBMITTED 2026-09-09 as reference 2 under Video Media,
   with Codex as the primary agent and every supported harness marked compatible.
   The catalog reviews source quality, security, license, and compatibility
   before publication.

Do not treat Smithery or other MCP directories as skill catalogs. Do not open
a second near-identical skill to farm installs. The conversion path that
matters is: agent cannot run CUDA locally, so it sends the user to raiw.cc.

## When to edit the skill

Update `SKILL.md` and `references/` in the same change as any CLI edit that
alters:

- command names or routing;
- extras a command needs;
- no-signal or exit-code behavior;
- registered visible mark keys;
- the CUDA-only invisible-image rule;
- the intended-use boundary.

Keep `SKILL.md` under 500 lines. Put flag lists in `references/`. Do not copy
calibration numbers or retired pipeline names into the skill.

Write for every harness and a weaker model, not only Claude Opus:

- third-person description, keywords first, so a truncated listing still matches;
- a numbered checklist in `SKILL.md`;
- `scripts/probe.py` for CUDA, ffmpeg, CLI, and installer detection. It measures
  CAPABILITY, not presence: it writes a blank PNG with `zlib` and `struct` and
  runs the installed `visible` on it, because exit `2` is a state only a working
  pixel stack reaches. Reporting a found binary as ready is what sent an agent
  into a missing-cv2 crash on the Homebrew build, which carries no extras. It
  also compares the installed release against `MIN_CLI_VERSION`, so a stale CLI
  is named instead of being reported as a broken flag. Video capability is
  checked separately in the installed CLI's Python environment:
  `video_stacks.pixels` reports `ok`, `missing`, or `unknown`, and
  `video_stacks.invisible` reports `installed`, `missing`, or `unknown`.
  These checks load no weights. An unrecognized launcher leaves video
  guidance unverified rather than inferring readiness from image support;
- uv, pipx, and pip as installer fallbacks;
- forward slashes only;
- no `allowed-tools` (that field is experimental and host-specific).

Raise `MIN_CLI_VERSION` in `scripts/probe.py` in the same change as any reference
that names a command, flag or value the previous release lacked. Left behind, it tells an agent its
CLI is current right before it types an option that build rejects.

Bump `metadata.version` in `SKILL.md` plus `version` in
`skills/.claude-plugin/plugin.json` and `.claude-plugin/marketplace.json` when
the skill instructions change. That version is independent of the PyPI package
version.

## Behavior evals

The deterministic suite proves the skill's prose matches the CLI. It cannot prove
the only thing a skill exists for: that a model reaches for it on a real request
and then obeys it. `scripts/skill_eval.py` measures that.

Each case builds a throwaway project with the skill installed and a recording
stand-in for the CLI first on `PATH`, runs a headless agent on one user prompt,
and grades the recorded command trace. Grading is mechanical wherever it can be:
"did it run `all` on a machine the probe said writes nothing" is a fact in the
trace, and "did it name an output file that is not on disk" is how a fabricated
result is caught without asking a model to judge itself.

```bash
python3 scripts/skill_eval.py --model haiku --repeat 3
python3 scripts/skill_eval.py --case visible_ru --model sonnet --out /tmp/eval.json
```

Cases live in `data/evaluations/skill/cases.json`; each carries the prompt, the
scenario the stand-in replays, and the expectations. Run outputs stay outside the
repository, per `data/README.md`. This never runs in `maintain.sh`: it spends model
tokens and needs the `claude` CLI.

Agents are nondeterministic, so the report is a pass RATE per check over `--repeat`
runs, never one verdict. Weak models are the point: run haiku before opus.

Four things the first runs found, all on haiku and none visible to the
deterministic suite:

- a skill that never triggered on "скажи, видно ли, что она сгенерирована ИИ",
  because the description sold removal and not the question;
- a finished-looking removal report, with a size table for a file nobody wrote,
  on a machine whose pixel stack was missing;
- `--mark gemini-sparkle`, a key that does not exist, guessed instead of read;
- a removal command run to answer a question that only needed `identify`.

`tests/test_skill_evals.py` keeps the harness honest: the stand-in's exit codes are
compared against the real CLI where the state is reproducible locally, the grader is
driven over hand-built traces, and every case is checked for internal consistency. A
scenario where `identify` names a visible mark that `visible` then cannot find sends
any agent into a retry loop and measures the case, not the skill.

## Validate

```bash
claude plugin validate .
claude plugin validate --strict .
claude plugin validate ./skills
```

The suite also checks frontmatter and manifest agreement in
`tests/test_agent_skill.py`, plus two things a catalog cannot see: that the
skill's prose still matches the live click application (mark keys, flag choices,
batch modes, exit codes, extras) and that a machine without the pixel stack gets
an install hint from the CLI, a caveat from `identify`, and `pixel_stack:
missing` from the probe.
