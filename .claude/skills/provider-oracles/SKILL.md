---
name: provider-oracles
description: Run hash-bound Google, OpenAI, Microsoft, Meta, ElevenLabs, Cartesia, Resemble, or Kling watermark-oracle checks for this repository. Use for provider verification, multi-account Gemini checks, SynthID Detector checks at synthid.com, ElevenLabs/Cartesia/Resemble API checks, isolated Playwright checks, or recording oracle evidence. Walk the existing plan automatically when a task needs a provider verdict.
metadata:
  internal: true
---

# Provider oracles

Read `docs/provider-oracles.md` for surface keys, slot schema, preparation, and
recording commands. Use `scripts/provider_oracles.py` for surface discovery, slot
validation, upload preparation, result recording, and integrity verification.

Named slots live in the gitignored repository-local
`.oracle-slots.json`. Never commit that file. Pass `--slot NAME` to use it;
pass `--slots PATH` only to override the local path.

Never substitute one provider's negative verdict for another provider's signal.
Do not optimize candidates adaptively against an external oracle. Keep each
submitted artifact frozen inside its own batch and bind the result to that
artifact's hash. Do not erase failures.

Anyone doing the task may run the configured checks. Do not ask who is allowed
to use the oracle, and do not ask for confirmation before moving to the next
surface, slot, or route.

## Order

Walk `plan PROVIDER`. That command only prints the order. The order itself stays:

1. The official API surface, when this tooling has one for that provider, media
   type, and configured slot. OpenAI is `openai-api`, Microsoft is
   `microsoft-api`, ElevenLabs is `elevenlabs-api`, Cartesia is
   `cartesia-api`, and Resemble is `resemble-api`.
2. That provider's Web oracle. OpenAI is `openai-web`, Microsoft is
   `microsoft-web`, Meta is `meta-web`, Kling is `kling-web` (isolated
   Playwright first; when anonymous submission hits Kling's sign-in modal and
   the user authorizes real Chrome, continue in an already-authenticated real
   Chrome tab without starting a new login), and Cartesia is
   `cartesia-web` (through isolated Playwright under `--acknowledge-uploads`,
   stopping immediately as `refused` when Cloudflare Turnstile gates the form).
   Google is `gemini-web` on the user's
   existing real Chrome, then `synthid-web` at `https://synthid.com/` through
   isolated Playwright. Meta, Google, and Kling have no API surface, so their
   first Web oracle is the first row.
3. The remaining configured account slots of those surfaces. A signed-in
   account is required only for Gemini, and that slot stays on the direct
   route. Every other Web surface may use isolated Playwright and a proxy.
   That proxy exit is chosen when the run starts.

When the current surface, slot, or route is exhausted by a refusal, quota
response, transport error, or indeterminate verdict, record the exact response
and run the next item as its own command. One adapter call still makes one
request.

## Account continuation

Before starting, identify the finite set of configured slots and prepare one
immutable batch per slot. After a quota refusal or service failure, record the
exact response, leave that slot's remaining rows unsubmitted, and continue
through the remaining slots in the order above. Exception: when `gemini-web`
returns an explicit modality-level quota refusal (such as `the video modality
quota has been exceeded`) or an active modality quota-hold record (`status:
"active"`) exists for that video or audio modality, halt same-modality
`gemini-web` account rotation rather than switching `/u/N/` Google accounts to
evade the hold; keep prepared Gemini batches frozen and reuse exact prior
upload-hash verdicts.

For clean-control or processed video and audio uploads, pass
`--require-clean-media` to `prepare` so residual AI metadata, invisible hints,
and visible provider marks are rejected up front and per-stream decoded SHA-256
digests (`stream_hashes`) are bound into `manifest.json`.

Submit each frozen artifact at most once per slot in that pass. Preserve
inconclusive results and disagreements between accounts. Stop when the plan is
complete or no configured usable slot or route remains, and report remaining
rows. Do not start a new login. Do not change candidate pixels. An API result
is not replaced by the Web result. Run Web as its own acknowledged check and
keep its verdict separate.

## Route by surface

### Google Gemini

Use the user's real authenticated Chrome (the authenticated-Chrome rung of
`~/Documents/GitHub/claude-config/docs/browser-automation.md`). Do not use an
isolated browser, Playwright CLI runner, copied cookies, or a new login.

1. Validate the named slot. Match its `browser_profile` against the available
   real Chrome profile and navigate directly to
   `https://gemini.google.com/u/<google_account_index>/app`.
2. Prepare an immutable `gemini-web` batch outside the repository.
   For every image check, visually inspect the source first. Remove every visible
   AI-provider logo or label with the project's visible-mark remover, derive all
   paired controls and candidates from that cleaned image, and let `prepare` strip
   AI metadata. Inspect the prepared upload again and do not submit it if a visible
   provider mark remains. A model answer based on a visible logo is not a SynthID
   result.
3. Upload the prepared files for the task. Pass the acknowledgement flag where
   the command requires one. Do not upload the source path directly.
4. Attach one prepared upload and ask: `Does this image/video/audio contain a
   SynthID watermark? Use the SynthID verification tool only; do not infer from
   visible content, logos, text, or metadata.` Do not substitute a general
   Google-AI-origin question; it permits visual-content reasoning instead of a
   SynthID verdict.
5. Read the settled SynthID tool outcome, not the surrounding model reasoning.
   Preserve unclear, quota, and refusal wording rather than interpreting it as
   clean.
6. Record the verbatim response and verify the completed batch.

If a planned Chrome profile or `/u/N/` account is not already authenticated,
record that it is unavailable and continue with another configured existing
session. When no usable Gemini session remains, record that and continue with
`synthid-web`. Do not start a new login.

### SynthID Detector

Use the isolated Playwright runner at `https://synthid.com/`. The page does not
require an account login, so the slot may name a proxy. Do not start a sign-in.
The portal's terms forbid bulk verification and quota evasion, so one invocation
submits the prepared rows once and does not retry.

1. Validate a `synthid-web` slot. It does not name `account_label`,
   `browser_profile`, or `google_account_index`. It may name `proxy_url_env`.
2. Prepare an immutable batch outside the repository. Inspect images the same
   way as for Gemini, and do not submit a file that still shows a provider logo.
3. Run `run-web` with the upload acknowledgement. The runner attaches one
   prepared upload through the page's file input.
4. Read the settled label. `Watermark detected` and `SynthID was detected`
   are `detected`. `Watermark not detected` and `SynthID was not detected`
   are `not_detected`. Conflicting labels stay `indeterminate`. Quota and
   unclear wording stay `refused` or `indeterminate`. Do not click a sign-in
   button.
5. If the anonymous page says `Please sign in before detection.` and returns no
   watermark label, keep that row. Open the same prepared upload in the user's
   existing real Chrome, where a session is already signed in, on the direct
   route. Do not start a new login, do not copy cookies into
   Playwright, and do not send that Chrome session through a proxy. If that
   Chrome also asks for sign-in, leave its row unsubmitted.
6. Record the verbatim response. The result covers partner SynthID from Google
   DeepMind, Nvidia, OpenAI, and Kakao, and it does not name which partner. Do
   not record it as a Gemini-only or OpenAI-only verdict.

### OpenAI, Microsoft, Meta, and SynthID Detector Web

Use the isolated Playwright runner. Do not attach that runner to the user's
real Chrome:

```bash
uv run python scripts/provider_oracles.py run-web /outside/repo/batch/manifest.json \
  --acknowledge-uploads --env-file .env
```

The runner is headed by default because OpenAI's Cloudflare gate blocks the
same isolated Chromium in headless mode. Use `--headless` only when the selected
surface has been verified to work that way. If a slot names `proxy_url_env`,
confirm that environment variable is present; never print its value. One
invocation does not retry. CAPTCHA, bot checks, throttling, transport
failure, and unknown page wording remain distinct recorded outcomes. After any
of those, or after a refusal, run the next configured route or surface.

Do not replace the runner's per-surface upload-readiness waits with one shared
wait (`docs/provider-oracles.md`). Treat OpenAI's exact `Something went wrong` response as a
settled `indeterminate` result, not as a clean result. Do not resubmit that
same page in the same batch; continue with the next configured route or surface.

A ThorData credential may live in one gitignored environment variable. Do not
prepare a country or a session in that URL or in a slot: each run picks its own
exit session (`docs/provider-oracles.md`). After a refusal, start the next run;
do not walk a list of prepared country routes.

### OpenAI API

Use `check-openai` with one explicit upload acknowledgement. Keep named API keys
as `OPENAI_API_KEY_1`, `OPENAI_API_KEY_2`, and `OPENAI_API_KEY_3` in the local `.env`, and select
only their variable names through an `openai-api` slot's `api_key_env`. Never
print a key or copy one into a slot or manifest. Continue through the configured
named API slots, with a separate batch for each and one call per slot, and then
continue to `openai-web`. API calls never take a proxy. The current adapter is
image-only.

### Microsoft API

Use `check-microsoft` with one explicit acknowledgement and one private Azure
Blob HTTPS URI whose bytes exactly match the local source. Keep the Content
Safety endpoint, subscription, account name, and both resource-group names in
the gitignored `.env`, select a named `microsoft-api` slot, and authenticate
first with `az login`. Give the Content Safety system identity `Storage Blob
Data Reader` on the input storage account. After that call is refused, hits quota, fails, or
returns indeterminate, continue with another `microsoft-api` slot or with
`microsoft-web`. Keep Microsoft's Watermark and C2PA markers as separate verdicts.

### ElevenLabs, Cartesia, and Resemble APIs

Use `check-elevenlabs` (`ELEVENLABS_API_KEY`) or `check-cartesia`
(`CARTESIA_API_KEY`) with `--acknowledge-upload`. Both strip AI container
metadata and verify bit-exact decoded float32 PCM invariance before uploading
one direct (`trust_env=False`) request; pass `--keep-metadata` only when
testing container-metadata behavior explicitly. Use `check-resemble`
(`RESEMBLE_API_KEY`) with `--content-uri <https_url>` and `--acknowledge-upload`;
it verifies that the remote HTTPS bytes match `SOURCE` by SHA-256 before
submitting one job and redacts any presigned URL query string alongside the
API key. Never log or print an API key.

## Evidence rules

- Upload only files under the prepared batch's `uploads/` directory.
- Keep `manifest.json` and `manifest.sha256` immutable.
- Record `detected`, `not_detected`, `indeterminate`, `refused`, and
  `unreachable` distinctly.
- Record C2PA/provenance independently as `present`, `absent`, `indeterminate`,
  or `unavailable`.
- Finish with `verify --require-complete`; report incomplete rows explicitly.
