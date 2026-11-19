# Provider oracle tooling

`scripts/provider_oracles.py` is the development-only entry point for the eight
remote watermark-oracle families used by this project. It drives Gemini through
the user's real authenticated Chrome and drives every other public Web surface,
including SynthID Detector, Kling AI Content Detection, and Cartesia Watermark
Detector, through isolated Playwright. It keeps provider scope, execution identity,
immutable media, and verbatim results in one format without exposing any oracle
through the installed CLI or public Python API.

## Surfaces

| Surface key | Provider signal | Access | Project evidence |
| --- | --- | --- | --- |
| `cartesia-api` | Cartesia Sonic watermark | Official Cartesia Watermark Detection API (`X-API-Key`) | None recorded |
| `elevenlabs-api` | ElevenLabs waveform watermark | Official ElevenLabs Content Provenance API (`xi-api-key`) | Audio |
| `openai-api` | OpenAI SynthID | Official Content Provenance API | Images |
| `microsoft-api` | Microsoft InvisMark | Azure Content Provenance Detection API | Images |
| `resemble-api` | Resemble Perth v1/v2 (+ auxiliary SynthID) | Official Resemble Watermark Detection API (`Bearer`) | None recorded |
| `cartesia-web` | Cartesia Sonic watermark | Public Cartesia detector (Cloudflare Turnstile gated) | None recorded |
| `gemini-web` | Google SynthID | User's authenticated Chrome | Images and videos |
| `kling-web` | Kling blind watermark (H.264 SEI + TC260) | Isolated Playwright probe; user's authenticated Chrome on sign-in gate | Images and videos |
| `synthid-web` | Partner SynthID | Anonymous Playwright page at synthid.com | None recorded |
| `meta-web` | Meta Content Seal (`VideoSeal v4`) | Anonymous Meta identification page | Images and videos |
| `microsoft-web` | Microsoft InvisMark | Public Azure AI validation page | Images |
| `openai-web` | OpenAI SynthID | Public web verifier | Images |

## Surface priority

The default selection order is API first, then Web. OpenAI therefore plans
`openai-api` before `openai-web`, Microsoft plans `microsoft-api` before
`microsoft-web`, and Cartesia plans `cartesia-api` before `cartesia-web`.
ElevenLabs and Resemble have only an API surface. Google, Meta, and Kling
currently have no usable API adapter in this tooling, so each of their Web
surfaces is primary. Google has two: `gemini-web`
and `synthid-web` at `https://synthid.com/`.

Inspect the machine-readable order before preparing a batch:

```bash
uv run python scripts/provider_oracles.py plan openai --json
uv run python scripts/provider_oracles.py plan microsoft
uv run python scripts/provider_oracles.py plan google
```

API-first is the default order. Run the API surface when this tooling has a
usable route for the provider, media, and local configuration. Google, Meta,
and Kling have no API surface, so each of their Web oracles is primary. For Google that
is `gemini-web`, then `synthid-web`. After a refusal, quota response, transport
failure, or indeterminate verdict, record that result and continue with the next
surface as its own acknowledged command, then the remaining configured account
slots of those surfaces. One call does not upload the same file to the next
surface. Keep the verdicts separate. Do not ask for confirmation before the
switch. Anyone doing the task may run the configured checks.

The old Google SynthID Detector portal at `labs.google/synthid` does not work.
On 2026-09-07 that entry point redirected a signed-in browser to a Google 404;
on 2026-09-23 it redirected to `deepmind.google.com/frontiers/synthid`, which
also returned 404. Historical corpus rows may still name `synthid-portal`; that
value describes how an old result was obtained and is not an operational surface.

The public SynthID Detector is a different site, [synthid.com](https://synthid.com/),
and it is the `synthid-web` surface. On 2026-10-17 the page title was
`SynthID Detector`. After the terms dialog, the uploader was present without a
completed sign-in in that session. The Account menu still offered Sign in, and
no file was submitted, so this check does not show whether a result requires an
account. The live FAQ named the partners as Google DeepMind, Nvidia, OpenAI,
and Kakao, and it did not list Apple. It defined two settled labels:
`Watermark detected` and `Watermark not detected`. The file input accepted
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.avif`, `.heic`, `.heif`, `.tiff`,
`.tif`, `.gif`, `.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`, `.m4a`, `.mp4`,
`.mov`, and `.webm`. This tooling still prepares only the project's existing
JPEG, PNG, WebP, MP4, MP3, and WAV suffixes. The portal's additional terms
forbid bulk verification and quota evasion. `run-web` may submit one prepared
upload through isolated Playwright, with a proxy when the slot names one, and
it does not start a sign-in. If the page says `Please sign in before detection.`
and returns no watermark label, keep that row and open the same prepared upload
in the user's existing real Chrome. That session stays on the direct route, with
no proxy and no new login. If that Chrome also asks for sign-in, leave its row
unsubmitted. A `Watermark detected` result is partner SynthID, not a Gemini-only
or OpenAI-only verdict.
Google's newer Search verification entry point did not provide an independent
oracle allowance in the session tested on 2026-09-13. Lens accepted the
metadata-stripped, previously Gemini-positive image
`sha256:c86feaf788df2c423eeb53fcaee52b74f12a8e83829f94414e6dec2966e6e62e`,
but ordinary Lens results only described and matched the visible content. An
explicit `Is this made with AI? Check whether it contains SynthID.` request in
Search AI Mode invoked the technical verifier and returned `The verification
tool's daily limit has been exceeded.` after that session's Gemini image
verification allowance was already exhausted. Record this as a quota refusal,
not a watermark verdict and not a second quota pool.

A second Search check on 2026-09-15 used a real Qwen 0.15 no-face/no-text
variant rather than a control. Lens and Search AI Mode accepted the upload, but
both answered from visible-content reasoning and Web matches instead of
returning a SynthID result. The exact Google-AI verification question even
misattributed the file to an Imagen 4 Fast demonstration and invented a matching
prompt. Search is therefore neither a usable independent oracle nor a substitute
verdict when its response does not explicitly report the verification tool's
SynthID outcome.

Fresh independent-session rechecks on 2026-09-14 split: one returned `detected`
for the same control while another returned `unable to process your request due
to a tool error or quota limit`. A later session returned five usable real-variant
verdicts and refused the sixth, while the other tested sessions still refused real
variants. Recovery is therefore not a global calendar event. Google [documents approximately ten
image checks in a rolling 24-hour window](https://support.google.com/gemini/answer/16722517?hl=en-GB),
but that approximation is orientation rather than a per-session reset schedule.
When a campaign is conserving that allowance, probe each explicitly selected
session with its next real pending variant and record a refusal without assigning
a watermark verdict. Reserve a positive-control probe for surface diagnosis, not
ordinary batch allocation.

A second independent-session sweep used one real pending variant per session and
again obtained no watermark verdict: three responses explicitly reported a temporary
verification quota, while one reported a verification tool-call error. Four completed immutable result batches preserve
the exact uploads and settled responses. Twenty-four newly generated
no-face/no-text scene variants form six four-rung ladders; the six lowest rungs
were prepared in immutable batches, and none had been submitted at that point.
The later completed sweep followed the planned lowest-rung-first escalation. Do
not treat batch preparation or visual quality scores as SynthID results.

A session recovered earlier than the estimate derived from the oldest usable call
preserved locally. Its next real pending
variant, Chroma 0.16 on the face carrier, returned `detected`; the immutable batch
is complete and hash-verified. This demonstrates why estimated rolling-window
times are retry guidance rather than reset timestamps, and closes the tested
face bracket at `0.16 detected -> 0.22 not_detected` without spending a duplicate
request on the already settled 0.22 file.

The same recovered session returned `not_detected` for Chroma 0.40 on the Cyrillic
carrier, then explicitly quota-refused SDXL CJK 0.35. The other selected sessions
were checked with real pending CJK variants, not controls, and returned quota or
tool limitations. One settled response exposed a request-specific instruction to
wait 18 hours. Treat that as the retry point for that session and request class, not as a replacement
for Google's documented approximate rolling 24-hour allowance or a universal
reset time. All four sessions were unavailable again at their latest checks.
Because one response did not distinguish service failure from quota, its real
Chroma CJK 0.40 candidate was retried. This time the
verifier explicitly named a temporary image-analysis quota. The refusal was
recorded and the same bytes were prepared in a new immutable retry batch; no
watermark verdict was inferred. One session was probed once more with a
real Qwen 0.15 no-face/no-text candidate. It returned a verification-tool
technical error rather than a SynthID result. That failure was recorded as
`refused`, the batch was hash-verified complete, and a fresh immutable retry
batch was prepared for the evidence-based session window.

All four sessions returned usable results again on 2026-09-15, earlier than the
previous retry guidance. The remaining real
variants were submitted adaptively without controls. SDXL CJK remained detected
at 0.35, 0.40, and 0.45 and cleared at 0.50. Chroma CJK remained detected at
0.40 and cleared at 0.50. The two no-face/no-text carriers cleared Qwen at 0.15;
Chroma cleared the flat carrier at 0.12 and the photographic carrier at 0.30
after detections at 0.12 and 0.20; SDXL cleared the photographic carrier at 0.15
and the quality-qualified flat carrier at 0.35. One quota refusal and one tool-call
failure encountered while routing SDXL CJK 0.40 were kept as separate completed
batches; another session then returned the usable `detected` result for the same
upload hash. Every settled batch passed immutable-source, upload, manifest, and
result verification. The observed early recovery reinforces that a retry time is
guidance for the next real request, not a promised reset.

A same-day production-boundary replication then used three new independent
Gemini-generated controls: busy CJK plus many faces, flat CJK without faces, and a
portrait plus dense CJK. Only processed variants were submitted. Across Qwen 0.35,
Chroma 0.50, and SDXL 0.50, all nine profile-by-carrier cells returned
`not_detected`. The night-market lower rungs Chroma 0.40 and SDXL 0.45 also returned
`not_detected`; they do not supersede the earlier harder-source detections at those
rungs. Two sessions returned explicit quota failures during this pass, one
candidate retry also hit quota, and a byte-normalized duplicate after the usable
SDXL result returned a tool-call error. Those four calls were recorded as
`unreachable`, not spliced into the nine usable verdicts. All result batches were
hash-verified.
Google did not expose a reset timestamp: the returned quota text supports only
waiting and retrying, not a universal 24-hour claim.

On 2026-09-27 one account refused after its tenth image check of the day, first
as `a tool error or quota limit` and later as `the usage quota for checks on
similar images has been exceeded`. Three other signed-in accounts in the same
Chrome profile still answered at the same time, so the allowance is per
account; the "similar images" wording is Google's and was not tested further.
The refused row went to another account only on the operator's explicit
instruction, recorded under that account's own slot.

The documented Vertex AI Media Studio verification flow was also unavailable
in the live console on 2026-09-14. Its old console URL redirected to Agent Media
Studio after the Vertex-to-Agent Platform rename. The current Image playground
was inspected in eligible configurations; its task menu exposed generation and image
editing but no watermark upload or validation action. The deprecated Python SDK
can still resolve `imageverification@001` to its publisher-model endpoint. A
2026-09-15 request reached `predict`, but the real Qwen 0.15 variant returned
HTTP 403 `The caller does not have permission` in both tested configurations. The current
Python reference still documents `WatermarkVerificationModel.verify_image`, but
the backing model was access-restricted in both tested configurations and the SDK path is
past its announced removal date. Recheck the live console and a real pending
variant before adding Search or Agent Media Studio to the surface table.
Adobe Inspect and local C2PA, DWT-DCT, TrustMark, and research detectors are
useful evidence readers, not interchangeable provider pixel-watermark oracles.

The surface capability and this project's evidence coverage are separate. The
current Google, Microsoft, and Meta pages, including `https://synthid.com/`,
accept image, video, and audio inputs; the OpenAI API accepts image and audio.
A media type does not become certified merely because a surface accepts it.
`synthid-web` therefore lists image, video, and audio as accepted media and
records no project evidence until a batch is verified. Run `list --json` to read
both fields.

## Execution slots

Keep account, browser-profile, and network labels in the gitignored repository-local
`.oracle-slots.json`. Labels and proxy environment-variable names are provenance,
not credentials. Do not commit this file or put cookies, tokens, passwords, proxy
URLs, or account secrets in it. When `--slot` is present, the CLI reads this path by
default; `--slots PATH` explicitly overrides it.

```json
{
  "format_version": 1,
  "slots": [
    {
      "name": "gemini-personal",
      "surface": "gemini-web",
      "account_label": "google-account-a",
      "browser_profile": "Your Chrome",
      "google_account_index": 0,
      "network_label": "home"
    },
    {
      "name": "gemini-research",
      "surface": "gemini-web",
      "account_label": "google-account-b",
      "browser_profile": "Your Chrome",
      "google_account_index": 2,
      "network_label": "office"
    },
    {
      "name": "synthid-proxy",
      "surface": "synthid-web",
      "account_label": null,
      "browser_profile": null,
      "network_label": "thordata",
      "proxy_url_env": "THORDATA_PROXY"
    },
    {
      "name": "openai-web-proxy",
      "surface": "openai-web",
      "account_label": null,
      "browser_profile": null,
      "network_label": "thordata",
      "proxy_url_env": "THORDATA_PROXY"
    },
    {
      "name": "openai-api-1",
      "surface": "openai-api",
      "account_label": "project-1",
      "browser_profile": null,
      "network_label": "home",
      "api_key_env": "OPENAI_API_KEY_1"
    },
    {
      "name": "openai-api-2",
      "surface": "openai-api",
      "account_label": "project-2",
      "browser_profile": null,
      "network_label": "home",
      "api_key_env": "OPENAI_API_KEY_2"
    },
    {
      "name": "openai-api-3",
      "surface": "openai-api",
      "account_label": "project-3",
      "browser_profile": null,
      "network_label": "home",
      "api_key_env": "OPENAI_API_KEY_3"
    },
    {
      "name": "microsoft-api-1",
      "surface": "microsoft-api",
      "account_label": "content-safety-project",
      "browser_profile": null,
      "network_label": "home"
    }
  ]
}
```

A routed batch selects one slot explicitly. A Gemini slot must name the Google
account, its real Chrome profile, and its nonnegative `/u/N/` account index. An
API slot must name its account or project and cannot name a browser profile. A
Playwright slot for `openai-web`, `microsoft-web`, `meta-web`, or `synthid-web`
does not name an account or a browser profile. The
tool records the network route that the operator has already established.
Those four pages may name `proxy_url_env`; Playwright reads the
secret proxy URL from that environment variable without writing or logging it.
The country and session are not part of that preparation. Each run replaces
them with a new session in a country in Europe or the United States.
A signed-in account is required only for `gemini-web`. API calls stay on the
direct SDK transport and do not take `proxy_url_env`. Every other Web surface
uses isolated Playwright and may use a proxy. The tool does
not merge sessions inside one adapter call. Account continuation is orchestrated
by the skill, not by an individual adapter call. A request to verify files with
a provider covers its configured account slots, surfaces, and network routes.
Anyone doing the task may run those checks. Identify the finite set of
configured slots before starting and prepare one immutable batch per slot.
Record quota refusals as `refused` and service failures according to their
actual response, leave remaining rows in that slot unsubmitted, and continue
with the next slot, surface, or configured route. Exception: when `gemini-web`
returns an explicit modality-level quota refusal (such as `the video modality
quota has been exceeded`) or an active modality quota-hold file (`status:
"active"`) is on record for video or audio, stop same-modality `gemini-web`
account rotation instead of switching Google `/u/N/` accounts to evade the
hold; keep prepared Gemini batches frozen and unsubmitted until the hold clears
and reuse exact prior upload-hash verdicts. Pass `--quota-hold PATH` to
`run-web` (or call `assert_no_active_quota_hold`) to enforce active holds
mechanically. Do not ask for confirmation before an allowed surface switch.

Submit each frozen artifact at most once per slot in one pass. Keep every
verbatim refusal, inconclusive reply, and disagreement between accounts. Stop
when the plan is complete or no configured usable slot, surface, or route
remains, and report anything still open. Do not start a new login. Each
submitted artifact stays frozen inside its own batch. Do not change candidate
pixels during the continuation.

Each `openai-api`, `elevenlabs-api`, `cartesia-api`, and `resemble-api` slot
must name exactly one `api_key_env`. Put the corresponding values in the
gitignored local `.env`, never in the slot JSON:

```dotenv
OPENAI_API_KEY_1=replace-with-first-key
OPENAI_API_KEY_2=replace-with-second-key
OPENAI_API_KEY_3=replace-with-third-key
THORDATA_PROXY=http://td-customer-USERNAME:PASSWORD@HOST.pr.thordata.net:9999
```

The tool reads only the variable selected by the requested slot. A nonempty
process environment value overrides the matching `.env` entry. It never prints
the value. One call reads only its selected slot. After authentication,
entitlement, quota, or transport failure, continue with the next configured
slot in a new call. The no-slot compatibility path reads
`OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `CARTESIA_API_KEY`, or
`RESEMBLE_API_KEY` for the matching surface.

Several Google accounts may share one real Chrome profile: give them distinct
slot names and `google_account_index` values while keeping `browser_profile` the
same.

ThorData encodes the exit in the username with its [country parameter](https://doc.thordata.com/doc/proxies/residential-proxies/location-settings/country-region)
and a [session id](https://doc.thordata.com/doc/proxies/residential-proxies/session-control).
A stored `country-XX` or `sessid` is not the exit this run will use. The runner
discards both and appends one new session and one country from Europe or the
United States. That pair stays for the browser context of the run. The next run
chooses again, including after a refusal. Gemini and API slots never accept
`proxy_url_env`. Anonymous Playwright slots may.

Validate and inspect the slots:

```bash
uv run python scripts/provider_oracles.py slots .oracle-slots.json
uv run python scripts/provider_oracles.py guide gemini-web --slot gemini-personal
```

## Immutable manual batches

Prepare a batch outside the repository:

```bash
uv run python scripts/provider_oracles.py prepare gemini-web input.png \
  --output /tmp/gemini-check-001 \
  --slot gemini-personal
```

For PNG, JPEG, and WebP, preparation strips AI provenance metadata and proves
that the decoded RGBA pixels and format did not change. This prevents a C2PA
result from masquerading as a pixel-watermark result. Video and audio are copied
byte for byte; the manifest explicitly records that their metadata was not
stripped. Pass `--require-clean-media` when preparing clean-control or
processed audio/video uploads so `prepare` verifies zero residual AI metadata,
invisible watermark hints, or visible provider logos and records decoded
per-stream SHA-256 digests (`stream_hashes`). Use the matched-control protocols
in [`synthid.md`](synthid.md) for video claims.

Before preparing any image for a SynthID oracle, visually inspect it and remove
every visible AI-provider logo or label with the project's visible-mark remover.
Derive both the positive control and every processed candidate from that same
visible-clean source, then run `prepare` to strip metadata losslessly. Inspect the
prepared upload again and refuse the request if a visible provider mark remains.
This separation is mandatory: Gemini can answer from a visible sparkle or from
image-content reasoning without returning a SynthID tool verdict. Record such an
answer as `indeterminate`, never `detected`. The source-to-visible-clean pixel
change belongs in the experiment report; `prepare`'s pixel-preservation claim is
only from that cleaned source to the metadata-stripped upload.

The batch contains:

- `manifest.json`, immutable surface, slot, source, upload, and hash identities;
- `manifest.sha256`, the manifest digest;
- `uploads/`, sanitized images or byte-identical non-image media;
- `results.json`, the only mutable document.

Submit each upload in index order, one file per fresh provider result. Preserve
the settled response verbatim, including an unclear or refused response. Record
watermark and provenance separately:

```bash
uv run python scripts/provider_oracles.py record /tmp/gemini-check-001/manifest.json \
  --artifact-id 000-0123456789abcdef \
  --watermark-result detected \
  --provenance-result absent \
  --raw-response "SynthID watermark detected" \
  --checked-at 2026-09-07T18:30:00Z

uv run python scripts/provider_oracles.py verify \
  /tmp/gemini-check-001/manifest.json --require-complete
```

The verifier rereads every source and upload hash, binds result rows to exact
upload bytes, checks timezone-aware timestamps, and never interprets
`indeterminate` or `refused` as `not_detected`.

## Web automation

Install Playwright's isolated Chromium once after the development dependencies:

```bash
uv sync --extra dev
uv run playwright install chromium
```

Gemini is excluded from the standalone runner because its checks use the
user's signed-in Chrome account. Invoke the project `provider-oracles` skill
to drive that existing profile at the named `/u/N/` account. For `synthid-web`,
`run-web` first opens `https://synthid.com/` in isolated anonymous Playwright,
attaches one prepared upload through the page's file input, and reads the
explicit `Watermark detected`, `SynthID was detected`, `Watermark not detected`,
or `SynthID was not detected` label. Conflicting labels remain indeterminate.
The runner closes its isolated session after the first refused, indeterminate,
or unreachable result, leaving the remaining manifest rows unsubmitted. Its
summary reports `complete: false` for that partial pass. The portal has no
`/u/N/` index. If the page says `Please sign in before detection.` and
returns no watermark label, keep that row and continue the same prepared upload
in the user's existing real Chrome. That session is direct: no proxy, no new
login, and no cookie copy into the isolated browser. If that Chrome also asks
for sign-in, leave its row unsubmitted.

For Gemini, ask the direct question stored in the batch manifest: `Does this
image/video/audio contain a SynthID watermark? Use the SynthID verification
tool only; do not infer from visible content, logos, text, or metadata.` A
general question about whether Google AI created or edited the media is not
equivalent: it may elicit origin or visual-content reasoning without a SynthID
verdict. SynthID Detector has no prompt; its settled labels are the verdict.

OpenAI Web, Microsoft Web, Meta, and SynthID Detector run through a fresh
isolated Playwright browser. A direct-route example:

```bash
uv run python scripts/provider_oracles.py prepare meta-web input.png \
  --output /tmp/meta-check-001 \
  --slot meta-direct

uv run python scripts/provider_oracles.py run-web \
  /tmp/meta-check-001/manifest.json --acknowledge-uploads
```

For an OpenAI, Microsoft, Meta, or SynthID Detector slot that names a ThorData
credential, keep that URL in the gitignored `.env`. The run chooses the country
and the session:

```bash
uv run python scripts/provider_oracles.py run-web \
  /tmp/openai-check-001/manifest.json --acknowledge-uploads \
  --env-file .env
```

The runner reuses one context and one route for the batch, performs a fresh page
navigation before each upload, does not retry, and stores the settled page text.
Each completed verdict is persisted immediately with its own observation
timestamp. A later navigation or upload failure preserves those completed
rows and leaves unobserved rows unresolved.
ThorData [user-and-password endpoints without a `sessid` rotate the exit on each
request](https://doc.thordata.com/doc/proxies/residential-proxies/session-control).
That is unsuitable inside one browser context: page resources can
arrive through different exits and trigger the provider's bot checks. For a
ThorData hostname, each run therefore discards a prepared country and session,
then keeps one new session and one country in Europe or the United States for
that browser context. The next invocation, including a retry after a refusal,
chooses again.
For a selected proxy route it enables certificate handling in that isolated
Chromium launch and context; direct launches keep Playwright's normal TLS
verification. A 2026-09-07 preflight sent one read-only request through each
configured route and observed five distinct egress addresses with the requested
US, DE, GB, NL, and JP country codes. A deliberately invalid country code failed
instead of silently using a random route. The addresses are intentionally not
recorded because residential egress is ephemeral; this proves route selection,
not reachability of any provider page. One observation per surface is not a
reliability baseline.
It is headed by default: a live 2026-09-07 probe reached OpenAI's upload control
that way while the same isolated Chromium received a Cloudflare gate in headless
mode. Use `--headless` only for a surface already verified in that mode. Bot or
quota refusal is `refused`, a transport failure is `unreachable`, and unknown or
changed page wording is `indeterminate`.

Each provider mounts its file input before the upload handler is necessarily
ready. Microsoft and SynthID Detector reach a usable handler at network idle. OpenAI keeps
background requests alive and never reaches network idle, so its observable
readiness seam is the page `load` state instead. Meta keeps background
`/monitoring` requests open and never reaches network idle either, so its
readiness seam is `load` plus visibility of the hydrated `Upload or drag an
image, video, or audio file` dropzone. The adapter waits for the
surface-specific state before assigning the file. OpenAI's static FAQ contains
the lowercase phrase `generated with OpenAI tools`; only the exact title-cased
result label may settle a positive result. Its generic `Something went wrong`
response settles as `indeterminate`, not clean. Meta's live positive wording
changed from `AI signatures from Meta were found` to `AI signatures were found`
by 2026-09-07; both exact forms remain accepted.

Isolation is a transport property, not a verdict requirement. On 2026-09-15,
the isolated OpenAI runner returned one generic error and two Cloudflare
refusals for a prepared three-file batch. Uploading those exact prepared hashes
through an authenticated first-party browser session reached the official verifier
and returned explicit SynthID results. A later Chroma 0.12 upload remained in
`Verifying` for more than 90 seconds and was recorded as indeterminate rather
than clean. The official API then returned HTTP 200 and SynthID `not_detected`
for the same processed source hash. Keep each route in its own immutable batch;
do not splice an isolated-runner failure and a first-party browser or API verdict into
one response.

On 2026-09-16, a rotating ThorData browser batch demonstrated the failure mode:
one upload reached OpenAI's generic error, four encountered Cloudflare, and ten
became unreachable after the browser context closed. Five fresh connections to
the same rotating endpoint had five distinct exits, while five connections with
one sticky session had one exit. After session pinning, one direct route and
five independently prepared country routes all reached the upload handler with
the same prepared test image, but every route returned OpenAI's exact generic
error. Record those responses as `indeterminate`. The result distinguishes the
fixed session-continuity defect from the remaining anonymous-Web availability
problem; further route changes are not a substitute for an API verdict.

## OpenAI API

The one automated path calls the hardened metadata-independent adapter already
covered by `tests/test_openai_provenance.py`:

```bash
uv run python scripts/provider_oracles.py check-openai input.png \
  --acknowledge-upload \
  --slot openai-api-1 \
  --env-file .env
```

The command accepts one image and one explicit acknowledgement, makes at most
one request with SDK retries disabled, and emits the selected slot beside the
result. Its SDK transport sets `trust_env=False`, so `HTTP_PROXY`,
`HTTPS_PROXY`, and related process settings cannot silently proxy an API slot.
On 2026-09-07, a no-upload `models.list()` authentication probe succeeded for
all three configured keys over that direct transport. The model counts are not
recorded because they are operationally volatile. The official response keeps
SynthID separate from C2PA. The current adapter deliberately handles images
only even though the current API surface also accepts audio.

OpenAI documents `not_detected` as absence of a supported signal, not proof that
content was not AI-generated. The API must not become an adaptive removal loss
or automatic search loop. See [`synthid.md`](synthid.md) for the oracle boundary.

The fresh no-face/no-text calibration on 2026-09-15 used three independent
official API requests after the public Web surface stalled. All three returned HTTP 200
with C2PA `not_present`: Chroma 0.12 and SDXL 0.06 were SynthID `not_detected`,
while SDXL 0.04 was `detected`. Together with settled Web results for the same
carrier, the tested 0.02-grid brackets are Qwen `0.04 detected -> 0.06
not_detected`, Chroma `0.10 detected -> 0.12 not_detected`, and SDXL `0.04
detected -> 0.06 not_detected`. The generated source control was not uploaded.

A separate face-plus-multilingual-text carrier then returned Qwen `0.04
detected -> 0.06 not_detected`, Chroma 0.10 `not_detected`, and SDXL 0.04
`not_detected`. The lower Chroma and SDXL results did not change the global
candidate because the no-face/no-text carrier remained the limiting OpenAI
stratum.

## Microsoft API

The direct Microsoft path uses the Azure Content Provenance Detection preview
API. Run `az login` first, then keep the non-secret resource configuration in
the gitignored local `.env`. The script reads both resource keys into memory
through Azure CLI and never writes or logs them:

```dotenv
AZURE_CONTENT_SAFETY_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_CONTENT_SAFETY_SUBSCRIPTION_ID=your-subscription-id
AZURE_CONTENT_SAFETY_RESOURCE_GROUP=your-content-safety-resource-group
AZURE_CONTENT_SAFETY_ACCOUNT_NAME=your-content-safety-account
AZURE_ORACLE_STORAGE_RESOURCE_GROUP=your-storage-resource-group
```

The live preview API accepts only an Azure Blob Storage HTTPS endpoint, despite
the how-to page also describing a general public URL. Give the Content Safety
resource's system-assigned identity `Storage Blob Data Reader` on the private
input storage account. The command downloads that private blob through Azure
CLI and refuses to submit unless its SHA-256 equals the named local file:

```bash
uv run python scripts/provider_oracles.py check-microsoft input.png \
  --content-uri https://storage-account.blob.core.windows.net/oracle-inputs/input.png \
  --acknowledge-upload \
  --slot microsoft-api-1 \
  --env-file .env
```

The adapter sends one detection request with no submit retry, polls only that
operation, ignores process proxy settings, and records Watermark separately
from C2PA. The current adapter accepts JPEG, PNG, and WebP up to the provider's
100 MiB limit. See Microsoft's
[Content Provenance workflow](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/how-to-provenance-detection).

## ElevenLabs, Cartesia, and Resemble APIs, Kling Web, and Cartesia Web

`scripts/provider_oracle_audio_apis.py` provides direct `httpx` (`trust_env=False`)
adapters for ElevenLabs Content Provenance (`POST https://api.elevenlabs.io/v1/content-provenance/watermark`
via `check-elevenlabs`), Cartesia Sonic watermark detection
(`POST https://api.cartesia.ai/v1/watermark/detect` with `Cartesia-Version: 2026-08-14`
via `check-cartesia`), and Resemble Perth watermark detection
(`POST https://app.resemble.ai/api/v2/watermark/detect` + `GET .../detect/{uuid}/result`
via `check-resemble --content-uri <https_url>`). By default, `check-elevenlabs`
and `check-cartesia` strip AI container metadata into a
temporary sibling and verify bit-exact decoded float32 PCM invariance
(`ffmpeg -flags2 +skip_manual`) before transmitting one acknowledged multipart
upload, so C2PA manifests cannot mask waveform watermark verdicts (`--keep-metadata`
sends raw bytes when explicitly testing container-level behavior). `check-resemble`
streams the caller-supplied HTTPS `--content-uri` first (redacting any presigned
query signature from logs and stored responses) and verifies that its SHA-256
matches the local `SOURCE` file before submitting one job, mapping `overall_status`
(`present`/`degraded` -> `detected`, `absent` with complete coverage -> `not_detected`,
`inconclusive` or incomplete audio coverage -> `indeterminate`) separately from
the optional `synthid` boolean (`synthid_status`):

```bash
uv run python scripts/provider_oracles.py check-elevenlabs speech.mp3 \
  --acknowledge-upload --env-file .env
uv run python scripts/provider_oracles.py check-cartesia sonic.wav \
  --acknowledge-upload --env-file .env
uv run python scripts/provider_oracles.py check-resemble perth.wav \
  --content-uri https://cdn.example.com/perth.wav --acknowledge-upload --env-file .env
```

On 2026-11-17, live `check-elevenlabs` verification across native long-form
ElevenLabs exports (`Eleven Multilingual v2` `34.9 s` and `eleven_v3_dpo` `61.8 s`)
confirmed that ElevenLabs embeds an inaudible payload-carrying waveform watermark
(`unique_id`, `model`, `product`, `generated_at`) that survives lossless C2PA
metadata stripping (`is_detected: true`) and `--method spectral` (`is_detected: true`),
while `--method encodec`, `--method combined`, `--method focalcodec`, and
multi-window `--method dac` (`34.9 s` 2-window and `61.8 s` 3-window MP3s) all
returned HTTP 200 `{"is_detected": false, "details": null}`.

For Web surfaces:

- `kling-web` (`https://kling.ai/detection`): the frontend bundle renders three
  separate detector rows (`Invisible watermark`, `AIGC metadata`, and
  `C2PA manifest`, with a `10 detections` quota pool on authenticated sessions).
  `parse_provider_result("kling-web", ...)` reads the `Invisible watermark` row
  independently from `AIGC metadata` and `C2PA manifest`. Anonymous `run-web`
  clicks `Start Detection` and records `indeterminate` when Kling opens its
  mandatory sign-in modal (`Welcome to Kling AI`), never signing into a user
  account automatically. On 2026-11-18 UTC, an authorized run in the user's
  already-authenticated Chrome across 9 native Kling exports and ablations
  (`provider-original.mp4`, `Kling 3.0 Turbo`, `Kling O3 Standard` with AAC
  audio, and `Kling Image v3`) proved that what Kling labels
  `Invisible watermark` on MP4 video is the H.264 `user_data_unregistered` SEI
  `kling-ai` tag rather than a pixel or audio waveform mark: all three native
  MP4 originals returned `Invisible watermark: Detected (Kling AI)`, and
  lossless `remove_video_metadata` alone (stream-copying H.264 video and AAC
  audio while scrubbing `kling_sei` and TC260 `AIGC`) flipped all three to
  `Invisible watermark: Not detected` and `No supported signal was detected.`
  On `Kling Image v3` PNG, `Invisible watermark` returned `Failed` and lossless
  `remove_ai_metadata` cleared `AIGC metadata: Detected (Kling AI)` to
  `No supported signal was detected.`
- `cartesia-web` (`https://www.cartesia.ai/watermark-detector`): `run-web`
  proceeds directly under `--acknowledge-uploads` without an extra Terms flag,
  but on 2026-11-18 UTC the live form embedded a Cloudflare Turnstile gate
  (`input[name="cf-turnstile-response"]`) that keeps `Check for watermark`
  disabled in isolated Playwright; `run-web` detects the Turnstile gate and
  stops immediately with `refused` rather than waiting out or attempting to
  bypass the challenge. Prefer `cartesia-api` (`check-cartesia`) when an API key
  is available.
- `meta-web` (`https://www.meta.ai/identification/`): on 2026-11-17, encoding a
  native `Muse Image 1` export (`ID: AXZpZGVvc2VhbHY0X2NhX3NjbOQFLyN-TEfk`,
  `VideoSeal v4`) into an H.264 MP4 returned `AI signatures were found` (`detected`),
  and regenerating that MP4 with `remove_video_invisible(..., profile="synthid")`
  returned `No AI signatures from Meta were found` (`not_detected`).

## Live cross-provider smoke

On 2026-09-08, the public `data/contentseal/originals/gen_fox_forest.webp`
fixture, or a metadata-stripped pixel-identical copy of it, was submitted once
per configured account or surface. These are transport and parser smokes, not
error rate estimates:

| Surface | Route and repetitions | Recorded result |
| --- | --- | --- |
| Gemini Web | Official browser verifier, three independent authenticated sessions | 3/3 `not_detected`; Gemini reported no reliable Google AI signal. |
| OpenAI API | Official API, three independent requests | 3/3 SynthID `not_detected`; C2PA `not_present`; all responses were HTTP 200. |
| OpenAI Web | ThorData GB, isolated Playwright | `indeterminate`; after accepting the prepared upload, the provider returned `Something went wrong. Please try again.` |
| Microsoft API | Direct API, private hash-matched Azure blob | HTTP 202, operation `Succeeded`; Watermark `not_detected`, C2PA `absent`, provider outcome `NoProvenanceDetected`. |
| Microsoft Web | ThorData US, isolated Playwright | `not_detected`; the page's exact settled label was `Inconclusive`. |
| Meta Web | ThorData DE, isolated Playwright | `detected`; attribution `Muse Image 1 - Meta` and the existing fixture generation ID were preserved. |

The same setup verified distinct US, DE, GB, NL, and JP egress addresses whose
reported countries matched their requested routes. A deliberately invalid
country code failed rather than falling back to a random location. The Google
404 and the OpenAI Web error remain availability findings; neither is evidence
that a watermark is absent.
