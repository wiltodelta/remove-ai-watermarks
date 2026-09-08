# Provider oracle tooling

`scripts/provider_oracles.py` is the development-only entry point for the four
remote watermark-oracle families used by this project. It drives Google through
the user's real authenticated Chrome and drives the other public Web surfaces
through isolated Playwright. It keeps provider scope, execution identity,
immutable media, and verbatim results in one format without exposing any oracle
through the installed CLI or public Python API.

## Surfaces

| Surface key | Provider signal | Access | Project evidence |
| --- | --- | --- | --- |
| `openai-api` | OpenAI SynthID | Official Content Provenance API | Images |
| `microsoft-api` | Microsoft InvisMark | Azure Content Provenance Detection API | Images |
| `gemini-web` | Google SynthID | Signed-in Gemini web account | Images and videos |
| `meta-web` | Meta Content Seal | Anonymous Meta identification page | Images |
| `microsoft-web` | Microsoft InvisMark | Public Azure AI validation page | Images |
| `openai-web` | OpenAI SynthID | Public web verifier | Images |

## Surface priority

The default selection order is API first, then Web. OpenAI therefore plans
`openai-api` before `openai-web`, and Microsoft plans `microsoft-api` before
`microsoft-web`. Google and Meta currently have no usable API adapter in this
tooling, so their Web surfaces are primary rather than fallback routes.

Inspect the machine-readable order before preparing a batch:

```bash
uv run python scripts/provider_oracles.py plan openai --json
uv run python scripts/provider_oracles.py plan microsoft
```

API-first is a selection policy, not automatic failover. Use Web only when the
provider, requested media, or local configuration has no usable API route, or
when the operator explicitly requests a second surface. An API refusal, quota
response, transport failure, or indeterminate verdict is recorded as-is and
never silently triggers a Web upload.

The old Google SynthID Detector portal does not work. On 2026-09-07, the
official `labs.google/synthid` entry point redirected a signed-in browser to a
Google 404. Historical corpus rows may still name `synthid-portal`; that value
describes how an old result was obtained and is not an operational surface.
Adobe Inspect and local C2PA, DWT-DCT, TrustMark, and research detectors are
useful evidence readers, not interchangeable provider pixel-watermark oracles.

The surface capability and this project's evidence coverage are separate. The
current Google, Microsoft, and Meta pages accept image, video, and audio inputs;
the OpenAI API accepts image and audio. A media type does not become certified
merely because a surface accepts it. Run `list --json` to read both fields.

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
      "name": "openai-web-thordata-us",
      "surface": "openai-web",
      "account_label": null,
      "browser_profile": null,
      "network_label": "thordata-us-residential",
      "proxy_url_env": "THORDATA_ROUTE_US"
    },
    {
      "name": "microsoft-web-thordata-de",
      "surface": "microsoft-web",
      "account_label": null,
      "browser_profile": null,
      "network_label": "thordata-de-residential",
      "proxy_url_env": "THORDATA_ROUTE_DE"
    },
    {
      "name": "meta-web-thordata-us",
      "surface": "meta-web",
      "account_label": null,
      "browser_profile": null,
      "network_label": "thordata-us-residential",
      "proxy_url_env": "THORDATA_ROUTE_US"
    },
    {
      "name": "openai-web-thordata-de",
      "surface": "openai-web",
      "account_label": null,
      "browser_profile": null,
      "network_label": "thordata-de-residential",
      "proxy_url_env": "THORDATA_ROUTE_DE"
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
account, its real Chrome profile, and its nonnegative `/u/N/` account index; an
API slot must name its account or project and cannot name a browser profile. The
tool records the network route that the operator has already established.
Non-Google Web slots may name `proxy_url_env`; Playwright reads the secret proxy
URL from that environment variable without writing or logging it. The tool does
not rotate accounts or networks, retry through another IP after throttling, or
merge sessions. Record a throttled response as `refused`; choosing another
authorized slot is a separate operator decision and a separate batch. This
preserves per-account and per-IP history without turning provider limits into an
automatic evasion mechanism.

Each `openai-api` slot must name exactly one `api_key_env`. Put the corresponding
values in the gitignored local `.env`, never in the slot JSON:

```dotenv
OPENAI_API_KEY_1=replace-with-first-key
OPENAI_API_KEY_2=replace-with-second-key
OPENAI_API_KEY_3=replace-with-third-key
THORDATA_ROUTE_US=http://td-customer-USERNAME-country-US:PASSWORD@HOST.pr.thordata.net:9999
THORDATA_ROUTE_DE=http://td-customer-USERNAME-country-DE:PASSWORD@HOST.pr.thordata.net:9999
THORDATA_ROUTE_GB=http://td-customer-USERNAME-country-GB:PASSWORD@HOST.pr.thordata.net:9999
THORDATA_ROUTE_NL=http://td-customer-USERNAME-country-NL:PASSWORD@HOST.pr.thordata.net:9999
THORDATA_ROUTE_JP=http://td-customer-USERNAME-country-JP:PASSWORD@HOST.pr.thordata.net:9999
```

The tool reads only the variable selected by the requested slot. A nonempty
process environment value overrides the matching `.env` entry. It never prints
the value or falls through to another token after authentication, entitlement,
quota, or transport failure. The no-slot compatibility path reads
`OPENAI_API_KEY`.

Several Google accounts may share one real Chrome profile: give them distinct
slot names and `google_account_index` values while keeping `browser_profile` the
same.

The ThorData examples use its [residential proxy country parameter](https://doc.thordata.com/doc/proxies/residential-proxies/location-settings/country-region).
The proxy username encodes one explicit two-letter ISO country code. Route
selection remains explicit and never changes after a refusal. OpenAI,
Microsoft, and Meta Web slots may reuse any route variable; the surface remains
part of each distinct slot. OpenAI API slots never accept `proxy_url_env` and
use the current direct network.

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
stripped. Use the matched-control protocols in [`synthid.md`](synthid.md) for
video claims.

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

Google is intentionally excluded from the standalone runner because its oracle
requires the user's existing authenticated Chrome session. Invoke the project
`provider-oracles` skill to prepare the batch, select the named real Chrome
profile and `/u/N/` Google account, upload each prepared artifact, capture the
settled SynthID tool outcome, and record it.

OpenAI Web, Microsoft Web, and Meta run entirely through a fresh isolated
Playwright browser. A direct-route example:

```bash
uv run python scripts/provider_oracles.py prepare meta-web input.png \
  --output /tmp/meta-check-001 \
  --slot meta-direct

uv run python scripts/provider_oracles.py run-web \
  /tmp/meta-check-001/manifest.json --acknowledge-uploads
```

For a ThorData-routed OpenAI, Microsoft, or Meta Web slot, keep the generated
proxy URLs in the gitignored `.env`, then select one route explicitly:

```bash
uv run python scripts/provider_oracles.py run-web \
  /tmp/openai-check-001/manifest.json --acknowledge-uploads \
  --env-file .env
```

The runner reuses one context and one route for the batch, performs a fresh page
navigation before each upload, does not retry, and stores the settled page text.
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
ready. Microsoft and Meta reach a usable handler at network idle. OpenAI keeps
background requests alive and never reaches network idle, so its observable
readiness seam is the page `load` state instead. The adapter waits for the
surface-specific state before assigning the file. OpenAI's static FAQ contains
the lowercase phrase `generated with OpenAI tools`; only the exact title-cased
result label may settle a positive result. Its generic `Something went wrong`
response settles as `indeterminate`, not clean. Meta's live positive wording
changed from `AI signatures from Meta were found` to `AI signatures were found`
by 2026-09-07; both exact forms remain accepted.

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

## Live cross-provider smoke

On 2026-09-08, the public `data/contentseal/originals/gen_fox_forest.webp`
fixture, or a metadata-stripped pixel-identical copy of it, was submitted once
per configured account or surface. These are transport and parser smokes, not
error rate estimates:

| Surface | Route and repetitions | Recorded result |
| --- | --- | --- |
| Gemini Web | User's real Chrome, three authenticated account slots | 3/3 `not_detected`; Gemini reported no reliable Google AI signal. |
| OpenAI API | Three named API-key slots, direct transport | 3/3 SynthID `not_detected`; C2PA `not_present`; all responses were HTTP 200. |
| OpenAI Web | ThorData GB, isolated Playwright | `indeterminate`; after accepting the prepared upload, the provider returned `Something went wrong. Please try again.` |
| Microsoft API | Direct API, private hash-matched Azure blob | HTTP 202, operation `Succeeded`; Watermark `not_detected`, C2PA `absent`, provider outcome `NoProvenanceDetected`. |
| Microsoft Web | ThorData US, isolated Playwright | `not_detected`; the page's exact settled label was `Inconclusive`. |
| Meta Web | ThorData DE, isolated Playwright | `detected`; attribution `Muse Image 1 - Meta` and the existing fixture generation ID were preserved. |

The same setup verified distinct US, DE, GB, NL, and JP egress addresses whose
reported countries matched their requested routes. A deliberately invalid
country code failed rather than falling back to a random location. The Google
404 and the OpenAI Web error remain availability findings; neither is evidence
that a watermark is absent.
