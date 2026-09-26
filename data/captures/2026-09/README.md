# Provider captures, September 2026

Original outputs downloaded from the maintainer's own accounts between
2026-09-23 and 2026-09-25 and cleared for publication by the maintainer. The
services are the Gemini API, xAI Grok, Vidu, Wan, Apple Image Playground,
YouTube (re-encodes of our own uploads), and the paid Higgsfield, Runway and
Dreamina plans. Every prompt is a neutral coffee-mug scene; Higgsfield
Genjutsu and the video-edit samples take our own Higgsfield outputs as input.

Files are stored byte-for-byte as served. Nothing is re-encoded, because the
provenance under study (C2PA, TC260 labels, `hf-job-id` chunks) lives in the
container bytes.

`manifest.csv` lists every capture once, including the ones that are already
test fixtures: those rows point at their `data/fixtures/` path instead of a
second copy (`stored` is `existing`). Columns:

| Column | Meaning |
| --- | --- |
| `sha256` | Digest of the stored file |
| `path` | Canonical repository path |
| `stored` | `new` for files in this directory, `existing` for a fixture already tracked |
| `service` | Service the file was downloaded from |
| `source_name` | Local capture name, which names the model |
| `captured` | Capture date |
| `ai_from_metadata`, `platform` | `identify` (images) or `identify_video` (videos) verdict from metadata only, at capture time |

What each service does to the upstream provenance, and the per-model
findings, are summarized in the live-captures table of
[`docs/watermarking-landscape.md`](../../../docs/watermarking-landscape.md).
The `google-photos/` files are the maintainer's own iPhone photos edited in
Google Photos on iOS (an Ask edit replacing a dog with a cat, an eraser edit, and
two other AI edits), published unmodified with their EXIF GPS position by the
maintainer's decision on 2026-09-25. Google's checker in Gemini found SynthID in
all four with the metadata stripped. The Apple Clean Up photo is not stored.
The `google-photos-web/` files are a negative control from the Google Photos web
editor (2026-09-26): one of the maintainer's photos, downloaded untouched and after
Enhance, Sky (Vivid) and a square crop. The web editor offers no generative tool and
writes no Content Credentials, so none of the four carries C2PA, IPTC or an AI
credit, and Google's checker in Gemini found no SynthID in any of them (three
explicit negatives; the crop drew a reply that did not name SynthID and is recorded
as indeterminate).

- License: repository license; generated and explicitly cleared by the maintainer
