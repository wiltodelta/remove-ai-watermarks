# C2PA soft-binding resolution research

Development-only investigation, 2026-09-07, completing the privacy-tiered C2PA
resolution item from the original research map. The outcome is a product
decision rather than a feature: this tool resolves nothing over the network,
and this page records what the resolution ecosystem would learn if it did.

## The normative shape

C2PA 2.2 specification section 18.10.5 defines the soft binding resolution
API: a standard way of retrieving manifest stores from a resolution endpoint
given **a soft binding value, a manifest identifier, or an asset** - three
input modes that are exactly three privacy exposure levels. The algorithm
registry (`c2pa-org/softbinding-algorithm-list`) carries the endpoint URIs in
`softBindingResolutionApis`. As of the packaged revision `c0218cd628c9`,
8 of 53 entries publish them.

## Measured contracts

| Algorithm | Kind | Endpoint | Input | Content leaves | Content digest leaves |
| --- | --- | --- | --- | --- | --- |
| `com.aiwatermark.audioseal.1` / `videoseal` / `pixelseal` | watermark | `aiwatermark.com/api/v1/resolve` | `GET ?payload={extracted payload}&alg=` | no | no |
| `com.joinmonolith.sha256` | fingerprint | `api.joinmonolith.com/api/c2pa/matches/byBinding` | `GET ?alg&value=base64(SHA-256 of asset)` | no | yes |
| `me.deepmark.audio.vigil.128` | watermark | `resolution-api.deepmark.me` | payload-based per vendor site | no | no |
| `ai.trufo.pawprint.watermark` / `.fingerprint` | both | `c2pa.trufo.ai/v1` | not published publicly | ? | ? |
| `io.blockfact.audio.watermark.32` | watermark | `api.blockfact.io/api/soft-binding/v1` | not published publicly | ? | ? |

Sources, verified 2026-09-07 against the vendors' own pages: the Monolith
soft-binding documentation (lookup endpoints public by design, "a verifier is
usually a stranger"; only manifest minting is authenticated), the AIWatermark
algorithm page ("after extracting the watermark payload from an audio file,
validators can resolve the payload"; model source Meta FAIR AudioSeal), the
Deepmark product description ("the signal is detected and its payload
resolved"), the Trufo developer documentation and `trufo-py` repository
(signing-side only), and the BlockFact API reference (its public verify
endpoint takes a Starknet transaction hash, not a soft binding).

The privacy gradient, concretely:

- **Payload-only** resolution (AIWatermark, Deepmark) sends a locally
  extracted watermark payload - opaque bits that identify the mark, not the
  file. This is the mode our pinned local oracles could serve without ever
  touching the network with content or a digest.
- **Digest** resolution (Monolith) sends base64(SHA-256) of the whole asset.
  No bytes leave, but the digest is linkable to the exact content: the
  registry learns you are inspecting that precise file.
- **Undocumented** (Trufo, BlockFact): no public contract; any integration
  would be guesswork and is not attempted.

## The product decision

This tool does not resolve anything over the network, at any tier, and no
`local_only` / `binding_only` / `full_content` modes ship. The reasoning:

- The tool's user processes their own generated or edited content and wants
  to know which provenance signals it carries and to remove them. Local
  payload extraction - already available through the pinned development
  oracles - answers the identification question without a network.
- Querying a vendor's resolver tells that vendor someone is inspecting the
  specific content (directly for digest-based lookups, via the payload for
  payload-based ones). For a removal tool that is a leak in the opposite
  direction of its purpose, and it contradicts the offline runtime boundary
  the project already guarantees.
- No user scenario of this tool requires manifest recovery from a registry:
  that is a verifier-of-third-party-content function, which is outside the
  intended use.

The runtime boundary stays: identification reports locally decoded evidence
only, and `docs/known-limitations.md` states the resolution boundary in user
terms. A guard test pins the packaged registry's published-API set, so an
upstream ecosystem change surfaces as a conscious decision recorded against
this page rather than a silent regeneration.

## Investigation note

The first reconnaissance pass concluded "upstream publishes zero resolution
APIs" because the probe searched the registry JSON for guessed key names
instead of the `softBindingResolutionApis` key the sync actually reads. The
drift-guard test written from that wrong premise failed immediately against
the packaged snapshot and exposed the error - eight live endpoints, including
resolution hosting for the exact AudioSeal and VideoSeal schemes this project
benchmarks. The corrected guard now pins the real set. Field names are
verified against the consumer's schema, never guessed.
