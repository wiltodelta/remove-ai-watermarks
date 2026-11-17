# Real product sample corpus

The sampler in this research means examples of actual outputs users bring
from popular products. It is not a diffusion scheduler. The execution order is
in [the roadmap](media-removal-roadmap.md); compatible checks are in the
[verifier matrix](media-provider-verifiers.md).

## Provenance tiers

| Tier | Meaning | Allowed inference |
| --- | --- | --- |
| A | Own or explicitly authorized native product/API export with recorded lineage | Candidate for a provider cohort; watermark presence still requires a matching positive check |
| A-local | Reproducible native open-weight generator run, pinned source/weights and encoder path | Compatibility of that exact implementation; not hosted-product certification |
| B | Official downloadable demo/showcase | Inspect provenance and format; encoding history, watermark and transformation/publication rights remain separate checks |
| C | Repost, unknown source, edited export or unsigned provider hint | Discovery and robustness input; not a pristine provider baseline or known-clean negative |

A native export is the file returned directly by the product, not necessarily
lossless model output. A directory named `originals`, filename, app brand, or
AI-origin classifier is insufficient provenance. Retain original bytes before
any remux, audio extraction, visible-label cleanup or metadata strip.

## Acquisition map

### Start with the existing cleared corpus

The repository already contains own-account product exports. Use
[the capture manifest](../data/captures/2026-09/manifest.csv) and its
[permission/export record](../data/captures/2026-09/README.md), the
[fixture source records](../data/fixtures/README.md), and
[real video carrier index](../data/video-watermark-examples.csv) before
requesting new generations. Point at canonical binaries rather than copying
them into another tracked corpus.

This source includes Dreamina/Seedance, Gemini/Omni, Grok, Vidu, Runway, and
Higgsfield exports across multiple underlying model families. The manifest
records the exporter and the capture's model name. Imported model outputs,
native editor operations, YouTube round trips and original generation exports
remain distinct routes. Metadata-only AI/platform verdicts in that CSV are
not watermark-detector observations.

Keep linked Higgsfield editor inputs/outputs and Google/Grok social round trips
in conservative shared split groups until the exact lineage edges are mapped.
Byte-distinct exports do not by themselves establish independent generations.
The private retained-media inventory supplements this public source; it does
not replace it. Actual source hashes must match both sources' records.

### Fill remaining product cells

The following are acquisition routes checked in the research snapshot. Current
access/plan restrictions are defined by their official pages. Do not create
paid generations or bypass disabled downloads to fill a cell.

| Product cell | Native route or official example source | Required distinction |
| --- | --- | --- |
| Veo / Flow | [Flow assets download](https://support.google.com/flow/answer/16935308?hl=en), [Veo showcase](https://deepmind.google/models/veo/), [API status](https://ai.google.dev/gemini-api/docs/video) | Individual generation versus edited scene; exact API/product availability and model version |
| Lyria / Gemini music | [Music API](https://ai.google.dev/gemini-api/docs/music-generation), [download guide](https://support.google.com/gemini/answer/16901237), [showcase](https://deepmind.google/models/lyria/) | Consumer download versus API stream; actual model, codec and sample clock |
| NotebookLM / Gemini Audio Overview | [Official example](https://blog.google/innovation-and-ai/products/notebooklm-audio-overviews/), [Gemini Overview download](https://support.google.com/gemini/answer/16047373) | Separate products; conversation download is not a screen recording or proof of the current watermark version |
| Seedance / Dreamina | [Model showcase](https://seed.bytedance.com/en/seedance), [audio/video demo](https://seed.bytedance.com/en/blog/sound-and-vision-all-in-one-take-the-official-release-of-seedance-1-5-pro), [Dreamina download](https://dreamina.capcut.com/resource/dreamina-ai) | Underlying generator versus editor/front end; AI-label setting; own retained originals preferred over showcase trailers |
| CapCut / TikTok / Higgsfield paths | Actual owned original download and documented subsequent exports | Imported video, model generation, edited render and social repost are separate branches; app branding does not identify the model |
| ElevenLabs speech | [Product guide](https://elevenlabs.io/docs/eleven-creative/playground/text-to-speech), [API](https://elevenlabs.io/docs/overview/capabilities/text-to-speech) | Model, voice, generation time, native download/response format; public voice previews are not equivalent |
| ElevenLabs music / SFX | [Music download](https://elevenlabs.io/docs/eleven-creative/products/music), [SFX download](https://elevenlabs.io/docs/eleven-creative/playground/sound-effects) | Full mix, stem, isolated effect and ambience need independent cells |
| Suno library / Studio | [Song download](https://help.suno.com/en/articles/2409921), [formats](https://help.suno.com/en/articles/13926081), [Studio export](https://help.suno.com/en/articles/13925249) | Library original versus Studio render/stems; unsigned tags versus verified credentials; waveform status separately |
| Historical Udio | [Official service changes](https://help.udio.com/en/articles/12683565-changes-associated-with-the-universal-music-group-umg-partnership) | Retained own exports; playback ripping is not a replacement native-export route |
| Archived Chatterbox local | [Code and models](https://github.com/resemble-ai/chatterbox), [official demo](https://resemble-ai.github.io/chatterbox_demopage/) | Stock local run and paired clean/marked output are A-local; retired research, hosted services and demos must not be conflated |
| OpenAI speech | [Native audio API output](https://developers.openai.com/api/docs/guides/text-to-speech) | TTS, Realtime and ChatGPT Voice are different paths; exact model/voice/response format |
| Historical Sora | [Discontinuation](https://help.openai.com/en/articles/20001152-what-to-know-about-the-sora-discontinuation), [API status](https://developers.openai.com/api/docs/deprecations) | Retained own native export; a surviving showcase is not evidence that new official generation remains available |
| Runway | [Assets download](https://help.runwayml.com/hc/en-us/articles/4408611980563-Managing-assets) | Assets can contain uploads and third-party model outputs, not just own-model generations |
| Kling | [Official product](https://kling.ai/), [FAQ](https://kling.ai/explore/kling_ai_faq) | Actual generation record and download settings; generated Explore descriptions do not establish model/version |
| Luma | [Web download](https://lumalabs.ai/learning-hub/web-quick-start), [Ray export](https://lumalabs.ai/learning-hub/ray3-faqs) | Draft/HiFi, SDR/HDR and preview/download are distinct |
| Pika | [FAQ/download](https://pika.art/faq) | Native creation versus third-party model provided through the same front end |
| Hailuo / MiniMax | [Official video API guide](https://platform.minimax.io/docs/guides/video-generation) | Consumer download versus API response; native audio versus external soundtrack |
| Vidu | [Official platform documentation](https://platform.vidu.com/docs) | Exact download contract remains an acquisition gap in this pass; do not invent a response field |
| Hosted Wan / local Wan | [Official open implementation](https://github.com/Wan-Video/Wan2.2) | Hosted marking may differ from local open weights; supplied audio is not automatically generated by Wan |
| Ovi | [Official implementation and demos](https://github.com/character-ai/Ovi) | Local synchronized generation versus lower-bitrate demo/trailer; showcase history is explicitly non-pristine |
| HeyGen | [Download/export](https://help.heygen.com/en/articles/9834825-how-to-download-or-export-a-video) | Avatar generation, translation, separate audio and cinematic third-party models |
| Synthesia | [Native download](https://help.synthesia.io/en/articles/9317524-how-do-i-download-my-synthesia-video) | Downloadable MP4/WAV versus interactive projects or trial previews |
| Adobe Firefly | [Generated clip](https://helpx.adobe.com/firefly/web/unified-generation-and-editing-experience/generate-and-edit-content.html), [SFX](https://helpx.adobe.com/firefly/web/work-with-audio-and-video/work-with-audio/text-to-sound-effects.html), [Timeline](https://helpx.adobe.com/firefly/web/firefly-video-editor/finalize-and-export/export-projects.html) | Own model versus third-party generation; clip/SFX versus edited project; exclude Stock preview assets |
| Microsoft speech / VibeVoice | [Coverage matrix](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure), [VibeVoice](https://github.com/microsoft/VibeVoice) | MAI/Azure native export versus official VibeVoice demo; each needs its own marking and receiver evidence |
| Cartesia | [Watermark/output guide](https://docs.cartesia.ai/build-with-cartesia/capability-guides/watermarking) | Native Bytes/SSE/WebSocket capture with exact Sonic model and channel/format settings; older voice previews do not qualify a current positive |
| Stable Audio | [Product/examples](https://stableaudio.com/), [mix export](https://stableaudio.com/guide), [API provenance](https://stability.ai/explainers/is-ai-generated-music-copyright-safe) | Original API response versus web/DAW mix and local open weights; credentials and waveform measured separately |
| Nova Sonic / Polly | [Nova declaration](https://docs.aws.amazon.com/us_en/bedrock/latest/userguide/watermarking.html), [Polly native response](https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html) | Distinct engines and marking paths; explicit post-generation AudioSeal injection is a separate cohort |
| Flow Music / legacy Producer or Riffusion | [Current product](https://www.flowmusic.app/), [Google integration](https://blog.google/innovation-and-ai/models-and-research/google-labs/producerai/) | Actual model and export date; do not merge legacy Fuzz and current Lyria outputs |
| Fish Audio | [Current capabilities](https://docs.fish.audio/overview/capabilities) | Native TTS model/API header; reference-voice samples, local weights and current hosted synthesis differ |
| Mureka | [API FAQ](https://platform.mureka.ai/docs/en/faq.html), [platform](https://platform.mureka.ai/docs/) | Original MP3 versus WAV/stems/lyrics video; web and API access differ |
| Murf / PlayHT | [Murf routes](https://murf.ai/api/docs/text-to-speech/overview), [PlayHT voice previews](https://docs.play.ht/reference/list-of-prebuilt-voices) | Streaming/non-streaming TTS versus Dub; legacy voice preview is not a current generation. PlayHT service availability unconfirmed |
| PixVerse | [Current catalog](https://pixverse.ai/en/ai-info-page) | Native family versus selectable Seedance/Kling/Wan/MiniMax; original completed-job download versus showcase/interactive output |
| Tencent Hunyuan | [Cloud output](https://cloud.tencent.com/document/api/1616/126160), [open model](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | Cloud generation/export-logo setting versus local weights; Tencent Cloud also hosts other manufacturers |
| Krea / Freepik-Magnific | [Krea restyle](https://www.krea.ai/apps/video-to-video), [Magnific models](https://www.magnific.com/ai/docs/video-ai-models), [lineage export](https://www.magnific.com/ai/docs/metadata-export-lineage) | Actual underlying generator, original download and sidecar lineage; Auto/model-unknown and stock assets do not qualify a native model cohort |
| Grok Imagine / voice | [Video output](https://docs.x.ai/developers/model-capabilities/video/generation), [file persistence](https://docs.x.ai/developers/model-capabilities/imagine/files), [voice marking](https://docs.x.ai/developers/advanced-api-usage/watermarking) | Ephemeral native video download versus persisted copy; voice output and marking response header are a separate path |
| LTX model / Studio | [API download](https://docs.ltx.io/quickstart/), [Studio](https://help.ltx.io/hc/en-us/p/ltx-studio), [open marking](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/marking.md) | Direct generated clip versus edited timeline and open-weight run; record optional marking flag/pack, HDR and A2Vid mode; hosted marking is not inferred from open code |

## Concrete official showcase examples

These page-to-media links were checked in the research pass. They are Tier B
inspection candidates, not rights-cleared native processing fixtures. Their
presence on an official site establishes where the example was published,
not its unedited encoding history or watermark outcome.

| Service page | Direct media example | Qualification gap |
| --- | --- | --- |
| [Veo](https://deepmind.google/models/veo/) | [Owl WebM](https://storage.googleapis.com/gdm-deepmind-com-prod-public/media/media/veo__veo-3__owl.webm) | Page mixes model sections; exact generation model and export history unverified |
| [Lyria](https://deepmind.google/models/lyria/) | [Music showcase WebM](https://storage.googleapis.com/gdm-deepmind-com-prod-public/media/Sjv9yy5JBLju3Ghc/lyria-3-5__demo__hasta-que-salga-el-sol__16-9__lo_30s.webm) | Prepared web rendition, not an established native audio export |
| [Eleven Music announcement](https://elevenlabs.io/blog/eleven-music-is-here) | [MP3 example](https://eleven-public-cdn.elevenlabs.io/payloadcms/vtkop192j8g.mp3) | Track title, model revision and export settings unverified |
| [Kling](https://kling.ai/) | [Hero banner MP4](https://s16-kling.klingai.com/kos/s101/nlav112918/kling-homepage-aio/hero-v4-banner-3.a600a9e8a68ca89d.mp4) | Promotional asset may be a montage; current homepage model name is not a generation manifest |
| [Pika](https://pika.art/) | [Featured creation MP4](https://cdn.pika.art/v2/media/creation_5b400169-bb40-4fe7-95c6-e4ec14ee9307/output.mp4) | Officially featured user work; creator rights and actual underlying model require separate evidence |

No explicit license for transforming or publishing derivatives was established
for these selected examples. Use owned native exports to qualify processing.
Keep private inspection downloads, headers, hashes and stream probes outside
the repository. Do not replace a missing native cohort with these demos.

## Manifest and selection policy

Keep the private corpus and derived media outside tracked paths. The manifest
contains asset id, generation/lineage group, source path and SHA-256, provider,
product, exact model when known, generation/export record, original download
route, owner/processing/publication permissions, provenance tier, transformation
history, source media parameters and content labels. Unknown fields stay unknown.

Watermark observations are independent records: exact artifact hash, surface,
detector/checkpoint/version, payload/key/reference requirements, timestamp,
coverage, raw response, typed status and control relationship. Do not overwrite
source provenance with detector guesses.

Select by independent generation group, not by file. Deduplicate exact bytes;
keep alternate exports, stems, source/extracted tracks and processed versions
in the same split. Group uncertain near-duplicates for review instead of
claiming independence. Freeze tuning and holdout groups before viewing removal
outcomes. A small pilot is reported as a pilot, not statistical generalization.

Prioritize cells with both an owned native export and a compatible accessible
verifier. Keep unknown-verifier samples in a separate discovery cohort; do not
fill a verified-removal denominator with them. Report missing cells explicitly.
An insufficient corpus does not justify substituting synthetic tones for real
product outputs. Constructed controls remain useful and separately labelled.

Cover speech in multiple languages, singing, instrumental and vocal music,
ambience/SFX, quiet intervals, mono/stereo and native clocks. Video needs smooth
and fine-textured content, faces, text/UI, low/high motion, occlusion, cuts,
portrait/landscape, and native audiovisual output. Include long files and
multi-track/VFR/HDR cases in delivery tests rather than qualifying only short
mono/CFR examples.

The sampler must be deterministic for a frozen inventory and seed, read-only on
sources, and auditable about exclusions. A prepared experiment verifies live
file hashes before processing. It never automatically uploads, normalizes,
transcodes, infers a provider from duration/filename, or promotes a demo into
an owned native export.
