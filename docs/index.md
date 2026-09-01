# Documentation

This documentation is split by purpose. Start with the user guides if you want
to run the tool. Use the maintainer references only when changing the code.

## User guides

| Page | Use it when |
| --- | --- |
| [Installation](installation.md) | You need the CLI, an optional model backend, or a development environment. |
| [CLI guide](cli.md) | You want a command for an image, video metadata or visible marks, a directory, or a specific watermark type. |
| [Python API](python-api.md) | You want to call the package from Python. |
| [Supported signals](supported-signals.md) | You need to know which visible marks, metadata formats, and invisible signals are covered. |
| [Known limitations](known-limitations.md) | You need the quality, device, format, or verification boundaries. |
| [Photo pixel classification](photo-classify.md) | You want `classify` / `classify_pixels`: AI versus camera, optional provider. |
| [Photo classifier training](photo-classify-training.md) | You need the retrain pack, public sources, and what is not published. |
| [Scope, safety, and legal notes](legal-and-safety.md) | You need the intended use and legal context. |

## Maintainer references

| Page | Purpose |
| --- | --- |
| [Module internals](module-internals.md) | Current architecture, invariants, and regression guards by module. |
| [Development](development.md) | Environment setup, dependency recovery, CI behavior, and fixture policy. |
| [Code provenance](code-provenance.md) | Required notices for licensed derivative work. |
| [Verification plan](verification-plan.md) | Verification methods, completed measurements, and remaining validation gaps. |
| [Release and distribution](release-and-distribution.md) | PyPI, Homebrew, Hugging Face Space, photo-classify Hub model, ComfyUI Registry, and the release workflow. |
| [Watermarking landscape](watermarking-landscape.md) | Vendor signals and detection approaches. |
| [SynthID technical reference](synthid.md) | Mechanism, provenance, robustness, regeneration. |

## Research archive

These pages record experiments and the evidence behind past decisions. They are
not command references and may describe prototypes that were later removed.
The current behavior is defined by the code, tests, README, and user guides.

- [ControlNet removal research](controlnet-removal-pipeline-research.md)
- [Qwen improvement research](qwen-improvement-research.md)
- [Doubao reverse-alpha research](research-doubao-distillation.md)
- [AI-generated image classifiers](ai-generated-image-classifiers.md) (photo AI-versus-camera freeze consumed by `classify_pixels`, not `identify`)
- [Photo classifier Hugging Face card](photo-classify-hf/README.md) (Hub publication text of the 2026-08-31 freeze)
- [SynthID source classifiers](synthid-classifiers.md) (metadata-free OpenAI/Gemini source finding and provider-lineage experiments)
- [SynthID local detector research](synthid-detector-research.md)
- [SynthID mark removal research](synthid-removal-research.md)
- [SynthID identity research](synthid-robust-identity-research.md)
- [SynthID identity follow-up](synthid-robust-identity-research-2026-06-08.md)
- [Video SynthID quality research](video-synthid-quality-research.md)
- [OpenAI SynthID oracle ladders](synthid-oracle-ladders.md) (routing hub)
- [SynthID detector and removal plan](synthid-detector-removal-plan.md) (chronological mixed archive)
- [Text protection research](text-protection-research.md)
- [Chroma1 engine research](chroma1-engine-research.md)
