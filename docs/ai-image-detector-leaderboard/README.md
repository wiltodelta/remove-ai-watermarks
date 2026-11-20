---
title: Open AI Image Detector and Attribution Leaderboard
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: "20 open AI detectors on FPR traps, recall, and attribution"
models:
  - wiltodelta/raiw-source-classify
  - abacusai/Smaug-AI-Image-Detector
  - buildborderless/CommunityForensics-DeepfakeDet-ViT
  - TensorFold/detectra-v1
  - Thermostatic/community-forensics-frontier-detector-2026-08
  - s-gehring/detectra-openfake
  - bendavidsteel/openfake-swinv2-base
  - haywoodsloan/ai-image-detector-deploy
  - Organika/sdxl-detector
  - dima806/ai_vs_human_generated_image_detection
  - umm-maybe/AI-image-detector
  - Ateeqq/ai-vs-human-image-detector
  - OwensLab/commfor-model-384
  - Bombek1/ai-image-detector-siglip-dinov2
  - Sumsub/Sumsub-ffs-synthetic-2.0
  - NYUAD-ComNets/NYUAD_AI-generated_images_detector
  - Smogy/SMOGY-Ai-images-detector
  - prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2
  - prithivMLmods/Deep-Fake-Detector-v2-Model
---

# Open AI image detector and attribution leaderboard

Reproducible open-weights evaluation across 578 SHA-deduplicated third-party images streamed at immutable Hugging Face commit SHAs, built by [Victor Kuznetsov](https://wiltodelta.com/).

Run the entire benchmark locally with one `uv` command:

```bash
uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/benchmark.py --json-out leaderboard.json
```

Made by [Victor Kuznetsov](https://wiltodelta.com/).
