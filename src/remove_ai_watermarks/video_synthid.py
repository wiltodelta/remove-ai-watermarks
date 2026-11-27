"""Shared configuration for the calibrated video-pixel SynthID-removal profile."""

DEFAULT_VIDEO_SYNTHID_VAE = "stabilityai/sd-vae-ft-mse"
DEFAULT_VIDEO_SYNTHID_VAE_REVISION = "31f26fdeee1355a5c34592e401dd41e45d25a493"
# The certified profile is a perturbation-to-signal ratio, so it is pinned against
# this latent scaling factor as much as against noise_std. Rationale and the drift
# it guards against: docs/module-internals.md.
VIDEO_SYNTHID_VAE_SCALING_FACTOR = 0.18215
DEFAULT_VIDEO_SYNTHID_NOISE_STD = 0.15
DEFAULT_VIDEO_SYNTHID_LONG_SIDE = 512
DEFAULT_VIDEO_SYNTHID_FPS: float | None = None
VIDEO_SYNTHID_LATENT_MULTIPLE = 8
VIDEO_SYNTHID_VERIFICATION_PROMPT = (
    "For the video attached to this message, was it created or edited by "
    "Google AI? Use the built-in SynthID content verification result."
)
