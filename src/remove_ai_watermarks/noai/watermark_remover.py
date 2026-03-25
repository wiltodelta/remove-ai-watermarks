"""Watermark removal using diffusion model regeneration attack.

Based on the paper "Image Watermarks Are Removable Using Controllable
Regeneration from Clean Noise" (ICLR 2025).

This module implements a simple regeneration attack that:
1. Encodes the watermarked image to latent space
2. Adds noise via forward diffusion process
3. Denoises via reverse diffusion process
4. Decodes back to pixel space
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from remove_ai_watermarks.noai.watermark_profiles import (
    CTRLREGEN_MODEL_ID,
    DEFAULT_MODEL_ID,
    HIGH_STRENGTH,
    LOW_STRENGTH,
    MEDIUM_STRENGTH,
    detect_model_profile,
)

logger = logging.getLogger(__name__)

# Check for optional dependencies
_HAS_TORCH = False
_HAS_DIFFUSERS = False

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore

try:
    from diffusers import AutoPipelineForImage2Image as AutoImg2ImgPipeline

    _HAS_DIFFUSERS = True
except ImportError:
    AutoImg2ImgPipeline = None  # type: ignore


def is_watermark_removal_available() -> bool:
    """Check if watermark removal dependencies are installed."""
    return _HAS_TORCH and _HAS_DIFFUSERS


_CUDA_FIX_ENV_KEY = "NOAI_CUDA_FIXED"


def _auto_install(packages: list[str], index_url: str | None = None) -> bool:
    """Attempt to install missing packages via pip. Returns True on success."""
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages]
    if index_url:
        cmd.extend(["--index-url", index_url])
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _has_nvidia_gpu() -> bool:
    """Check if an NVIDIA GPU is present via nvidia-smi."""
    import subprocess

    try:
        subprocess.check_call(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _detect_cuda_index_url() -> str:
    """Detect the appropriate PyTorch CUDA index URL from nvidia-smi output."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if "CUDA Version" in line:
                version_str = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                major, minor = version_str.split(".")[:2]
                cuda_tag = f"cu{major}{minor}"
                return f"https://download.pytorch.org/whl/{cuda_tag}"
    except Exception:
        pass
    return "https://download.pytorch.org/whl/cu121"


def _reinstall_torch_cuda_and_restart() -> None:
    """Reinstall torch with CUDA support showing live progress, then restart."""
    import re
    import subprocess

    from remove_ai_watermarks.noai.progress import run_with_progress

    index_url = _detect_cuda_index_url()
    progress_state: dict[str, str] = {"message": "NVIDIA GPU detected — installing CUDA-enabled PyTorch..."}

    pct_re = re.compile(r"(\d+)%")
    pkg_re = re.compile(r"(?:Collecting|Downloading|Installing)\s+(\S+)")

    def _run_pip() -> bool:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "torch",
            "--index-url",
            index_url,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in iter(proc.stdout.readline, ""):  # type: ignore[union-attr]
            stripped = line.strip()
            if not stripped:
                continue
            pkg_m = pkg_re.search(stripped)
            pct_m = pct_re.search(stripped)
            if pct_m and pkg_m:
                progress_state["message"] = f"Downloading {pkg_m.group(1)} ({pct_m.group(1)}%)"
            elif pct_m:
                progress_state["message"] = f"Downloading CUDA packages ({pct_m.group(1)}%)"
            elif pkg_m:
                action = "Installing" if stripped.startswith("Installing") else "Downloading"
                progress_state["message"] = f"{action} {pkg_m.group(1)}"
            elif "Successfully installed" in stripped:
                progress_state["message"] = "CUDA-enabled PyTorch installed successfully"
        proc.wait()
        return proc.returncode == 0

    try:
        success = run_with_progress(_run_pip, progress_state)
    except Exception:
        success = False

    if not success:
        print(
            f"\n  Failed to install CUDA-enabled PyTorch.\n"
            f"  Install manually:\n"
            f"    pip install torch --index-url {index_url}\n",
            file=sys.stderr,
        )
        return

    os.environ[_CUDA_FIX_ENV_KEY] = "1"
    restart_code = f"import sys; sys.argv = {sys.argv!r}; from remove_ai_watermarks.cli import main; sys.exit(main())"
    os.execl(sys.executable, sys.executable, "-c", restart_code)


def _ensure_watermark_deps() -> None:
    """Auto-install and re-import missing watermark removal dependencies."""
    global _HAS_TORCH, _HAS_DIFFUSERS, torch, AutoImg2ImgPipeline
    missing_pkgs: list[str] = []
    if not _HAS_TORCH:
        missing_pkgs.append("torch")
    if not _HAS_DIFFUSERS:
        missing_pkgs.extend(["diffusers", "transformers", "accelerate"])
    logger.info("Auto-installing missing dependencies: %s", missing_pkgs)
    if not _auto_install(missing_pkgs):
        raise ImportError(
            f"Failed to auto-install missing dependencies: {', '.join(missing_pkgs)}. "
            "Try manually: pip install --force-reinstall noai-watermark"
        )
    import torch as _torch

    torch = _torch
    _HAS_TORCH = True
    from diffusers import AutoPipelineForImage2Image  # noqa: N813

    AutoImg2ImgPipeline = AutoPipelineForImage2Image  # noqa: N806
    _HAS_DIFFUSERS = True


def get_device() -> str:
    """Get the best available device for inference."""
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():  # type: ignore
        try:
            t = torch.tensor([1.0], device="cuda")
            _ = t + t
            del t
            return "cuda"
        except (AssertionError, RuntimeError):
            pass
    if _has_nvidia_gpu() and not os.environ.get(_CUDA_FIX_ENV_KEY):
        _reinstall_torch_cuda_and_restart()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Keep legacy name available for backwards compatibility
_detect_model_profile_from_id = detect_model_profile


class WatermarkRemover:
    """Remove watermarks from images using diffusion model regeneration.

    Attributes:
        model_id: HuggingFace model ID for the diffusion model.
        device: Device to run inference on (cuda, mps, or cpu).
    """

    DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
    CTRLREGEN_MODEL_ID = CTRLREGEN_MODEL_ID
    LOW_STRENGTH = LOW_STRENGTH
    MEDIUM_STRENGTH = MEDIUM_STRENGTH
    HIGH_STRENGTH = HIGH_STRENGTH

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        torch_dtype: Any = None,
        progress_callback: Callable[[str], None] | None = None,
        hf_token: str | None = None,
    ):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.model_profile = detect_model_profile(self.model_id)

        if not is_watermark_removal_available():
            _ensure_watermark_deps()
        self.device = (device or get_device()).lower()
        if self.device == "auto":
            self.device = get_device()
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, mps, cuda.")
        if torch_dtype is None:
            if self.device == "cpu" or self.device == "mps":
                self.torch_dtype = torch.float32  # type: ignore
            else:
                self.torch_dtype = torch.float16  # type: ignore
        else:
            self.torch_dtype = torch_dtype

        self._pipeline: AutoImg2ImgPipeline | None = None
        self._ctrlregen_engine: Any = None
        self._progress_callback = progress_callback
        self.hf_token: str | None = hf_token or os.environ.get("HF_TOKEN")

    def _set_progress(self, message: str) -> None:
        """Send a progress update through callback when available."""
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:
            pass

    # ── Preload ──────────────────────────────────────────────────────

    def preload(self) -> None:
        """Eagerly load the pipeline so download progress bars are visible."""
        if self.model_profile == "ctrlregen":
            self._run_ctrlregen_preload()
        else:
            self._load_pipeline()

    def _run_ctrlregen_preload(self) -> None:
        """Ensure the CtrlRegen engine and all its models are loaded."""
        from remove_ai_watermarks.noai.ctrlregen import is_ctrlregen_available

        if not is_ctrlregen_available():
            missing_pkgs = ["controlnet-aux", "color-matcher", "safetensors"]
            logger.info("Auto-installing missing CtrlRegen dependencies: %s", missing_pkgs)
            if not _auto_install(missing_pkgs):
                raise ImportError(
                    f"Failed to auto-install missing dependencies: {', '.join(missing_pkgs)}. "
                    "Try manually: pip install --force-reinstall noai-watermark"
                )
        if self._ctrlregen_engine is None:
            self._ctrlregen_engine = self._make_ctrlregen_engine()
        self._ctrlregen_engine.load()

    def _make_ctrlregen_engine(self) -> Any:
        """Create a new CtrlRegenEngine with current settings."""
        from remove_ai_watermarks.noai.ctrlregen import CtrlRegenEngine

        base_model = self.model_id if self.model_id != self.CTRLREGEN_MODEL_ID else None
        return CtrlRegenEngine(
            base_model_id=base_model,
            device=self.device,
            torch_dtype=self.torch_dtype,
            hf_token=self.hf_token,
            progress_callback=self._progress_callback,
        )

    # ── Pipeline loading ─────────────────────────────────────────────

    def _load_pipeline(self) -> AutoImg2ImgPipeline:
        """Load the diffusion pipeline lazily."""
        if self._pipeline is None:
            logger.info("Loading model %s on %s...", self.model_id, self.device)
            self._set_progress(f"Loading model weights: {self.model_id}")

            load_kwargs: dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            if self.hf_token:
                load_kwargs["token"] = self.hf_token

            self._pipeline = AutoImg2ImgPipeline.from_pretrained(  # type: ignore
                self.model_id,
                **load_kwargs,
            )

            self._set_progress(f"Moving model to device: {self.device}")
            try:
                self._pipeline = self._pipeline.to(self.device)  # type: ignore
            except (RuntimeError, AssertionError) as exc:
                if self.device == "cuda" and not os.environ.get(_CUDA_FIX_ENV_KEY):
                    self._set_progress("CUDA failed. Reinstalling torch with CUDA support...")
                    _reinstall_torch_cuda_and_restart()
                raise RuntimeError(
                    f"Failed to move model to {self.device} ({exc}). "
                    "Install CUDA-enabled PyTorch manually:\n"
                    f"  pip install torch --index-url {_detect_cuda_index_url()}"
                ) from exc

            if hasattr(self._pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    self._set_progress("Enabling memory optimizations...")
                    self._pipeline.enable_xformers_memory_efficient_attention()  # type: ignore
                except Exception:
                    pass

            # Mac Float32 memory slicing
            if self.device == "mps" and hasattr(self._pipeline, "enable_attention_slicing"):
                try:
                    self._pipeline.enable_attention_slicing("max")
                except Exception:
                    pass

            logger.info("Model loaded successfully")
            self._set_progress("Model initialized. Preparing input image...")

        return self._pipeline  # type: ignore

    # ── Core removal ─────────────────────────────────────────────────

    def remove_watermark(
        self,
        image_path: Path,
        output_path: Path | None = None,
        strength: float | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> Path:
        """Remove watermark from an image using regeneration attack.

        Args:
            image_path: Path to the watermarked image.
            output_path: Path for the cleaned image. If None, modifies in place.
            strength: Denoising strength (0.0-1.0).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            Path to the cleaned image.

        Raises:
            FileNotFoundError: If input image doesn't exist.
            ValueError: If strength is not in valid range.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if output_path is None:
            output_path = image_path

        strength = strength or self.LOW_STRENGTH

        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

        if guidance_scale is None:
            guidance_scale = 2.0 if self.model_profile == "ctrlregen" else 7.5

        self._set_progress("Loading and preprocessing input image...")
        init_image = Image.open(image_path).convert("RGB")
        w, h = init_image.size
        self._set_progress(f"Image loaded: {w}x{h}px | Model: {self.model_id}")

        generator = None
        if seed is not None and _HAS_TORCH:
            self._set_progress(f"Setting reproducible seed: {seed}")
            generator = torch.Generator(device=self.device).manual_seed(seed)  # type: ignore

        effective_steps = max(1, int(num_inference_steps * strength))
        self._set_progress(
            f"Config: strength={strength}, steps={num_inference_steps} "
            f"(~{effective_steps} effective), guidance={guidance_scale}, device={self.device}"
        )

        _total_start = time.monotonic()

        if self.model_profile == "ctrlregen":
            cleaned_image = self._run_ctrlregen(
                init_image,
                strength,
                num_inference_steps,
                guidance_scale,
                generator,
            )
        else:
            cleaned_image = self._run_img2img(
                init_image,
                strength,
                num_inference_steps,
                guidance_scale,
                generator,
            )

        self._set_progress(f"Regeneration complete · Output: {w}x{h}px {cleaned_image.mode}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower()
        if fmt in (".jpg", ".jpeg"):
            self._set_progress(f"Encoding as JPEG → {output_path.name}...")
        else:
            self._set_progress(f"Encoding as PNG → {output_path.name}...")
        cleaned_image.save(output_path)

        if output_path.exists():
            self._set_progress("Stripping AI metadata from output...")
            try:
                from remove_ai_watermarks.noai.cleaner import remove_ai_metadata

                remove_ai_metadata(output_path, output_path, keep_standard=True)
            except Exception:
                logger.debug("AI metadata stripping skipped", exc_info=True)

        total_time = time.monotonic() - _total_start

        size_str = ""
        try:
            file_size = output_path.stat().st_size
            if file_size < 1024 * 1024:
                size_str = f" ({file_size / 1024:.0f}KB)"
            else:
                size_str = f" ({file_size / (1024 * 1024):.1f}MB)"
        except OSError:
            pass

        logger.info("Cleaned image saved to %s", output_path)
        self._set_progress(f"✓ Saved {output_path.name}{size_str} · {w}x{h}px · {total_time:.0f}s total")

        return output_path

    # ── Img2img runner ───────────────────────────────────────────────

    def _run_img2img(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        """Execute the img2img pipeline with progress and MPS fallback."""
        from remove_ai_watermarks.noai.img2img_runner import run_img2img_with_mps_fallback

        result_image, final_device = run_img2img_with_mps_fallback(
            load_pipeline=self._load_pipeline,
            image=init_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            device=self.device,
            set_progress=self._set_progress,
            reload_on_cpu=self._reload_pipeline_on_cpu,
        )

        if final_device != self.device:
            self.device = final_device
            self.torch_dtype = torch.float32  # type: ignore[assignment]

        return result_image

    def _reload_pipeline_on_cpu(self) -> Any:
        """Reload pipeline on CPU after MPS failure."""
        self.device = "cpu"
        self.torch_dtype = torch.float32  # type: ignore[assignment]
        self._pipeline = None
        return self._load_pipeline()

    # ── CtrlRegen runner ─────────────────────────────────────────────

    def _run_ctrlregen(
        self,
        init_image: Image.Image,
        strength: float,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Any,
    ) -> Image.Image:
        """Run CtrlRegen pipeline with MPS fallback."""
        from remove_ai_watermarks.noai.ctrlregen import is_ctrlregen_available
        from remove_ai_watermarks.noai.progress import is_mps_error

        if not is_ctrlregen_available():
            missing_pkgs = ["controlnet-aux", "color-matcher", "safetensors"]
            logger.info("Auto-installing missing CtrlRegen dependencies: %s", missing_pkgs)
            if not _auto_install(missing_pkgs):
                raise ImportError(
                    f"Failed to auto-install missing dependencies: {', '.join(missing_pkgs)}. "
                    "Try manually: pip install --force-reinstall noai-watermark"
                )

        if self._ctrlregen_engine is None:
            self._ctrlregen_engine = self._make_ctrlregen_engine()

        seed = None
        if generator is not None and hasattr(generator, "initial_seed"):
            seed = generator.initial_seed()

        try:
            return self._ctrlregen_engine.run(
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except RuntimeError as error:
            if self.device == "mps" and is_mps_error(error):
                logger.warning("MPS out of memory during CtrlRegen. Falling back to CPU.")
                self._set_progress("MPS out of memory! Retrying CtrlRegen on CPU...")
                try:
                    if _HAS_TORCH and hasattr(torch, "mps"):
                        torch.mps.empty_cache()  # type: ignore[attr-defined]
                except Exception:
                    pass

                self.device = "cpu"
                self.torch_dtype = torch.float32  # type: ignore[assignment]
                self._ctrlregen_engine = self._make_ctrlregen_engine()

                return self._ctrlregen_engine.run(
                    image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
            raise

    # ── Batch ────────────────────────────────────────────────────────

    def remove_watermark_batch(
        self,
        input_dir: Path,
        output_dir: Path,
        strength: float | None = None,
        num_inference_steps: int = 50,
        extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
    ) -> list[Path]:
        """Remove watermarks from all images in a directory."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        cleaned_paths: list[Path] = []

        for ext in extensions:
            for image_path in input_dir.glob(f"*{ext}"):
                output_path = output_dir / image_path.name
                try:
                    result_path = self.remove_watermark(
                        image_path=image_path,
                        output_path=output_path,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                    )
                    cleaned_paths.append(result_path)
                except Exception as e:
                    logger.error("Failed to process %s: %s", image_path, e)

        return cleaned_paths


# ── Convenience function ─────────────────────────────────────────────


def remove_watermark(
    image_path: Path,
    output_path: Path | None = None,
    strength: float = 0.04,
    model_id: str | None = None,
    device: str | None = None,
    hf_token: str | None = None,
) -> Path:
    """Convenience function to remove watermark from an image."""
    remover = WatermarkRemover(model_id=model_id, device=device, hf_token=hf_token)
    return remover.remove_watermark(
        image_path=image_path,
        output_path=output_path,
        strength=strength,
    )
