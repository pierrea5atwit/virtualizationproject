from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class HardwareProfile:
    has_nvidia_gpu: bool
    gpu_name: str
    virtualization_mode: str
    source: str
    error: str

    @property
    def is_virtualized(self) -> bool:
        return self.virtualization_mode in {"vgpu", "passthrough", "virtualized"}

    @property
    def is_physical_verified(self) -> bool:
        return self.virtualization_mode == "none" and self.has_nvidia_gpu


def detect_hardware_profile(device_index: int = 0) -> HardwareProfile:
    """Detect NVIDIA GPU presence and virtualization mode via NVML."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception as exc:
        return HardwareProfile(
            has_nvidia_gpu=False,
            gpu_name="",
            virtualization_mode="no_gpu",
            source="import",
            error=str(exc),
        )

    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True

        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
        gpu_name = gpu_name_raw.decode("utf-8") if isinstance(gpu_name_raw, bytes) else str(gpu_name_raw)

        virtualization_mode = "unknown"
        source = "nvml"

        try:
            mode = pynvml.nvmlDeviceGetVirtualizationMode(handle)
            none_mode = getattr(pynvml, "NVML_GPU_VIRTUALIZATION_MODE_NONE", None)
            passthrough_mode = getattr(pynvml, "NVML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH", None)
            vgpu_mode = getattr(pynvml, "NVML_GPU_VIRTUALIZATION_MODE_VGPU", None)

            if mode == none_mode:
                virtualization_mode = "none"
            elif mode == passthrough_mode:
                virtualization_mode = "passthrough"
            elif mode == vgpu_mode:
                virtualization_mode = "vgpu"
            else:
                virtualization_mode = "virtualized"
        except Exception as exc:
            return HardwareProfile(
                has_nvidia_gpu=True,
                gpu_name=gpu_name,
                virtualization_mode="unknown",
                source=source,
                error=f"virtualization_mode_detection_failed: {exc}",
            )

        return HardwareProfile(
            has_nvidia_gpu=True,
            gpu_name=gpu_name,
            virtualization_mode=virtualization_mode,
            source=source,
            error="",
        )
    except Exception as exc:
        return HardwareProfile(
            has_nvidia_gpu=False,
            gpu_name="",
            virtualization_mode="no_gpu",
            source="nvml",
            error=str(exc),
        )
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def validate_environment_request(profile: HardwareProfile, requested_environment: str) -> Dict[str, str | bool]:
    """Return decision metadata for environment safety checks."""
    requested = requested_environment.lower().strip()

    if requested != "physical":
        return {"ok": True, "reason": "virtual config allowed on all hardware profiles"}

    if not profile.has_nvidia_gpu:
        return {"ok": False, "reason": "physical config requires a detectable NVIDIA GPU"}

    if profile.virtualization_mode == "none":
        return {"ok": True, "reason": "detected non-virtualized NVIDIA hardware"}

    if profile.virtualization_mode in {"vgpu", "passthrough", "virtualized"}:
        return {
            "ok": False,
            "reason": f"physical config blocked: virtualization mode is '{profile.virtualization_mode}'",
        }

    return {
        "ok": False,
        "reason": "physical config blocked: unable to verify non-virtualized hardware",
    }
