"""GPU power monitoring helpers with graceful fallbacks."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional


class PowerMonitorBase:
    backend_name = "base"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def read_power_w(self) -> Optional[float]:
        return None

    @property
    def available(self) -> bool:
        return self.backend_name != "null"


@dataclass
class NullPowerMonitor(PowerMonitorBase):
    reason: str = "disabled"

    backend_name = "null"

    def read_power_w(self) -> Optional[float]:
        return None


class NVMLPowerMonitor(PowerMonitorBase):
    backend_name = "nvml"

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._pynvml = None
        self._device = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._device = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._started = True

    def stop(self) -> None:
        if self._started and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._started = False
        self._device = None

    def read_power_w(self) -> Optional[float]:
        if not self._started or self._pynvml is None or self._device is None:
            return None
        try:
            milli_watts = self._pynvml.nvmlDeviceGetPowerUsage(self._device)
            return float(milli_watts) / 1000.0
        except Exception:
            return None


class NvidiaSMIPowerMonitor(PowerMonitorBase):
    backend_name = "nvidia_smi"

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index

    def read_power_w(self) -> Optional[float]:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except Exception:
            return None

        if result.returncode != 0:
            return None

        raw_line = result.stdout.strip().splitlines()
        if not raw_line:
            return None

        try:
            return float(raw_line[0].strip())
        except ValueError:
            return None


def _can_import_pynvml() -> bool:
    return importlib.util.find_spec("pynvml") is not None


def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def build_power_monitor(
    *,
    enabled: bool = True,
    backend: str = "auto",
    gpu_index: int = 0,
) -> PowerMonitorBase:
    if not enabled:
        return NullPowerMonitor(reason="disabled")

    choice = (backend or "auto").strip().lower()
    if choice in {"off", "none", "null"}:
        return NullPowerMonitor(reason="off")

    if choice == "nvml":
        if _can_import_pynvml():
            return NVMLPowerMonitor(gpu_index=gpu_index)
        return NullPowerMonitor(reason="pynvml_unavailable")

    if choice in {"nvidia_smi", "nvidia-smi"}:
        if _has_nvidia_smi():
            return NvidiaSMIPowerMonitor(gpu_index=gpu_index)
        return NullPowerMonitor(reason="nvidia_smi_unavailable")

    if choice != "auto":
        return NullPowerMonitor(reason=f"unsupported_backend:{choice}")

    if _can_import_pynvml():
        return NVMLPowerMonitor(gpu_index=gpu_index)
    if _has_nvidia_smi():
        return NvidiaSMIPowerMonitor(gpu_index=gpu_index)
    return NullPowerMonitor(reason="no_power_backend")
