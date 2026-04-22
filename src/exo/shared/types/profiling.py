import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import psutil

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import CamelCaseModel


class MemoryUsage(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )

    @classmethod
    def from_nvidia_smi(cls, *, override_memory: int | None) -> Self:
        """Report VRAM across all visible NVIDIA GPUs on the node.

        Used on Linux/Nvidia nodes so the master constrains model placement by
        VRAM rather than system RAM. Values are summed across all GPUs.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        total_mib = 0
        used_mib = 0
        lines = [line for line in result.stdout.strip().splitlines() if line]
        if not lines:
            raise RuntimeError("nvidia-smi returned no GPUs")
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                raise RuntimeError(f"Unexpected nvidia-smi line: {line!r}")
            total_mib += int(parts[0])
            used_mib += int(parts[1])
        mib = 1024 * 1024
        total_bytes = total_mib * mib
        available_bytes = (total_mib - used_mib) * mib
        return cls.from_bytes(
            ram_total=total_bytes,
            ram_available=available_bytes if override_memory is None else override_memory,
            swap_total=0,
            swap_available=0,
        )


class DiskUsage(CamelCaseModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(CamelCaseModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(CamelCaseModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(CamelCaseModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(CamelCaseModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(CamelCaseModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
