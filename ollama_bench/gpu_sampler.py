"""
Background GPU sampler — samples power (W), memory (MiB), utilization (%), and
GPU affinity per second while a benchmark question runs.

Usage:
    sampler = GPUSampler()
    sampler.start()          # start sampling
    ... run question ...
    sampler.stop()           # stop sampling
    metrics = sampler.metrics  # dict of results
"""

import threading
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pynvml")
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetName, nvmlDeviceGetComputeRunningProcesses,
        NVMLError,
    )
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


class GPUSampler:
    """Samples GPU metrics in a background thread at ~1 Hz."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._samples = []  # list of per-tick dicts
        self._gpu_names = []
        self._nvml_initialized = False
        self.metrics = {}

    def start(self):
        """Start background sampling."""
        if not _NVML_AVAILABLE:
            self.metrics = {"error": "pynvml not available"}
            return
        try:
            nvmlInit()
            self._nvml_initialized = True
        except NVMLError:
            self.metrics = {"error": "nvmlInit failed"}
            return

        self._stop_event.clear()
        self._samples = []
        self._gpu_names = []
        try:
            count = nvmlDeviceGetCount()
            for i in range(count):
                h = nvmlDeviceGetHandleByIndex(i)
                self._gpu_names.append(nvmlDeviceGetName(h))
        except NVMLError:
            pass
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop sampling and compute metrics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.metrics = self._compute_metrics()
        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except NVMLError:
                pass
            self._nvml_initialized = False
        return self.metrics

    def _sample_loop(self):
        while not self._stop_event.is_set():
            tick = {"ts": time.time(), "gpus": []}
            try:
                count = nvmlDeviceGetCount()
                for i in range(count):
                    h = nvmlDeviceGetHandleByIndex(i)
                    power_mw = nvmlDeviceGetPowerUsage(h)
                    mi = nvmlDeviceGetMemoryInfo(h)
                    try:
                        util = nvmlDeviceGetUtilizationRates(h)
                        gpu_util = util.gpu
                        mem_util = util.memory
                    except NVMLError:
                        gpu_util = 0
                        mem_util = 0
                    # Get PIDs on this GPU
                    pids = []
                    try:
                        procs = nvmlDeviceGetComputeRunningProcesses(h)
                        pids = [p.pid for p in procs]
                    except NVMLError:
                        pass
                    tick["gpus"].append({
                        "index": i,
                        "power_w": round(power_mw / 1000, 1),
                        "mem_used_mib": mi.used // 1024 // 1024,
                        "mem_total_mib": mi.total // 1024 // 1024,
                        "gpu_util": gpu_util,
                        "mem_util": mem_util,
                        "pids": pids,
                    })
            except NVMLError:
                pass
            self._samples.append(tick)
            self._stop_event.wait(1.0)  # ~1 Hz

    def _compute_metrics(self):
        if not self._samples:
            return {"error": "no samples collected"}

        n_gpus = len(self._samples[0]["gpus"]) if self._samples[0]["gpus"] else 0
        if n_gpus == 0:
            return {"error": "no GPUs found"}

        result = {
            "gpu_names": self._gpu_names,
            "sample_count": len(self._samples),
            "gpus": [],
            "total_power_w": {},
        }

        # Per-GPU aggregated metrics
        for gi in range(n_gpus):
            powers = []
            mem_used = []
            gpu_utils = []
            mem_utils = []
            all_pids = set()
            mem_peak = 0

            for tick in self._samples:
                if gi < len(tick["gpus"]):
                    g = tick["gpus"][gi]
                    powers.append(g["power_w"])
                    mem_used.append(g["mem_used_mib"])
                    gpu_utils.append(g["gpu_util"])
                    mem_utils.append(g["mem_util"])
                    all_pids.update(g["pids"])
                    if g["mem_used_mib"] > mem_peak:
                        mem_peak = g["mem_used_mib"]

            gpu_result = {
                "index": gi,
                "power_avg_w": round(sum(powers) / len(powers), 1) if powers else 0,
                "power_peak_w": round(max(powers), 1) if powers else 0,
                "mem_peak_mib": mem_peak,
                "mem_total_mib": self._samples[0]["gpus"][gi]["mem_total_mib"] if self._samples[0]["gpus"] else 0,
                "gpu_util_avg": round(sum(gpu_utils) / len(gpu_utils), 1) if gpu_utils else 0,
                "mem_util_avg": round(sum(mem_utils) / len(mem_utils), 1) if mem_utils else 0,
                "pids": sorted(all_pids),
            }
            result["gpus"].append(gpu_result)

        # Total power across all GPUs
        total_powers = []
        for tick in self._samples:
            total_p = sum(g["power_w"] for g in tick["gpus"])
            total_powers.append(total_p)
        result["total_power_w"] = {
            "avg": round(sum(total_powers) / len(total_powers), 1) if total_powers else 0,
            "peak": round(max(total_powers), 1) if total_powers else 0,
        }

        # Which GPUs are actually used (mem > 1GB or power > 30W)
        active_gpus = []
        for g in result["gpus"]:
            if g["mem_peak_mib"] > 1024 or g["power_avg_w"] > 30:
                active_gpus.append(g["index"])
        result["active_gpus"] = active_gpus

        # Energy estimate: avg_power * wall_time_seconds / 3600 = Wh
        # (caller should set this using wall_time)
        result["energy_wh"] = None  # set by caller

        return result

    def set_energy(self, wall_time_s):
        """Compute energy consumption in Wh given wall time in seconds."""
        if self.metrics and "total_power_w" in self.metrics:
            avg_w = self.metrics["total_power_w"].get("avg", 0)
            self.metrics["energy_wh"] = round(avg_w * wall_time_s / 3600, 2)