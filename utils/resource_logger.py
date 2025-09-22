# utils/resource_logger.py
import atexit
import json
import os
import psutil
import signal
import threading
import time
from typing import List

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class ResourceLogger:
    """
    Signal-safe resource logger:
      - instant wake on SIGINT/SIGTERM
      - autosaves partial logs every `flush_every` seconds
      - atomic writes (<outfile>.tmp -> <outfile>)
      - creates parent dirs for <path>/<file>.json
      - multi-GPU autodetect; no args required
    """

    def __init__(
        self,
        interval: float = 120,
        outfile: str = "experiment_log.json",
        job_type: str = "unspecified",
        metadata: dict | None = None,
        flush_every: float = 60,
    ):
        self.interval = float(interval)
        self.flush_every = float(flush_every)
        self.outfile = outfile
        self.job_type = job_type
        self.metadata = metadata or {}

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._finalized = False

        self.cpu: List[float] = []
        self.ram: List[float] = []

        # --- GPUs (auto) ---
        self.gpu_indices: List[int] = []
        self._gpu_meta: dict[int, dict] = {}
        self.gpu_util: dict[int, List[float]] = {}
        self.gpu_mem: dict[int, List[float]] = {}

        if GPU_AVAILABLE:
            self.gpu_indices = self._discover_visible_gpus()
            for i in self.gpu_indices:
                self.gpu_util[i] = []
                self.gpu_mem[i] = []
            self._prime_gpu_meta()

        # Signal hooks
        for sig_name in ("SIGINT", "SIGTERM", "SIGUSR1", "SIGUSR2"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, RuntimeError):
                pass

    # ---------- context manager ----------

    def __enter__(self):
        self.start_time = time.time()
        print(f"ResourceLogger → {os.path.abspath(self.outfile)}", flush=True)
        atexit.register(self._write_log)
        self.thread = threading.Thread(target=self._loop, name="ResourceLogger", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._finalized:
            return
        self._stop.set()
        if getattr(self, "thread", None):
            self.thread.join(timeout=self.interval + 2.0)
        self._write_log()
        self._finalized = True
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # ---------- signal handling ----------

    def _handle_signal(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping ResourceLogger...", flush=True)
        self._stop.set()
        try:
            self._write_log()
            if getattr(self, "thread", None) and self.thread.is_alive():
                self.thread.join(timeout=2.0)
        finally:
            raise SystemExit(128 + int(signum))

    # ---------- GPU helpers ----------

    def _discover_visible_gpus(self) -> List[int]:
        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return []

        # Default: all NVML devices
        indices = list(range(count))

        # Respect NVIDIA_VISIBLE_DEVICES first (UUIDs or "all"), else CUDA_VISIBLE_DEVICES.
        raw = os.environ.get("NVIDIA_VISIBLE_DEVICES")
        if not raw or raw == "all":
            raw = os.environ.get("CUDA_VISIBLE_DEVICES")

        if not raw or raw == "all":
            return indices

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        # Build maps from NVML to support UUID and PCI bus ID filtering
        uuid_to_idx, busid_to_idx = {}, {}
        for i in range(count):
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                # UUID
                try:
                    uuid = pynvml.nvmlDeviceGetUUID(h)
                except AttributeError:
                    uuid = pynvml.nvmlDeviceGetUUIDString(h)  # very old pynvml
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8", "replace")
                # PCI bus ID
                pci = pynvml.nvmlDeviceGetPciInfo(h).busId
                if isinstance(pci, bytes):
                    pci = pci.decode("utf-8", "replace")
                uuid_to_idx[uuid] = i
                busid_to_idx[pci] = i
            except Exception:
                continue

        filtered: List[int] = []
        seen = set()
        for tok in tokens:
            # Ordinal
            if tok.isdigit():
                j = int(tok)
                if 0 <= j < count and j not in seen:
                    filtered.append(j)
                    seen.add(j)
                continue
            # UUID (GPU-xxxx)
            if tok in uuid_to_idx and uuid_to_idx[tok] not in seen:
                filtered.append(uuid_to_idx[tok])
                seen.add(uuid_to_idx[tok])
                continue
            # PCI Bus ID (0000:xx:yy.z)
            if tok in busid_to_idx and busid_to_idx[tok] not in seen:
                filtered.append(busid_to_idx[tok])
                seen.add(busid_to_idx[tok])
                continue

        return filtered or indices

    def _prime_gpu_meta(self):
        for i in self.gpu_indices:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                total_mb = pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e6
                self._gpu_meta[i] = {"name": name, "mem_total_MB": total_mb}
            except Exception:
                pass

    # ---------- sampling & writing ----------

    def _loop(self):
        self._last_flush = time.time()
        while not self._stop.is_set():
            self._sample_once()
            if (time.time() - self._last_flush) >= self.flush_every:
                self._write_log()
                self._last_flush = time.time()
            self._stop.wait(self.interval)
        self._sample_once()

    def _sample_once(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().used / 1e6
        with self._lock:
            self.cpu.append(cpu)
            self.ram.append(ram)

        if GPU_AVAILABLE and self.gpu_indices:
            for i in self.gpu_indices:
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    mem_used = pynvml.nvmlDeviceGetMemoryInfo(h).used / 1e6
                except Exception:
                    continue
                with self._lock:
                    self.gpu_util.setdefault(i, []).append(util)
                    self.gpu_mem.setdefault(i, []).append(mem_used)

    def _stats(self):
        with self._lock:
            cpu = list(self.cpu)
            ram = list(self.ram)
            gu = {k: list(v) for k, v in self.gpu_util.items()}
            gm = {k: list(v) for k, v in self.gpu_mem.items()}

        def avg(xs): return sum(xs) / len(xs) if xs else None
        def peak(xs): return max(xs) if xs else None

        now = time.time()
        start = getattr(self, "start_time", now)

        gpu_devices = {}
        for idx in self.gpu_indices:
            u = gu.get(idx, [])
            m = gm.get(idx, [])
            meta = self._gpu_meta.get(idx, {})
            gpu_devices[str(idx)] = {
                "name": meta.get("name"),
                "mem_total_MB": meta.get("mem_total_MB"),
                "util_avg": avg(u),
                "util_peak": peak(u),
                "mem_used_MB_avg": avg(m),
                "mem_used_MB_peak": peak(m),
                "samples": len(u) or len(m) or 0,
            }

        util_avgs = [d["util_avg"] for d in gpu_devices.values() if d["util_avg"] is not None]
        util_peaks = [d["util_peak"] for d in gpu_devices.values() if d["util_peak"] is not None]
        mem_avgs = [d["mem_used_MB_avg"] for d in gpu_devices.values() if d["mem_used_MB_avg"] is not None]
        mem_peaks = [d["mem_used_MB_peak"] for d in gpu_devices.values() if d["mem_used_MB_peak"] is not None]

        overall_gpu = {
            "indices": self.gpu_indices,
            "util_avg_across_devices": avg(util_avgs),
            "util_peak_any_device": max(util_peaks) if util_peaks else None,
            "mem_used_MB_avg_across_devices": avg(mem_avgs),
            "mem_used_MB_peak_any_device": max(mem_peaks) if mem_peaks else None,
            "devices": gpu_devices,
        }

        # legacy overall fields
        return {
            "job_type": self.job_type,
            "start_time": start,
            "end_time": now,
            "wall_clock_sec": now - start,
            "cpu_percent_avg": avg(cpu),
            "cpu_percent_peak": peak(cpu),
            "ram_used_MB_avg": avg(ram),
            "ram_used_MB_peak": peak(ram),
            "gpu_util_avg": overall_gpu["util_avg_across_devices"],
            "gpu_util_peak": overall_gpu["util_peak_any_device"],
            "gpu_mem_MB_avg": overall_gpu["mem_used_MB_avg_across_devices"],
            "gpu_mem_MB_peak": overall_gpu["mem_used_MB_peak_any_device"],
            "gpu": overall_gpu,
            "samples": len(cpu),
            "interval_sec": self.interval,
            "metadata": self.metadata,
        }

    def _ensure_outdir(self):
        d = os.path.dirname(self.outfile)
        if d:
            os.makedirs(d, exist_ok=True)

    def _write_log(self):
        if self._finalized:
            return
        self._ensure_outdir()
        payload = self._stats()
        tmp = f"{self.outfile}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.outfile)
        print(f"Resource log written to {self.outfile}", flush=True)

    def flush(self):
        self._write_log()
