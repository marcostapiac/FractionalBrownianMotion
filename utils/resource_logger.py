import time
import json
import psutil
import threading
import subprocess

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ResourceLogger:
    def __init__(self, interval=120, outfile="experiment_log.json", job_type="unspecified", metadata=None):
        self.interval = interval
        self.outfile = outfile
        self.job_type = job_type
        self.metadata = metadata or {}

        self.cpu = []
        self.ram = []
        self.gpu_util = []
        self.gpu_mem = []
        self._stop_flag = False

    def _log_resources(self):
        while not self._stop_flag:
            # CPU + RAM
            self.cpu.append(psutil.cpu_percent(interval=None))
            self.ram.append(psutil.virtual_memory().used / 1e6)

            # GPU
            if GPU_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e6
                self.gpu_util.append(util)
                self.gpu_mem.append(mem)

            time.sleep(self.interval)

    def __enter__(self):
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._log_resources)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_flag = True
        self.thread.join()
        self.end_time = time.time()

        log = {
            "job_type": self.job_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "wall_clock_sec": self.end_time - self.start_time,
            "cpu_percent_avg": sum(self.cpu)/len(self.cpu) if self.cpu else None,
            "cpu_percent_peak": max(self.cpu) if self.cpu else None,
            "ram_used_MB_avg": sum(self.ram)/len(self.ram) if self.ram else None,
            "ram_used_MB_peak": max(self.ram) if self.ram else None,
            "gpu_util_avg": sum(self.gpu_util)/len(self.gpu_util) if self.gpu_util else None,
            "gpu_util_peak": max(self.gpu_util) if self.gpu_util else None,
            "gpu_mem_MB_avg": sum(self.gpu_mem)/len(self.gpu_mem) if self.gpu_mem else None,
            "gpu_mem_MB_peak": max(self.gpu_mem) if self.gpu_mem else None,
            "metadata": self.metadata
        }

        with open(self.outfile, "w") as f:
            json.dump(log, f, indent=2)

        print(f"Resource log written to {self.outfile}\n")
