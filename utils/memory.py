# utils/memory.py
import os, time, threading, psutil, ctypes
from statistics import mean

class MemoryTracker:
    def __init__(self, sample_interval=0.001):
        self.process = psutil.Process(os.getpid())
        self.interval = sample_interval
        self._baseline = None
        self._samples = []
        self._running = False
        self._thread = None
        self.page_faults = 0

    def __enter__(self):
        import gc
        gc.collect()
        # trim the working set to drop caches
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        time.sleep(0.05)
        self._baseline = self._get_usage()
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sampler, daemon=True)
        self._thread.start()
        return self

    def _get_usage(self):
        try:
            info = self.process.memory_full_info()
            self.page_faults = info.pfaults
            return info.uss / 1024
        except (AttributeError, psutil.AccessDenied):
            return self.process.memory_info().rss / 1024

    def _sampler(self):
        # pin to core 0 to reduce scheduling jitter
        try:
            self.process.cpu_affinity([0])
        except Exception:
            pass

        get = self._get_usage
        base = self._baseline
        while self._running:
            usage = get() - base
            self._samples.append(max(0.0, usage))
            time.sleep(self.interval)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._thread:
            self._thread.join()
        if not self._samples:
            self._samples = [0.0]
        self.min_usage = min(self._samples)
        self.avg_usage = mean(self._samples)
        self.max_usage = max(self._samples)
