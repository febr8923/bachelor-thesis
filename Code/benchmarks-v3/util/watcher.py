import subprocess
import threading
import time
from datetime import datetime
import psutil
import logging
from pathlib import Path
from util.result import StreamingResult
from util.config import RESULTS_GPU_MEMORY, RESULTS_CPU_MEMORY


class GpuWatcher:
    def __init__(self, gpu_id, save_loc):
        self.gpu_id = gpu_id
        self.stop_event = None
        self.watcher = None
        self.save_loc = save_loc
        self.result = None
        self.poi = []

    def add_poi(self, description):
        """Add a point of interest with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.poi.append((timestamp, description))

    def track_gpu(self, stop_event):
        # Initialize streaming result
        self.result = StreamingResult(RESULTS_GPU_MEMORY, self.save_loc)

        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,memory.reserved",
                "--format=csv,noheader,nounits",
                "-lms=100",
                f"--id={self.gpu_id}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in iter(process.stdout.readline, ''):
                if stop_event.is_set():
                    break
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                fields = [f.strip() for f in line.split(',')]
                if len(fields) == 6:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    self.result.add_row(
                        timestamp=timestamp,
                        poi_type=None,
                        gpu_id=self.gpu_id,
                        **{
                            "gpu_util%": fields[0],
                            "mem_util%": fields[1],
                            "mem_total_mb": fields[2],
                            "mem_free_mb": fields[3],
                            "mem_used_mb": fields[4],
                            "mem_reserved_mb": fields[5]
                        }
                    )

                    process.poll()
                    if process.returncode is not None:
                        break
        finally:
            process.terminate()
            process.wait()

    

    def start(self):
        if self.watcher and self.watcher.is_alive():
            print(f"watcher running for {self.gpu_id}")
            return
        self.stop_event = threading.Event()
        self.watcher = threading.Thread(target=self.track_gpu, args=(self.stop_event,), daemon=True)
        self.watcher.start()


    def stop(self):
        if self.stop_event:
            self.stop_event.set()

        if self.watcher:
            self.watcher.join()

        self.watcher = None

        # Append POI rows at the end
        if self.result:
            for timestamp, description in self.poi:
                self.result.add_row(
                    timestamp=timestamp,
                    poi_type=description,
                    gpu_id=None,
                    **{
                        "gpu_util%": None,
                        "mem_util%": None,
                        "mem_total_mb": None,
                        "mem_free_mb": None,
                        "mem_used_mb": None,
                        "mem_reserved_mb": None
                    }
                )

            # Close the streaming result file
            self.result.close()


class CpuWatcher:
    def __init__(self, id: int, save_loc: str, interval: float = 0.5):
        self.id = id
        self.save_loc = Path(save_loc)
        self.csv_file = self.save_loc
        self.running = False
        self.watcher = None
        self.interval = interval
        self.result = None
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"CpuWatcher_{id}")

        self.poi = []

    def add_poi(self, description):
        """Add a point of interest with timestamp."""
        timestamp = datetime.now().isoformat()
        self.poi.append((timestamp, description))

    def track_cpu(self):
        """Efficiently track CPU and memory usage using psutil."""
        # Initialize streaming result
        self.result = StreamingResult(RESULTS_CPU_MEMORY, str(self.save_loc))

        while self.running:
            try:
                # Get CPU usage (requires interval for accurate %)
                cpu_util = psutil.cpu_percent(interval=None)

                # Get memory info
                mem = psutil.virtual_memory()
                mem_total = mem.total
                mem_used = mem.used
                mem_free = mem.available  # Use available instead of free for accuracy

                timestamp = datetime.now().isoformat()

                self.result.add_row(
                    timestamp=timestamp,
                    poi_type=None,
                    cpu_util_pct=cpu_util,
                    mem_total_bytes=mem_total,
                    mem_used_bytes=mem_used,
                    mem_available_bytes=mem_free
                )

                self.logger.debug(f"Logged: CPU={cpu_util:.1f}%, Mem={mem_used/1e9:.1f}GB used")

            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")

            time.sleep(self.interval)
    
    def start(self):
        if not self.running:
            self.running = True
            self.watcher = threading.Thread(target=self.track_cpu, daemon=True)
            self.watcher.start()
            self.logger.info(f"Started CPU watcher {self.id} (interval: {self.interval}s)")
    
    def stop(self):
        if self.running:
            self.running = False
            if self.watcher and self.watcher.is_alive():
                self.watcher.join(timeout=2.0)
            self.watcher = None

            # Append POI rows at the end
            if self.result:
                for timestamp, description in self.poi:
                    self.result.add_row(
                        timestamp=timestamp,
                        poi_type=description,
                        cpu_util_pct=None,
                        mem_total_bytes=None,
                        mem_used_bytes=None,
                        mem_available_bytes=None
                    )

                # Close the streaming result file
                self.result.close()

            self.logger.info(f"Stopped CPU watcher {self.id}")


def start_watcher(execution_location: str, save_loc: str, id: int = 0):
    if execution_location == "gpu":
        watcher = GpuWatcher(gpu_id=0, save_loc=save_loc)
    else:
        watcher = CpuWatcher(id=0, save_loc=save_loc)
    
    watcher.start()
    return watcher

def stop_watcher(watcher):
    if watcher:
        time.sleep(1) 
        watcher.stop()