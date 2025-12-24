import subprocess
import threading
import csv
import time
from datetime import datetime
import psutil
import logging
from pathlib import Path

class GpuWatcher:
    def __init__(self, gpu_id, save_loc):
        self.gpu_id = gpu_id
        self.stop_event = None
        self.watcher = None
        self.save_loc = save_loc

    def track_gpu(self, stop_event):
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

        lines = []

        with open(self.save_loc, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "gpu_id", "gpu_util%", "mem_util%", "mem_total_mb", "mem_free_mb", "mem_used_mb", "mem_reserved_mb"])

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
                            gpu_util = fields[0]
                            mem_util = fields[1]
                            mem_total = fields[2]
                            mem_free = fields[3]
                            mem_used = fields[4]
                            mem_reserved = fields[5]
                            
                            writer.writerow([timestamp, self.gpu_id, gpu_util, mem_util, 
                                       mem_total, mem_free, mem_used, mem_reserved])
                            
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


class CpuWatcher:
    def __init__(self, id: int, save_loc: str, interval: float = 1.0):
        self.id = id
        self.save_loc = Path(save_loc)
        self.csv_file = self.save_loc
        self.running = False
        self.watcher = None
        self.interval = interval
        self._lock = threading.Lock()
        self._first_row = True
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"CpuWatcher_{id}")

    def track_cpu(self):
        """Efficiently track CPU and memory usage using psutil."""
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
                
                with self._lock:
                    with self.csv_file.open('a', newline='') as f:
                        writer = csv.writer(f)
                        if self._first_row:
                            writer.writerow(['timestamp', 'cpu_util_pct', 'mem_total_bytes', 
                                           'mem_used_bytes', 'mem_available_bytes'])
                            self._first_row = False
                        writer.writerow([timestamp, cpu_util, mem_total, mem_used, mem_free])
                
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
            self.logger.info(f"Stopped CPU watcher {self.id}")
