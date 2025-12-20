import subprocess
import threading
import csv
import time
from datetime import datetime

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
    def __init__(self, id, save_loc):
        self.id = id
        self.running = False
        self.watcher = None
        self.save_loc = save_loc

    def track_cpu(self):
        cmd_util = ['top', "-bn1"]
        cmd_mem = ['free', "--si", "--bytes"]
        while self.running:
            try:
                cpu_line = subprocess.check_output(["top", "-bn1"]).decode().splitlines()[7]
                idle = float(cpu_line.split('%id')[0].split(',')[-1].strip('%'))
                cpu_util = 100 - idle

                mem_line = subprocess.check_output(["free", "--si", "--bytes"]).decode().splitlines()[1]
                mem_parts = mem_line.split()
                mem_total, mem_used, mem_free = map(int, [mem_parts[1], mem_parts[2], mem_parts[3]])



                timestamp = datetime.now().isoformat()
                with open(self.csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if first_row:
                            writer.writerow(['timestamp', 'cpu_util_pct', 'mem_total_bytes', 
                                        'mem_used_bytes', 'mem_free_bytes'])
                            first_row = False
                        writer.writerow([timestamp, cpu_util, mem_total, mem_used, mem_free])
            except Exception as e:
                pass

            time.sleep(0.1)
    
    def start(self):
        if not self.running:
            self.running = True
            self.watcher = threading.Thread(target=self.track_cpu, daemon=True)
            self.watcher.start()

    def stop(self):
        self.running = False
        
        if self.watcher:
            self.watcher.join()
        
        self.watcher = None