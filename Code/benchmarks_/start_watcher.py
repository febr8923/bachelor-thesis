#!/usr/bin/env python
import sys
import signal
import time
from watcher import GpuWatcher, CpuWatcher

def main():
    if len(sys.argv) < 4:
        print("Usage: python start_watcher.py <type> <id> <output_file>")
        print("  type: 'gpu' or 'cpu'")
        print("  id: GPU ID or CPU watcher ID")
        print("  output_file: Path to save monitoring data")
        sys.exit(1)
    
    watcher_type = sys.argv[1]
    watcher_id = int(sys.argv[2])
    output_file = sys.argv[3]
    
    if watcher_type == "gpu":
        watcher = GpuWatcher(watcher_id, output_file)
    elif watcher_type == "cpu":
        watcher = CpuWatcher(watcher_id, output_file, interval=0.1)
    else:
        print(f"Unknown watcher type: {watcher_type}")
        sys.exit(1)
    
    # Handle SIGTERM and SIGINT gracefully
    def signal_handler(sig, frame):
        print(f"\nStopping {watcher_type} watcher...")
        watcher.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    watcher.start()
    print(f"Started {watcher_type} watcher {watcher_id}, saving to {output_file}")
    print("Press Ctrl+C to stop (or send SIGTERM)")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\nStopping {watcher_type} watcher...")
        watcher.stop()

if __name__ == "__main__":
    main()
