import subprocess
import os
import time


print("[CONTROLLER] Starting benchmark subprocess...")
proc = subprocess.Popen(["python", "benchmark.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

pid = None
for line in proc.stdout:
    print("[BENCHMARK]", line.strip())

    if "PID:" in line:
        pid = int(line.strip().split("PID:")[1])
        print(f"[CONTROLLER] Found benchmark PID: {pid}")
    elif "READY_FOR_PERF" in line:
        break


print("[CONTROLLER] Launching perf...")
perf = subprocess.Popen(["perf", "stat", "-e", "cache-misses,cache-references,cycles,instructions", "-p", str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

proc.wait()
perf.terminate()
stdout, stderr = perf.communicate()
print(f"[PERF OUTPUT]: {stderr}")
print("[CONTROLLER] Done")