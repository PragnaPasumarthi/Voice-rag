import subprocess
import time
import re
import sys

proc = subprocess.Popen(
    [r"C:\Users\pragn\AppData\Local\Temp\cloudflared.exe", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

url_found = False
for line in proc.stdout:
    print(line.strip())
    if "trycloudflare.com" in line:
        match = re.search(r"(https://[^\s]+trycloudflare\.com[^\s]*)", line)
        if match:
            print(f"\n\nPUBLIC_URL={match.group(1)}")
            sys.stdout.flush()
            url_found = True
            break

if not url_found:
    time.sleep(20)
