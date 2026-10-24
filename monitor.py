"""
monitor.py - live system + GPU stats for the text-to-mp3 workload

Run in a separate terminal:
    .venv/Scripts/python.exe monitor.py

Press Ctrl+C to quit.
"""

import subprocess
import sys
import time

import psutil

NVIDIA_SMI_FIELDS = (
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "clocks.mem",
)
QUERY = ",".join(NVIDIA_SMI_FIELDS)


def query_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        temp, gpu_util, mem_util, mem_used, mem_total, pwr, pwr_limit, clk_sm, clk_mem = parts
        return {
            "temp": int(float(temp)),
            "gpu_util": int(float(gpu_util)),
            "mem_util": int(float(mem_util)),
            "mem_used_mb": int(float(mem_used)),
            "mem_total_mb": int(float(mem_total)),
            "power_w": float(pwr),
            "power_limit_w": float(pwr_limit),
            "clk_sm_mhz": int(float(clk_sm)),
            "clk_mem_mhz": int(float(clk_mem)),
        }
    except Exception as e:
        return {"error": str(e)}


def bar(value, total, width=20, fill="#", empty="-"):
    filled = int(round(value / total * width)) if total else 0
    return fill * filled + empty * (width - filled)


def render(gpu, cpu_pct, ram):
    lines = []

    # ── GPU ──────────────────────────────────────────────────
    if "error" in gpu:
        lines.append(f"  GPU  nvidia-smi error: {gpu['error']}")
    else:
        mem_pct = gpu["mem_used_mb"] / gpu["mem_total_mb"] * 100
        pwr_pct = gpu["power_w"] / gpu["power_limit_w"] * 100 if gpu["power_limit_w"] else 0

        temp_c = gpu["temp"]
        temp_color = ""
        if temp_c >= 85:
            temp_color = "\033[91m"   # red
        elif temp_c >= 70:
            temp_color = "\033[93m"   # yellow
        reset = "\033[0m"

        lines.append("  +- GPU --------------------------------------------------------+")
        lines.append(f"  |  Temp      {temp_color}{temp_c:3d} C {reset}  {bar(temp_c, 110)}  ({temp_c}/110 C)           |")
        lines.append(f"  |  Compute   {gpu['gpu_util']:3d} %   {bar(gpu['gpu_util'], 100)}                           |")
        lines.append(f"  |  VRAM BW   {gpu['mem_util']:3d} %   {bar(gpu['mem_util'], 100)}                           |")
        lines.append(f"  |  VRAM      {gpu['mem_used_mb']:5d} / {gpu['mem_total_mb']} MiB  {bar(mem_pct, 100, 14)}                  |")
        lines.append(f"  |  Power     {gpu['power_w']:5.1f} / {gpu['power_limit_w']:.0f} W    {bar(pwr_pct, 100, 14)}                  |")
        lines.append(f"  |  SM clock  {gpu['clk_sm_mhz']:5d} MHz   Mem clock {gpu['clk_mem_mhz']:6d} MHz                |")
        lines.append("  +-------------------------------------------------------------+")

    # ── CPU + RAM ─────────────────────────────────────────────
    ram_used_gb = (ram.total - ram.available) / 1024**3
    ram_total_gb = ram.total / 1024**3
    ram_pct = 100.0 * (ram.total - ram.available) / ram.total

    lines.append("  +- CPU / RAM --------------------------------------------------+")
    lines.append(f"  |  CPU       {cpu_pct:5.1f} %  {bar(cpu_pct, 100)}                           |")
    lines.append(f"  |  RAM       {ram_used_gb:4.1f} / {ram_total_gb:.0f} GB    {bar(ram_pct, 100, 14)}                  |")
    lines.append("  +-------------------------------------------------------------+")

    return "\n".join(lines)


def clear_lines(n):
    # Move cursor up n lines and clear each one.
    for _ in range(n):
        sys.stdout.write("\033[1A\033[2K")


def main(interval: float = 1.0):
    print("  RTX 5060 Ti + System Monitor  (Ctrl+C to quit)\n")
    last_lines = 0

    # Prime the CPU measurement (first call returns 0.0)
    psutil.cpu_percent(interval=None)

    try:
        while True:
            gpu = query_gpu()
            cpu_pct = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()

            output = render(gpu, cpu_pct, ram)
            num_lines = output.count("\n") + 1

            if last_lines:
                clear_lines(last_lines)

            sys.stdout.write(output + "\n")
            sys.stdout.flush()
            last_lines = num_lines

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    # Ensure the terminal can handle ANSI colour codes on Windows
    if sys.platform == "win32":
        import os
        os.system("")   # enables VT processing in conhost/Windows Terminal
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
