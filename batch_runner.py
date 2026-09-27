#!/usr/bin/env python3
"""
Batch Runner for Full Video Evaluation & Annotation
===================================================
Processes the 3 full videos from /home/azizbek/Загрузки:
  1. C3905.MP4 (127.6s, ~2m 7s)
  2. C3897.MP4 (317.8s, ~5m 17s)
  3. C3896.MP4 (340.3s, ~5m 40s)

Outputs to: /home/azizbek/Видео/test_uchun/
"""
import os
import sys
import subprocess
import time
import json

VIDEOS = [
    "/home/azizbek/Загрузки/C3905.MP4",
    "/home/azizbek/Загрузки/C3897.MP4",
    "/home/azizbek/Загрузки/C3896.MP4",
]

OUTPUT_DIR = "/home/azizbek/Видео/test_uchun"
PYTHON_BIN = "/home/azizbek/miniconda3/bin/python3.14"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    
    print("=" * 72)
    print("  BATCH RUNNER: FULL VIDEO EVALUATION & ANNOTATION")
    print("=" * 72)
    print(f"Total videos to process: {len(VIDEOS)}")
    for v in VIDEOS:
        sz_gb = os.path.getsize(v) / (1024**3)
        print(f" - {os.path.basename(v)} ({sz_gb:.2f} GB)")
    print("-" * 72)
    
    t_start_all = time.time()
    
    for idx, v_path in enumerate(VIDEOS, 1):
        v_name = os.path.basename(v_path)
        print(f"\n[{idx}/{len(VIDEOS)}] STARTING: {v_name}")
        t0 = time.time()
        
        # Command: run with duration 0 (full length)
        cmd = [
            PYTHON_BIN, "export_test_video.py",
            "--video", v_path,
            "--duration", "0",
            "--stride", "3",
            "--imgsz", "640",
            "--output-dir", OUTPUT_DIR,
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        output_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output_lines.append(line)
            
        proc.wait()
        el = time.time() - t0
        
        # Parse output for summary
        saved_file = None
        ev_count = None
        events = []
        
        for line in output_lines:
            if "Saved:" in line:
                saved_file = line.split("Saved:")[-1].strip()
            elif "Detected Events Count:" in line:
                try:
                    ev_count = int(line.split(":")[-1].strip())
                except:
                    pass
            elif line.strip().startswith("[") and "]" in line and ("ACCIDENT" in line or "JAYWALKING" in line or "WRONG_WAY" in line or "CONGESTION" in line or "STOPPED" in line or "FAILURE" in line or "SOLID" in line or "TURN" in line or "RED" in line or "STOP" in line or "OBSTACLE" in line or "FIRE" in line):
                events.append(line.strip())
                
        results[v_name] = {
            "elapsed_sec": round(el, 1),
            "exit_code": proc.returncode,
            "saved_file": saved_file,
            "events_count": ev_count if ev_count is not None else len(events),
            "events": events,
        }
        
        print(f"[{idx}/{len(VIDEOS)}] FINISHED {v_name} in {el/60:.1f} minutes. Code: {proc.returncode}")
        print("-" * 72)
        
    t_all = time.time() - t_start_all
    summary_path = os.path.join(OUTPUT_DIR, "batch_evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 72)
    print(f"ALL VIDEOS PROCESSED in {t_all/60:.1f} minutes!")
    print(f"Summary written to: {summary_path}")
    print("=" * 72)
    
if __name__ == "__main__":
    main()
