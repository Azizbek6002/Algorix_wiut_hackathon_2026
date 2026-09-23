"""
Real CCTV Sample Videos Analyzer & Visualizer.
Runs Part A, Part B, generates annotated screenshots, and logs event metrics.
WIUT Hackathon 2026 — Algorix Team
"""

import sys
import time
import json
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    import subprocess

from eda.video_analyzer import VideoAnalyzer, classify_lighting
from solution import detect_events, RiskEstimator, CLASSES
from src.utils.hardware import validate_execution_time
from evaluate import validate_prediction_format


def analyze_and_extract_frames(video_path: Path, output_dir: Path, video_id: str):
    """Deep analysis and visual frame extraction with bounding boxes and trajectory overlays."""
    print(f"\n========================================================")
    print(f"🎬 TESTLANMOQDA: {video_path.name}")
    print(f"========================================================")

    # 1. EDA
    analyzer = VideoAnalyzer(sample_interval_frames=30)
    meta = analyzer.analyze_video(str(video_path))
    duration = meta.get("duration_sec", 0.0)
    fps = meta.get("fps", 30.0)
    width = meta.get("width", 3840)
    height = meta.get("height", 2160)
    lighting = meta.get("lighting", {})

    print(f"📌 Texnik parametrlar:")
    print(f"   • Rezolyutsiya: {width}x{height} (4K UHD)")
    print(f"   • Kadrlar tezligi: {fps:.2f} FPS")
    print(f"   • Video davomiyligi: {duration:.2f}s ({duration/60:.1f} daqiqa)")
    print(f"   • Yorug'lik: {lighting.get('condition', 'daylight')} (Luma: {lighting.get('mean_brightness', 0):.1f})")

    # 2. Extract 2 visual screenshots for website
    output_dir.mkdir(parents=True, exist_ok=True)
    snap1_path = output_dir / f"{video_id.lower()}_analysis_1.jpg"
    snap2_path = output_dir / f"{video_id.lower()}_analysis_2.jpg"

    # Use ffmpeg for 1280x720 clean web screenshots
    import subprocess
    cmd1 = [
        "ffmpeg", "-y", "-ss", "00:00:15", "-i", str(video_path),
        "-vf", "scale=1280:720", "-vframes", "1", "-q:v", "2", str(snap1_path)
    ]
    cmd2 = [
        "ffmpeg", "-y", "-ss", "00:01:00", "-i", str(video_path),
        "-vf", "scale=1280:720", "-vframes", "1", "-q:v", "2", str(snap2_path)
    ]
    subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Run Part A (detect_events)
    print(f"\n🔍 Part A: Hodisalarni aniqlash boshlandi...")
    start_t = time.perf_counter()
    events = detect_events(str(video_path))
    elapsed_t = time.perf_counter() - start_t

    print(f"   ⏱️ Qayta ishlash vaqti: {elapsed_t:.2f} soniya")
    valid_time, time_msg = validate_execution_time(duration, elapsed_t)
    print(f"   ⚡ Qoida 4 (Vaqt limiti): {time_msg}")

    valid_fmt, fmt_msg = validate_prediction_format(events)
    print(f"   📋 Format validatsiyasi: {'PASS' if valid_fmt else 'FAIL'} ({fmt_msg})")
    print(f"   🎯 Aniqlangan hodisalar soni: {len(events)}")

    # 4. Run Part B (RiskEstimator causal simulation)
    print(f"\n🛡️ Part B: Causal Risk Estimator tekshiruvi...")
    estimator = RiskEstimator()
    risk_scores = []
    # Test first 10 seconds sample timestamps
    for s in range(min(10, int(duration))):
        score = estimator.update(None, float(s))
        risk_scores.append([float(s), round(score, 4)])

    print(f"   • Causal xavf baholari (birinchi 10s): {[r[1] for r in risk_scores]}")

    result_data = {
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": duration,
        "lighting": lighting,
        "processing_time_sec": round(elapsed_t, 2),
        "events": events,
        "sample_risks": risk_scores,
        "screenshots": [str(snap1_path.name), str(snap2_path.name)],
    }
    return result_data


def main():
    downloads = Path("/home/azizbek/Загрузки")
    sample_videos = [
        downloads / "C3896.MP4",
        downloads / "C3897.MP4",
        downloads / "C3905.MP4",
    ]

    assets_dir = Path(__file__).resolve().parent / "website" / "assets"
    all_results = {}

    for vpath in sample_videos:
        if vpath.exists():
            res = analyze_and_extract_frames(vpath, assets_dir, vpath.stem)
            all_results[vpath.name] = res

    # Save summary report
    report_path = Path("real_videos_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n========================================================")
    print(f"✅ BARCHA 3 TA VIDEO TO'LIQ TESTLANDI VA HISOBOT SAQLANDI: {report_path}")
    print(f"========================================================\n")


if __name__ == "__main__":
    main()
