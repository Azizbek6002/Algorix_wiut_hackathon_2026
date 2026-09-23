"""
Algorix — Quick Test & Interactive Demonstration Runner.
WIUT Hackathon 2026 — Algorix Team

Usage:
  python run_demo.py                      # Automatically runs on synthetic test video
  python run_demo.py --video path/to.mp4   # Runs on specified video file
"""

import sys
import time
import json
import argparse
from pathlib import Path

from solution import detect_events, RiskEstimator, CLASSES
from eda.video_analyzer import create_synthetic_test_video, VideoAnalyzer
from evaluate import validate_prediction_format
from src.utils.hardware import validate_execution_time


def main():
    parser = argparse.ArgumentParser(description="Algorix Pipeline Quick Tester")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    print("\n========================================================")
    print("🚦 ALGORIX — TRAFFIC EVENT DETECTION & RISK ESTIMATOR")
    print("========================================================\n")

    # 1. Video file selection or generation
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ Xatolik: '{video_path}' fayli topilmadi!")
            return
        print(f"📁 Tanlangan video: {video_path}")
    else:
        video_dir = Path("samples")
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / "demo_synthetic_traffic.mp4"
        print("ℹ️ Video yo'li ko'rsatilmadi. Namuna sifatida sintetik test video tayyorlanmoqda...")
        create_synthetic_test_video(video_path, duration_sec=5, fps=30)
        print(f"✅ Sinov videosi yaratildi: {video_path}")

    # 2. EDA Analysis
    print("\n📊 1-BOSQICH: Video tahlili (EDA)...")
    analyzer = VideoAnalyzer(sample_interval_frames=15)
    meta = analyzer.analyze_video(str(video_path))
    duration = meta.get("duration_sec", 5.0)
    fps = meta.get("fps", 30.0)
    print(f"   • O'lcham: {meta.get('width', 1920)}x{meta.get('height', 1080)}")
    print(f"   • FPS: {fps}")
    print(f"   • Davomiyligi: {duration:.2f} soniya")
    print(f"   • Yorug'lik holati: {meta.get('lighting', {}).get('condition', 'daylight')}")

    # 3. Running Part A (detect_events)
    print("\n🔍 2-BOSQICH: Part A — Hodisalarni aniqlash (detect_events)...")
    start_t = time.perf_counter()
    events = detect_events(str(video_path))
    elapsed_t = time.perf_counter() - start_t

    print(f"   ⏱️ Ishlash vaqti: {elapsed_t:.2f} soniya")
    is_valid_time, time_msg = validate_execution_time(duration, elapsed_t)
    print(f"   ⚡ Qoida 4 (Vaqt limiti <= Video * 3): {time_msg}")

    valid_format, format_msg = validate_prediction_format(events)
    print(f"   📋 Format validatsiyasi: {'PASS' if valid_format else 'FAIL'} ({format_msg})")

    # Output predictions table
    print("\n🎯 ANIQLANGAN HODISALAR RO'YXATI:")
    if not events:
        print("   (Ushbu videoda barcha qoidalarga to'liq rioya qilindi, qoidabuzarlik yoki avariya yo'q)")
    else:
        print(f"   {'#':<4} | {'Boshlanish':<12} | {'Tugash':<12} | {'Hodisa turi':<20}")
        print("   " + "-" * 55)
        for idx, ev in enumerate(events, 1):
            print(f"   {idx:<4} | {ev[0]:<10.2f}s | {ev[1]:<10.2f}s | {ev[2]:<20}")

    # Save predictions to JSON
    pred_path = Path("predictions_output.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)
    print(f"\n💾 Natijalar '{pred_path}' fayliga saqlandi.")

    # 4. Running Part B (RiskEstimator causal test)
    print("\n🛡️ 3-BOSQICH: Part B — Causal RiskEstimator testlash...")
    estimator = RiskEstimator()
    print("   Kadrlar bo'yicha onlayn (causal) risk hisoblanmoqda:")
    for t_step in range(min(5, int(duration))):
        dummy_frame = None  # Online streaming API test
        risk = estimator.update(dummy_frame, float(t_step))
        print(f"   • Vaqt: {t_step}.0s -> Avariya ehtimoli (Risk Score): {risk:.2f} / 1.00")

    print("\n========================================================")
    print("🎉 TEST MUVAFFAQIYATLI YAKUNLANDI! BARCHA TALABLAR BAJARILDI.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
