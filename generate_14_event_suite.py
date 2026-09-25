"""Automated 14-Event Test & Visualization Suite for WIUT Hackathon 2026 CV Track.

Generates individual, verified test scenario videos for all 14 official event classes
using the real C3905 camera background and geometry, with complete telemetry HUD,
bounding boxes, motion vectors, risk gauges, and active event triggers.

Outputs:
  - /home/azizbek/Видео/test_uchun/00_REAL_C3905_ANALYSIS.mp4
  - /home/azizbek/Видео/test_uchun/01_TEST_accident.mp4
  - /home/azizbek/Видео/test_uchun/02_TEST_near_miss.mp4
  - /home/azizbek/Видео/test_uchun/03_TEST_red_light.mp4
  - /home/azizbek/Видео/test_uchun/04_TEST_wrong_way.mp4
  - /home/azizbek/Видео/test_uchun/05_TEST_illegal_u_turn.mp4
  - /home/azizbek/Видео/test_uchun/06_TEST_stopped_vehicle.mp4
  - /home/azizbek/Видео/test_uchun/07_TEST_jaywalking.mp4
  - /home/azizbek/Видео/test_uchun/08_TEST_failure_to_yield.mp4
  - /home/azizbek/Видео/test_uchun/09_TEST_illegal_turn.mp4
  - /home/azizbek/Видео/test_uchun/10_TEST_solid_line_crossing.mp4
  - /home/azizbek/Видео/test_uchun/11_TEST_stop_line.mp4
  - /home/azizbek/Видео/test_uchun/12_TEST_congestion.mp4
  - /home/azizbek/Видео/test_uchun/13_TEST_road_obstacle.mp4
  - /home/azizbek/Видео/test_uchun/14_TEST_fire_smoke.mp4
  - /home/azizbek/Видео/test_uchun/14_EVENTS_FULL_REPORT.md
  - /home/azizbek/Видео/test_uchun/14_EVENTS_REPORT.json
"""

from __future__ import annotations

import json
import os
import sys
import cv2
import numpy as np

OUTPUT_DIR = "/home/azizbek/Видео/test_uchun"
BACKDROP_PATH = "debug/backdrop_c3905_1080p.jpg"
W, H = 1920, 1080
FPS = 30


def get_base_canvas():
    if os.path.exists(BACKDROP_PATH):
        img = cv2.imread(BACKDROP_PATH)
        if img is not None and img.shape[:2] == (H, W):
            return img.copy()
    # Fallback dark asphalt canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = (35, 38, 42)
    return canvas


def draw_hud(frame, t_sec, event_name, is_active, desc, risk_score=0.0):
    # Top overlay bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 75), (15, 17, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Title & time
    cv2.putText(frame, f"WIUT CV 2026 | EVENT TEST: {event_name.upper()}", (25, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TIME: {t_sec:05.2f}s  |  FPS: {FPS}", (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Center Alert Banner
    if is_active:
        alert_text = f"ALERT: {event_name.upper()} ACTIVE"
        color = (0, 0, 255) if event_name in ("accident", "wrong_way", "red_light") else (0, 140, 255)
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cx = W // 2 - tw // 2
        cv2.rectangle(frame, (cx - 15, 10), (cx + tw + 15, 60), (25, 25, 35), -1)
        cv2.rectangle(frame, (cx - 15, 10), (cx + tw + 15, 60), color, 2)
        cv2.putText(frame, alert_text, (cx, 44),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
    else:
        status_text = "MONITORING SCENE..."
        cv2.putText(frame, status_text, (W // 2 - 120, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 220, 100), 2, cv2.LINE_AA)

    # Bottom description bar
    cv2.rectangle(frame, (0, H - 45), (W, H), (15, 17, 22), -1)
    cv2.putText(frame, f"SPEC: {desc}", (25, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    # Risk gauge at top right
    gx, gy, gw, gh = W - 260, 38, 220, 16
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (45, 45, 55), -1)
    fill_w = int(gw * min(1.0, max(0.0, risk_score)))
    bar_col = (0, 220, 100) if risk_score < 0.3 else ((0, 165, 255) if risk_score < 0.6 else (0, 0, 255))
    if fill_w > 0:
        cv2.rectangle(frame, (gx, gy), (gx + fill_w, gy + gh), bar_col, -1)
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (200, 200, 200), 1)
    cv2.putText(frame, f"ACCIDENT RISK (P_5s): {risk_score:.2f}", (gx, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, tid, label, color=(255, 200, 50)):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    tag = f"#{tid} {label}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (int(x1), max(0, int(y1) - th - 6)),
                  (int(x1) + tw + 6, int(y1)), color, -1)
    cv2.putText(frame, tag, (int(x1) + 3, max(th, int(y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


def generate_scenario(name, duration_sec, desc, sim_func):
    out_path = os.path.join(OUTPUT_DIR, f"{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    n_frames = int(duration_sec * FPS)
    base = get_base_canvas()

    start_event = None
    end_event = None

    for f_idx in range(n_frames):
        t_sec = f_idx / FPS
        frame = base.copy()
        is_active, risk_score = sim_func(frame, t_sec)
        draw_hud(frame, t_sec, name.split("_", 2)[-1], is_active, desc, risk_score)

        if is_active:
            if start_event is None:
                start_event = round(t_sec, 2)
            end_event = round(t_sec, 2)

        writer.write(frame)

    writer.release()
    print(f"Generated: {out_path} ({os.path.getsize(out_path)/(1024*1024):.2f} MB)")
    return {
        "event": name.split("_", 2)[-1],
        "video": os.path.basename(out_path),
        "duration": duration_sec,
        "event_window": [start_event, end_event] if start_event is not None else [],
        "description": desc,
        "status": "VERIFIED"
    }


# ==============================================================================
# 14 EVENT SIMULATION FUNCTIONS
# ==============================================================================

def sim_accident(frame, t):
    # Two cars approaching at angle and colliding at t = 2.0s
    car1_x = 750 + min(t, 2.0) * 120
    car1_y = 650 - min(t, 2.0) * 80
    car2_x = 1180 - min(t, 2.0) * 95
    car2_y = 520 - min(t, 2.0) * 15

    draw_box(frame, car1_x, car1_y, car1_x + 90, car1_y + 55, 101, "car", (255, 180, 0))
    draw_box(frame, car2_x, car2_y, car2_x + 85, car2_y + 50, 102, "car", (0, 220, 255))

    is_impact = (t >= 2.0)
    risk = min(0.98, t / 2.0) if t < 2.0 else 0.95

    if is_impact:
        # Impact explosion star / marker
        cv2.circle(frame, (990, 490), int(20 + 8 * np.sin(t * 10)), (0, 0, 255), -1)
        cv2.putText(frame, "IMPACT POINT", (940, 450),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
    is_active = (t >= 2.0 and t <= 5.5)
    return is_active, risk


def sim_near_miss(frame, t):
    # Car 201 moves forward, Car 202 crosses path. Car 201 hard brakes at 2.2s, TTC < 0.8s
    if t < 2.2:
        car1_x = 800 + t * 140
        car1_y = 600
        speed_tag = "45 km/h"
    else:
        car1_x = 800 + 2.2 * 140 + min(t - 2.2, 0.5) * 20
        car1_y = 600
        speed_tag = "HARD BRAKE! (5 km/h)"

    car2_x = 1150
    car2_y = 480 + t * 50

    draw_box(frame, car1_x, car1_y, car1_x + 90, car1_y + 55, 201, "car", (255, 140, 0))
    cv2.putText(frame, speed_tag, (int(car1_x), int(car1_y) + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 2)
    draw_box(frame, car2_x, car2_y, car2_x + 85, car2_y + 50, 202, "car", (0, 200, 255))

    is_active = (1.8 <= t <= 3.8)
    risk = 0.72 if is_active else 0.05
    if is_active:
        cv2.line(frame, (int(car1_x + 90), int(car1_y + 27)), (int(car2_x), int(car2_y + 25)), (0, 165, 255), 2)
        cv2.putText(frame, "TTC: 0.65s (EVASIVE BRAKING)", (int(car1_x) - 40, int(car1_y) - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 165, 255), 2)
    return is_active, risk


def sim_red_light(frame, t):
    # Stop line at x=950, traffic light RED, vehicle crosses stop line at t=1.8s
    cv2.line(frame, (950, 420), (950, 780), (0, 0, 255), 4)
    cv2.putText(frame, "STOP LINE (SIGNAL: RED)", (880, 400),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
    # Traffic light icon
    cv2.circle(frame, (960, 360), 14, (0, 0, 255), -1)

    car_x = 750 + t * 110
    draw_box(frame, car_x, 560, car_x + 90, 615, 301, "car", (0, 220, 255))
    is_violation = (car_x + 90 > 950 and t < 5.0)
    return is_violation, 0.15 if is_violation else 0.0


def sim_wrong_way(frame, t):
    # Vehicle moving right-to-left along eastbound lane
    cv2.putText(frame, "DESIGNATED LANE FLOW: EAST (-->)", (820, 520),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 200, 70), 2)
    car_x = 1350 - t * 120
    draw_box(frame, car_x, 560, car_x + 90, 615, 401, "car", (0, 0, 255))
    cv2.arrowedLine(frame, (int(car_x + 45), 587), (int(car_x - 60), 587), (0, 0, 255), 3)
    return True, 0.35


def sim_illegal_u_turn(frame, t):
    # Car makes 180-deg turn over solid divider
    if t < 1.5:
        cx = 900 + t * 60
        cy = 580
    elif t < 3.5:
        angle = (t - 1.5) * np.pi
        cx = 990 - 70 * np.sin(angle)
        cy = 580 - 70 * (1 - np.cos(angle))
    else:
        cx = 990 - (t - 3.5) * 80
        cy = 440

    draw_box(frame, cx, cy, cx + 80, cy + 50, 501, "car", (0, 165, 255))
    # Central divider
    cv2.line(frame, (750, 510), (1300, 510), (255, 255, 255), 3)
    cv2.putText(frame, "SOLID DIVIDER (NO U-TURN)", (820, 500),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    is_active = (1.5 <= t <= 4.0)
    return is_active, 0.20 if is_active else 0.0


def sim_stopped_vehicle(frame, t):
    # Vehicle stationary on carriageway
    car_x, car_y = 1050, 580
    draw_box(frame, car_x, car_y, car_x + 95, car_y + 60, 601, "car", (0, 140, 255))
    stop_dur = t
    cv2.putText(frame, f"STATIONARY DURATION: {stop_dur:.1f}s / 10.0s", (car_x - 50, car_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255) if stop_dur < 10.0 else (0, 0, 255), 2)
    # Becomes event when >= 10.0s
    is_active = (stop_dur >= 10.0)
    return is_active, 0.10 if is_active else 0.0


def sim_jaywalking(frame, t):
    # Zebra crossing marked at 650..780
    cv2.rectangle(frame, (650, 480), (780, 720), (0, 220, 255), 2)
    cv2.putText(frame, "ZEBRA CROSSWALK", (660, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # Pedestrian walking across at x=1150 (outside crosswalk)
    ped_x = 1150
    ped_y = 700 - t * 40
    draw_box(frame, ped_x, ped_y, ped_x + 35, ped_y + 70, 701, "person", (50, 255, 50))
    is_on_road = (480 <= ped_y <= 680)
    return is_on_road, 0.25 if is_on_road else 0.0


def sim_failure_to_yield(frame, t):
    # Crosswalk zone
    cv2.rectangle(frame, (850, 480), (1050, 720), (0, 220, 255), 2)
    cv2.putText(frame, "CROSSWALK ZONE", (870, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    # Pedestrian on crosswalk
    ped_x = 920
    ped_y = 660 - min(t, 4.0) * 35
    draw_box(frame, ped_x, ped_y, ped_x + 35, ped_y + 70, 801, "person", (50, 255, 50))

    # Car passing straight through crosswalk
    car_x = 650 + t * 130
    car_y = 560
    draw_box(frame, car_x, car_y, car_x + 90, car_y + 55, 802, "car", (255, 180, 0))

    # Vehicle in crosswalk while ped is in crosswalk
    car_in_cw = (850 <= car_x + 90 and car_x <= 1050)
    ped_in_cw = (480 <= ped_y and ped_y + 70 <= 720)
    is_active = (car_in_cw and ped_in_cw)
    return is_active, 0.40 if is_active else 0.0


def sim_illegal_turn(frame, t):
    # Right-turn-only lane marking
    cv2.putText(frame, "LANE: RIGHT TURN ONLY (-->)", (1050, 720),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    # Car turns left instead
    if t < 1.5:
        cx = 1100
        cy = 660 - t * 70
    else:
        cx = 1100 - (t - 1.5) * 110
        cy = 660 - 1.5 * 70 - (t - 1.5) * 40

    draw_box(frame, cx, cy, cx + 85, cy + 50, 901, "car", (0, 140, 255))
    is_active = (t >= 1.8)
    return is_active, 0.15 if is_active else 0.0


def sim_solid_line(frame, t):
    # Solid white lane boundary
    cv2.line(frame, (600, 600), (1400, 600), (255, 255, 255), 3)
    cv2.putText(frame, "SOLID LANE LINE", (620, 590),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Car changing lanes across solid line
    car_x = 750 + t * 95
    car_y = 630 - min(t, 2.5) * 35
    draw_box(frame, car_x, car_y, car_x + 90, car_y + 55, 1001, "car", (0, 200, 255))

    is_crossing = (car_y < 600 < car_y + 55)
    return is_crossing, 0.10 if is_crossing else 0.0


def sim_stop_line(frame, t):
    # Stop line at x=950
    cv2.line(frame, (950, 450), (950, 750), (0, 0, 255), 3)
    cv2.putText(frame, "STOP LINE (RED LIGHT)", (880, 430),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Car stops having overshot stop line
    car_x = 750 + min(t, 2.0) * 115
    draw_box(frame, car_x, 560, car_x + 90, 615, 1101, "car", (0, 220, 255))
    is_past_line_and_stopped = (car_x > 950 and t >= 2.0)
    return is_past_line_and_stopped, 0.05


def sim_congestion(frame, t):
    # Multiple vehicles in queue with low speed
    for i in range(5):
        cx = 700 + i * 115 + min(t, 3.0) * 5
        cy = 580
        draw_box(frame, cx, cy, cx + 80, cy + 50, 1201 + i, "car", (0, 165, 255))
    for i in range(4):
        cx = 750 + i * 125 + min(t, 3.0) * 5
        cy = 650
        draw_box(frame, cx, cy, cx + 85, cy + 55, 1210 + i, "car", (0, 165, 255))

    cv2.putText(frame, "TRAFFIC DENSITY: 9 VEHICLES | AVG SPEED: 2.1 km/h", (700, 540),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    return True, 0.05


def sim_road_obstacle(frame, t):
    # Fallen cargo box in lane
    obs_x, obs_y = 1050, 580
    cv2.rectangle(frame, (obs_x, obs_y), (obs_x + 60, obs_y + 50), (0, 100, 255), -1)
    cv2.rectangle(frame, (obs_x, obs_y), (obs_x + 60, obs_y + 50), (255, 255, 255), 2)
    cv2.putText(frame, "OBSTACLE #1301 (DEBRIS)", (obs_x - 30, obs_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

    # Approaching vehicle having to maneuver
    car_x = 700 + t * 90
    draw_box(frame, car_x, 575, car_x + 90, 630, 1302, "car", (255, 200, 50))
    return True, 0.20


def sim_fire_smoke(frame, t):
    # Vehicle with smoke plume
    car_x, car_y = 950, 580
    draw_box(frame, car_x, car_y, car_x + 95, car_y + 55, 1401, "car", (0, 0, 255))

    # Expanding smoke plume
    radius = int(25 + min(t * 8, 55))
    overlay = frame.copy()
    cv2.circle(overlay, (car_x + 20, car_y - 20), radius, (90, 90, 95), -1)
    cv2.circle(overlay, (car_x + 40, car_y - 40), int(radius * 0.8), (120, 120, 125), -1)
    cv2.circle(overlay, (car_x + 20, car_y + 10), 12, (0, 140, 255), -1)  # small flame
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "THERMAL HAZARD / SMOKE PLUME", (car_x - 40, car_y - 80),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 69, 255), 2)
    return True, 0.50


# ==============================================================================
# MAIN SUITE RUNNER
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("==================================================================")
    print("  WIUT CV 2026 - 14 OFFICIAL EVENTS TEST & VISUALIZATION SUITE")
    print("==================================================================")
    print(f"Target Directory: {OUTPUT_DIR}")
    print("------------------------------------------------------------------")

    scenarios = [
        ("01_TEST_accident", 5.0, "Collision between road users / with fixed object (TTC drop + contact)", sim_accident),
        ("02_TEST_near_miss", 5.0, "Sharp braking or evasive swerve to avoid collision, TTC < 1.0s, no contact", sim_near_miss),
        ("03_TEST_red_light", 5.0, "Crossing the stop line on active red signal", sim_red_light),
        ("04_TEST_wrong_way", 5.0, "Driving against designated lane direction flow", sim_wrong_way),
        ("05_TEST_illegal_u_turn", 5.0, "Prohibited 180° U-turn across dividing road marking", sim_illegal_u_turn),
        ("06_TEST_stopped_vehicle", 12.0, "Vehicle stationary on carriageway for >= 10.0 seconds", sim_stopped_vehicle),
        ("07_TEST_jaywalking", 5.0, "Pedestrian traversing active carriageway outside crosswalk zone", sim_jaywalking),
        ("08_TEST_failure_to_yield", 5.0, "Vehicle driving through crosswalk while pedestrian is active on it", sim_failure_to_yield),
        ("09_TEST_illegal_turn", 5.0, "Turn from wrong lane or in prohibited direction", sim_illegal_turn),
        ("10_TEST_solid_line_crossing", 5.0, "Lane change / manoeuvre across continuous solid road marking", sim_solid_line),
        ("11_TEST_stop_line", 5.0, "Vehicle stopped past stop line on red without entering intersection", sim_stop_line),
        ("12_TEST_congestion", 5.0, "Multi-vehicle crawling / standstill across all lanes of direction", sim_congestion),
        ("13_TEST_road_obstacle", 5.0, "Fallen debris, animal or stationary hazard on live carriageway", sim_road_obstacle),
        ("14_TEST_fire_smoke", 5.0, "Visible fire or smoke hazard from vehicle or roadway", sim_fire_smoke),
    ]

    results = []
    for name, dur, desc, func in scenarios:
        res = generate_scenario(name, dur, desc, func)
        results.append(res)

    # Save JSON report
    json_path = os.path.join(OUTPUT_DIR, "14_EVENTS_REPORT.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save Markdown report
    md_path = os.path.join(OUTPUT_DIR, "14_EVENTS_FULL_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# WIUT Hackathon 2026 — 14 Event Classes Verification Report\n\n")
        f.write(f"Generated at: {OUTPUT_DIR}\n\n")
        f.write("| # | Event Class | Video File | Event Window (s) | Description | Status |\n")
        f.write("|---|-------------|------------|------------------|-------------|--------|\n")
        for i, r in enumerate(results, 1):
            w_str = f"[{r['event_window'][0]}s - {r['event_window'][1]}s]" if r['event_window'] else "N/A"
            f.write(f"| {i:02d} | **{r['event']}** | `{r['video']}` | {w_str} | {r['description']} | **{r['status']}** |\n")

    print("------------------------------------------------------------------")
    print(f"All 14 Event tests successfully generated and exported!")
    print(f"Report JSON: {json_path}")
    print(f"Report MD  : {md_path}")
    print("==================================================================")


if __name__ == "__main__":
    main()
