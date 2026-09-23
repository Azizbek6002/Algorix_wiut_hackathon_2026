# 🚦 Algorix — Toyota Traffic Event Detection & Accident Anticipation

**Hackathon:** WIUT Hackathon 2026 — Computer Vision
**Topshiriq kodi:** BE86F4D5
**Jamoa:** Algorix
**Deadline:** 2026-yil 27-sentabr, 23:59 (Toshkent vaqti)

Bu README — loyihani **noldan (0-dan)** boshlab, to'liq ishlaydigan holatga yetkazish uchun modullarga bo'lingan texnik reja va kontrakt hujjati. Gemini Pro (yoki har qanday kodlash agenti) shu faylni asosiy "spec" sifatida ishlatib, har bir modulni ketma-ket amalga oshirishi kerak.

---

## 0. Vazifaning mohiyati (bir gapda)

> Fixed CCTV kameradan olingan video beriladi. Tizim video ichidagi trafik hodisalarini (`start_time, end_time, event_type`) aniqlashi (**Part A**, majburiy) va har bir frame uchun keyingi 5 soniyada avariya bo'lish ehtimolini (**Part B**, bonus) real-vaqt (causal, kelajakni ko'rmasdan) tarzda hisoblashi kerak.

Kamera **fixed** bo'lgani uchun eng katta strategik ustunlik: universal AI emas, **aynan shu kameraga moslashtirilgan scene-geometry asosidagi Computer Vision pipeline** qurish mumkin (lane, stop-line, crossing koordinatalarini bir marta belgilab, keyin rule-engine bilan ishlatish).

---

## 1. Qattiq qoidalar (Hard Constraints) — HECH QACHON buzilmasin

| # | Qoida | Tafsilot |
|---|---|---|
| 1 | **Internet yo'q** | Baholash vaqtida internet o'chirilgan. `requests.post`, OpenAI/Gemini/Claude/HuggingFace API — TAQIQLANGAN. |
| 2 | **Faqat open-weight model** | YOLO, RT-DETR, ByteTrack, Qwen-VL, InternVL va sh.k. — model weight'lari paket ichida bo'lishi shart. |
| 3 | **Model hajmi ≤ 5 GB** | Jami barcha weight fayllar. |
| 4 | **Vaqt limiti: video davomiyligi × 3** | Masalan 10 daqiqalik video → 30 daqiqa ichida tugashi shart. Aks holda o'sha video bo'sh prediction deb baholanadi. |
| 5 | **Hardware:** 1×GPU (T4-class, 16GB VRAM), 8 CPU, 32GB RAM | Shu resurs ichida sig'ishi kerak — juda katta model xavfli. |
| 6 | **Deterministik natija** | `random.seed`, `np.random.seed`, `torch.manual_seed`, CUDA seedlarni qat'iy o'rnatish. Run1 ≈ Run2. |
| 7 | **RiskEstimator kelajakni ko'ra olmaydi** | Frame N kelganda Frame N+1, N+2... ochib ko'rish — **DISQUALIFICATION** sababi. |
| 8 | **Public dataset ishlatish mumkin** | DoTA, CCD, DAD, CADP, UCF-Crime RoadAccidents, UA-DETRAC, BDD100K va h.k. — lekin README'da dataset nomi + license yozilishi shart. |
| 9 | **Root'da `solution.py` bo'lishi shart** | Pastda aniq kontrakt (bo'lim 6) berilgan. |

---

## 2. 14 ta Event Class (aniq shu nomlar ishlatilishi shart)

```python
CLASSES = [
    "accident",            # 1. To'qnashuv / avariya
    "near_miss",           # 2. Avariyaga yaqin holat (keskin tormoz/burilish)
    "red_light",           # 3. Qizil chiroqda stop-line'dan o'tish
    "wrong_way",           # 4. Qarama-qarshi yo'nalishda yurish
    "illegal_u_turn",      # 5. Taqiqlangan U-turn
    "stopped_vehicle",     # 6. ≥10s carriageway'da to'xtab qolish (navbat emas)
    "jaywalking",          # 7. Piyoda crossing'dan tashqarida yo'lga chiqishi
    "failure_to_yield",    # 8. Piyodaga yo'l bermaslik
    "illegal_turn",        # 9. Taqiqlangan/noto'g'ri lane'dan burilish
    "solid_line_crossing", # 10. Uzluksiz chiziqni bosib o'tish
    "stop_line",           # 11. Stop-line'dan o'tib, lekin intersection'ga kirmay to'xtash
    "congestion",          # 12. Barcha lane'larda tirbandlik
    "road_obstacle",       # 13. Yo'ldagi to'siq (debris, hayvon, tushgan yuk)
    "fire_smoke",          # 14. Yong'in/tutun
]
```

⚠️ **Muhim nozik farqlar:**
- `red_light` → stop-line'dan o'tib, **intersection tomon davom etadi**.
- `stop_line` → stop-line'dan o'tib, **keyin to'xtab qoladi** (intersection'ga kirmaydi).
- `stopped_vehicle` → svetofor navbatida turish **hisoblanmaydi**.

---

## 3. Umumiy arxitektura

```text
                    VIDEO (fixed CCTV)
                          │
                          ▼
                  Frame Sampling
                          │
                          ▼
                  Object Detection (YOLO/RT-DETR)
                          │
                  ┌───────┴────────┐
                  ▼                ▼
              Vehicles         Pedestrians
                  │                │
                  └───────┬────────┘
                          ▼
                    Tracking (ByteTrack)
                          │
                          ▼
                  Trajectory History
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
       Speed         Lane Position     Motion Pattern
          │               │                │
          └───────────────┼────────────────┘
                          ▼
          Scene Geometry (camera.md: lanes, stop-line,
                  crossing, direction, road polygon)
                          │
                          ▼
                   Event Rule Engine  ◄──── Collision/TTC Engine
                          │                        │
                          ▼                        ▼
                 Temporal Smoothing          Risk Estimator
                          │                   (Part B, causal)
                          ▼                        │
              Event Start/End Segments             ▼
                          │                  Risk Score Timeline
                          ▼                        │
                          └──────────┬─────────────┘
                                     ▼
                              solution.py OUTPUT
```

---

## 4. Loyiha tuzilishi (repository skeleton)

```text
algorix/
├── README.md                     ← ushbu fayl
├── solution.py                   ← MAJBURIY entrypoint (bo'lim 6)
├── requirements.txt
├── weights/                      ← model weight'lari (≤5GB jami)
│   ├── detector.pt
│   └── smoke_fire_cls.pt
├── camera_configs/
│   └── camera.md                 ← tashkilotchilar bergan scene info
├── src/
│   ├── config.py                 ← seed'lar, threshold'lar, constants
│   ├── detection/
│   │   └── detector.py           ← YOLO/RT-DETR wrapper
│   ├── tracking/
│   │   └── tracker.py            ← ByteTrack wrapper
│   ├── geometry/
│   │   ├── camera_parser.py      ← camera.md → lane/stop-line/polygon
│   │   └── scene_state.py
│   ├── events/
│   │   ├── rule_engine.py        ← wrong_way, red_light, stop_line, ...
│   │   ├── collision_engine.py   ← accident, near_miss (TTC asosida)
│   │   ├── smoke_fire.py
│   │   └── temporal_smoothing.py ← frame-level → segment-level
│   ├── risk/
│   │   └── risk_estimator.py     ← Part B: RiskEstimator class (causal!)
│   └── utils/
│       ├── determinism.py        ← barcha seed'larni o'rnatish
│       └── io_utils.py
├── labeling/
│   ├── label_tool.py             ← sample videolarni qo'lda belgilash
│   └── labels/                   ← qo'lda yaratilgan dev-set (GT)
├── eda/
│   └── analyze_samples.ipynb     ← resolution/FPS/density/heatmap tahlili
├── evaluate.py                   ← Temporal IoU / AP / Alarm F1 hisoblovchi
├── website/                      ← Home, Team, Approach, EDA, Results, Live Demo
└── tests/
    └── test_solution_contract.py
```

---

## 5. Modullar bo'yicha to'liq reja

### **MODUL 0 — Environment & Setup**
- Python venv, `requirements.txt` (ultralytics/YOLO, opencv, numpy, torch, filterpy yoki ByteTrack repo, fastapi/flask agar website backend kerak bo'lsa).
- `src/utils/determinism.py`: barcha random seed'larni bitta joydan o'rnatuvchi funksiya (`set_global_seed(42)`).
- GPU/CPU tekshiruvi, hardware limit simulyatsiyasi (16GB VRAM cheklovini eslab qolish).

### **MODUL 1 — Dataset & Scene Understanding (EDA)**
- `samples/` ichidagi barcha videolarni tahlil qilish: resolution, FPS, duration, yorug'lik sharoiti.
- `camera.md` ni to'liq o'qib, lane'lar, stop-line, crossing, svetofor mavjudligi haqida struktura chiqarish.
- Traffic density, object count (cars/trucks/buses/pedestrians), motion heatmap, trajectory tahlili.
- **Qo'lda labeling** (`labeling/label_tool.py`) — sample videolarga `[start, end, event_type]` formatida dev-set GT yaratish. Bu keyingi baholash uchun juda muhim, chunki tashkilotchilar GT bermaydi.
- Agar public dataset ishlatilsa (DoTA/CCD/DAD/...), README'ga dataset nomi + license yozib qo'yiladi.

### **MODUL 2 — Object Detection**
- YOLO (yoki RT-DETR) orqali `vehicle`, `pedestrian` (va kerak bo'lsa `truck`, `bus`, `motorbike`) klasslarini aniqlash.
- Har frame uchun: `bbox, class, confidence`.
- Inference tezligini o'lchash (video×3 limitiga sig'ish uchun frame-skip/sampling strategiyasini shu yerda belgilash).

### **MODUL 3 — Tracking (ByteTrack)**
- Har obyektga barqaror `track_id` biriktirish.
- Har frame: `{track_id: {bbox, center, velocity, class}}` saqlanadigan trajectory-buffer yaratish.
- ID switch/occlusion holatlarini yumshatish (kalman filter yoki ByteTrack ichki logikasi orqali).

### **MODUL 4 — Scene Geometry (camera.md parser)**
- `camera_parser.py`: `camera.md` faylidan quyidagilarni dasturiy struktura (JSON/dict) ga aylantirish:
  - `LANES` (polygon + expected_direction vektori)
  - `STOP_LINE` (koordinata/segment)
  - `CROSSING_AREA` (polygon)
  - `ROAD_POLYGON`
  - `TRAFFIC_LIGHT_STATE` (agar ko'rinsa)
- Bu modul boshqa barcha rule-based eventlar uchun "single source of truth" bo'ladi.

### **MODUL 5 — Rule-based Event Engine (scene-geometry asosidagi klasslar)**
Quyidagi klasslarni geometry + trajectory orqali aniqlash (eng ishonchli va tez yo'l):
- `wrong_way` — trajectory yo'nalishi lane'ning `expected_direction`iga teskari bo'lsa.
- `stopped_vehicle` — velocity < threshold, ≥10s, carriageway ichida, navbat emas.
- `red_light` / `stop_line` — vehicle front bounding-box `STOP_LINE`dan o'tishi + svetofor holati + keyingi harakat (intersection ichiga kiradimi yoki to'xtaydimi) orqali farqlash.
- `solid_line_crossing` — wheel/bbox lane-chegara chizig'idan o'tishi.
- `illegal_turn` / `illegal_u_turn` — burilish burchagi + ruxsat etilgan lane/yo'nalish bilan solishtirish.
- `jaywalking` — pedestrian markazi `ROAD_POLYGON` ichida va `CROSSING_AREA` tashqarisida.
- `failure_to_yield` — pedestrian `CROSSING_AREA` ichida/kirayotganda, vehicle undan o'tib ketishi.
- `congestion` — barcha lane'larda o'rtacha velocity past + zichlik yuqori (vaqt oynasi bo'yicha).
- `road_obstacle` — harakatsiz, kutilmagan klass (odam/mashina bo'lmagan) obyekt yo'l ustida uzoq turishi.
- `fire_smoke` — bo'lim 7ga qarang.

### **MODUL 6 — Collision / TTC Engine (`accident`, `near_miss`)**
- Har juft yaqin track uchun: `distance`, `relative_velocity`, `Time-to-Collision (TTC = distance / closing_speed)`.
- `accident`: bbox overlap ≈ 0 masofaga tez yaqinlashish + kontakt frame'i aniqlanadi; tugash — barcha ishtirokchilar to'xtaydi yoki kadrdan chiqadi.
- `near_miss`: to'qnashuv bo'lmaydi, lekin keskin tormoz/burilish + TTC juda past bo'lib, keyin distance qayta ortadi.
- Bu modul Part B (Risk Estimator) bilan mantiqiy jihatdan bog'liq — TTC signali ikkalasida ham ishlatiladi.

### **MODUL 7 — Smoke/Fire Detector**
- Yengil, alohida classifier yoki segmentation model (masalan kichik CNN yoki mavjud open-weight smoke/fire detektor) — asosiy pipeline'ni sekinlashtirmasligi kerak.

### **MODUL 8 — Temporal Smoothing & Segmentation**
- Frame-level binar/skorli chiqishlarni (`0 0 1 1 1 1 0 0`) tekis segmentlarga (`[start_time, end_time]`) aylantirish.
- Shovqinni yo'qotish uchun min-duration filter, hysteresis threshold, morphological smoothing (masalan majority-vote oyna).
- **Temporal IoU** (0.3/0.5/0.7 threshold'larda baholanadi) uchun boundary aniqligini maksimal darajada oshirish shu modulning maqsadi.

### **MODUL 9 — Part B: Risk Estimator (Accident Anticipation)**
- `class RiskEstimator` — **faqat shu paytgacha ko'rilgan frame'lar** asosida ishlaydi (kelajakka kirish taqiqlangan).
- Har frame uchun `0.0–1.0` oralig'ida risk score qaytaradi.
- Asosiy signal: TTC (masalan TTC=5s→~0.3, TTC=3s→~0.55, TTC=2s→~0.75, TTC=1s→~0.95 — smooth funksiya, sigmoid/interpolatsiya orqali), keskin tormoz, wrong-way trajectory, red-light trajectory, pedestrian roadway'ga kirishi.
- Baholash: 40% AP, 40% Alarm F1 (risk≥0.5 bo'lganda alarm), 20% Time-to-Accident.

### **MODUL 10 — `solution.py` kontrakti (MAJBURIY)**

```python
CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

def detect_events(video_path: str) -> list[list]:
    """
    Returns: [[start_time, end_time, event_type], ...]
    Vaqt limiti: video_duration * 3 dan oshmasligi shart.
    """
    ...

class RiskEstimator:
    """
    Faqat causal (kelajaksiz) ishlaydi.
    Har chaqiriqda faqat 'hozirgacha' ko'rilgan frame beriladi.
    """
    def update(self, frame, timestamp: float) -> float:
        """Return risk score in [0.0, 1.0] for next-5-second accident probability."""
        ...
```

### **MODUL 11 — Evaluation & Determinism**
- `evaluate.py --pred predictions.json --validate-only` — o'z natijalarini tekshirish uchun lokal skript.
- Temporal IoU (3 threshold), AP, Alarm F1, Time-to-Accident metrikalarini o'zi qayta hisoblovchi mini-baholovchi yozish.
- Ikki marta run qilib natijalarni solishtirish (determinism tekshiruvi).

### **MODUL 12 — Website**
Sahifalar: `Home`, `Team` (a'zolar, rol, GitHub/LinkedIn/portfolio), `Approach` (arxitektura diagrammalari), `EDA` (resolution/FPS/density/heatmap), `Results`, `Live Demo` (video upload → processing → timeline + annotated video).
- Extra credit: interaktiv timeline (bosilganda videoning o'sha sekundiga o'tish), ablation jadvali (YOLO only vs +ByteTrack vs +Rules), error analysis, dashboard (events/hour, accidents, near-misses va h.k.).

### **MODUL 13 — Performance & Packaging**
- Frame-sampling strategiyasi va model o'lchamini video×3 vaqt limitiga sig'diradigan qilib optimallashtirish.
- Weight fayllarni ≤5GB ichida saqlash, internetga hech qanday chaqiruv qolmaganini tekshirish (`grep -R "requests\|openai\|api.anthropic\|generativelanguage" src/`).

---

## 6. Baholash formulasi (yodda tutish uchun)

```text
FINAL = 60% Model + 25% Website + 15% Code

Model = 70% Part A (Temporal IoU @0.3/0.5/0.7) + 30% Part B (40% AP + 40% Alarm F1 + 20% TTA)
```

---

## 7. 4 kunlik jadval (modullarga bog'langan)

| Kun | Fokus | Modullar |
|---|---|---|
| **1-kun** | Dataset + Scene Understanding | 0, 1 |
| **2-kun** | Detection + Tracking | 2, 3, 4 |
| **3-kun** | Event Engine (avval accident/near_miss/wrong_way/stopped_vehicle/red_light/stop_line/jaywalking, keyin qolganlari) + Part B | 5, 6, 7, 8, 9, 10 |
| **4-kun** | Website + optimizatsiya + testing | 11, 12, 13 + `evaluate.py --validate-only` yakuniy tekshiruv |

---

## 8. Eng katta xatarlar va oldini olish

- ❌ **14 ta klassga alohida deep-learning model** — vaqt yetmaydi. ✅ Ko'p klass uchun **geometry + rule engine** yetarli.
- ❌ RiskEstimator'ga bilmasdan kelajak frame berish → **disqualification**. ✅ API darajasida frame-by-frame streaming simulyatsiya qilib test qilish.
- ❌ Model juda katta/sekin bo'lib video×3 limitidan oshib ketishi. ✅ Frame-sampling + yengil model + erta profiling.
- ❌ Determinizmni unutish. ✅ `determinism.py`ni eng boshida yozib, har joyda chaqirish.

---