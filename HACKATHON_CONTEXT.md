# Контекст чата: WIUT Hackathon 2026 — Toyota Traffic Event Detection

> Файл-перенос контекста для новой сессии AI. Скопируй/загрузи его в новый чат как стартовый контекст.

---

## 1. Соревнование

- **Название:** Toyota Traffic Event Detection and Accident Anticipation from a Fixed Road Camera (Elimination Task · Computer Vision)
- **Организатор:** WIUT Hackathon 2026, Westminster International University in Tashkent (10–11 October 2026)
- **Команда:** (3 человека)
- **Дедлайн:** Sunday, **27 September 2026, 23:59 Tashkent time** (на момент чата ~4 дня оставалось)
- **Ограничение:** не более 5 команд с одного университета проходят в финал; одна команда — одна посылка (затаганный коммит).

### Суть задачи
Фиксированная CCTV-камера (один угол, без движения). Сэмпл-видео выдают без лейблов; скрытый тест — тот же угол. Решение работает **offline** на машине организаторов.

**Три части посылки:**
1. **Репозиторий:** `solution.py` + веса + один рабочий вход (запуск двумя командами).
2. **Публичный сайт команды** с live demo (загрузка видео → события), EDA, визуализациями, отчётом.
3. **Короткий технический отчёт** (можно страницей на сайте).

---

## 2. Требования к интерфейсу (solution.py — строго)

```python
CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]

def detect_events(video_path: str) -> list[list]:  # Part A
    # return [[start_sec, end_sec, label], ...]
    ...

class RiskEstimator:  # Part B (bonus)
    def reset(self, meta: dict) -> None: ...  # meta = {video_id, fps, width, height, n_frames}
    def step(self, frame: np.ndarray, t_sec: float) -> float: ...  # 0..1, только прошлые кадры
```

- Part B **каузален**: нельзя открывать видео внутри `step`, нельзя использовать вывод Part A.
- Хардкод сцены из `camera.md` (полосы, направления, стоп-линии) — разрешён и ожидается.
- Сегменты одного класса не пересекаются; границы важны (temporal IoU).

### 14 классов (определения)
- `accident` — контакт двух+ участников; старт = первый видимый контакт; конец = остановка/выбытие
- `near_miss` — резкое торможение/манёвр избежания без контакта
- `red_light` — пересечение стоп-линии на красный; старт = перед пересекает линию
- `wrong_way` — движение против направления полосы
- `illegal_u_turn` — разворот в запрещённом месте
- `stopped_vehicle` — стоит на проезжей ≥10 с вне очереди на светофор
- `jaywalking` — пешеход на проезжей вне перехода
- `failure_to_yield` — машина через переход, когда пешеход на нём
- `illegal_turn` — поворот не из той полосы/запрещённого направления
- `solid_line_crossing` — пересечение сплошной разметки
- `stop_line` — остановка за стоп-линией на красный без въезда в перекрёсток
- `congestion` — трафик стоит/ползёт по всем полосам направления
- `road_obstacle` — мусор/животное/упавший объект на проезжей
- `fire_smoke` — видимый огонь/дым

---

## 3. Скоринг

- **Score A (Part A):** macro F1 по классам, усреднённый по IoU-порогам {0.3, 0.5, 0.7} (greedy matching по убыванию IoU).
- **Score B (Part B, только accident, H=5s, W=10s, θ=0.5):**
  - Frame labels: положительный кадр если `s − 5 ≤ t < s`; внутри accident и около near_miss — ignored
  - AP (chance-normalised) + F1_alarm + mTTA/W
  - `Score_B = 0.4×AP + 0.4×F1_alarm + 0.2×(mTTA/W)`
- **Model M = 0.7×Score_A + 0.3×Score_B**
- **Elimination = 0.6×M + 0.25×Website + 0.15×Code**

### Website rubric (0–1): Live demo 30%, визуализации сэмплов 20%, EDA 15%, подход+отчёт 15%, команда 10%, дизайн/бонусы 10%
### Code rubric (0–1): Runs as submitted 40%, reproducibility 25%, structure 20%, engineering judgement 15%
> Пакет, не запустившийся после одной попытки фикса очевидной проблемы, получает Model score = 0.

---

## 4. Ограничения «железо и лимиты»

| Параметр | Значение |
|---|---|
| GPU | 1× NVIDIA 16 GB VRAM (T4-class), 8 CPU cores, 32 GB RAM |
| Интернет при оценке | **Нет** — все веса внутри пакета |
| Время на видео (A+B) | ≤ 3× длительности видео (wall-clock), иначе пустой скор |
| Веса | ≤ 5 GB |
| Python | 3.10+, зависимости из requirements.txt или Dockerfile |

Правила: только open weights (никаких OpenAI/Gemini/Anthropic API на инференсе); публичные датасеты (DoTA, CCD, DAD, CADP, UCF-Crime RoadAccidents, UA-DETRAC, BDD100K) с указанием лицензий в README; детерминизм (фикс. сиды); переиспользование open-source с атрибуцией; сборка запускается двумя командами:
```
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
```
Перед отправкой: `python evaluate.py --pred predictions.json --validate-only`.

---

## 5. Исследованная библиотека: UCMCTrack

### Вариант A: C++ (`LSH9832/UCMCTrack-cpp`) — ОТКЛОНЁН
- Apache-2.0, ~3000 FPS чистый трекинг, конверсия Python→C++ версии автора.
- Требует `libopencv-dev`, `libeigen3-dev`, `libyaml-cpp0.6`, CMake — **риск «не соберётся на чистой машине»** (критично для Code rubric 40%).
- Нет детектора (только трекинг по готовым `bboxes.yaml`).

### Вариант B: Python (`corfyi/UCMCTrack`) — ВЫБРАН ✅
- **MIT-лицензия**, AAAI 2024, SOTA MOT17/MOT20/DanceTrack/KITTI **без appearance-признаков** (только motion + ground-plane homography).
- Dependencies: `numpy, filterpy, lap, scipy, ultralytics, opencv-python` — всё кладётся в requirements.txt.
- Скорость: <100 Hz CPU → для видео 25 fps **запас хватает** (Python-версия пригодна, C++ не нужна).
- Детектор не включён; demo = **YOLOv8x** (`class_list = [2,5,7]` car/bus/truck).
- Ключевая особенность: нужен файл камеры `cam_para.txt` (intrinsic/extrinsic) для гомографии на ground plane; есть инструмент оценки из одного кадра: `python util/estimate_cam_para.py --img ... --cam_para ...` (или C++ `scripts/estimate_cam_param.py`).
- Интерфейс: `tracker.update(detections, frame_id)`, классы `Detection`, `Mapper.mapto(bbox)->(y,R)`.
- API детектора (demo.py): `Detector.get_dets(img, conf_thresh, det_classes)` → Detection с `bb_left, bb_top, bb_width, bb_height, conf, det_class, track_id`.
- Прочие фолбеки: ByteTrack/BoT-SORT (ultralytics `track()`) — если калибровка UCMCTrack упрётся.

### Почему UCMCTrack хорош для этой задачи
- Фиксированная камера = идеальный случай ground-plane модели; разовые параметры камеры действуют на весь тест-сет.
- Даёт траектории в **метрах** (не пикселях) → физически осмысленный TTC для Part B.
- Быстрый → укладывается в лимит 3× с запасом.

### Какие классы закрываются треками/правилами
- Trajectory-правила: `wrong_way`, `stopped_vehicle` (≥10s), `congestion`, `red_light`/`stop_line` (пересечение стоп-линии), `solid_line_crossing`, `jaywalking`, `failure_to_yield`, `illegal_turn`, `illegal_u_turn`.
- Rule-heavy/hard: `accident`, `near_miss` (пересечение треков + резкая остановка / TTC), `fire_smoke`, `road_obstacle`.
- Part B: TTC между треками + резкое торможение + пешеход на проезжей → риск в [0,1].

---

## 6. Принятая архитектура

```
samples/*.mp4 → детектор (YOLO11x/ultralytics, COCO-веса)
              → трекер (UCMCTrack Python + cam_para)
              → траектории (track_id, xy-метры, скорость, класс)
                 ├→ rules.py     → события [[start,end,label]] + postprocess
                 └→ risk.py (TTC) → RiskEstimator.step()
```

### Структура репозитория
```
repo/
├── solution.py            # интерфейс (детект + RiskEstimator)
├── run_submission.py      # из starter kit, без изменений
├── evaluate.py            # из starter kit, без изменений
├── requirements.txt
├── weights/download.sh    # yolov11x.pt и пр. (≤5 ГБ)
├── src/
│   ├── detector.py        # YOLO11, NMS, фильтр классов COCO
│   ├── tracker_wrap.py    # обёртка UCMCTrack
│   ├── scene.py           # hard-code camera.md: полосы, направления, стоп-линии, зоны перехода
│   ├── calibrate_cam.py   # разовая калибровка камеры
│   ├── rules.py           # правила событий
│   ├── postprocess.py     # слияние фрагментов, отсев <0.5с, границы IoU
│   └── risk.py            # Part B
├── predictions_samples.json
└── README.md              # установка, подход, лицензии, сиды, состав команды
```

**Решение по детектору:** YOLO11x (ultralytics, open-weights) как база; опционально RT-DETR — но за 4 дня YOLO11 надёжнее. Детектор ≠ скриптор событий.

---

## 7. План по дням (23–27 сентября)

- **23.09 (день 1):** starter kit, `camera.md`, скелет solution.py, YOLO11 + UCMCTrack протокол, калибровка камеры, решение по видимости светофора; ч.2 начинает ручную разметку сэмплов (dev-set); ч.3 ставит сайт-скелет.
- **24.09 (день 2):** правила для 6–8 траекторных классов; прогон на 1–2 видео, compare с разметкой; сайт: структура страниц.
- **25.09 (день 3):** `accident`/`near_miss` + Part B (TTC) + постпроцессинг границ; полная разметка.
- **26.09 (день 4):** полный прогон, predictions_samples.json, evaluate.py, замер времени, интерактивный сайт + live demo, README.
- **27.09 (день 5):** проверка на clean-machine, детерминизм, ≤5 ГБ веса, сабмит до 23:59.

### Роли (3 чел.)
1. Код/модель: solution.py, пайплайн, правила.
2. Данные/оценка: разметка сэмплов, dev-labels, evaluate.py, анализ ошибок, predictions_samples.json.
3. Сайт: Team, Approach, EDA, Results, Live demo, Report, UX.

---

## 8. Риски / открытые вопросы

1. **Виден ли светофор в кадре** (из camera.md) → если да, нужен ROI-классификатор сигнала для `red_light`/`stop_line`; если нет — эти классы почти недостижимы правилами.
2. **Калибровка UCMCTrack** (сегодня закрыть; фолбек ByteTrack — TTC тогда по пикселям, грубее).
3. Есть ли `fire_smoke`/`road_obstacle` в сэмплах — если нет, не тратить время.
4. Разметка = узкое место: если dev-set не готов к 25.09, итерации невозможны → сэмплировать кадры (1 из 2–3), начинать с 2 видео.
5. C++ UCMCTrack не использовать (компиляционный риск).
6. Скор boundary precision: слияние фрагментов + отсев коротких, проверять IoU 0.7.

---

## 9. Следующий шаг (для новой сессии)

Собрать рабочий скелет:
- `solution.py` (detect_events + RiskEstimator),
- `src/detector.py` (YOLO11) + `src/tracker_wrap.py` (UCMCTrack) + `src/scene.py` (camera.md),
- начало `src/rules.py` для траекторных классов.

Затем прогнать на одном сэмпл-видео, сверить с ручной разметкой через `evaluate.py`.
