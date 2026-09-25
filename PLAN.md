# План реализации — WIUT Hackathon 2026, CV Track

Deadline 27.09.2026 23:59. Исходники правил: `HACKATHON_CONTEXT.md`, `README.md` стартового кита, полный текст задания (часть правил собрана в сообщениях чата).

## 1. Цель

`Elimination = 0.6·M + 0.25·Website + 0.15·Code`, где `M = 0.7·Score_A + 0.3·Score_B`.

Приоритет: **Score_A мощнее Score_B и дешевле** → сначала стабильный Score_A, потом разреженный Score_B, потом сайт и чистота пакета.

## 2. Механика Score_A (правит архитектуру)

Классы C = GT ∪ predictions.

- Класс, который предскажем, но нет в тесте → F1=0, добавлен в C → тянет среднее вниз.
- Класс из теста, который не предскажем → тоже F1=0.
- → **мало классов с высоким F1 лучше, чем много с посредственным**.
- Class-конфузия = двойное наказание (FP нашему классу + FN истинному).
- tIoU {0.3, 0.5, 0.7}: на 0.7 важны точные границы.

### Решётка по классам (решение по `CLASSES`)

- Гарантировано (траекторные правила, низкая цена): `wrong_way`, `stopped_vehicle`, `congestion`, `jaywalking`, `failure_to_yield`, `solid_line_crossing`, `illegal_turn`, `illegal_u_turn`.
- `red_light`/`stop_line` — **только если светофор виден** (решение из `camera.md`).
- Осознанный риск: `accident`, `near_miss` (контакт треков + TTC; = ядро Part B).
- `fire_smoke`, `road_obstacle` — только если встретятся в сэмплах; иначе не тратить время.
- На dev-set: класс с F1 < ~0.2 → **удаляем из `CLASSES`**.

### Правила генерации сегментов

- Пересечение разных классов — разрешено (например, `wrong_way` + `accident`). Не подавлять.
- Одновременные события одного класса → **один сегмент на оба**.
- Один класс не пересекается сам с собой (`clean_events` срежет поздний сегмент).
- Постпроцесс: merge фрагменты (разрыв < 1 с), отсев < 0.5 с, clamp `end ≤ duration`. Проверка на tIoU@0.7.

## 3. Механика Score_B (0.3)

- Позитив: `[s−5, s)`; ignored: внутри аварий и `[s−5, e]` near_miss.
- AP (0.4) chance-normalized → кривая **разреженная**: 0 на норме, рост только перед аварией. Константа 1.0 = 0.
- F1_alarm (0.4): runs ≥ 0.5, склейка gap < 2 с, матч в окне `[s−10, s)`. Цель: **ровно один аларм на аварию в окне**.
- mTTA (0.2): раньше старт в окне = лучше, но старт до `s−10` = FP.
- Sign: `max` по сигналам — TTC ≤ 5 с, резкое торможение, wrong_way/red_light-кандидат, пешеход у машины. Норм-кадры → 0.0. Калибровка: 0.5 ≈ авария в течение 5 с.
- `step()` каузален; сэмплинг кадров с повтором последнего score разрешён.
- **Part A может использовать risk-кривую Part B** (в одну сторону); сам RiskEstimator каузален.

## 4. Архитектура

```
video → src/detector.py (YOLO11x, сэмплинг 1/2–1/5, батчи)
      → src/tracker_wrap.py (UCMCTrack + cam_para; фолбек ByteTrack)
      → src/scene.py (хардкод из camera.md: полосы/направления/стоп-линии/переходы/сплошные/ROI светофора)
      → src/features.py (метры, скорость, ускорение, стационарность, направление, зоны)
         ├→ src/rules.py        → [[start,end,label]]
         ├→ src/postprocess.py  → merge/отсев/clamp
         ├→ src/risk.py         → RiskEstimator.step() [0,1]
         └→ src/calibrate_cam.py (разовая калибровка)
```

Стек (из HACKATHON_CONTEXT.md): YOLO11x (ultralytics, COCO: car/truck/bus/motorcycle/bicycle/person, опц. traffic light), UCMCTrack Python (MIT). Детерминизм: фикс. сиды, проверка двух прогонов.

## 5. Правила по классам

| Группа | Механика | Цена/риск |
|---|---|---|
| wrong_way, illegal_turn/u_turn | направление трека vs направление полосы | низкая |
| stopped_vehicle, congestion | стационарность ≥ 10 с; затор = все полосы направления стоят/ползут | низкая; риск конфузии |
| red_light, stop_line | стоп-линия + сигнал (видимый светофор) | высокая без ROI-светофора |
| jaywalking, failure_to_yield | пешеход вне перехода / на переходе + машина | средняя |
| solid_line_crossing | пересечение сплошной треком | средняя |
| accident | контакт треков (IoU bbox / метры + остановка) | высокая, но = ядро Part B |
| near_miss | резкое торможение, манёвр, TTC-пик без контакта | высокая |
| road_obstacle, fire_smoke | фоновая статика/цвет; спец-детектор | очень высокая → скип |

## 6. Дорожная карта

### День 0 (сейчас)
- [x] Получить `samples/` + `camera.md` (видео есть; **camera.md организаторы не дали** → геометрия сцены восстановлена программно из разведки).
- [x] git init + коммит кита.
- [x] Установка окружения, веса YOLO11x в `weights/`, smoke-тест кита (GPU: CUDA torch в `C:\pylibs` через `.pth`, 45 мс/кадр).
- [ ] Начать разметку dev-set (2 видео, кадры 1/1–1/3) ← **в работе (ручная разметка).**

### День 1 (23–24)
- [x] `detector.py` + сэмплинг + замер FPS/батча (imgsz 800, GPU 45 мс/кадр ≈ 22 fps).
- [ ] Калибровка камеры / UCMCTrack с фолбеком ByteTrack — решение до конца дня (пока ByteTrack, UCMCTrack не подключён).
- [x] `scene.py` (полигоны из разведки, camera.md нет).
- [x] `rules.py`: wrong_way, stopped_vehicle, congestion, jaywalking, failure_to_yield (v0).
- [x] `postprocess.py`, `solution.detect_events` → прогон на 1 видео, сравнение с разметкой (прогон есть, сравнение без GT невозможно).

### День 2 (24–25)
- [ ] `rules.py`: solid_line_crossing, illegal_turn/u_turn, red_light/stop_line.
- [ ] Замер времени (цель < 60% бюджета 3×).
- [ ] `risk.py` (TTC+brake+context) → RiskEstimator.
- [ ] `accident`/`near_miss`.
- [ ] Полная разметка dev-set.

### День 3 (25–26)
- [ ] Прогон на всех сэмплах → `predictions_samples.json`.
- [ ] `evaluate.py --gt my_labels.json` → таблица per-class → решётка `CLASSES`.
- [ ] Оптимизация границ, анализ конфузии классов.
- [ ] Детерминизм: два прогона = одинаковый выход.

### День 4 (26–27)
- [ ] Clean-machine проверка двумя командами; README (install/run, weights, подход, датасеты+лицензии, сиды, роли).
- [ ] Структура пакета: `solution.py` в корне, `src/`, `weights/`, `requirements.txt`, `predictions_samples.json`, `run_submission.py`/`evaluate.py` неизменные.
- [ ] Сайт параллельно (см. ниже).

## 7. Сайт (0.25) и Code (0.15)

Сайт (рубрика): Live-demo 30%, визуализации сэмплов 20%, EDA 15%, approach+report 15%, team 10%, дизайн 10%. Делает участник 3 (Vercel/Streamlit/HF Spaces; CPU-инференс, лимит 2 мин).

Code: Runs-as-submitted 40% (две команды на чистой машине), Reproducibility 25% (веса+сиды+`predictions_samples.json` совпадает), Structure 20%, Engineering 15% (сэмплинг, батчи, запас по времени).

## 8. Риски

| Риск | Митигация |
|---|---|
| Нет сэмплов/camera.md вовремя | Блокер; скелет писать параллельно |
| Светофор не виден | `red_light`/`stop_line` → вон из `CLASSES` |
| UCMCTrack не сходится | Фолбек ByteTrack к концу Дня 1 |
| Малый dev-сет | Начать разметку с 2 видео |
| Class-конфузия | Таблица F1 per-class в День 3, решётка CLASSES |
| Поздний старт сайта | Участник 3 с Дня 0 |
| Тайм-бюджет | Замер с Дня 2, цель < 60% |

## 9. Роли

1. Код/модель — детектор, трекер, правила, риск, интерфейс.
2. Данные/оценка — camera.md, разметка, evaluate.py, ошибки-анализ, predictions_samples.json.
3. Сайт — EDA, визуализации, live demo, report.

## 10. Структура пакета (финал)

```
repo/
├── solution.py          # интерфейс (у нас в wiut_cv_scripts/wiut_cv_scripts/ до сборки финала)
├── run_submission.py    # неизменный
├── evaluate.py          # неизменный
├── requirements.txt
├── weights/             # yolov11x.pt и т.п. (≤ 5 ГБ) + download.sh
├── src/                 # detector, tracker_wrap, scene, features, rules, postprocess, risk, calibrate_cam
├── notebooks/           # опционально EDA
├── predictions_samples.json
└── README.md
```

Финальная сборка (День 4): распаковать пакет в корень репо или сделать пакет корнем — `solution.py` обязателен в корне.