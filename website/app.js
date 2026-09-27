// Algorix — Toyota Traffic Event Detection & Accident Anticipation
// Multilingual (UZ/RU) Engine & Minimalist Interactive Controller

const translations = {
  uz: {
    // Navigation
    "nav.home": "Bosh sahifa",
    "nav.approach": "Yondashuv",
    "nav.geometry": "Geometriya & EDA",
    "nav.events": "14 ta Klass",
    "nav.results": "Natijalar",
    "nav.demo": "Live Demo",
    "nav.team": "Jamoa",
    "nav.github": "GitHub Repo ↗",

    // Hero
    "hero.badge": "WIUT Hackathon 2026 • CV Track",
    "hero.title": "Toyota Traffic Event Detection & Accident Anticipation",
    "hero.desc": "Fixed CCTV kameraga moslashtirilgan deterministik Scene-Geometry Computer Vision tizimi. 14 ta murakkab hodisani yuqori aniqlikda aniqlash va kelajakka qaramagan (causal) real-vaqt avariya xavfi prognozi.",
    "hero.btnDemo": "Live Demon Ko'rish",
    "hero.btnArchitecture": "Tizim Arxitekturasi",

    // Stats
    "stat.events.num": "14",
    "stat.events.label": "Trafik Hodisalari Klassi",
    "stat.latency.num": "< 1.3×",
    "stat.latency.label": "Inference Vaqt Ko'rsatkichi",
    "stat.risk.num": "81.5%",
    "stat.risk.label": "Avariya Xavfi Prognozi (Alarm F1)",
    "stat.offline.num": "0.0 MB",
    "stat.offline.label": "Tashqi API (Offline Ready)",

    // Approach
    "approach.tag": "Muhandislik Yechimi",
    "approach.title": "Tizim Arxitekturasi va Yondashuv",
    "approach.subtitle": "Og'ir va sekin umumiy modellardan voz kechib, aynan doimiy CCTV kameraning fazoviy xususiyatlariga tayangan yuqori tezlikdagi pipeline.",
    "approach.c1.title": "1. Object Detection & Tracking",
    "approach.c1.desc": "YOLO / RT-DETR modellari orqali avtomobillar, mototsikllar va piyodalar kadrma-kadr aniqlanadi. ByteTrack assotsiatsiyasi har bir obyektning traektoriyasi va tezlik vektorlarini uzluksiz kuzatadi.",
    "approach.c2.title": "2. Scene Geometry Engine",
    "approach.c2.desc": "Kameraning o'zgarmas koordinatalari (harakat yo'laklari, stop-chiziq, piyodalar o'tish joyi va uzluksiz chiziqlar) Ray-Casting va poligon kesishish qoidalari asosida 100% deterministik tahlil qilinadi.",
    "approach.c3.title": "3. Causal Risk & TTC Engine",
    "approach.c3.desc": "Avariyalar va xavfli yaqinlashuvlar Time-to-Collision (TTC) formulasi orqali oldindan bashorat qilinadi. Algoritm keyingi kadrlarni ko'rmasdan (anti-leakage) to'liq real vaqt rejimida ishlaydi.",

    // Geometry & EDA
    "geo.tag": "Fazoviy Tahlil",
    "geo.title": "Scene Geometriyasi va Hududlar",
    "geo.subtitle": "CCTV kameraning kalibrlangan poligonlari, harakat yo'nalishlari va real-vaqt telemetriyasi",
    "geo.laneSouth": "Yo'lak 1 & 2 (Janubiy oqim)",
    "geo.laneNorth": "Qarama-qarshi yo'lak (Oncoming)",
    "geo.stopLine": "Stop Chizig'i (Stop Line)",
    "geo.crosswalk": "Piyodalar o'tish joyi (Zebra)",
    "geo.boxTitle": "Kamera Geometriyasining Asosiy Qoidalari",
    "geo.rule1": "Har bir piksel koordinatasi offline kalibrlangan — yashil poligon qatnov qismi, sariq poligonlar esa zebra va burilish hududlarini belgilaydi.",
    "geo.rule2": "Qoidabuzarliklar (wrong_way, failure_to_yield, jaywalking) zonalarga kirish va harakat vektorlarining kesishishi orqali aniqlanadi.",
    "geo.rule3": "Obyektlar tezlik vektorlari (to'q sariq chiziqlar) orqali kuzatilib, to'xtashlar va xavfli holatlar mikrosekundlarda qayd etiladi.",

    // 14 Classes
    "classes.tag": "Standartlashtirilgan Voqealar",
    "classes.title": "14 ta Rasmiy Event Klasslari",
    "classes.subtitle": "Shahar harakatidagi sabab-oqibat zanjirlari va ularning kompyuter ko'rishi (CV) orqali aniqlanishi",
    "filter.all": "🌐 Barchasi (14)",
    "filter.collision": "💥 Avariya & To'qnashuv",
    "filter.pedestrian": "🚸 Piyodalar Xavfsizligi",
    "filter.discipline": "🚦 Yo'l Intizomi & Qatorlar",
    "filter.flow": "🛑 To'siqlar & Tirbandlik",
    "card.scenarioLabel": "Real Hayotiy Ko'rinishi:",
    "card.chainLabel": "Zanjir Bog'liqligi (Sabab-Oqibat):",
    "card.logicLabel": "CV / Deteksiya Qoidasi:",

    // Ablation Table
    "results.tag": "Benchmark Natijalari",
    "results.title": "Ablation Tahlili va Hakamlar Baholash Mezonlari",
    "results.subtitle": "Rasmiy M = 0.7·Score_A + 0.3·Score_B mezoni va vaqt byudjeti bo'yicha har bir modul hissasi",
    "th.config": "Konfiguratsiya / Modullar",
    "th.partA": "Part A (Macro tIoU F1)",
    "th.partB": "Part B (Alarm F1)",
    "th.scoreM": "Ball M (0.7A + 0.3B)",
    "th.fps": "FPS (Tezlik)",
    "th.constraint": "Vaqt Faktori (≤ 3.0×)",
    "results.h1.title": "Qat'iy Vaqt Cheklovi Kafolati",
    "results.h1.desc": "Vaqt limiti 3.0× bo'lib, Algorix 1.22× ko'rsatkichda 60% xavfsizlik buferiga ega — diskvalifikatsiya ehtimoli 0%.",
    "results.h2.title": "Causal & Zero Data Leakage",
    "results.h2.desc": "Part B Risk Estimator birorta ham kelajak kadrini ko'rmaydi (anti-leakage) va to'liq real vaqtda faqat o'tmishga tayanadi.",
    "results.h3.title": "100% Offline Grading Ready",
    "results.h3.desc": "Tashqi API talab etilmaydi. yolo11x.pt vaznlari va NPZ zonalari cheklovdan (≤5GB) ancha kam bo'lib, to'liq offline baholanadi.",

    // Live Demo
    "demo.tag": "Interaktiv Sinov",
    "demo.title": "Jonli Monitoring va Causal Risk Simulyatori",
    "demo.subtitle": "Voqealarni tanlab, tizimning javob reaksiyasini va xavf darajasini kuzating",
    "demo.liveBadge": "CCTV MONITORING",
    "demo.timelineTitle": "Aniqlangan Hodisalar Xronologiyasi",
    "demo.clickTip": "Hodisani tanlash uchun bosing:",
    "demo.riskLabel": "Hisoblangan Xavf Darajasi:",
    "demo.ttcLabel": "Time-to-Collision (TTC):",

    // Team
    "team.tag": "Ishlab Chiquvchilar",
    "team.title": "Algorix Jamoasi",
    "team.subtitle": "WIUT Hackathon 2026 ishtirokchilari — intellektual transport tizimlari muhandislari",
    "team.member1.role": "Developer",
    "team.member1.bio": "Backend arxitekturasi, model inferensi optimizatsiyasi va real-vaqt ma'lumotlar oqimini qayta ishlash bo'yicha dasturchi.",
    "team.member2.badge": "Team Captain",
    "team.member2.role": "AI Engineer & Team Captain",
    "team.member2.bio": "Computer Vision arxitekturasi, Scene-Geometry algoritmlari, Causal Risk Estimator va loyihaning umumiy texnik yetakchisi.",
    "team.member3.role": "Speaker",
    "team.member3.bio": "Loyiha taqdimoti, biznes qiymati va hakamlar hay'ati oldida texnik yechimlarni mukammal yetkazib berish bo'yicha mas'ul.",

    // Footer
    "footer.text": "© 2026 Algorix. WIUT Hackathon 2026 uchun maxsus ishlab chiqildi. Barcha huquqlar himoyalangan."
  },

  ru: {
    // Navigation
    "nav.home": "Главная",
    "nav.approach": "Подход",
    "nav.geometry": "Геометрия & EDA",
    "nav.events": "14 Классов",
    "nav.results": "Результаты",
    "nav.demo": "Демо",
    "nav.team": "Команда",
    "nav.github": "GitHub Репо ↗",

    // Hero
    "hero.badge": "WIUT Hackathon 2026 • CV Track",
    "hero.title": "Toyota Traffic Event Detection & Accident Anticipation",
    "hero.desc": "Детерминированный Scene-Geometry пайплайн компьютерного зрения, оптимизированный под фиксированную камеру CCTV. Детекция 14 сложных дорожных событий и строго каузальное (без взгляда в будущее) прогнозирование риска аварий.",
    "hero.btnDemo": "Смотреть Демо",
    "hero.btnArchitecture": "Архитектура Системы",

    // Stats
    "stat.events.num": "14",
    "stat.events.label": "Классов Дорожных Событий",
    "stat.latency.num": "< 1.3×",
    "stat.latency.label": "Фактор Времени Инференса",
    "stat.risk.num": "81.5%",
    "stat.risk.label": "Прогноз Риска ДТП (Alarm F1)",
    "stat.offline.num": "0.0 MB",
    "stat.offline.label": "Внешние API (Полный Offline)",

    // Approach
    "approach.tag": "Инженерное Решение",
    "approach.title": "Архитектура Системы и Подход",
    "approach.subtitle": "Отказ от медленных универсальных ИИ-моделей в пользу сверхбыстрого пайплайна, опирающегося на неизменную геометрию перекрестка.",
    "approach.c1.title": "1. Object Detection & Tracking",
    "approach.c1.desc": "Детекция транспортных средств и пешеходов на базе YOLO / RT-DETR. Двухэтапный трекинг ByteTrack строит плавные траектории движения и векторы скорости объектов.",
    "approach.c2.title": "2. Scene Geometry Engine",
    "approach.c2.desc": "Фиксированные координаты камеры (полосы, стоп-линия, пешеходный переход, сплошные линии) анализируются с помощью Ray-Casting и векторной математики со 100% детерминизмом.",
    "approach.c3.title": "3. Causal Risk & TTC Engine",
    "approach.c3.desc": "Прогнозирование ДТП и опасных сближений на основе метрики Time-to-Collision (TTC). Алгоритм строго исключает доступ к будущим кадрам (anti-leakage).",

    // Geometry & EDA
    "geo.tag": "Пространственный Анализ",
    "geo.title": "Геометрия Сцены и Зоны Интереса",
    "geo.subtitle": "Откалиброванные полигоны камеры CCTV, направления движения и онлайн телеметрия",
    "geo.laneSouth": "Полосы 1 и 2 (Южный поток)",
    "geo.laneNorth": "Встречная полоса (Oncoming)",
    "geo.stopLine": "Стоп-линия (Stop Line)",
    "geo.crosswalk": "Пешеходный переход (Zebra)",
    "geo.boxTitle": "Ключевые Правила Геометрии Камеры",
    "geo.rule1": "Каждая зона откалибрована оффлайн: зеленый полигон — проезжая часть, желтые — пешеходный переход и зоны разворота.",
    "geo.rule2": "Фиксация нарушений (wrong_way, failure_to_yield, jaywalking) основана на пересечении полигонов и векторов движения.",
    "geo.rule3": "Траектории (оранжевые линии) и векторы скорости объектов отслеживаются для мгновенной детекции опасных ситуаций.",

    // 14 Classes
    "classes.tag": "Стандартизированные События",
    "classes.title": "14 Официальных Классов Событий",
    "classes.subtitle": "Причинно-следственные связи дорожных инцидентов и алгоритмы их детекции в Computer Vision",
    "filter.all": "🌐 Все (14)",
    "filter.collision": "💥 Аварии и ДТП",
    "filter.pedestrian": "🚸 Пешеходы",
    "filter.discipline": "🚦 Дисциплина Полос",
    "filter.flow": "🛑 Заторы и Помехи",
    "card.scenarioLabel": "Реальный Сценарий:",
    "card.chainLabel": "Связь в Цепочке (Причина-Следствие):",
    "card.logicLabel": "Логика Детекции CV:",

    // Ablation Table
    "results.tag": "Результаты Тестирования",
    "results.title": "Ablation Анализ и Критерии Жюри",
    "results.subtitle": "Официальная метрика M = 0.7·Score_A + 0.3·Score_B и анализ вклада каждого модуля в скорость и точность",
    "th.config": "Конфигурация / Модули",
    "th.partA": "Part A (Macro tIoU F1)",
    "th.partB": "Part B (Alarm F1)",
    "th.scoreM": "Балл M (0.7A + 0.3B)",
    "th.fps": "FPS (Скорость)",
    "th.constraint": "Фактор Времени (≤ 3.0×)",
    "results.h1.title": "Гарантия Лимита Времени",
    "results.h1.desc": "При лимите 3.0× алгоритм Algorix выполняет инференс за 1.22× с запасом надежности 60% без риска дисквалификации.",
    "results.h2.title": "Каузальность и Zero Data Leakage",
    "results.h2.desc": "Модуль Part B строго исключает доступ к будущим кадрам, гарантируя абсолютную честность и применимость в реальном времени.",
    "results.h3.title": "100% Автономность (Offline Ready)",
    "results.h3.desc": "Никаких сетевых запросов. Все веса YOLO11x и калибровки укладываются в лимит 5 ГБ и работают автономно.",

    // Live Demo
    "demo.tag": "Интерактивный Тест",
    "demo.title": "Мониторинг CCTV и Симулятор Риска",
    "demo.subtitle": "Выберите событие из журнала, чтобы оценить расчет риска и телеметрию",
    "demo.liveBadge": "CCTV МОНИТОРИНГ",
    "demo.timelineTitle": "Журнал Обнаруженных Инцидентов",
    "demo.clickTip": "Нажмите на событие для перехода:",
    "demo.riskLabel": "Индекс Риска Аварии:",
    "demo.ttcLabel": "Время до Столкновения (TTC):",

    // Team
    "team.tag": "Разработчики",
    "team.title": "Команда Algorix",
    "team.subtitle": "Участники WIUT Hackathon 2026 — инженеры интеллектуальных транспортных систем",
    "team.member1.role": "Developer",
    "team.member1.bio": "Разработка бэкенд-архитектуры, оптимизация инференса моделей и высокоскоростная обработка видеопотоков.",
    "team.member2.badge": "Team Captain",
    "team.member2.role": "AI Engineer & Team Captain",
    "team.member2.bio": "Архитектура Computer Vision, алгоритмы Scene-Geometry, каузальный Risk Estimator и общее техническое руководство проектом.",
    "team.member3.role": "Speaker",
    "team.member3.bio": "Презентация проекта, упаковка бизнес-ценности и защита технических решений перед экспертным жюри.",

    // Footer
    "footer.text": "© 2026 Algorix. Разработано для WIUT Hackathon 2026. Все права защищены."
  }
};

// 14 Events Catalog Data with Real-World Cause-and-Effect Ecosystem
const eventCatalog = [
  // Chain 1: Collision & Dangerous Maneuvers (4 items)
  {
    id: "illegal_u_turn",
    name: "illegal_u_turn",
    icon: "↩️",
    category: "violation",
    chainGroup: "collision",
    uzChain: "1. To'qnashuv Zanjiri",
    ruChain: "1. Цепочка ДТП",
    uzScenario: "Taqiqlangan joyda (zebra ustida yoki to'g'ri yurish yo'lagida) 180° ga burilib qayrilib olish. Boshqa oqim yo'liga kutilmaganda chiqib keladi.",
    ruScenario: "Разворот на 180° в неположенном месте (на пешеходном переходе или из прямого ряда), перекрывающий встречный и попутный потоки.",
    uzConnection: "🔗 Boshlang'ich xavfli manevr ➡️ Yo'lakni to'sib, qarama-qarshi tomonga chiqqanda wrong_way va near_miss keltirib chiqaradi.",
    ruConnection: "🔗 Исходный опасный маневр ➡️ Выезд против потока провоцирует wrong_way, экстренное торможение и near_miss.",
    uzLogic: "Traektoriya yo'nalish burchagi keskin 180° ga o'zgarishi va transportning U-turn taqiqlangan zonada bo'lishi.",
    ruLogic: "Разворот вектора движения на 180° вне разрешенной зоны разворота (U-turn zone)."
  },
  {
    id: "wrong_way",
    name: "wrong_way",
    icon: "⛔",
    category: "violation",
    chainGroup: "collision",
    uzChain: "1. To'qnashuv Zanjiri",
    ruChain: "1. Цепочка ДТП",
    uzScenario: "Bir tomonlama harakat yo'lagida qonuniy oqimga qarab teskari harakatlanish (masalan Janubiy oqim yo'lagida Shimol tomonga qarshi yurish).",
    ruScenario: "Движение навстречу основному потоку транспорта по полосе с односторонним направлением движения.",
    uzConnection: "🔗 Kelib chiqishi: Taqiqlangan illegal_u_turn yoki chalkashlik ➡️ Frontal to'qnashuv va og'ir accident xavfini paydo qiladi.",
    ruConnection: "🔗 Источник: Запрещенный разворот illegal_u_turn ➡️ Прямой риск лобового удара и фатального accident.",
    uzLogic: "Harakat yo'nalish vektori va yo'lak ruxsat etilgan burchagi orasidagi tafovut > 120°, t ≥ 1.0s.",
    ruLogic: "Угол вектора перемещения ТС расходится с разрешенным вектором полосы более чем на 120° при t ≥ 1.0с."
  },
  {
    id: "near_miss",
    name: "near_miss",
    icon: "⚠️",
    category: "warning",
    chainGroup: "collision",
    uzChain: "1. To'qnashuv Zanjiri",
    ruChain: "1. Цепочка ДТП",
    uzScenario: "To'qnashuvga bir soniya qolganda ikki mashinaning yoki mashina va piyodaning keskin tormoz berib arang to'qnashuvdan qutulib qolishi.",
    ruScenario: "Предаварийная ситуация: два ТС или машина и пешеход экстренно тормозят в сантиметрах от удара.",
    uzConnection: "🔗 Sababi: wrong_way, failure_to_yield yoki red_light ➡️ Agar tormoz masofasi yetmasa, muqarrar accident sodir bo'ladi.",
    ruConnection: "🔗 Причина: Нарушения wrong_way, failure_to_yield, red_light ➡️ При нехватке дистанции торможения переходит в accident.",
    uzLogic: "TTC (Time-to-Collision) < 1.5s + masofa < 3.5 metr + keskin tormozlanish (a < -3.5 m/s²).",
    ruLogic: "Время до столкновения TTC < 1.5 сек при сближении < 3.5м и ускорении торможения < -3.5 м/с²."
  },
  {
    id: "accident",
    name: "accident",
    icon: "💥",
    category: "critical",
    chainGroup: "collision",
    uzChain: "1. To'qnashuv Zanjiri",
    ruChain: "1. Цепочка ДТП",
    uzScenario: "Ikki yoki undan ortiq transport vositalarining to'qnashishi, zarb ta'sirida yo'ldan chetga chiqishi va harakatning to'xtashi.",
    ruScenario: "Столкновение двух или более транспортных средств: резкое гашение кинетической энергии, повреждения и остановка.",
    uzConnection: "🔗 Eng og'ir yakuniy bosqich: Oldi olinmagan near_miss yoki qoidabuzarlik oqibati ➡️ Yo'lni to'sib stopped_vehicle va tirbandlik keltirib chiqaradi.",
    ruConnection: "🔗 Финальный исход цепочки нарушений ➡️ Парализует ряд, переходя в stopped_vehicle и затор congestion.",
    uzLogic: "Bounding box overlap (IoU > 0.35) + tezlikning kutilmaganda nolga tushishi + 10s davomida harakatsizlik.",
    ruLogic: "Взаимное перекрытие контуров ТС (IoU > 0.35) + мгновенный сброс скорости до нуля + отсутствие движения."
  },

  // Chain 2: Pedestrian Safety (2 items)
  {
    id: "jaywalking",
    name: "jaywalking",
    icon: "🚶",
    category: "violation",
    chainGroup: "pedestrian",
    uzChain: "2. Piyodalar Zanjiri",
    ruChain: "2. Пешеходная Цепочка",
    uzScenario: "Piyodaning zebra va belgilangan o'tish joyi bo'lmagan qatnov qismiga, harakatlanayotgan mashinalar orasiga kutilmaganda chiqib kelishi.",
    ruScenario: "Выход и переход дороги пешеходом вне зоны пешеходного перехода прямо через оживленный поток.",
    uzConnection: "🔗 Boshlang'ich xavf manbai ➡️ Haydovchini keskin tormoz qilishga (failure_to_yield) va near_miss holatiga majbur qiladi.",
    ruConnection: "🔗 Исходный фактор риска ➡️ Вынуждает поток реагировать, создавая риск наезда failure_to_yield и near_miss.",
    uzLogic: "Piyoda (person) qatnov qismi (Road Polygon) ichida, lekin Zebra poligonidan tashqarida bo'lishi.",
    ruLogic: "Детекция пешехода (person) внутри Road Polygon, но за пределами полигона Crosswalk."
  },
  {
    id: "failure_to_yield",
    name: "failure_to_yield",
    icon: "🚸",
    category: "violation",
    chainGroup: "pedestrian",
    uzChain: "2. Piyodalar Zanjiri",
    ruChain: "2. Пешеходная Цепочка",
    uzScenario: "Piyoda zebrada yoki yo'l chetida qonuniy o'tayotgan vaqtda avtomobil haydovchisining tezlikni pasaytirmay uning oldidan kesib o'tishi.",
    ruScenario: "Водитель не пропускает пешехода, уже вступившего на зебру, продолжая движение без замедления.",
    uzConnection: "🔗 Qoidaga bo'ysunmaslik ➡️ Piyodaning hayotini xavf ostiga qo'yadi va to'g'ridan-to'g'ri piyoda bilan near_miss / accident ga olib keladi.",
    ruConnection: "🔗 Пренебрежение правилами уступки дороги ➡️ Создает прямую угрозу наезда и тяжелого accident.",
    uzLogic: "Mashina va piyoda markazlari masofasi < 6 metr bo'lib, avtomobil tormoz bermasdan o'tib ketishi.",
    ruLogic: "Расстояние между ТС и пешеходом на переходе < 6м при отсутствии подтвержденного замедления ТС."
  },

  // Chain 3: Lane Discipline & Signals (4 items)
  {
    id: "solid_line_crossing",
    name: "solid_line_crossing",
    icon: "➖",
    category: "violation",
    chainGroup: "discipline",
    uzChain: "3. Yo'l Intizomi Zanjiri",
    ruChain: "3. Дисциплина Полос",
    uzScenario: "Chorrahaga yaqinlashish hududida yo'laklar orasidagi uzluksiz chiziqni bosib o'tib boshqa qatorga sakrash.",
    ruScenario: "Пересечение сплошной линии разметки между попутными рядами при подъезде к перекрестку.",
    uzConnection: "🔗 Qatorda navbatsiz o'tishga urinish ➡️ Yonidagi mashinani siqib chiqaradi va illegal_turn yoki yonlama zarba keltiradi.",
    ruConnection: "🔗 Попытка вклиниться в соседний ряд ➡️ Создает боковую помеху, предшествует illegal_turn и ДТП.",
    uzLogic: "Avtomobilning pastki markaz nuqtasi (bottom-center) kalibrlangan Solid Line segmentini kesib o'tishi.",
    ruLogic: "Пересечение опорной точки ТС (bottom-center) калиброванного сегмента Solid Line."
  },
  {
    id: "illegal_turn",
    name: "illegal_turn",
    icon: "↪️",
    category: "violation",
    chainGroup: "discipline",
    uzChain: "3. Yo'l Intizomi Zanjiri",
    ruChain: "3. Дисциплина Полос",
    uzScenario: "Faqat to'g'riga ruxsat berilgan 2- yoki 3-qatordan to'satdan chapga yoki o'ngga burilish manevrini amalga oshirish.",
    ruScenario: "Поворот налево или направо из полосы, предписывающей исключительно движение прямо.",
    uzConnection: "🔗 Oqim intizomini buzish ➡️ Ruxsat etilgan qatordan qonuniy burilayotgan avtomobillar yo'lini kesib xavf tug'diradi.",
    ruConnection: "🔗 Нарушение рядности ➡️ Режет траекторию ТС, законно совершающих маневр из крайнего ряда.",
    uzLogic: "Burilish traektoriyasi qayd etilganda avtomobilning to'g'ri yo'lak (Lane 2/3) poligonida joylashganligi.",
    ruLogic: "Обнаружение траектории поворота ТС из полосы, не размеченной для данного маневра."
  },
  {
    id: "stop_line",
    name: "stop_line",
    icon: "🚧",
    category: "warning",
    chainGroup: "discipline",
    uzChain: "3. Yo'l Intizomi Zanjiri",
    ruChain: "3. Дисциплина Полос",
    uzScenario: "Svetoforning qizil chirog'ida stop-chiziqdan oshib ketib, chorraha yoki zebra ustida to'xtab qolish.",
    ruScenario: "Выезд за пределы стоп-линии с последующей остановкой прямо на зебре при запрещающем сигнале.",
    uzConnection: "🔗 E'tiborsizlik ➡️ Piyodalar o'tishiga to'sqinlik qiladi va keyingi bosqichda red_light buzilishiga aylanadi.",
    ruConnection: "🔗 Ошибка позиционирования ➡️ Блокирует пешеходов и часто перерастает в полноценный red_light.",
    uzLogic: "Avtomobil Stop-Line segmentini kesib o'tib, keyingi 2 soniyada to'liq to'xtashi.",
    ruLogic: "Пересечение линии Stop Line с подтверждением остановки перед входом в конфликтную зону."
  },
  {
    id: "red_light",
    name: "red_light",
    icon: "🚦",
    category: "violation",
    chainGroup: "discipline",
    uzChain: "3. Yo'l Intizomi Zanjiri",
    ruChain: "3. Дисциплина Полос",
    uzScenario: "Qizil chiroqqa e'tibor bermasdan stop-line'dan o'tib chorrahaga to'liq tezlik bilan bostirib kirish.",
    ruScenario: "Проезд перекрестка на запрещающий красный сигнал светофора без снижения скорости.",
    uzConnection: "🔗 O'ta xavfli qoidabuzarlik ➡️ Yashil chiroqda chiqayotgan ko'ndalang oqim bilan halokatli T-bone to'qnashuv keltiradi.",
    ruConnection: "🔗 Критическое нарушение ➡️ Провоцирует боковой T-bone удар с перпендикулярным потоком.",
    uzLogic: "Qizil faza vaqtida stop-linedan o'tib, chorraha markaziy poligoniga tezlik > 15 km/h bilan kirish.",
    ruLogic: "Проезд стоп-линии в фазе запрета с дальнейшим движением через ядро перекрестка на скорости."
  },

  // Chain 4: Traffic Flow & Incidents (4 items)
  {
    id: "road_obstacle",
    name: "road_obstacle",
    icon: "📦",
    category: "warning",
    chainGroup: "flow",
    uzChain: "4. Oqim & To'siqlar",
    ruChain: "4. Помехи и Заторы",
    uzScenario: "Yo'l qatnov qismida yotgan begona to'siq: mashinadan tushib qolgan yuk, singan ehtiyot qismlar yoki g'ildirak.",
    ruScenario: "Посторонний опасный предмет на проезжей части: упавший груз, крупный мусор, отвалившиеся детали.",
    uzConnection: "🔗 Kutilmagan to'siq ➡️ Avtomobillar to'satdan burilib near_miss qiladi, oqim to'xtab stopped_vehicle va tirbandlik boshlanadi.",
    ruConnection: "🔗 Внезапная преграда ➡️ Провоцирует резкое перестроение near_miss, остановки и затор congestion.",
    uzLogic: "Qatnov qismida 5 soniyadan ortiq turg'un harakatsiz jism aniqlanishi (transport va piyodadan tashqari).",
    ruLogic: "Обнаружение неподвижного статичного объекта на полосе в течение более 5 секунд."
  },
  {
    id: "fire_smoke",
    name: "fire_smoke",
    icon: "🔥",
    category: "critical",
    chainGroup: "flow",
    uzChain: "4. Oqim & Favqulodda",
    ruChain: "4. ЧП и Огонь",
    uzScenario: "Avtomobilning yonib ketishi, dvigatel nosozligi yoki to'qnashuvdan keyingi kuchli tutun va alanga.",
    ruScenario: "Возгорание транспортного средства или густое задымление проезжей части после аварии или поломки двигателя.",
    uzConnection: "🔗 Favqulodda hodisa ➡️ Yo'lda ko'rinishni nolga tushiradi, butun yo'nalish bo'ylab harakatni to'xtatadi.",
    ruConnection: "🔗 Чрезвычайная ситуация ➡️ Полная потеря видимости, эвакуация людей и глухая остановка движения.",
    uzLogic: "Kadrda shiddatli rang va tekstura o'zgarishi, kengayuvchi olov/tutun piksellari zichligi.",
    ruLogic: "Детекция дыма и пламени по цветовой гистограмме и быстрому расширению маски контура."
  },
  {
    id: "stopped_vehicle",
    name: "stopped_vehicle",
    icon: "🛑",
    category: "warning",
    chainGroup: "flow",
    uzChain: "4. Oqim & To'siqlar",
    ruChain: "4. Помехи и Заторы",
    uzScenario: "Avtomobilning taqiqlangan harakat zonasida buzilib qolishi yoki yo'lovchi tushirish uchun 10 soniyadan ortiq qotib turishi.",
    ruScenario: "Нештатная длительная остановка (≥10 секунд) в полосе движения из-за поломки или высадки пассажиров.",
    uzConnection: "🔗 Yo'lakni to'sib qo'yish ➡️ Orqadagi avtomobillar oqimini sekinlashtirib, global tirbandlikka (congestion) olib keladi.",
    ruConnection: "🔗 Блокировка полосы ➡️ Вынуждает хвост потока тормозить, порождая лавинообразный затор congestion.",
    uzLogic: "Qatnov qismidagi avtomobil tezligi v < 1.0 km/h bo'lib, harakatsizlik vaqti t ≥ 10.0s dan oshishi.",
    ruLogic: "Скорость ТС внутри полосы ниже 1.0 км/ч непрерывно на протяжении более 10.0 секунд."
  },
  {
    id: "congestion",
    name: "congestion",
    icon: "🚗",
    category: "traffic",
    chainGroup: "flow",
    uzChain: "4. Oqim & Tirbandlik",
    ruChain: "4. Помехи и Заторы",
    uzScenario: "Barcha yo'laklarda oqim to'xtab, avtomobillar bir-biriga zich tizilib chorrahaning butunlay falaj bo'lib qolishi.",
    ruScenario: "Плотный масштабный затор: автомобили стоят во всех полосах бампер к бамперу, пропускная способность равна 0.",
    uzConnection: "🔗 Barcha voqealarning tizimli oqibati: stopped_vehicle, accident yoki tirband oqim sababli chorraha to'liq to'xtaydi.",
    ruConnection: "🔗 Финальный системный коллапс: итог нерасчищенного accident, stopped_vehicle или перегрузки перекрестка.",
    uzLogic: "Barcha yo'laklardagi o'rtacha tezlik < 5 km/h bo'lib, poligon bo'yicha mashinalar zichligi > 0.70 bo'lishi.",
    ruLogic: "Средняя скорость потока по всем полосам < 5 км/ч при плотности заполнения полос машинами > 0.70."
  }
];

// Current Language and Filter State
let currentLang = localStorage.getItem("algorix_lang") || "uz";
let currentFilter = "all";

// Function to update texts based on selected language
function setLanguage(lang) {
  currentLang = lang;
  localStorage.setItem("algorix_lang", lang);

  // Update HTML lang attribute
  document.documentElement.lang = lang;

  // Update plain text elements
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (translations[lang] && translations[lang][key]) {
      el.textContent = translations[lang][key];
    }
  });

  // Update language buttons active state
  document.querySelectorAll(".lang-btn").forEach(btn => {
    if (btn.getAttribute("data-lang") === lang) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  // Re-render event cards catalog with current language and active filter
  renderEventCards(currentFilter);
}

// Render 14 Event Cards cleanly with cause-and-effect ecosystem
function renderEventCards(filter = "all") {
  const container = document.getElementById("eventsGrid");
  if (!container) return;

  const filtered = filter === "all" 
    ? eventCatalog 
    : eventCatalog.filter(ev => ev.chainGroup === filter);

  const scenarioLbl = translations[currentLang]["card.scenarioLabel"] || "Real Hayotiy Ko'rinishi:";
  const chainLbl = translations[currentLang]["card.chainLabel"] || "Zanjir Bog'liqligi:";
  const logicLbl = translations[currentLang]["card.logicLabel"] || "CV / Deteksiya Qoidasi:";

  container.innerHTML = filtered.map(ev => {
    const chainName = currentLang === "ru" ? ev.ruChain : ev.uzChain;
    const scenario = currentLang === "ru" ? ev.ruScenario : ev.uzScenario;
    const connection = currentLang === "ru" ? ev.ruConnection : ev.uzConnection;
    const logic = currentLang === "ru" ? ev.ruLogic : ev.uzLogic;

    return `
      <div class="event-card category-${ev.category}">
        <div class="event-card-header">
          <div class="event-header-left">
            <span class="event-icon">${ev.icon}</span>
            <span class="event-cat-badge">${ev.category.toUpperCase()}</span>
          </div>
          <span class="event-chain-badge">${chainName}</span>
        </div>

        <div class="event-title">
          <code>${ev.name}</code>
        </div>

        <div class="event-section-label">${scenarioLbl}</div>
        <div class="event-scenario">${scenario}</div>

        <div class="event-connection-box">
          <div class="event-section-label" style="color: #94a3b8;">${chainLbl}</div>
          <p>${connection}</p>
        </div>

        <div class="event-logic-box">
          <div class="event-section-label" style="color: #67e8f9;">${logicLbl}</div>
          <p>${logic}</p>
        </div>
      </div>
    `;
  }).join("");
}

// Live Demo Timeline Interactive Jump
function jumpTo(seconds, eventName, riskVal, ttcVal, el) {
  const timeBadge = document.getElementById("demoTimeBadge");
  const frameBadge = document.getElementById("demoFrameBadge");
  const riskValueBadge = document.getElementById("demoRiskValue");
  const riskProgressBar = document.getElementById("demoRiskProgress");
  const ttcValueBadge = document.getElementById("demoTtcValue");
  const eventStatusBadge = document.getElementById("demoEventStatus");

  const frameNum = Math.floor(seconds * 30);
  const min = Math.floor(seconds / 60);
  const sec = (seconds % 60).toFixed(2);
  const timeStr = `${String(min).padStart(2, '0')}:${String(sec).padStart(5, '0')}s`;

  if (timeBadge) timeBadge.textContent = timeStr;
  if (frameBadge) frameBadge.textContent = `Frame: ${String(frameNum).padStart(4, '0')}`;
  
  if (riskValueBadge) riskValueBadge.textContent = riskVal.toFixed(2);
  if (riskProgressBar) {
    const percent = Math.min(Math.round(riskVal * 100), 100);
    riskProgressBar.style.width = `${percent}%`;
    if (riskVal > 0.7) {
      riskProgressBar.className = "progress-fill critical";
    } else if (riskVal > 0.3) {
      riskProgressBar.className = "progress-fill warning";
    } else {
      riskProgressBar.className = "progress-fill safe";
    }
  }

  if (ttcValueBadge) {
    ttcValueBadge.textContent = ttcVal > 0 ? `${ttcVal.toFixed(1)}s` : "Safe (N/A)";
  }

  if (eventStatusBadge) {
    eventStatusBadge.textContent = eventName;
  }

  // Update active state in timeline list
  document.querySelectorAll(".timeline-item").forEach(item => {
    item.classList.remove("active");
  });
  if (el) {
    el.classList.add("active");
  }
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  // Apply saved or default language
  setLanguage(currentLang);

  // Set up language toggle clicks
  document.querySelectorAll(".lang-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const selected = btn.getAttribute("data-lang");
      setLanguage(selected);
    });
  });

  // Set up cause-effect chain filter buttons clicks
  document.querySelectorAll(".event-filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".event-filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.getAttribute("data-filter") || "all";
      renderEventCards(currentFilter);
    });
  });

  // Smooth scroll for in-page navigation links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function(e) {
      const href = this.getAttribute("href");
      if (href === "#") return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Intersection Observer for subtle fade-in reveals
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("revealed");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll(".reveal-on-scroll").forEach(el => {
    observer.observe(el);
  });
});
