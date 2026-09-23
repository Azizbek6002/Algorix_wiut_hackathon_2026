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
    "hero.badge": "WIUT Hackathon 2026 • Topshiriq: BE86F4D5",
    "hero.title": "Toyota Traffic Event Detection & Accident Anticipation",
    "hero.desc": "Fixed CCTV kameraga moslashtirilgan deterministik Scene-Geometry Computer Vision tizimi. 14 ta murakkab hodisani yuqori aniqlikda aniqlash va kelajakka qaramagan (causal) real-vaqt avariya xavfi prognozi.",
    "hero.btnDemo": "Live Demon Ko'rish",
    "hero.btnArchitecture": "Tizim Arxitekturasi",

    // Stats
    "stat.events.num": "14",
    "stat.events.label": "Trafik Hodisalari Klassi",
    "stat.latency.num": "< 1.3×",
    "stat.latency.label": "Inference Vaqt Ko'rsatkichi",
    "stat.risk.num": "100%",
    "stat.risk.label": "Causal Risk Baholash",
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
    "geo.subtitle": "1920×1080 kadr o'lchamidagi har bir zonaning qat'iy matematik chegaralari",
    "geo.laneSouth": "Yo'lak 1 & 2 (Janubiy oqim)",
    "geo.laneNorth": "Qarama-qarshi yo'lak (Oncoming)",
    "geo.stopLine": "Stop Chizig'i (Stop Line)",
    "geo.crosswalk": "Piyodalar o'tish joyi (Zebra)",
    "geo.boxTitle": "Kamera Geometriyasining Asosiy Qoidalari",
    "geo.rule1": "Har bir piksel koordinatasi offline kalibrlangan, hisoblashlar mikrosekundlarda bajariladi.",
    "geo.rule2": "Chiziq bosish (solid_line_crossing) va qarama-qarshi yurish (wrong_way) vektor burchaklari orqali aniqlanadi.",
    "geo.rule3": "Svetofor navbatida kutib turish stopped_vehicle hisoblanmaydi; faqat harakat zonasidagi to'xtashlar inobatga olinadi.",

    // 14 Classes
    "classes.tag": "Standartlashtirilgan Voqealar",
    "classes.title": "14 ta Rasmiy Event Klasslari",
    "classes.subtitle": "Part A vazifasi bo'yicha aniqlanadigan barcha hodisalar ro'yxati",

    // Ablation Table
    "results.tag": "Benchmark Natijalari",
    "results.title": "Ablation Tahlili va Samaradorlik",
    "results.subtitle": "Har bir komponent qo'shilgandagi aniqlik va tezlik o'sishi ko'rsatkichlari",
    "th.config": "Konfiguratsiya",
    "th.partA": "Part A (Mean tIoU F1)",
    "th.partB": "Part B (Alarm F1)",
    "th.fps": "FPS (Tezlik)",
    "th.constraint": "Vaqt Limiti Qoidasi",

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
    "hero.badge": "WIUT Hackathon 2026 • Задача: BE86F4D5",
    "hero.title": "Toyota Traffic Event Detection & Accident Anticipation",
    "hero.desc": "Детерминированный Scene-Geometry пайплайн компьютерного зрения, оптимизированный под фиксированную камеру CCTV. Детекция 14 сложных дорожных событий и строго каузальное (без взгляда в будущее) прогнозирование риска аварий.",
    "hero.btnDemo": "Смотреть Демо",
    "hero.btnArchitecture": "Архитектура Системы",

    // Stats
    "stat.events.num": "14",
    "stat.events.label": "Классов Дорожных Событий",
    "stat.latency.num": "< 1.3×",
    "stat.latency.label": "Фактор Времени Инференса",
    "stat.risk.num": "100%",
    "stat.risk.label": "Каузальный Расчет Риска",
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
    "geo.subtitle": "Точная математическая разметка зон перекрестка в разрешении 1920×1080",
    "geo.laneSouth": "Полосы 1 и 2 (Южный поток)",
    "geo.laneNorth": "Встречная полоса (Oncoming)",
    "geo.stopLine": "Стоп-линия (Stop Line)",
    "geo.crosswalk": "Пешеходный переход (Zebra)",
    "geo.boxTitle": "Ключевые Правила Геометрии Камеры",
    "geo.rule1": "Все ключевые полигоны откалиброваны оффлайн, расчет принадлежности происходит за микросекунды.",
    "geo.rule2": "Пересечение сплошной (solid_line) и выезд на встречную (wrong_way) вычисляются по углам векторов движения.",
    "geo.rule3": "Ожидание на светофоре не классифицируется как stopped_vehicle; фиксируется только нештатная остановка в полосе.",

    // 14 Classes
    "classes.tag": "Стандартизированные События",
    "classes.title": "14 Официальных Классов Событий",
    "classes.subtitle": "Список всех классифицируемых инцидентов для части Part A",

    // Ablation Table
    "results.tag": "Результаты Тестирования",
    "results.title": "Ablation Анализ и Метрики",
    "results.subtitle": "Вклад каждого модуля в итоговую точность F1 и частоту кадров (FPS)",
    "th.config": "Конфигурация",
    "th.partA": "Part A (Mean tIoU F1)",
    "th.partB": "Part B (Alarm F1)",
    "th.fps": "FPS (Скорость)",
    "th.constraint": "Лимит Времени (≤3×)",

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

// 14 Events Catalog Data with Descriptions
const eventCatalog = [
  { id: "accident", name: "accident", icon: "💥", category: "critical", uzDesc: "To'qnashuv / Avariya holati", ruDesc: "Столкновение / Аварийная ситуация" },
  { id: "near_miss", name: "near_miss", icon: "⚠️", category: "warning", uzDesc: "Avariyaga yaqin holat (keskin tormoz/burilish)", ruDesc: "Опасное сближение (экстренное торможение)" },
  { id: "red_light", name: "red_light", icon: "🚦", category: "violation", uzDesc: "Qizil chiroqda stop-line'dan o'tish", ruDesc: "Проезд стоп-линии на красный сигнал" },
  { id: "wrong_way", name: "wrong_way", icon: "⛔", category: "violation", uzDesc: "Qarama-qarshi yo'nalishda harakatlanish", ruDesc: "Движение по встречной полосе" },
  { id: "illegal_u_turn", name: "illegal_u_turn", icon: "↩️", category: "violation", uzDesc: "Taqiqlangan joyda qayrilib olish", ruDesc: "Запрещенный разворот (U-turn)" },
  { id: "stopped_vehicle", name: "stopped_vehicle", icon: "🛑", category: "warning", uzDesc: "Qatnov qismida ≥10s to'xtab qolish", ruDesc: "Остановка на проезжей части ≥10 сек" },
  { id: "jaywalking", name: "jaywalking", icon: "🚶", category: "violation", uzDesc: "Zebradan tashqarida yo'lga chiqish", ruDesc: "Переход дороги вне пешеходного перехода" },
  { id: "failure_to_yield", name: "failure_to_yield", icon: "🚸", category: "violation", uzDesc: "Piyodaga yo'l bermaslik holati", ruDesc: "Непредоставление преимущества пешеходу" },
  { id: "illegal_turn", name: "illegal_turn", icon: "↪️", category: "violation", uzDesc: "Noto'g'ri qatordan burilish", ruDesc: "Поворот из неразрешенной полосы" },
  { id: "solid_line_crossing", name: "solid_line_crossing", icon: "➖", category: "violation", uzDesc: "Uzluksiz chiziqni bosib o'tish", ruDesc: "Пересечение сплошной линии разметки" },
  { id: "stop_line", name: "stop_line", icon: "🚧", category: "warning", uzDesc: "Stop-line'dan o'tib to'xtab qolish", ruDesc: "Остановка сразу за стоп-линией" },
  { id: "congestion", name: "congestion", icon: "🚗", category: "traffic", uzDesc: "Barcha yo'laklarda kuchli tirbandlik", ruDesc: "Плотный затор на всех полосах движения" },
  { id: "road_obstacle", name: "road_obstacle", icon: "📦", category: "warning", uzDesc: "Yo'l ustidagi begona jism yoki to'siq", ruDesc: "Препятствие или посторонний предмет на дороге" },
  { id: "fire_smoke", name: "fire_smoke", icon: "🔥", category: "critical", uzDesc: "Yo'l hududida yong'in yoki quyuq tutun", ruDesc: "Возгорание или задымление на проезжей части" }
];

// Current Language State
let currentLang = localStorage.getItem("algorix_lang") || "uz";

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

  // Re-render event cards catalog with current language
  renderEventCards();
}

// Render 14 Event Cards cleanly
function renderEventCards() {
  const container = document.getElementById("eventsGrid");
  if (!container) return;

  container.innerHTML = eventCatalog.map(ev => {
    const desc = currentLang === "ru" ? ev.ruDesc : ev.uzDesc;
    return `
      <div class="event-card category-${ev.category}">
        <div class="event-card-header">
          <span class="event-icon">${ev.icon}</span>
          <span class="event-cat-badge">${ev.category.toUpperCase()}</span>
        </div>
        <h4 class="event-title"><code>${ev.name}</code></h4>
        <p class="event-desc">${desc}</p>
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
