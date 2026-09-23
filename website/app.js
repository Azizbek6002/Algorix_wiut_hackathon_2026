// Algorix Interactive Showcase JS

function jumpTo(seconds) {
  const overlay = document.querySelector('.video-overlay-text');
  const riskBadge = overlay.querySelector('.badge.green') || overlay.querySelector('.badge.red') || overlay.querySelector('.badge.yellow');
  const frameBadge = overlay.querySelector('.badge.blue');

  const frameNum = Math.floor(seconds * 30);
  const min = Math.floor(seconds / 60);
  const sec = (seconds % 60).toFixed(2);
  const timeStr = `${String(min).padStart(2, '0')}:${String(sec).padStart(5, '0')}s`;

  if (frameBadge) {
    frameBadge.textContent = `Frame: ${String(frameNum).padStart(4, '0')} | ${timeStr}`;
  }

  // Dynamic risk score calculation based on timestamp
  let risk = 0.08;
  let status = "Normal";
  let cls = "green";

  if (seconds >= 12.0 && seconds <= 18.0) {
    risk = 0.45;
    status = "Red Light Incursion";
    cls = "yellow";
  } else if (seconds >= 45.0 && seconds <= 54.0) {
    risk = 0.35;
    status = "Obstruction Risk";
    cls = "yellow";
  } else if (seconds >= 75.0 && seconds <= 85.0) {
    risk = 0.92;
    status = "CRITICAL: Near-Miss TTC=1.2s";
    cls = "red";
  }

  if (riskBadge) {
    riskBadge.className = `badge ${cls}`;
    riskBadge.textContent = `Risk Score: ${risk.toFixed(2)} (${status})`;
  }
}

// Smooth scrolling for navigation links
document.querySelectorAll('.nav-links a').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    const targetId = this.getAttribute('href');
    const targetElement = document.querySelector(targetId);
    if (targetElement) {
      targetElement.scrollIntoView({ behavior: 'smooth' });
      document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
      this.classList.add('active');
    }
  });
});

console.log("🚦 Algorix Interactive Showcase initialized.");
