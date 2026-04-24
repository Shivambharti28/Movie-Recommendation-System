:root {
  --bg: #07111f;
  --bg-soft: rgba(11, 23, 40, 0.72);
  --panel: rgba(12, 24, 40, 0.72);
  --panel-strong: rgba(10, 21, 37, 0.9);
  --line: rgba(255, 255, 255, 0.08);
  --text: #f4f7fb;
  --muted: #9cb2c9;
  --accent: #ff6b3d;
  --accent-soft: #ffb077;
  --blue: #61d0ff;
  --shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
  --radius-xl: 32px;
  --radius-lg: 24px;
  --radius-md: 18px;
}

* {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  min-height: 100%;
  background:
    radial-gradient(circle at top left, rgba(255, 107, 61, 0.22), transparent 30%),
    radial-gradient(circle at 80% 10%, rgba(97, 208, 255, 0.2), transparent 25%),
    linear-gradient(180deg, #06101d 0%, #091421 55%, #040912 100%);
  color: var(--text);
  font-family: "Manrope", sans-serif;
}

body {
  overflow-x: hidden;
}

.app-shell {
  position: relative;
  min-height: 100vh;
  padding: 28px;
}

.ambient {
  position: fixed;
  inset: auto;
  border-radius: 999px;
  filter: blur(60px);
  opacity: 0.35;
  pointer-events: none;
}

.ambient-one {
  top: 10%;
  left: -8%;
  width: 360px;
  height: 360px;
  background: rgba(255, 107, 61, 0.3);
}

.ambient-two {
  right: -6%;
  top: 30%;
  width: 420px;
  height: 420px;
  background: rgba(97, 208, 255, 0.16);
}

.topbar,
.content {
  position: relative;
  z-index: 2;
}

.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  margin-bottom: 26px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 14px;
}

.brand-mark {
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  border-radius: 18px;
  background: linear-gradient(135deg, var(--accent), #ff8d63);
  color: #fff;
  font-weight: 800;
  font-size: 1.35rem;
  box-shadow: var(--shadow);
}

.brand h1,
.section-header h2,
.explore-panel h2,
.hero h2 {
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.03em;
  margin: 0;
}

.eyebrow {
  margin: 0 0 6px;
  color: var(--accent-soft);
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 0.74rem;
}

.ghost-button,
.search-box button,
.mini-button {
  border: 0;
  cursor: pointer;
  transition: transform 180ms ease, opacity 180ms ease, background 180ms ease;
}

.ghost-button,
.mini-button,
.search-box button {
  border-radius: 999px;
}

.ghost-button {
  padding: 12px 18px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text);
  backdrop-filter: blur(18px);
}

.ghost-button:hover,
.search-box button:hover,
.mini-button:hover {
  transform: translateY(-1px);
}

.content {
  display: grid;
  gap: 22px;
}

.glass-panel,
.hero,
.movie-card {
  backdrop-filter: blur(26px);
}

.hero {
  min-height: 520px;
  border-radius: 34px;
  overflow: hidden;
  position: relative;
  display: flex;
  align-items: end;
  padding: 34px;
  background:
    linear-gradient(180deg, rgba(4, 8, 14, 0.08), rgba(5, 10, 18, 0.84)),
    linear-gradient(120deg, rgba(255, 107, 61, 0.14), transparent 30%),
    #0d1b2a;
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
}

.hero.loading::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent);
  animation: shimmer 2.4s infinite;
}

.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center;
  transform: scale(1.04);
  z-index: -2;
}

.hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(6, 11, 18, 0.08) 0%, rgba(4, 9, 15, 0.95) 100%);
  z-index: -1;
}

.hero-inner {
  display: grid;
  gap: 18px;
  max-width: 640px;
}

.hero h2 {
  font-size: clamp(2.4rem, 5vw, 4.9rem);
  line-height: 0.95;
}

.hero p {
  margin: 0;
  color: #d9e5f2;
  max-width: 60ch;
  font-size: 1rem;
  line-height: 1.75;
}

.hero-metrics,
.meta-row,
.genre-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.pill {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #edf5fe;
}

.pill.active {
  background: rgba(255, 107, 61, 0.2);
  border-color: rgba(255, 176, 119, 0.4);
}

.clear-pill {
  background: rgba(97, 208, 255, 0.08);
  border-color: rgba(97, 208, 255, 0.24);
  color: #d7f3ff;
}

.explore-panel {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  gap: 24px;
  padding: 28px;
  border-radius: var(--radius-xl);
  background: linear-gradient(135deg, rgba(13, 26, 44, 0.88), rgba(8, 17, 30, 0.72));
  border: 1px solid var(--line);
}

.panel-copy p:last-child {
  color: var(--muted);
  line-height: 1.75;
  margin-bottom: 0;
}

.search-stack {
  display: grid;
  gap: 14px;
  align-content: center;
}

.search-box {
  display: flex;
  gap: 12px;
  padding: 12px;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.search-box input {
  flex: 1;
  background: transparent;
  border: 0;
  outline: 0;
  color: var(--text);
  font-size: 1rem;
  padding: 0 8px;
}

.search-box button {
  padding: 14px 18px;
  color: #08121c;
  font-weight: 800;
  background: linear-gradient(135deg, #ffd26f, var(--accent));
}

.results-layout,
.rail {
  display: grid;
  gap: 16px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
}

.status-text {
  margin: 0;
  color: var(--muted);
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 18px;
}

.results-group-stack {
  display: grid;
  gap: 24px;
}

.results-group {
  display: grid;
  gap: 14px;
}

.results-group-header {
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 16px;
  padding-bottom: 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.results-group-header h3 {
  margin: 0;
  font-size: 1.2rem;
  font-family: "Space Grotesk", sans-serif;
  letter-spacing: -0.03em;
}

.results-group-meta {
  margin: 0;
  color: var(--muted);
}

.group-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 18px;
}

.movie-card {
  overflow: hidden;
  border-radius: 24px;
  background: linear-gradient(180deg, rgba(13, 24, 38, 0.88), rgba(8, 15, 26, 0.94));
  border: 1px solid rgba(255, 255, 255, 0.07);
  transition: transform 220ms ease, border-color 220ms ease;
}

.movie-card:hover {
  transform: translateY(-6px);
  border-color: rgba(255, 176, 119, 0.4);
}

.poster-wrap {
  position: relative;
  aspect-ratio: 0.76;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), transparent);
}

.poster {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.poster-overlay {
  position: absolute;
  inset: auto 12px 12px 12px;
  display: flex;
  justify-content: space-between;
  gap: 8px;
  opacity: 0;
  transform: translateY(10px);
  transition: opacity 180ms ease, transform 180ms ease;
}

.movie-card:hover .poster-overlay {
  opacity: 1;
  transform: translateY(0);
}

.mini-button {
  padding: 10px 12px;
  text-decoration: none;
  background: rgba(6, 15, 25, 0.88);
  color: #fff;
  font-weight: 700;
}

.card-copy {
  padding: 16px;
}

.card-topline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 0.9rem;
  margin-bottom: 8px;
}

.card-title {
  margin: 0 0 8px;
  font-size: 1.08rem;
}

.card-overview {
  margin: 0 0 14px;
  color: #c8d6e5;
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-genres .pill {
  padding: 7px 11px;
  font-size: 0.8rem;
}

.rails {
  display: grid;
  gap: 22px;
  margin-bottom: 42px;
}

.rail-track {
  display: grid;
  grid-auto-flow: column;
  grid-auto-columns: minmax(220px, 260px);
  gap: 18px;
  overflow-x: auto;
  padding-bottom: 8px;
}

.rail-track::-webkit-scrollbar {
  height: 8px;
}

.rail-track::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.12);
  border-radius: 999px;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

@media (max-width: 980px) {
  .explore-panel {
    grid-template-columns: 1fr;
  }

  .section-header,
  .topbar,
  .results-group-header {
    align-items: start;
    flex-direction: column;
  }
}

@media (max-width: 720px) {
  .app-shell {
    padding: 16px;
  }

  .hero {
    min-height: 460px;
    padding: 24px;
  }

  .hero h2 {
    font-size: 2.5rem;
  }

  .search-box {
    flex-direction: column;
  }

  .results-grid {
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
  }
}
