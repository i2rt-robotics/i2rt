---
layout: page
---

<script setup>
import { withBase } from 'vitepress'
</script>

<div class="i2rt-home">

<section class="hero">
  <div class="hero-text">
    <div class="hero-eyebrow">Open Robotics Platform</div>
    <h1 class="hero-title">
      <span class="grad">I2RT</span> Robotics
    </h1>
    <p class="hero-sub">
      Cost-effective, high-performance hardware built for learning-based robotics —
      teleoperation-ready, policy-deployment-ready, and battle-tested in the real world.
    </p>
    <div class="hero-actions">
      <a class="btn-primary" :href="withBase('/getting-started/sw-setup')">Get Started →</a>
      <a class="btn-ghost"   :href="withBase('/products/')">Explore Products</a>
      <a class="btn-ghost"   href="https://github.com/i2rt-robotics/i2rt" target="_blank">GitHub</a>
    </div>
    <div class="hero-chips">
      <span class="chip">Fully Open Source</span>
      <span class="chip">Production Ready</span>
      <span class="chip">Python SDK</span>
    </div>
  </div>
  <div class="hero-viewer">
    <ClientOnly>
      <RobotViewer />
    </ClientOnly>
  </div>
</section>

<section class="features">
  <a class="feat" :href="withBase('/products/yam')">
    <span class="feat-icon">🦾</span>
    <h3>YAM Arm Family</h3>
    <p>Low-cost, low-inertia 6-DOF arms designed for AI training and teleoperation — four tiers from research to production.</p>
    <span class="feat-link">Learn more →</span>
  </a>
  <a class="feat" :href="withBase('/products/flow-base')">
    <span class="feat-icon">🛞</span>
    <h3>Flow Base</h3>
    <p>Omnidirectional holonomic platform with on-board Pi, remote control, and full Python API.</p>
    <span class="feat-link">Learn more →</span>
  </a>
  <a class="feat" :href="withBase('/products/yam-cell')">
    <span class="feat-icon">📡</span>
    <h3>Teleoperation Ready</h3>
    <p>Bilateral leader-follower teleop. YAM Cell collects bimanual demonstrations for embodied AI.</p>
    <span class="feat-link">Learn more →</span>
  </a>
  <a class="feat" :href="withBase('/getting-started/demos/yam')">
    <span class="feat-icon">🐍</span>
    <h3>Python-First SDK</h3>
    <p>Plug-and-play API: gravity compensation, PD control, trajectory recording, and simulation.</p>
    <span class="feat-link">Quick start →</span>
  </a>
</section>

<section class="products">
  <h2 class="section-title">Product Family</h2>
  <div class="product-grid">
    <a class="pcard" :href="withBase('/products/yam')">
      <span class="pcard-icon">🦾</span>
      <div>
        <h3>YAM Series</h3>
        <p>6-DOF arms in four tiers: YAM · Pro · Ultra · BIG YAM.</p>
        <span class="pcard-price">From $2,999</span>
      </div>
    </a>
    <a class="pcard" :href="withBase('/products/yam-cell')">
      <span class="pcard-icon">🔬</span>
      <div>
        <h3>YAM Cell</h3>
        <p>Bimanual teleoperation station — 2 leader + 2 follower arms.</p>
        <span class="pcard-badge">Python SDK</span>
      </div>
    </a>
    <a class="pcard" :href="withBase('/products/yam-box')">
      <span class="pcard-icon">📦</span>
      <div>
        <h3>YAM Box</h3>
        <p>Enclosed manipulation station. Hardware available.</p>
        <span class="pcard-badge muted">Code Coming Soon</span>
      </div>
    </a>
    <a class="pcard" :href="withBase('/products/flow-base')">
      <span class="pcard-icon">🛞</span>
      <div>
        <h3>Flow Base</h3>
        <p>Omnidirectional mobile base with Python API and remote.</p>
      </div>
    </a>
    <a class="pcard" :href="withBase('/products/linear-bot')">
      <span class="pcard-icon">📏</span>
      <div>
        <h3>Linear Bot</h3>
        <p>Flow Base + vertical linear rail — full mobile manipulation.</p>
        <span class="pcard-price">From $18,999</span>
      </div>
    </a>
  </div>
</section>

<section class="quickstart">
  <div class="qs-text">
    <h2>Up and running in minutes</h2>
    <p>Install the Python SDK, plug in the CAN adapter, and start controlling the arm in a single script.</p>
    <a class="btn-primary" :href="withBase('/getting-started/demos/yam')">Full Quick Start →</a>
  </div>
  <div class="qs-code">

```python
from i2rt.robots.get_robot import get_yam_robot
import numpy as np

# Connect (zero-gravity mode by default)
robot = get_yam_robot(channel="can0")

# Read current observations
obs = robot.get_observations()
print(obs["joint_pos"])  # (6,) arm joints in radians

# Command the arm home (6 joints + 1 gripper)
robot.command_joint_pos(np.zeros(7))
```

  </div>
</section>

<section class="cta-row">
  <a class="btn-primary" :href="withBase('/getting-started/sw-setup')">Get Started →</a>
  <a class="btn-ghost"   href="https://i2rt.com" target="_blank">i2rt.com</a>
  <a class="btn-ghost"   href="mailto:support@i2rt.com">support@i2rt.com</a>
</section>

</div>

<style>
.i2rt-home {
  max-width: 1400px;
  margin: 0 auto;
  padding: 0 48px 80px;
}
@media (max-width: 1100px) { .i2rt-home { padding: 0 32px 80px; } }
@media (max-width: 600px)  { .i2rt-home { padding: 0 20px 60px; } }

/* HERO */
.hero {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 48px;
  align-items: center;
  padding: 72px 0 80px;
  min-height: 580px;
}
@media (max-width: 1100px) { .hero { gap: 32px; padding: 56px 0 64px; } }
@media (max-width: 860px) {
  .hero { grid-template-columns: 1fr; padding: 40px 0 48px; gap: 24px; }
  .hero-viewer { order: -1; }
}
@media (max-width: 480px) { .hero { padding: 28px 0 36px; } }

/* Eyebrow — i2rt.com style: 12px, uppercase, letter-spaced, muted */
.hero-eyebrow {
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(17, 21, 28, 0.50);
  margin-bottom: 16px;
}
.dark .hero-eyebrow { color: rgba(232, 228, 223, 0.45); }

.hero-title {
  font-family: 'Platypi', Georgia, serif;
  font-size: clamp(2.4rem, 5vw, 3.8rem);
  font-weight: 600;
  line-height: 1.1;
  letter-spacing: -0.03em;
  margin: 0 0 20px;
  border: none;
  padding: 0;
}

.grad {
  background: linear-gradient(135deg, #855832 0%, #4C6762 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-sub {
  font-size: 1.05rem;
  line-height: 1.7;
  color: rgba(17, 21, 28, 0.65);
  margin: 0 0 32px;
  max-width: 440px;
}
.dark .hero-sub { color: rgba(232, 228, 223, 0.65); }

.hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 28px; }

/* Buttons: i2rt.com style — minimal, border-based */
.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 11px 22px;
  border-radius: 4px;
  background: #855832;
  color: #DFDFD8 !important;
  font-size: 0.9rem;
  font-weight: 500;
  text-decoration: none !important;
  transition: opacity 0.15s;
}
.btn-primary:hover { opacity: 0.85; }

.btn-ghost {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 11px 20px;
  border-radius: 4px;
  color: rgba(17, 21, 28, 1) !important;
  border: 1px solid rgba(17, 21, 28, 0.25);
  font-size: 0.9rem;
  font-weight: 500;
  text-decoration: none !important;
  transition: border-color 0.15s;
}
.btn-ghost:hover { border-color: rgba(17, 21, 28, 0.6); }
.dark .btn-ghost { color: rgba(232,228,223,1) !important; border-color: rgba(232,228,223,0.25); }
.dark .btn-ghost:hover { border-color: rgba(232,228,223,0.6); }

/* Chips */
.hero-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  font-size: 0.70rem;
  font-weight: 500;
  padding: 3px 10px;
  border-radius: 2px;
  background: rgba(17, 21, 28, 0.06);
  border: 1px solid rgba(17, 21, 28, 0.12);
  color: rgba(17, 21, 28, 0.60);
}
.dark .chip { background: rgba(232,228,223,0.06); border-color: rgba(232,228,223,0.12); color: rgba(232,228,223,0.55); }

/* ── FEATURES — i2rt.com 1px gap grid trick ─────────── */
.features {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: rgba(17, 21, 28, 0.10);     /* gap color = border color */
  border: 1px solid rgba(17, 21, 28, 0.10);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 72px;
}
@media (max-width: 1000px) { .features { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px)  { .features { grid-template-columns: 1fr; margin-bottom: 48px; } }

.feat {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 28px 24px;
  background: var(--vp-c-bg);           /* same as page = gaps become visible lines */
  text-decoration: none !important;
  transition: background 0.15s;
}
.feat:hover { background: var(--vp-c-bg-soft); }
.feat-icon { font-size: 1.8rem; line-height: 1; }
.feat h3 {
  font-family: 'Platypi', Georgia, serif;
  font-size: 0.95rem; font-weight: 500; margin: 0; border: none; padding: 0;
  color: var(--vp-c-text-1);
}
.feat p { font-size: 0.85rem; color: var(--vp-c-text-2); line-height: 1.6; margin: 0; flex: 1; }
.feat-link { font-size: 0.78rem; font-weight: 500; color: #855832; }

/* ── PRODUCTS — same grid trick ─────────────────────── */
.products { margin-bottom: 72px; }
.section-title {
  font-family: 'Platypi', Georgia, serif;
  font-size: 1.5rem; font-weight: 600; letter-spacing: -0.02em; margin: 0 0 16px; border: none; padding: 0;
}
.product-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1px;
  background: rgba(17, 21, 28, 0.10);
  border: 1px solid rgba(17, 21, 28, 0.10);
  border-radius: 8px;
  overflow: hidden;
}
@media (max-width: 960px) { .product-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .product-grid { grid-template-columns: 1fr; } }

.pcard {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  padding: 24px;
  background: var(--vp-c-bg);
  text-decoration: none !important;
  transition: background 0.15s;
}
.pcard:hover { background: var(--vp-c-bg-soft); }
.pcard-icon { font-size: 1.6rem; flex-shrink: 0; padding-top: 2px; }
.pcard h3 {
  font-family: 'Platypi', Georgia, serif;
  font-size: 0.95rem; font-weight: 500; margin: 0 0 5px; border: none; padding: 0; color: var(--vp-c-text-1);
}
.pcard p { font-size: 0.83rem; color: var(--vp-c-text-2); margin: 0 0 8px; line-height: 1.55; }
.pcard-price { font-size: 0.78rem; font-weight: 600; color: #855832; }
.pcard-badge {
  display: inline-block; font-size: 0.68rem; font-weight: 500;
  padding: 2px 8px; border-radius: 2px;
  background: rgba(76,103,98,0.08); color: #4C6762; border: 1px solid rgba(76,103,98,0.20);
}
.pcard-badge.muted {
  background: rgba(17,21,28,0.05); color: rgba(17,21,28,0.40);
  border-color: rgba(17,21,28,0.12);
}
.dark .pcard-badge.muted {
  background: rgba(232,228,223,0.05); color: rgba(232,228,223,0.40);
  border-color: rgba(232,228,223,0.12);
}

/* ── QUICK START ─────────────────────────────────────── */
.quickstart {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 40px;
  align-items: center;
  padding: 40px 48px;
  background: var(--vp-c-bg-soft);
  border: 1px solid rgba(17, 21, 28, 0.10);
  border-radius: 8px;
  margin-bottom: 56px;
}
@media (max-width: 900px) { .quickstart { gap: 28px; padding: 32px 36px; } }
@media (max-width: 640px) { .quickstart { grid-template-columns: 1fr; padding: 24px; } }
.qs-text h2 {
  font-family: 'Platypi', Georgia, serif;
  font-size: 1.3rem; font-weight: 600; margin: 0 0 12px; border: none; padding: 0;
}
.qs-text p { color: var(--vp-c-text-2); line-height: 1.65; margin: 0 0 24px; }
.qs-code div[class*="language-"] { margin: 0 !important; }

/* ── CTA ─────────────────────────────────────────────── */
.cta-row { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; padding-top: 8px; }

/* ── DARK MODE ───────────────────────────────────────── */
.dark .features {
  background: rgba(255,255,255,0.07);
  border-color: rgba(255,255,255,0.07);
}
.dark .feat:hover { background: var(--vp-c-bg-soft); }
.dark .product-grid {
  background: rgba(255,255,255,0.07);
  border-color: rgba(255,255,255,0.07);
}
.dark .pcard:hover { background: var(--vp-c-bg-soft); }
.dark .quickstart { border-color: rgba(133,88,50,0.12); }
</style>
