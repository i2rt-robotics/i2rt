---
layout: page
---

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
      <a class="btn-primary" href="/i2rt/getting-started/installation">Get Started →</a>
      <a class="btn-ghost"   href="/i2rt/products/">Explore Products</a>
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
  <a class="feat" href="/i2rt/products/yam">
    <span class="feat-icon">🦾</span>
    <h3>YAM Arm Family</h3>
    <p>Low-cost, low-inertia 6-DOF arms designed for AI training and teleoperation — four tiers from research to production.</p>
    <span class="feat-link">Learn more →</span>
  </a>
  <a class="feat" href="/i2rt/products/flow-base">
    <span class="feat-icon">🛞</span>
    <h3>Flow Base</h3>
    <p>Omnidirectional holonomic platform with on-board Pi, remote control, and full Python API.</p>
    <span class="feat-link">Learn more →</span>
  </a>
  <a class="feat" href="/i2rt/products/yam-cell">
    <span class="feat-icon">📡</span>
    <h3>Teleoperation Ready</h3>
    <p>Bilateral leader-follower teleop. YAM Cell collects bimanual demonstrations for embodied AI.</p>
    <span class="feat-link">Learn more →</span>
  </a>
  <a class="feat" href="/i2rt/getting-started/quick-start">
    <span class="feat-icon">🐍</span>
    <h3>Python-First SDK</h3>
    <p>Plug-and-play API: gravity compensation, PD control, trajectory recording, and simulation.</p>
    <span class="feat-link">Quick start →</span>
  </a>
</section>

<section class="products">
  <h2 class="section-title">Product Family</h2>
  <div class="product-grid">
    <a class="pcard" href="/i2rt/products/yam">
      <span class="pcard-icon">🦾</span>
      <div>
        <h3>YAM Series</h3>
        <p>6-DOF arms in four tiers: YAM · Pro · Ultra · BIG YAM.</p>
        <span class="pcard-price">From $2,999</span>
      </div>
    </a>
    <a class="pcard" href="/i2rt/products/yam-cell">
      <span class="pcard-icon">🔬</span>
      <div>
        <h3>YAM Cell</h3>
        <p>Bimanual teleoperation station — 2 leader + 2 follower arms.</p>
        <span class="pcard-badge">Python SDK</span>
      </div>
    </a>
    <a class="pcard" href="/i2rt/products/yam-box">
      <span class="pcard-icon">📦</span>
      <div>
        <h3>YAM Box</h3>
        <p>Enclosed manipulation station. Hardware available.</p>
        <span class="pcard-badge muted">Code Coming Soon</span>
      </div>
    </a>
    <a class="pcard" href="/i2rt/products/flow-base">
      <span class="pcard-icon">🛞</span>
      <div>
        <h3>Flow Base</h3>
        <p>Omnidirectional mobile base with Python API and remote.</p>
      </div>
    </a>
    <a class="pcard" href="/i2rt/products/linear-bot">
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
    <a class="btn-primary" href="/i2rt/getting-started/quick-start">Full Quick Start →</a>
  </div>
  <div class="qs-code">

```python
from i2rt.robots.motor_chain_robot import get_yam_robot
import numpy as np

# Connect (zero-gravity mode by default)
robot = get_yam_robot(channel="can0")

# Read current joint positions
q = robot.get_joint_pos()   # shape: (6,) radians

# Command the arm home
robot.command_joint_pos(np.zeros(6))
```

  </div>
</section>

<section class="cta-row">
  <a class="btn-primary" href="/i2rt/getting-started/installation">Get Started →</a>
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
@media (max-width: 1100px) {
  .hero { gap: 32px; padding: 56px 0 64px; }
}
@media (max-width: 860px) {
  .hero { grid-template-columns: 1fr; padding: 40px 0 48px; gap: 24px; }
  .hero-viewer { order: -1; }
}
@media (max-width: 480px) {
  .hero { padding: 28px 0 36px; }
}

.hero-eyebrow {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #FF7A29;
  margin-bottom: 16px;
}

.hero-title {
  font-size: clamp(2.4rem, 5vw, 3.8rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.04em;
  margin: 0 0 20px;
  border: none;
  padding: 0;
}

.grad {
  background: linear-gradient(135deg, #FF7A29 0%, #4CCFB0 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-sub {
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--vp-c-text-2);
  margin: 0 0 32px;
  max-width: 440px;
}

.hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 28px;
}

.btn-primary {
  display: inline-block;
  padding: 10px 22px;
  border-radius: 8px;
  background: linear-gradient(135deg, #FF7A29, #cc5500);
  color: #fff !important;
  font-size: 0.92rem;
  font-weight: 600;
  text-decoration: none !important;
  transition: transform 0.15s, opacity 0.15s;
}
.btn-primary:hover { transform: translateY(-1px); opacity: 0.9; }

.btn-ghost {
  display: inline-block;
  padding: 10px 20px;
  border-radius: 8px;
  background: rgba(255, 122, 41, 0.07);
  color: #FF7A29 !important;
  border: 1px solid rgba(255, 122, 41, 0.28);
  font-size: 0.92rem;
  font-weight: 600;
  text-decoration: none !important;
  transition: background 0.15s, transform 0.15s;
}
.btn-ghost:hover { background: rgba(255,122,41,0.14); transform: translateY(-1px); }

.hero-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 12px;
  background: rgba(76, 207, 176, 0.08);
  border: 1px solid rgba(76, 207, 176, 0.22);
  color: #4CCFB0;
}

/* FEATURES */
.features {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-bottom: 72px;
}
@media (max-width: 1000px) { .features { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px)  { .features { grid-template-columns: 1fr; margin-bottom: 48px; } }

.feat {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 22px 18px;
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
  border: 1px solid rgba(255, 122, 41, 0.08);
  text-decoration: none !important;
  transition: border-color 0.2s, transform 0.2s;
}
.feat:hover { border-color: rgba(255,122,41,0.35); transform: translateY(-2px); }
.feat-icon { font-size: 1.8rem; line-height: 1; }
.feat h3 { font-size: 0.95rem; font-weight: 700; margin: 0; border: none; padding: 0; color: var(--vp-c-text-1); }
.feat p { font-size: 0.85rem; color: var(--vp-c-text-2); line-height: 1.55; margin: 0; flex: 1; }
.feat-link { font-size: 0.8rem; font-weight: 600; color: #FF7A29; }

/* PRODUCTS */
.products { margin-bottom: 72px; }
.section-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 24px; border: none; padding: 0; }
.product-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 960px) { .product-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px) { .product-grid { grid-template-columns: 1fr; } }

.pcard {
  display: flex;
  gap: 14px;
  align-items: flex-start;
  padding: 18px;
  background: var(--vp-c-bg-soft);
  border: 1px solid rgba(255, 122, 41, 0.08);
  border-radius: 12px;
  text-decoration: none !important;
  transition: border-color 0.2s, transform 0.2s;
}
.pcard:hover { border-color: rgba(255,122,41,0.38); transform: translateY(-2px); }
.pcard-icon { font-size: 1.6rem; flex-shrink: 0; padding-top: 2px; }
.pcard h3 { font-size: 0.95rem; font-weight: 700; margin: 0 0 5px; border: none; padding: 0; color: var(--vp-c-text-1); }
.pcard p { font-size: 0.83rem; color: var(--vp-c-text-2); margin: 0 0 8px; line-height: 1.5; }
.pcard-price { font-size: 0.78rem; font-weight: 700; color: #FF7A29; }
.pcard-badge {
  display: inline-block; font-size: 0.7rem; font-weight: 600;
  padding: 2px 9px; border-radius: 10px;
  background: rgba(76,207,176,0.1); color: #4CCFB0; border: 1px solid rgba(76,207,176,0.28);
}
.pcard-badge.muted { background: rgba(156,163,184,0.07); color: #9ca3b8; border-color: rgba(156,163,184,0.22); }

/* QUICK START */
.quickstart {
  display: grid;
  grid-template-columns: 1fr 1.5fr;
  gap: 40px;
  align-items: center;
  padding: 48px;
  background: var(--vp-c-bg-soft);
  border: 1px solid rgba(255,122,41,0.1);
  border-radius: 16px;
  margin-bottom: 56px;
}
@media (max-width: 900px)  { .quickstart { gap: 28px; padding: 36px; } }
@media (max-width: 640px)  { .quickstart { grid-template-columns: 1fr; padding: 24px; } }
.qs-text h2 { font-size: 1.4rem; font-weight: 700; margin: 0 0 12px; border: none; padding: 0; }
.qs-text p { color: var(--vp-c-text-2); line-height: 1.6; margin: 0 0 24px; }
.qs-code div[class*="language-"] { margin: 0 !important; }

/* CTA */
.cta-row { display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; padding-top: 8px; }
</style>
