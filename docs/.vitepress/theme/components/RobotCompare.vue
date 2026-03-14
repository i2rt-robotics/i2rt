<template>
  <div class="compare-wrapper">
    <div ref="container" class="compare-canvas"></div>

    <!-- Robot selector pills -->
    <div class="compare-pills">
      <button
        :class="['pill', selected === 'yam' && 'pill-active']"
        @click="selectRobot('yam')"
      >YAM <span class="pill-sub">(ST / Pro / Ultra)</span></button>
      <button
        :class="['pill', selected === 'big_yam' && 'pill-active']"
        @click="selectRobot('big_yam')"
      >BIG YAM</button>
    </div>

    <!-- Joint sliders panel (shown when a robot is selected) -->
    <div v-if="selected && showSliders" class="slider-panel">
      <div class="slider-header">
        <span class="slider-title">{{ selected === 'yam' ? 'YAM' : 'BIG YAM' }} Joints</span>
        <button class="slider-close" @click="showSliders = false">&times;</button>
      </div>
      <div v-for="(j, i) in activeJoints" :key="i" class="slider-row">
        <label class="slider-label">J{{ i + 1 }}</label>
        <input
          type="range"
          :min="j.min"
          :max="j.max"
          :step="0.01"
          :value="j.angle"
          class="slider-input"
          @input="e => setJoint(i, +e.target.value)"
        />
        <span class="slider-val">{{ (j.angle * 180 / Math.PI).toFixed(0) }}&deg;</span>
      </div>
      <div class="slider-row">
        <label class="slider-label">Grip</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          :value="activeGripper"
          class="slider-input"
          @input="e => setGripper(+e.target.value)"
        />
        <span class="slider-val">{{ (activeGripper * 100).toFixed(0) }}%</span>
      </div>
      <button class="reset-btn" @click="resetJoints">Reset Pose</button>
    </div>

    <!-- Control button -->
    <button v-if="selected && !showSliders" class="control-toggle" @click="showSliders = true">
      Control Joints
    </button>

    <div class="compare-overlay">
      <span class="compare-hint">Click a robot to control it · Drag to orbit · Scroll to zoom</span>
    </div>

    <div v-if="loading" class="compare-loading">
      <div class="loading-ring"></div>
      <span>Loading models…</span>
    </div>
    <div v-if="error" class="compare-error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, reactive, computed } from 'vue'

const container = ref(null)
const loading = ref(true)
const error = ref(null)
const selected = ref(null)        // 'yam' | 'big_yam' | null
const showSliders = ref(false)

// Per-robot gripper state
const yamGripperOpen = ref(0.5)
const bigGripperOpen = ref(0.5)

// Joint state for sliders
const yamJoints = reactive([])
const bigYamJoints = reactive([])

const activeJoints = computed(() =>
  selected.value === 'yam' ? yamJoints : selected.value === 'big_yam' ? bigYamJoints : []
)
const activeGripper = computed(() =>
  selected.value === 'yam' ? yamGripperOpen.value : bigGripperOpen.value
)

let renderer, scene, camera, controls, animId
let yamGroup, bigYamGroup
let yamBodies = [], bigYamBodies = []
let yamBaseQs = [], bigYamBaseQs = []
let yamGripL, yamGripR, yamGripBaseL, yamGripBaseR
let bigGripL, bigGripR, bigGripBaseL, bigGripBaseR
let raycaster, mouse
let THREE

function selectRobot(name) {
  if (selected.value === name) {
    selected.value = null
    showSliders.value = false
    return
  }
  selected.value = name
  showSliders.value = true
}

function setJoint(idx, val) {
  if (selected.value === 'yam') yamJoints[idx].angle = val
  else bigYamJoints[idx].angle = val
}

function setGripper(val) {
  if (selected.value === 'yam') yamGripperOpen.value = val
  else bigGripperOpen.value = val
}

function resetJoints() {
  const joints = selected.value === 'yam' ? yamJoints : bigYamJoints
  for (const j of joints) j.angle = j.default
  if (selected.value === 'yam') yamGripperOpen.value = 0.5
  else bigGripperOpen.value = 0.5
}

onMounted(async () => {
  if (typeof window === 'undefined') return

  try {
    THREE = await import('three')
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js')
    const { STLLoader } = await import('three/addons/loaders/STLLoader.js')

    const el = container.value
    const W = el.clientWidth
    const H = el.clientHeight

    // ── Renderer ──────────────────────────────────────────────────────────
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    el.appendChild(renderer.domElement)

    // ── Scene ─────────────────────────────────────────────────────────────
    const isDark = document.documentElement.classList.contains('dark')
    const bgColor = isDark ? 0x080c18 : 0xf2f4f8
    scene = new THREE.Scene()
    scene.background = new THREE.Color(bgColor)
    scene.fog = new THREE.FogExp2(bgColor, 0.08)

    // ── Camera ────────────────────────────────────────────────────────────
    camera = new THREE.PerspectiveCamera(40, W / H, 0.005, 50)

    // ── Controls ──────────────────────────────────────────────────────────
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.minDistance = 0.4
    controls.maxDistance = 6
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.3

    // ── Raycaster for click-to-select ─────────────────────────────────────
    raycaster = new THREE.Raycaster()
    mouse = new THREE.Vector2()
    renderer.domElement.addEventListener('pointerdown', onPointerDown)

    // ── Lighting ──────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0xffffff, 2.0))

    const key = new THREE.DirectionalLight(0xfff4e0, 5)
    key.position.set(3, 6, 4)
    key.castShadow = true
    key.shadow.mapSize.set(2048, 2048)
    scene.add(key)

    const fill = new THREE.DirectionalLight(0xffffff, 2.5)
    fill.position.set(-3, 3, -2)
    scene.add(fill)

    const amberPt = new THREE.PointLight(0xFF7A29, 6, 4, 2)
    amberPt.position.set(-0.5, 1.5, 0.5)
    scene.add(amberPt)

    const tealPt = new THREE.PointLight(0x4CCFB0, 4, 4, 2)
    tealPt.position.set(1.0, 1.0, 0.8)
    scene.add(tealPt)

    // ── Floor grid ────────────────────────────────────────────────────────
    const grid = new THREE.GridHelper(4, 30, 0x1a3344, 0x0d1f2a)
    scene.add(grid)

    // ── 1m × 1m Table ─────────────────────────────────────────────────────
    const tableW = 1.0, tableD = 1.0, tableH = 0.02, legH = 0.72, legR = 0.02
    const tableSurface = legH + tableH  // 0.74m
    const tableGroup = new THREE.Group()

    // Tabletop
    const topGeo = new THREE.BoxGeometry(tableW, tableH, tableD)
    const tableMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a35, metalness: 0.3, roughness: 0.6,
      transparent: true, opacity: 0.85
    })
    const top = new THREE.Mesh(topGeo, tableMat)
    top.position.y = legH + tableH / 2
    top.castShadow = true
    top.receiveShadow = true
    tableGroup.add(top)

    // Table legs
    const legGeo = new THREE.CylinderGeometry(legR, legR, legH, 8)
    const legMat = new THREE.MeshStandardMaterial({ color: 0x3a3a45, metalness: 0.5, roughness: 0.4 })
    const offsets = [
      [-tableW/2 + 0.04, -tableD/2 + 0.04],
      [ tableW/2 - 0.04, -tableD/2 + 0.04],
      [-tableW/2 + 0.04,  tableD/2 - 0.04],
      [ tableW/2 - 0.04,  tableD/2 - 0.04],
    ]
    for (const [lx, lz] of offsets) {
      const leg = new THREE.Mesh(legGeo, legMat)
      leg.position.set(lx, legH / 2, lz)
      tableGroup.add(leg)
    }

    // "1m" dimension lines on the table edge
    const dimMat = new THREE.LineBasicMaterial({ color: 0xFF7A29, transparent: true, opacity: 0.7 })
    const frontY = tableSurface + 0.005
    const frontZ = tableD / 2 + 0.03
    const sideX = tableW / 2 + 0.03

    tableGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-tableW/2, frontY, frontZ),
        new THREE.Vector3( tableW/2, frontY, frontZ),
      ]), dimMat))

    tableGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(sideX, frontY, -tableD/2),
        new THREE.Vector3(sideX, frontY,  tableD/2),
      ]), dimMat))

    // Center table at origin
    tableGroup.position.set(0, 0, 0)
    scene.add(tableGroup)

    // ── Dimension label sprites ───────────────────────────────────────────
    const mkLabel = (text, px, py, pz) => {
      const canvas = document.createElement('canvas')
      canvas.width = 128
      canvas.height = 48
      const ctx = canvas.getContext('2d')
      ctx.font = 'bold 28px Inter, sans-serif'
      ctx.fillStyle = '#FF7A29'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(text, 64, 24)
      const tex = new THREE.CanvasTexture(canvas)
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
      const sprite = new THREE.Sprite(mat)
      sprite.position.set(px, py, pz)
      sprite.scale.set(0.2, 0.075, 1)
      return sprite
    }

    const tblY = tableSurface + 0.06
    tableGroup.add(mkLabel('1 m', 0, tblY, frontZ + 0.03))
    tableGroup.add(mkLabel('1 m', sideX + 0.03, tblY, 0))

    // ── Materials ─────────────────────────────────────────────────────────
    const mkMat = (hex, emH = 0x000000, emI = 0, met = 0.75, rou = 0.25) =>
      new THREE.MeshStandardMaterial({ color: hex, metalness: met, roughness: rou, emissive: emH, emissiveIntensity: emI })

    // YAM materials (amber/teal)
    const yamMats = [
      mkMat(0xCC5500, 0xFF7A29, 0.15),
      mkMat(0xFF8C3A, 0xFF7A29, 0.2),
      mkMat(0xE06020, 0xFF7A29, 0.15),
      mkMat(0x2EB89A, 0x4CCFB0, 0.2),
      mkMat(0xFF7A29, 0xFF7A29, 0.25),
      mkMat(0x25A88A, 0x4CCFB0, 0.2),
    ]
    const yamGripMat = mkMat(0x3a3a4a, 0xFF7A29, 0.12, 0.9, 0.15)
    const yamTipMat  = mkMat(0x5DDFC0, 0x4CCFB0, 0.5, 0.5, 0.2)

    // BIG YAM materials (silver/blue)
    const bigMats = [
      mkMat(0x8888aa, 0x6699cc, 0.1),
      mkMat(0x9999bb, 0x6699cc, 0.12),
      mkMat(0x7777aa, 0x6699cc, 0.1),
      mkMat(0x9999bb, 0x6699cc, 0.12),
      mkMat(0x8888aa, 0x6699cc, 0.1),
      mkMat(0x7777aa, 0x6699cc, 0.12),
    ]
    const bigGripMat = mkMat(0x555570, 0x6699cc, 0.1, 0.9, 0.15)
    const bigTipMat  = mkMat(0x88aacc, 0x6699cc, 0.3, 0.5, 0.2)

    // ── Load STLs ─────────────────────────────────────────────────────────
    const loader = new STLLoader()
    const base = import.meta.env.BASE_URL || '/i2rt/'

    const loadSTL = path => new Promise((res, rej) =>
      loader.load(base + path, g => { g.computeVertexNormals(); res(g) }, undefined, rej))

    const [gBase, g1, g2, g3, g4, g5, gGrip, gTL, gTR,
           bj1, bj2, bj3, bj4, bj5, bj6] = await Promise.all([
      // YAM
      loadSTL('models/yam/assets/base.stl'),
      loadSTL('models/yam/assets/link1.stl'),
      loadSTL('models/yam/assets/link2.stl'),
      loadSTL('models/yam/assets/link3.stl'),
      loadSTL('models/yam/assets/link4.stl'),
      loadSTL('models/yam/assets/link5.stl'),
      loadSTL('models/yam/assets/gripper.stl'),
      loadSTL('models/yam/assets/tip_left.stl'),
      loadSTL('models/yam/assets/tip_right.stl'),
      // BIG YAM
      loadSTL('models/big_yam/assets/joint1.STL'),
      loadSTL('models/big_yam/assets/joint2_link.STL'),
      loadSTL('models/big_yam/assets/joint3_link.STL'),
      loadSTL('models/big_yam/assets/joint4_link.STL'),
      loadSTL('models/big_yam/assets/joint5_link.STL'),
      loadSTL('models/big_yam/assets/joint6_link.STL'),
    ])

    // ── Helpers ───────────────────────────────────────────────────────────
    const mkGroup = (px, py, pz, qw = 1, qx = 0, qy = 0, qz = 0) => {
      const g = new THREE.Group()
      g.position.set(px, py, pz)
      g.quaternion.set(qx, qy, qz, qw)
      return g
    }

    const mkMesh = (geom, mat, px, py, pz, qw = 1, qx = 0, qy = 0, qz = 0) => {
      const m = new THREE.Mesh(geom, mat)
      m.position.set(px, py, pz)
      m.quaternion.set(qx, qy, qz, qw)
      m.castShadow = true
      return m
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── BUILD YAM (MuJoCo Z-up) ─────────────────────────────────────────
    // ═══════════════════════════════════════════════════════════════════════
    yamGroup = new THREE.Group()
    yamGroup.userData.robotName = 'yam'

    yamGroup.add(mkMesh(gBase, yamMats[0],
      -0.0374966, -0.0464005, 0.187501, 0.707105, 0, 0.707108, 0))

    const y1 = mkGroup(0, 0, 0.067, 0.707105, 0, 0, -0.707108)
    yamGroup.add(y1)
    y1.add(mkMesh(g1, yamMats[1],
      -0.0464, 0.0374968, 0.119501, 0.499998, 0.5, 0.5, -0.500002))

    const y2 = mkGroup(-0.0329, 0.02, 0.0455, 0.499998, 0.5, -0.500002, -0.5)
    y1.add(y2)
    y2.add(mkMesh(g2, yamMats[2],
      -0.0174977, -0.0740001, -0.07925, 0.499998, 0.5, 0.500002, 0.5))

    const y3 = mkGroup(0.264, 4.08431e-7, -0.06375,
      9.38184e-7, -0.707105, -0.707108, 9.38187e-7)
    y2.add(y3)
    y3.add(mkMesh(g3, yamMats[3],
      0.0740003, -0.281499, -0.0813, 9.38184e-7, 9.38187e-7, -0.707108, -0.707105))

    const y4 = mkGroup(0.0600003, -0.244999, -0.00205, 1.32679e-6, 0, 0, -1)
    y3.add(y4)
    y4.add(mkMesh(g4, yamMats[4],
      -0.0138003, 0.0364989, -0.0787882, 0.707105, 0.707108, 0, 0))

    const y5 = mkGroup(-0.0403003, 0.0703851, -0.0323887,
      9.38184e-7, -0.707105, -9.38187e-7, -0.707108)
    y4.add(y5)
    y5.add(mkMesh(g5, yamMats[5],
      -0.0463995, 0.0311519, 0.0265, 0.499998, -0.5, -0.5, -0.500002))

    const y6 = mkGroup(2.39858e-7, -0.0419481, 0.0404996,
      0.499998, -0.5, -0.5, -0.500002)
    y5.add(y6)

    // Gripper body
    y6.add(mkMesh(gGrip, yamGripMat, -0.014, -0.0463995, 0.0731, 1, 0, 0, 0))

    yamGripL = mkGroup(-0.0238981, 0.0450619, -0.0545599,
      0.499998, -0.5, -0.5, -0.500002)
    y6.add(yamGripL)
    yamGripL.add(mkMesh(gTL, yamTipMat,
      0.129783, 0.00999321, -0.0914614, 0.499998, 0.5, 0.500002, 0.5))

    yamGripR = mkGroup(0.0238981, -0.0450619, -0.0545599,
      0.707105, 0.707108, 0, 0)
    y6.add(yamGripR)
    yamGripR.add(mkMesh(gTR, yamTipMat,
      -0.0379932, 0.129783, 0.00133753, 0.707105, -0.707108, 0, 0))

    yamGripBaseL = yamGripL.position.clone()
    yamGripBaseR = yamGripR.position.clone()

    // MuJoCo Z-up → Three.js Y-up, then rotate 90° CCW around Y
    const qX = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2)
    const qY = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2)
    yamGroup.quaternion.copy(qY.multiply(qX))

    yamBodies = [y1, y2, y3, y4, y5, y6]
    yamBaseQs = yamBodies.map(b => b.quaternion.clone())

    const yamLimits = [
      { min: -2.618, max: 3.054 },
      { min: 0, max: 3.65 },
      { min: 0, max: 3.665 },
      { min: -1.5708, max: 1.5708 },
      { min: -1.5708, max: 1.5708 },
      { min: -2.0944, max: 2.0944 },
    ]
    const yamDefaults = [0, Math.PI/2, Math.PI/2, 0, 0, 0]
    for (let i = 0; i < 6; i++) {
      yamJoints.push({
        min: yamLimits[i].min, max: yamLimits[i].max,
        angle: yamDefaults[i], default: yamDefaults[i],
      })
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ── BUILD BIG YAM (MuJoCo Z-up, different joint axes) ───────────────
    // ═══════════════════════════════════════════════════════════════════════
    bigYamGroup = new THREE.Group()
    bigYamGroup.userData.robotName = 'big_yam'

    // joint1 base geom (fixed)
    bigYamGroup.add(mkMesh(bj1, bigMats[0], 0, 0, 0))

    // joint2_link (joint1: axis Z)
    const b2l = mkGroup(0, 0, 0.07735)
    bigYamGroup.add(b2l)
    b2l.add(mkMesh(bj2, bigMats[1], 0, 0, 0))

    // joint3_link (joint2: axis -X)
    const b3l = mkGroup(0, 0.021912, 0.053202)
    b2l.add(b3l)
    b3l.add(mkMesh(bj3, bigMats[2], 0, 0, 0))

    // joint4_link (joint3: axis X)
    const b4l = mkGroup(0, -0.3787, 0.068)
    b3l.add(b4l)
    b4l.add(mkMesh(bj4, bigMats[3], 0, 0, 0))

    // joint5_link (joint4: axis X)
    const b5l = mkGroup(0, 0.38198, 0)
    b4l.add(b5l)
    b5l.add(mkMesh(bj5, bigMats[4], 0, 0, 0))

    // joint6_link (joint5: axis Z)
    const b6l = mkGroup(0, 0.074, 0.0455)
    b5l.add(b6l)
    b6l.add(mkMesh(bj6, bigMats[5], 0, 0, 0))

    // ── BIG YAM gripper (via gripper_adapter → link_6) ──────────────────
    // gripper_adapter: pos="0 -0.0085 -0.006" quat="0 1 0 0"
    // MuJoCo quat (w,x,y,z) = (0,1,0,0) → THREE.Quaternion(x,y,z,w) = (1,0,0,0)
    const bigAdapter = mkGroup(0, -0.0085, -0.006, 0, 1, 0, 0)
    b6l.add(bigAdapter)

    // link_6 (joint6: axis -Z)
    const bigLink6 = mkGroup(2.39858e-7, -0.0419481, 0.0404996,
      0.499998, -0.5, -0.5, -0.500002)
    bigAdapter.add(bigLink6)

    // Gripper body mesh
    bigLink6.add(mkMesh(gGrip, bigGripMat, -0.014, -0.0463995, 0.0731, 1, 0, 0, 0))

    // Tip left
    bigGripL = mkGroup(-0.0238981, 0.0450619, -0.0545599,
      0.499998, -0.5, -0.5, -0.500002)
    bigLink6.add(bigGripL)
    bigGripL.add(mkMesh(gTL, bigTipMat,
      0.129783, 0.00999321, -0.0914614, 0.499998, 0.5, 0.500002, 0.5))

    // Tip right
    bigGripR = mkGroup(0.0238981, -0.0450619, -0.0545599,
      0.707105, 0.707108, 0, 0)
    bigLink6.add(bigGripR)
    bigGripR.add(mkMesh(gTR, bigTipMat,
      -0.0379932, 0.129783, 0.00133753, 0.707105, -0.707108, 0, 0))

    bigGripBaseL = bigGripL.position.clone()
    bigGripBaseR = bigGripR.position.clone()

    // MuJoCo Z-up → Three.js Y-up
    bigYamGroup.rotation.x = -Math.PI / 2

    // 6 joints: j1-j5 on arm bodies, j6 on bigLink6
    bigYamBodies = [b2l, b3l, b4l, b5l, b6l, bigLink6]
    bigYamBaseQs = bigYamBodies.map(b => b.quaternion.clone())

    const bigLimits = [
      { min: -2.618, max: 3.13 },
      { min: 0, max: 3.65 },
      { min: 0, max: 3.13 },
      { min: -1.65, max: 1.65 },
      { min: -1.5708, max: 1.5708 },
      { min: -2.0944, max: 2.0944 },
    ]
    const bigDefaults = [0, Math.PI/2, Math.PI/2, 0, 0, 0]
    for (let i = 0; i < 6; i++) {
      bigYamJoints.push({
        min: bigLimits[i].min, max: bigLimits[i].max,
        angle: bigDefaults[i], default: bigDefaults[i],
      })
    }

    // ── Position robots ON the table, side by side ────────────────────────
    yamGroup.position.set(-0.25, tableSurface, 0)
    bigYamGroup.position.set(0.25, tableSurface, 0)

    scene.add(yamGroup)
    scene.add(bigYamGroup)

    // ── Name label sprites ────────────────────────────────────────────────
    const mkRobotLabel = (text, subtext, x) => {
      const canvas = document.createElement('canvas')
      canvas.width = 512
      canvas.height = 96
      const ctx = canvas.getContext('2d')
      ctx.font = 'bold 36px Inter, sans-serif'
      ctx.fillStyle = '#ffffff'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillText(text, 256, 8)
      if (subtext) {
        ctx.font = '22px Inter, sans-serif'
        ctx.fillStyle = '#FF7A29'
        ctx.fillText(subtext, 256, 52)
      }
      const tex = new THREE.CanvasTexture(canvas)
      const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
      const sprite = new THREE.Sprite(mat)
      sprite.position.set(x, tableSurface - 0.06, 0.15)
      sprite.scale.set(0.45, 0.085, 1)
      return sprite
    }

    scene.add(mkRobotLabel('YAM', 'ST / Pro / Ultra', -0.25))
    scene.add(mkRobotLabel('BIG YAM', '', 0.25))

    // ── Camera framing ────────────────────────────────────────────────────
    controls.target.set(0, tableSurface + 0.25, 0)
    camera.position.set(1.6, tableSurface + 0.6, 1.8)
    camera.lookAt(controls.target)
    controls.update()

    loading.value = false

    // ── Joint axes ────────────────────────────────────────────────────────
    const zAxis = new THREE.Vector3(0, 0, 1)
    const xAxisNeg = new THREE.Vector3(-1, 0, 0)
    const xAxis = new THREE.Vector3(1, 0, 0)
    const zAxisNeg = new THREE.Vector3(0, 0, -1)

    // YAM: all 6 joints around local Z
    const yamAxes = [zAxis, zAxis, zAxis, zAxis, zAxis, zAxis]

    // BIG YAM: j1=Z, j2=-X, j3=X, j4=X, j5=Z, j6=-Z
    const bigAxes = [zAxis, xAxisNeg, xAxis, xAxis, zAxis, zAxisNeg]

    const dq = new THREE.Quaternion()
    let t = 0
    const STROKE = 0.0475

    // ── Apply default joint angles on first frame ─────────────────────────
    const applyJoints = (bodies, baseQs, joints, axes) => {
      for (let i = 0; i < bodies.length; i++) {
        dq.setFromAxisAngle(axes[i], joints[i].angle)
        bodies[i].quaternion.copy(baseQs[i]).multiply(dq)
      }
    }
    applyJoints(yamBodies, yamBaseQs, yamJoints, yamAxes)
    applyJoints(bigYamBodies, bigYamBaseQs, bigYamJoints, bigAxes)

    // Default gripper position (50%)
    const defaultSlide = 0.5 * STROKE
    yamGripL.position.y = yamGripBaseL.y - defaultSlide
    yamGripR.position.y = yamGripBaseR.y + defaultSlide
    bigGripL.position.y = bigGripBaseL.y - defaultSlide
    bigGripR.position.y = bigGripBaseR.y + defaultSlide

    // ── Animate ───────────────────────────────────────────────────────────
    const animate = () => {
      animId = requestAnimationFrame(animate)
      t += 0.008

      // ── YAM joints (always from slider values) ──
      for (let i = 0; i < yamBodies.length; i++) {
        dq.setFromAxisAngle(yamAxes[i], yamJoints[i].angle)
        yamBodies[i].quaternion.copy(yamBaseQs[i]).multiply(dq)
      }

      // YAM gripper
      const ySlide = yamGripperOpen.value * STROKE
      yamGripL.position.y = yamGripBaseL.y - ySlide
      yamGripR.position.y = yamGripBaseR.y + ySlide

      // ── BIG YAM joints (always from slider values) ──
      for (let i = 0; i < bigYamBodies.length; i++) {
        dq.setFromAxisAngle(bigAxes[i], bigYamJoints[i].angle)
        bigYamBodies[i].quaternion.copy(bigYamBaseQs[i]).multiply(dq)
      }

      // BIG YAM gripper
      const bSlide = bigGripperOpen.value * STROKE
      bigGripL.position.y = bigGripBaseL.y - bSlide
      bigGripR.position.y = bigGripBaseR.y + bSlide

      // ── Highlight selected robot (dim the other) ──
      if (selected.value) {
        const dimTarget = selected.value === 'yam' ? bigYamGroup : yamGroup
        const brightTarget = selected.value === 'yam' ? yamGroup : bigYamGroup
        dimTarget.traverse(c => { if (c.isMesh && c.material) { c.material.opacity = 0.4; c.material.transparent = true } })
        brightTarget.traverse(c => { if (c.isMesh && c.material) { c.material.opacity = 1.0 } })
      } else {
        yamGroup.traverse(c => { if (c.isMesh && c.material) { c.material.opacity = 1.0; c.material.transparent = false } })
        bigYamGroup.traverse(c => { if (c.isMesh && c.material) { c.material.opacity = 1.0; c.material.transparent = false } })
      }

      amberPt.intensity = 6 + Math.sin(t * 1.2) * 1.5
      tealPt.intensity  = 4 + Math.cos(t * 0.9) * 1.2

      controls.update()
      renderer.render(scene, camera)
    }
    animate()

  } catch (e) {
    loading.value = false
    error.value = 'Could not load 3D models — check browser console.'
    console.error('[RobotCompare]', e)
  }
})

function onPointerDown(e) {
  if (!raycaster || !camera) return
  const rect = renderer.domElement.getBoundingClientRect()
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
  raycaster.setFromCamera(mouse, camera)

  const yamHits = raycaster.intersectObject(yamGroup, true)
  if (yamHits.length > 0) { selectRobot('yam'); return }

  const bigHits = raycaster.intersectObject(bigYamGroup, true)
  if (bigHits.length > 0) { selectRobot('big_yam'); return }
}

function onResize() {
  if (!container.value || !renderer || !camera) return
  const w = container.value.clientWidth
  const h = container.value.clientHeight
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  renderer.setSize(w, h)
}

onMounted(() => window.addEventListener('resize', onResize))
onUnmounted(() => {
  cancelAnimationFrame(animId)
  window.removeEventListener('resize', onResize)
  renderer?.domElement?.removeEventListener('pointerdown', onPointerDown)
  renderer?.dispose()
})
</script>

<style scoped>
.compare-wrapper {
  position: relative;
  width: 100%;
  height: 600px;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255, 122, 41, 0.18);
  box-shadow: 0 0 60px rgba(255, 122, 41, 0.07), 0 0 120px rgba(76, 207, 176, 0.04);
  margin: 24px 0;
}

.compare-canvas { width: 100%; height: 100%; }

.compare-pills {
  position: absolute;
  top: 16px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  z-index: 10;
}

.pill {
  padding: 6px 16px;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.15);
  background: rgba(8, 12, 24, 0.75);
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  backdrop-filter: blur(8px);
  user-select: none;
}

.pill:hover {
  border-color: rgba(255, 122, 41, 0.5);
  color: #fff;
}

.pill-active {
  border-color: #FF7A29;
  background: rgba(255, 122, 41, 0.2);
  color: #FF7A29;
  box-shadow: 0 0 16px rgba(255, 122, 41, 0.3);
}

.pill-sub {
  font-weight: 400;
  opacity: 0.6;
  font-size: 0.7rem;
}

.slider-panel {
  position: absolute;
  top: 56px;
  right: 16px;
  width: 220px;
  background: rgba(8, 12, 24, 0.88);
  border: 1px solid rgba(255, 122, 41, 0.2);
  border-radius: 12px;
  padding: 12px 14px;
  z-index: 10;
  backdrop-filter: blur(12px);
}

.slider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.slider-title {
  font-size: 0.75rem;
  font-weight: 700;
  color: #FF7A29;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.slider-close {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.4);
  font-size: 1.2rem;
  cursor: pointer;
  padding: 0 4px;
  line-height: 1;
}
.slider-close:hover { color: #fff; }

.slider-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.slider-label {
  font-size: 0.68rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.5);
  width: 30px;
  text-align: right;
  flex-shrink: 0;
}

.slider-input {
  flex: 1;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}

.slider-input::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: #FF7A29;
  cursor: pointer;
  box-shadow: 0 0 6px rgba(255, 122, 41, 0.5);
}

.slider-val {
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.4);
  width: 32px;
  text-align: left;
  flex-shrink: 0;
  font-family: 'IBM Plex Mono', monospace;
}

.reset-btn {
  width: 100%;
  margin-top: 8px;
  padding: 5px;
  border-radius: 6px;
  border: 1px solid rgba(76, 207, 176, 0.3);
  background: rgba(76, 207, 176, 0.1);
  color: #4CCFB0;
  font-size: 0.7rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.reset-btn:hover {
  background: rgba(76, 207, 176, 0.2);
  border-color: #4CCFB0;
}

.control-toggle {
  position: absolute;
  top: 56px;
  right: 16px;
  padding: 6px 14px;
  border-radius: 8px;
  border: 1px solid rgba(255, 122, 41, 0.3);
  background: rgba(8, 12, 24, 0.8);
  color: #FF7A29;
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  z-index: 10;
  backdrop-filter: blur(8px);
  transition: all 0.2s;
}
.control-toggle:hover {
  background: rgba(255, 122, 41, 0.15);
  border-color: #FF7A29;
}

.compare-overlay {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  pointer-events: none;
  user-select: none;
}

.compare-hint {
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  color: rgba(255, 255, 255, 0.3);
}

.compare-loading {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  background: rgba(8, 12, 24, 0.92);
  color: rgba(255, 255, 255, 0.45);
  font-size: 0.85rem;
}

.loading-ring {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 2px solid rgba(255, 122, 41, 0.15);
  border-top-color: #FF7A29;
  animation: spin 0.85s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.compare-error {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(8, 12, 24, 0.92);
  color: #f87171;
  font-size: 0.9rem;
  padding: 24px;
  text-align: center;
}
</style>
