<template>
  <div class="robot-viewer-wrapper">
    <div ref="container" class="robot-canvas"></div>
    <div class="viewer-overlay">
      <span class="viewer-hint">Drag to rotate · Scroll to zoom</span>
    </div>
    <div v-if="loading" class="viewer-loading">
      <div class="loading-ring"></div>
      <span>Loading model…</span>
    </div>
    <div v-if="error" class="viewer-error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const container = ref(null)
const loading = ref(true)
const error = ref(null)

let renderer, scene, camera, controls, animId

// ── MuJoCo quat helper: (w,x,y,z) → THREE.Quaternion(x,y,z,w) ──────────────
const mq = (w, x, y, z) => {
  const { Quaternion } = window._THREE
  return new Quaternion(x, y, z, w)
}

onMounted(async () => {
  if (typeof window === 'undefined') return

  try {
    const THREE = await import('three')
    window._THREE = THREE
    const { OrbitControls } = await import('three/addons/controls/OrbitControls.js')
    const { STLLoader } = await import('three/addons/loaders/STLLoader.js')

    const el = container.value
    const W = el.clientWidth
    const H = el.clientHeight

    // ── Renderer ──────────────────────────────────────────────────────────────
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.1
    el.appendChild(renderer.domElement)

    // ── Scene ─────────────────────────────────────────────────────────────────
    const isDark = document.documentElement.classList.contains('dark')
    const bgColor = isDark ? 0x080c18 : 0xf2f4f8
    const fogColor = isDark ? 0x080c18 : 0xf2f4f8
    scene = new THREE.Scene()
    scene.background = new THREE.Color(bgColor)
    scene.fog = new THREE.FogExp2(fogColor, 0.22)

    // ── Camera ────────────────────────────────────────────────────────────────
    camera = new THREE.PerspectiveCamera(40, W / H, 0.005, 30)
    camera.position.set(0.75, 0.55, 0.85)

    // ── Controls ──────────────────────────────────────────────────────────────
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.07
    controls.minDistance = 0.2
    controls.maxDistance = 4
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.5
    controls.enablePan = false

    // ── Lighting ──────────────────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0xffffff, 2.5))

    const key = new THREE.DirectionalLight(0xfff4e0, 6)
    key.position.set(2, 5, 3)
    key.castShadow = true
    key.shadow.mapSize.set(2048, 2048)
    scene.add(key)

    const fill = new THREE.DirectionalLight(0xffffff, 3)
    fill.position.set(-3, 2, -2)
    scene.add(fill)

    const amberPt = new THREE.PointLight(0xFF7A29, 8, 2.5, 2)
    amberPt.position.set(-0.5, 0.6, 0.3)
    scene.add(amberPt)

    const tealPt = new THREE.PointLight(0x4CCFB0, 6, 2.5, 2)
    tealPt.position.set(0.7, 0.15, 0.6)
    scene.add(tealPt)

    // ── Floor ─────────────────────────────────────────────────────────────────
    const grid = new THREE.GridHelper(3, 22, 0x1a3344, 0x0d1f2a)
    grid.position.y = 0
    scene.add(grid)

    const glowGeo = new THREE.CircleGeometry(0.2, 48)
    const glowMat = new THREE.MeshBasicMaterial({ color: 0xFF7A29, transparent: true, opacity: 0.1, depthWrite: false })
    const glow = new THREE.Mesh(glowGeo, glowMat)
    glow.rotation.x = -Math.PI / 2
    glow.position.y = 0.001
    scene.add(glow)

    // ── Materials ─────────────────────────────────────────────────────────────
    const mkMat = (hex, emissiveHex = 0x000000, emissiveI = 0, metal = 0.75, rough = 0.25) =>
      new THREE.MeshStandardMaterial({ color: hex, metalness: metal, roughness: rough, emissive: emissiveHex, emissiveIntensity: emissiveI })

    // Bright brand-color scheme — amber body, teal accents
    const matBase   = mkMat(0xCC5500, 0xFF7A29, 0.15)          // deep amber
    const matLink1  = mkMat(0xFF8C3A, 0xFF7A29, 0.2)           // bright amber
    const matLink2  = mkMat(0xE06020, 0xFF7A29, 0.15)          // mid amber
    const matLink3  = mkMat(0x2EB89A, 0x4CCFB0, 0.2)           // teal
    const matLink4  = mkMat(0xFF7A29, 0xFF7A29, 0.25)          // brand orange
    const matLink5  = mkMat(0x25A88A, 0x4CCFB0, 0.2)           // teal
    const matGrip   = mkMat(0x3a3a4a, 0xFF7A29, 0.12, 0.9, 0.15) // dark metal housing
    const matTip    = mkMat(0x5DDFC0, 0x4CCFB0, 0.5, 0.5, 0.2)   // bright teal fingers

    // ── Load all STLs ─────────────────────────────────────────────────────────
    const loader = new STLLoader()
    const assetBase = (import.meta.env.BASE_URL || '/i2rt/') + 'models/yam/assets/'

    const loadSTL = name => new Promise((res, rej) => loader.load(assetBase + name, res, undefined, rej))

    const [gBase, g1, g2, g3, g4, g5, gGrip, gTL, gTR] = await Promise.all([
      loadSTL('base.stl'),
      loadSTL('link1.stl'),
      loadSTL('link2.stl'),
      loadSTL('link3.stl'),
      loadSTL('link4.stl'),
      loadSTL('link5.stl'),
      loadSTL('gripper.stl'),
      loadSTL('tip_left.stl'),
      loadSTL('tip_right.stl'),
    ])

    for (const g of [gBase, g1, g2, g3, g4, g5, gGrip, gTL, gTR]) g.computeVertexNormals()

    // ── Scene graph helpers ───────────────────────────────────────────────────
    // All transforms straight from yam.xml / linear_4310.xml
    // MuJoCo quat format is (w x y z); THREE.Quaternion is (x y z w)

    const mkGroup = (px, py, pz, qw, qx, qy, qz) => {
      const g = new THREE.Group()
      g.position.set(px, py, pz)
      g.quaternion.set(qx, qy, qz, qw)
      return g
    }

    const mkMesh = (geom, mat, px, py, pz, qw, qx, qy, qz) => {
      const m = new THREE.Mesh(geom, mat)
      m.position.set(px, py, pz)
      m.quaternion.set(qx, qy, qz, qw)
      m.castShadow = true
      return m
    }

    // ── Assemble robot in MuJoCo (Z-up) space ────────────────────────────────
    const robotMJ = new THREE.Group()

    // BASE GEOM — fixed in worldbody
    // pos="-0.0374966 -0.0464005 0.187501" quat="0.707105 0 0.707108 0"
    robotMJ.add(mkMesh(gBase, matBase,
      -0.0374966, -0.0464005, 0.187501,
      0.707105, 0, 0.707108, 0))

    // LINK 1 body — pos="0 0 0.067" quat="0.707105 0 0 -0.707108"
    // joint1 axis="0 0 1"  range="-2.618 3.13"
    const b1 = mkGroup(0, 0, 0.067,  0.707105, 0, 0, -0.707108)
    robotMJ.add(b1)
    // link1 geom — pos="-0.0464 0.0374968 0.119501" quat="0.499998 0.5 0.5 -0.500002"
    b1.add(mkMesh(g1, matLink1,
      -0.0464, 0.0374968, 0.119501,
      0.499998, 0.5, 0.5, -0.500002))

    // LINK 2 body (child of b1) — pos="-0.0329 0.02 0.0455" quat="0.499998 0.5 -0.500002 -0.5"
    // joint2 axis="0 0 1"  range="0 3.65"
    const b2 = mkGroup(-0.0329, 0.02, 0.0455,  0.499998, 0.5, -0.500002, -0.5)
    b1.add(b2)
    // link2 geom — pos="-0.0174977 -0.0740001 -0.07925" quat="0.499998 0.5 0.500002 0.5"
    b2.add(mkMesh(g2, matLink2,
      -0.0174977, -0.0740001, -0.07925,
      0.499998, 0.5, 0.500002, 0.5))

    // LINK 3 body (child of b2) — pos="0.264 4.08e-07 -0.06375" quat="9.38e-07 -0.707105 -0.707108 9.38e-07"
    // joint3 axis="0 0 1"  range="0 3.13"
    const b3 = mkGroup(0.264, 4.08431e-7, -0.06375,
      9.38184e-7, -0.707105, -0.707108, 9.38187e-7)
    b2.add(b3)
    // link3 geom — pos="0.0740003 -0.281499 -0.0813" quat="9.38e-07 9.38e-07 -0.707108 -0.707105"
    b3.add(mkMesh(g3, matLink3,
      0.0740003, -0.281499, -0.0813,
      9.38184e-7, 9.38187e-7, -0.707108, -0.707105))

    // LINK 4 body (child of b3) — pos="0.0600003 -0.244999 -0.00205" quat="1.32679e-06 0 0 -1"
    // joint4 axis="0 0 1"  range="-1.5708 1.5708"
    const b4 = mkGroup(0.0600003, -0.244999, -0.00205,  1.32679e-6, 0, 0, -1)
    b3.add(b4)
    // link4 geom — pos="-0.0138003 0.0364989 -0.0787882" quat="0.707105 0.707108 0 0"
    b4.add(mkMesh(g4, matLink4,
      -0.0138003, 0.0364989, -0.0787882,
      0.707105, 0.707108, 0, 0))

    // LINK 5 body (child of b4) — pos="-0.0403003 0.0703851 -0.0323887" quat="9.38e-07 -0.707105 -9.38e-07 -0.707108"
    // joint5 axis="0 0 1"  range="-1.5708 1.5708"
    const b5 = mkGroup(-0.0403003, 0.0703851, -0.0323887,
      9.38184e-7, -0.707105, -9.38187e-7, -0.707108)
    b4.add(b5)
    // link5 geom — pos="-0.0463995 0.0311519 0.0265" quat="0.499998 -0.5 -0.5 -0.500002"
    b5.add(mkMesh(g5, matLink5,
      -0.0463995, 0.0311519, 0.0265,
      0.499998, -0.5, -0.5, -0.500002))

    // LINK 6 / WRIST body (child of b5) — from both yam.xml & linear_4310.xml
    // pos="2.39858e-07 -0.0419481 0.0404996" quat="0.499998 -0.5 -0.5 -0.500002"
    // joint6 axis="0 0 1"  range="-2.0944 2.0944"
    const b6 = mkGroup(2.39858e-7, -0.0419481, 0.0404996,
      0.499998, -0.5, -0.5, -0.500002)
    b5.add(b6)

    // GRIPPER BODY geom — pos="-0.014 -0.0463995 0.0731" quat="1 0 0 0"
    b6.add(mkMesh(gGrip, matGrip,
      -0.014, -0.0463995, 0.0731,
      1, 0, 0, 0))

    // TIP LEFT body (child of b6) — pos="-0.0238981 0.0450619 -0.0545599" quat="0.499998 -0.5 -0.5 -0.500002"
    const bTL = mkGroup(-0.0238981, 0.0450619, -0.0545599,
      0.499998, -0.5, -0.5, -0.500002)
    b6.add(bTL)
    // tip_left geom — pos="0.129783 0.00999321 -0.0914614" quat="0.499998 0.5 0.500002 0.5"
    bTL.add(mkMesh(gTL, matTip,
      0.129783, 0.00999321, -0.0914614,
      0.499998, 0.5, 0.500002, 0.5))

    // TIP RIGHT body (child of b6) — pos="0.0238981 -0.0450619 -0.0545599" quat="0.707105 0.707108 0 0"
    const bTR = mkGroup(0.0238981, -0.0450619, -0.0545599,
      0.707105, 0.707108, 0, 0)
    b6.add(bTR)
    // tip_right geom — pos="-0.0379932 0.129783 0.00133753" quat="0.707105 -0.707108 0 0"
    bTR.add(mkMesh(gTR, matTip,
      -0.0379932, 0.129783, 0.00133753,
      0.707105, -0.707108, 0, 0))

    // ── Convert MuJoCo (Z-up) → Three.js (Y-up) ──────────────────────────────
    // Rotate entire robot -90° around X: MuJoCo (x,y,z) → Three.js (x,z,-y)
    robotMJ.rotation.x = -Math.PI / 2

    // Scale to a good display size first, then place base plate at origin
    scene.add(robotMJ)
    const box0 = new THREE.Box3().setFromObject(robotMJ)
    const size0 = box0.getSize(new THREE.Vector3())
    const maxDim = Math.max(size0.x, size0.y, size0.z)
    const scale = 0.85 / maxDim
    robotMJ.scale.setScalar(scale)

    // After scaling: sit the bottom of the arm on the floor (y = 0),
    // and center horizontally (X/Z) so the base plate is at origin.
    const box1 = new THREE.Box3().setFromObject(robotMJ)
    const cx = (box1.min.x + box1.max.x) / 2
    const cz = (box1.min.z + box1.max.z) / 2
    robotMJ.position.set(-cx, -box1.min.y, -cz)

    // Camera orbit target = mid-height of the arm above the base plate
    const box2 = new THREE.Box3().setFromObject(robotMJ)
    const armH = (box2.min.y + box2.max.y) / 2
    const target = new THREE.Vector3(0, armH * 1.2, 0)
    controls.target.copy(target)
    camera.position.set(1.1, armH * 2.8 + 0.75, 2.2)
    camera.lookAt(target)
    controls.update()

    // ── Animation setup ───────────────────────────────────────────────────────
    // All angles in radians. Each revolute joint oscillates around a center.
    // body.quat = baseQ * R(z, center + amp * sin(t * freq + phase))
    const D = Math.PI / 180
    const zAxis = new THREE.Vector3(0, 0, 1)

    const joints = [
      // j1: -30° to +30°, center = 0°
      { body: b1, center:   0*D, amp: 30*D, freq: 0.30, phase: 0.0 },
      // j2: 80° to 100°, center = 90°
      { body: b2, center:  90*D, amp: 10*D, freq: 0.22, phase: 1.2 },
      // j3: 80° to 100°, center = 90°
      { body: b3, center:  90*D, amp: 10*D, freq: 0.28, phase: 2.5 },
      // j4: -20° to +20°, center = 0°
      { body: b4, center:   0*D, amp: 20*D, freq: 0.35, phase: 0.8 },
      // j5: -20° to +20°, center = 0°
      { body: b5, center:   0*D, amp: 20*D, freq: 0.40, phase: 1.9 },
      // j6: -20° to +20°, center = 0°
      { body: b6, center:   0*D, amp: 20*D, freq: 0.45, phase: 3.1 },
    ]
    for (const j of joints) j.baseQ = j.body.quaternion.clone()

    // Gripper slide joints (tip_left / tip_right), range 0 → 0.0475 m
    // Computed: tip_left slides in (0,-1,0) in b6 frame; tip_right in (0,+1,0)
    const basePosL = bTL.position.clone()  // (-0.0239, +0.0451, -0.0546)
    const basePosR = bTR.position.clone()  // (+0.0239, -0.0451, -0.0546)
    const STROKE = 0.0475                   // max slide in metres

    loading.value = false

    // ── Animate ───────────────────────────────────────────────────────────────
    const dq = new THREE.Quaternion()
    let t = 0

    const animate = () => {
      animId = requestAnimationFrame(animate)
      t += 0.008   // slow tick

      // Revolute joints
      for (const j of joints) {
        const θ = j.center + j.amp * Math.sin(t * j.freq + j.phase)
        dq.setFromAxisAngle(zAxis, θ)
        j.body.quaternion.copy(j.baseQ).multiply(dq)
      }

      // Gripper: normalized 0→1, slow open-close cycle
      const norm = 0.5 + 0.5 * Math.sin(t * 0.25)   // 0 = open, 1 = closed
      const slide = norm * STROKE
      bTL.position.y = basePosL.y - slide   // tip_left moves in -y of b6
      bTR.position.y = basePosR.y + slide   // tip_right moves in +y of b6

      // Soft light breathing
      amberPt.intensity = 8 + Math.sin(t * 1.2) * 2.5
      tealPt.intensity  = 6 + Math.cos(t * 0.9) * 2

      controls.update()
      renderer.render(scene, camera)
    }
    animate()

  } catch (e) {
    loading.value = false
    error.value = 'Could not load 3D model — check browser console for details.'
    console.error('[RobotViewer]', e)
  }
})

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
  renderer?.dispose()
  delete window._THREE
})
</script>

<style scoped>
.robot-viewer-wrapper {
  position: relative;
  width: 100%;
  height: 540px;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255, 122, 41, 0.18);
  box-shadow: 0 0 60px rgba(255, 122, 41, 0.07), 0 0 120px rgba(76, 207, 176, 0.04);
}

.robot-canvas { width: 100%; height: 100%; }

.viewer-overlay {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 3px;
  pointer-events: none;
  user-select: none;
}

.viewer-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #FF7A29;
  text-shadow: 0 0 12px rgba(255, 122, 41, 0.7);
}

.viewer-hint {
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  color: rgba(255, 255, 255, 0.3);
}

.viewer-loading {
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
  letter-spacing: 0.05em;
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

.viewer-error {
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
