"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";

// ─── Types ────────────────────────────────────────────────────────────────────
type Stage = "pointcloud" | "pillars" | "pseudoimage";

interface PillarData {
  col: number;
  row: number;
  density: number;
  height: number;
  x: number;
  z: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────
const GRID_COLS = 24;
const GRID_ROWS = 24;
const GRID_SIZE = 0.45;
const POINT_COUNT = 2800;
const SCENE_HALF = (GRID_COLS * GRID_SIZE) / 2;

// ─── Colour helpers ───────────────────────────────────────────────────────────
function heightColor(t: number): THREE.Color {
  // cyan → lime → yellow → orange
  const c = new THREE.Color();
  c.setHSL(0.55 - t * 0.45, 1.0, 0.45 + t * 0.2);
  return c;
}
function densityColor(d: number): THREE.Color {
  const c = new THREE.Color();
  c.setHSL(0.65 - d * 0.65, 0.95, 0.35 + d * 0.3);
  return c;
}
function pseudoColor(v: number): THREE.Color {
  const c = new THREE.Color();
  if (v < 0.001) { c.set(0x050a12); return c; }
  c.setHSL(0.7 - v * 0.7, 1.0, 0.15 + v * 0.55);
  return c;
}

// ─── Generate synthetic LiDAR scene ──────────────────────────────────────────
function generateScene() {
  const pts: { x: number; y: number; z: number; intensity: number }[] = [];
  const rng = () => Math.random();

  // Ground plane
  for (let i = 0; i < 900; i++) {
    pts.push({
      x: (rng() - 0.5) * SCENE_HALF * 2.2,
      y: -0.05 + rng() * 0.08,
      z: (rng() - 0.5) * SCENE_HALF * 2.2,
      intensity: rng() * 0.3,
    });
  }
  // Cars (boxes)
  const cars = [
    { cx: -1.5, cz: 1.2, w: 1.8, d: 0.9, h: 0.7 },
    { cx: 2.8,  cz: -1.5, w: 1.6, d: 0.85, h: 0.65 },
    { cx: -3.2, cz: -2.0, w: 1.7, d: 0.88, h: 0.68 },
    { cx: 0.4,  cz: 3.5, w: 1.75, d: 0.9, h: 0.72 },
  ];
  for (const car of cars) {
    for (let i = 0; i < 280; i++) {
      const face = Math.floor(rng() * 5);
      let lx = 0, ly = 0, lz = 0;
      if (face === 0) { lx = (rng()-0.5)*car.w; ly = rng()*car.h; lz = car.d/2 * (rng()>0.5?1:-1); }
      else if (face===1) { lx = car.w/2*(rng()>0.5?1:-1); ly = rng()*car.h; lz = (rng()-0.5)*car.d; }
      else { lx = (rng()-0.5)*car.w; ly = car.h; lz = (rng()-0.5)*car.d; }
      pts.push({ x: car.cx+lx, y: ly, z: car.cz+lz, intensity: 0.6+rng()*0.4 });
    }
  }
  // Pedestrians
  for (const [px, pz] of [[-0.5,2.1],[1.9,0.6],[-2.8,1.5]]) {
    for (let i = 0; i < 60; i++) {
      pts.push({ x: px+(rng()-0.5)*0.2, y: rng()*1.7, z: pz+(rng()-0.5)*0.2, intensity: 0.5+rng()*0.3 });
    }
  }
  // Road markings / scattered noise
  for (let i = pts.length; i < POINT_COUNT; i++) {
    pts.push({ x:(rng()-0.5)*SCENE_HALF*2, y:rng()*0.15, z:(rng()-0.5)*SCENE_HALF*2, intensity:rng()*0.2 });
  }
  return pts.slice(0, POINT_COUNT);
}

// ─── Build pillar density map ─────────────────────────────────────────────────
function buildPillars(points: { x: number; y: number; z: number; intensity: number }[]): PillarData[] {
  const grid: { count: number; maxH: number }[][] = Array.from({ length: GRID_ROWS }, () =>
    Array.from({ length: GRID_COLS }, () => ({ count: 0, maxH: 0 }))
  );
  for (const p of points) {
    const col = Math.floor((p.x + SCENE_HALF) / GRID_SIZE);
    const row = Math.floor((p.z + SCENE_HALF) / GRID_SIZE);
    if (col >= 0 && col < GRID_COLS && row >= 0 && row < GRID_ROWS) {
      grid[row][col].count++;
      grid[row][col].maxH = Math.max(grid[row][col].maxH, p.y);
    }
  }
  const maxCount = Math.max(...grid.flat().map(c => c.count), 1);
  const pillars: PillarData[] = [];
  for (let r = 0; r < GRID_ROWS; r++) {
    for (let c = 0; c < GRID_COLS; c++) {
      const cell = grid[r][c];
      if (cell.count > 0) {
        pillars.push({
          col: c, row: r,
          density: cell.count / maxCount,
          height: Math.max(cell.maxH, 0.05),
          x: -SCENE_HALF + c * GRID_SIZE + GRID_SIZE / 2,
          z: -SCENE_HALF + r * GRID_SIZE + GRID_SIZE / 2,
        });
      }
    }
  }
  return pillars;
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function LidarPipeline() {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const frameRef = useRef<number>(0);
  const groupRef = useRef<THREE.Group | null>(null);
  const animStateRef = useRef({ t: 0, rotating: true });

  const [stage, setStage] = useState<Stage>("pointcloud");
  const [animating, setAnimating] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const stageRef = useRef<Stage>("pointcloud");

  const pointsDataRef = useRef<ReturnType<typeof generateScene>>([]);
  const pillarsDataRef = useRef<PillarData[]>([]);
  const [pillarCount, setPillarCount] = useState(0);

  // ── Build Three.js objects for each stage ──
  const buildPointCloud = useCallback((group: THREE.Group) => {
    const pts = pointsDataRef.current;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(pts.length * 3);
    const colors = new Float32Array(pts.length * 3);
    pts.forEach((p, i) => {
      positions[i*3] = p.x; positions[i*3+1] = p.y; positions[i*3+2] = p.z;
      const c = heightColor(Math.min(p.y / 1.5, 1));
      colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
    });
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({ size: 0.045, vertexColors: true, sizeAttenuation: true });
    group.add(new THREE.Points(geo, mat));

    // Grid floor
    const gridHelper = new THREE.GridHelper(SCENE_HALF*2, GRID_COLS, 0x112244, 0x0a1a33);
    gridHelper.position.y = -0.01;
    group.add(gridHelper);
  }, []);

  const buildPillarScene = useCallback((group: THREE.Group) => {
    const pillars = pillarsDataRef.current;
    for (const p of pillars) {
      const h = p.height + p.density * 0.8;
      const geo = new THREE.BoxGeometry(GRID_SIZE * 0.88, h, GRID_SIZE * 0.88);
      const col = densityColor(p.density);
      const mat = new THREE.MeshPhongMaterial({
        color: col,
        emissive: col.clone().multiplyScalar(0.35),
        transparent: true,
        opacity: 0.55 + p.density * 0.45,
        shininess: 80,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(p.x, h / 2, p.z);
      group.add(mesh);

      // Glow outline
      const edge = new THREE.EdgesGeometry(geo);
      const lineMat = new THREE.LineBasicMaterial({ color: col.clone().addScalar(0.3), transparent: true, opacity: 0.3 });
      const lines = new THREE.LineSegments(edge, lineMat);
      lines.position.copy(mesh.position);
      group.add(lines);
    }
    const gridHelper = new THREE.GridHelper(SCENE_HALF*2, GRID_COLS, 0x112244, 0x0a1a33);
    gridHelper.position.y = -0.01;
    group.add(gridHelper);
  }, []);

  const buildPseudoImage = useCallback((group: THREE.Group) => {
    const pillars = pillarsDataRef.current;
    const densityMap = new Float32Array(GRID_ROWS * GRID_COLS);
    for (const p of pillars) densityMap[p.row * GRID_COLS + p.col] = p.density;

    // Flat grid of planes
    for (let r = 0; r < GRID_ROWS; r++) {
      for (let c = 0; c < GRID_COLS; c++) {
        const v = densityMap[r * GRID_COLS + c];
        const col = pseudoColor(v);
        const geo = new THREE.PlaneGeometry(GRID_SIZE * 0.96, GRID_SIZE * 0.96);
        const mat = new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.set(
          -SCENE_HALF + c * GRID_SIZE + GRID_SIZE / 2,
          0.001,
          -SCENE_HALF + r * GRID_SIZE + GRID_SIZE / 2
        );
        group.add(mesh);

        if (v > 0.3) {
          // "Feature activation" glow planes
          const gGeo = new THREE.PlaneGeometry(GRID_SIZE * 0.7, GRID_SIZE * 0.7);
          const gMat = new THREE.MeshBasicMaterial({
            color: col.clone().addScalar(0.5),
            transparent: true,
            opacity: v * 0.7,
            side: THREE.DoubleSide,
          });
          const glow = new THREE.Mesh(gGeo, gMat);
          glow.rotation.x = -Math.PI / 2;
          glow.position.set(mesh.position.x, 0.01, mesh.position.z);
          group.add(glow);
        }
      }
    }

    // CNN-style scan line (animated separately below)
    const lineGeo = new THREE.PlaneGeometry(SCENE_HALF * 2, GRID_SIZE * 1.5);
    const lineMat = new THREE.MeshBasicMaterial({
      color: 0x00ffcc,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
    });
    const scanLine = new THREE.Mesh(lineGeo, lineMat);
    scanLine.rotation.x = -Math.PI / 2;
    scanLine.position.y = 0.05;
    scanLine.name = "scanline";
    group.add(scanLine);
  }, []);

  // ── Setup Three.js ──
  // Seed random data client-side only — prevents SSR/client hydration mismatch
  useEffect(() => {
    pointsDataRef.current = generateScene();
    pillarsDataRef.current = buildPillars(pointsDataRef.current);
    setPillarCount(pillarsDataRef.current.length);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!mountRef.current) return;
    const W = mountRef.current.clientWidth;
    const H = mountRef.current.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.setClearColor(0x030810, 1);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x030810, 0.06);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(52, W / H, 0.1, 100);
    camera.position.set(7, 7, 9);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Lights
    scene.add(new THREE.AmbientLight(0x112244, 1.2));
    const dir = new THREE.DirectionalLight(0x88ccff, 1.5);
    dir.position.set(5, 10, 5);
    scene.add(dir);
    const pt = new THREE.PointLight(0x00ffcc, 1.2, 30);
    pt.position.set(-3, 4, -3);
    scene.add(pt);

    const group = new THREE.Group();
    scene.add(group);
    groupRef.current = group;
    buildPointCloud(group);

    let scanDir = 1;

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      const dt = 0.005;
      if (animStateRef.current.rotating) {
        group.rotation.y += 0.003;
      }

      // Animate scan line in pseudo-image stage
      if (stageRef.current === "pseudoimage") {
        const sl = group.getObjectByName("scanline");
        if (sl) {
          sl.position.z += scanDir * 0.04;
          if (sl.position.z > SCENE_HALF || sl.position.z < -SCENE_HALF) scanDir *= -1;
        }
      }

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      if (!mountRef.current) return;
      const W2 = mountRef.current.clientWidth;
      const H2 = mountRef.current.clientHeight;
      camera.aspect = W2 / H2;
      camera.updateProjectionMatrix();
      renderer.setSize(W2, H2);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [buildPointCloud]);

  // ── Stage transition ──
  const transitionTo = useCallback((next: Stage) => {
    if (!sceneRef.current || !groupRef.current || animating) return;
    setAnimating(true);
    setStage(next);
    stageRef.current = next;

    const group = groupRef.current;
    // Fade out existing
    group.traverse(obj => {
      if ((obj as THREE.Mesh).isMesh || (obj as THREE.Points).isPoints) {
        const mat = (obj as THREE.Mesh).material as THREE.Material;
        if (mat) (mat as any).transparent = true;
      }
    });

    let tick = 0;
    const fadeOut = setInterval(() => {
      tick++;
      group.traverse(obj => {
        const mat = ((obj as THREE.Mesh).material) as any;
        if (mat?.opacity !== undefined) mat.opacity = Math.max(0, mat.opacity - 0.06);
      });
      if (tick > 16) {
        clearInterval(fadeOut);
        // Clear group
        while (group.children.length > 0) group.remove(group.children[0]);
        // Build new stage
        if (next === "pointcloud") buildPointCloud(group);
        else if (next === "pillars") buildPillarScene(group);
        else buildPseudoImage(group);
        // Fade in
        let tick2 = 0;
        const fadeIn = setInterval(() => {
          tick2++;
          group.traverse(obj => {
            const mat = ((obj as THREE.Mesh).material) as any;
            if (mat?.opacity !== undefined) mat.opacity = Math.min((mat as any)._targetOpacity ?? 1, mat.opacity + 0.05);
          });
          if (tick2 > 20) { clearInterval(fadeIn); setAnimating(false); }
        }, 30);
      }
    }, 30);
  }, [animating, buildPointCloud, buildPillarScene, buildPseudoImage]);

  const toggleRotate = () => {
    animStateRef.current.rotating = !animStateRef.current.rotating;
    setAutoRotate(r => !r);
  };

  const STAGES: { id: Stage; label: string; icon: string; desc: string }[] = [
    { id: "pointcloud", label: "Point Cloud", icon: "⬡", desc: "Raw LiDAR PC2 — sparse 3D points with XYZ + intensity" },
    { id: "pillars",    label: "Pillarization", icon: "▬", desc: "PointPillars grid — vertical feature columns per cell" },
    { id: "pseudoimage",label: "Pseudo-Image", icon: "▦", desc: "2D CNN feature map — density projected onto BEV canvas" },
  ];

  return (
    <div style={{
      width: "100%", height: "100vh", background: "#030810",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      display: "flex", flexDirection: "column", overflow: "hidden",
      color: "#a8d8ff",
    }}>
      {/* Header */}
      <div style={{
        padding: "10px 20px", borderBottom: "1px solid #0d2a4a",
        background: "rgba(3,8,16,0.95)", backdropFilter: "blur(8px)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: "#00ffcc", boxShadow: "0 0 8px #00ffcc", animation: "pulse 1.5s infinite"
          }} />
          <span style={{ fontSize: 11, color: "#00ffcc", letterSpacing: 2 }}>LIDAR PERCEPTION PIPELINE</span>
          <span style={{ fontSize: 10, color: "#2a5070", letterSpacing: 1 }}>| PointPillars BEV Visualizer</span>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <span style={{ fontSize: 10, color: "#2a5070" }}>{POINT_COUNT} pts / {pillarCount > 0 ? pillarCount : "..."} pillars</span>
          <button onClick={toggleRotate} style={{
            background: autoRotate ? "rgba(0,255,204,0.12)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${autoRotate ? "#00ffcc44" : "#ffffff22"}`,
            color: autoRotate ? "#00ffcc" : "#446688",
            padding: "3px 10px", borderRadius: 3, cursor: "pointer",
            fontSize: 10, letterSpacing: 1,
          }}>⟳ ROTATE</button>
        </div>
      </div>

      {/* Stage selector */}
      <div style={{
        display: "flex", gap: 0, flexShrink: 0,
        borderBottom: "1px solid #0d2a4a",
        background: "rgba(3,8,16,0.9)",
      }}>
        {STAGES.map((s, i) => (
          <button
            key={s.id}
            onClick={() => transitionTo(s.id)}
            disabled={animating}
            style={{
              flex: 1, padding: "10px 12px",
              background: stage === s.id ? "rgba(0,255,204,0.07)" : "transparent",
              border: "none",
              borderRight: i < 2 ? "1px solid #0d2a4a" : "none",
              borderBottom: stage === s.id ? "2px solid #00ffcc" : "2px solid transparent",
              color: stage === s.id ? "#00ffcc" : "#2a5880",
              cursor: animating ? "not-allowed" : "pointer",
              transition: "all 0.2s",
              textAlign: "left",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 14 }}>{s.icon}</span>
              <div>
                <div style={{ fontSize: 10, letterSpacing: 1, fontWeight: 600 }}>
                  {`0${i+1}`.slice(-2)} · {s.label.toUpperCase()}
                </div>
                <div style={{ fontSize: 9, color: stage === s.id ? "#3a7a8a" : "#1a3a4a", marginTop: 1 }}>{s.desc}</div>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* 3D Viewport */}
      <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
        <div ref={mountRef} style={{ width: "100%", height: "100%" }} />

        {/* Overlay HUD */}
        <div style={{
          position: "absolute", top: 12, right: 12, pointerEvents: "none",
          display: "flex", flexDirection: "column", gap: 6,
        }}>
          <HudBox label="FRAME" value="0042" />
          <HudBox label="SENSOR" value="Hesai Pandar64" />
          <HudBox label="VIEW" value="BEV + 3D" />
          <HudBox label="GRID" value={`${GRID_COLS}×${GRID_ROWS} @ ${GRID_SIZE}m`} />
          {stage === "pillars" && <HudBox label="PILLARS" value={`${pillarCount} active`} accent="#7b5ea7" />}
          {stage === "pseudoimage" && <HudBox label="CNN INPUT" value={`${GRID_COLS}×${GRID_ROWS}×C`} accent="#d4822a" />}
        </div>

        {/* Stage legend */}
        <div style={{
          position: "absolute", bottom: 12, left: 12, pointerEvents: "none",
        }}>
          {stage === "pointcloud" && <Legend items={[
            { color: "#00b3ff", label: "Ground plane" },
            { color: "#00ff88", label: "Vehicles" },
            { color: "#ffcc00", label: "Pedestrians" },
          ]} title="Point Height Coloring" />}
          {stage === "pillars" && <Legend items={[
            { color: "#6655ff", label: "Low density" },
            { color: "#00aaff", label: "Medium density" },
            { color: "#ff8800", label: "High density (object)" },
          ]} title="Pillar Density Encoding" />}
          {stage === "pseudoimage" && <Legend items={[
            { color: "#050a12", label: "Empty voxel" },
            { color: "#441a6a", label: "Low activation" },
            { color: "#ff5500", label: "High activation" },
          ]} title="BEV Feature Map" />}
        </div>

        {animating && (
          <div style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(3,8,16,0.5)", pointerEvents: "none",
          }}>
            <div style={{
              border: "1px solid #00ffcc44", padding: "8px 20px",
              color: "#00ffcc", fontSize: 11, letterSpacing: 3,
              background: "rgba(0,20,30,0.8)",
            }}>TRANSFORMING...</div>
          </div>
        )}
      </div>

      {/* Pipeline flow footer */}
      <div style={{
        padding: "8px 20px", borderTop: "1px solid #0d2a4a",
        background: "rgba(3,8,16,0.95)", display: "flex",
        alignItems: "center", justifyContent: "center", gap: 0, flexShrink: 0,
      }}>
        {STAGES.map((s, i) => (
          <div key={s.id} style={{ display: "flex", alignItems: "center" }}>
            <div style={{
              padding: "3px 14px", borderRadius: 2,
              background: stage === s.id ? "rgba(0,255,204,0.1)" : "transparent",
              border: `1px solid ${stage === s.id ? "#00ffcc44" : "#0d2a4a"}`,
              color: stage === s.id ? "#00ffcc" : "#1a4060",
              fontSize: 9, letterSpacing: 1.5, cursor: "pointer",
            }} onClick={() => transitionTo(s.id)}>{s.label.toUpperCase()}</div>
            {i < 2 && <span style={{ color: "#0d3050", fontSize: 14, margin: "0 4px" }}>→</span>}
          </div>
        ))}
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
      `}</style>
    </div>
  );
}

function HudBox({ label, value, accent = "#00ffcc" }: { label: string; value: string; accent?: string }) {
  return (
    <div style={{
      background: "rgba(3,8,20,0.82)", border: `1px solid ${accent}22`,
      padding: "4px 10px", minWidth: 140,
    }}>
      <div style={{ fontSize: 8, color: "#2a5070", letterSpacing: 1.5 }}>{label}</div>
      <div style={{ fontSize: 10, color: accent, letterSpacing: 0.5 }}>{value}</div>
    </div>
  );
}

function Legend({ items, title }: { items: { color: string; label: string }[]; title: string }) {
  return (
    <div style={{
      background: "rgba(3,8,20,0.82)", border: "1px solid #0d2a4a",
      padding: "8px 12px", minWidth: 190,
    }}>
      <div style={{ fontSize: 8, color: "#2a5070", letterSpacing: 1.5, marginBottom: 6 }}>{title.toUpperCase()}</div>
      {items.map(it => (
        <div key={it.label} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
          <div style={{ width: 10, height: 10, background: it.color, flexShrink: 0 }} />
          <span style={{ fontSize: 9, color: "#4a8aa0" }}>{it.label}</span>
        </div>
      ))}
    </div>
  );
}