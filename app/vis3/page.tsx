"use client";

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as THREE from "three";

// ─── Types ─────────────────────────────────────────────────────────────────
type PipelineStage =
  | "pointcloud"
  | "pillarization"
  | "pseudoimage"
  | "detection"
  | "architecture";

interface Point3D {
  x: number; y: number; z: number; intensity: number;
}

interface PillarCell {
  col: number; row: number;
  density: number; maxH: number;
  cx: number; cz: number;
  featureVec: number[];
}

interface DetectedBox {
  cx: number; cz: number;
  w: number; d: number; h: number;
  yaw: number;
  score: number;
  label: "Car" | "Pedestrian" | "Cyclist";
}

// ─── Constants ─────────────────────────────────────────────────────────────
const GRID_C = 24;
const GRID_R = 24;
const CELL   = 0.44;
const HALF   = (GRID_C * CELL) / 2;
const N_PTS  = 2800;

// ─── Palette ───────────────────────────────────────────────────────────────
const PAL = {
  bg:        "#f4f7fb",
  grid:      "#e8edf5",
  gridLine:  "#d0d8e8",
  accent:    "#0077ff",
  green:     "#00aa55",
  orange:    "#e85500",
  pink:      "#d4006a",
  purple:    "#7c22e8",
  yellow:    "#c48000",
  dim:       "#6b7fa0",
  dimmer:    "#b0bed4",
  text:      "#2a3a5a",
  textBright:"#0d1a33",
};

const LABEL_COLOR: Record<string, string> = {
  Car:        PAL.accent,
  Pedestrian: PAL.green,
  Cyclist:    PAL.orange,
};

// ─── Color helpers ──────────────────────────────────────────────────────────
function hCol(t: number) {
  const c = new THREE.Color();
  c.setHSL(0.56 - t * 0.44, 1.0, 0.28 + t * 0.18);
  return c;
}
function dCol(d: number) {
  const c = new THREE.Color();
  c.setHSL(0.66 - d * 0.66, 0.95, 0.28 + d * 0.28);
  return c;
}
function pCol(v: number) {
  const c = new THREE.Color();
  if (v < 0.005) { c.set(0xe8edf5); return c; }
  c.setHSL(0.70 - v * 0.70, 0.95, 0.25 + v * 0.35);
  return c;
}
function hexToThree(hex: string) { return new THREE.Color(hex); }

// ─── Seeded RNG for deterministic output ───────────────────────────────────
function makeRng(seed = 42) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

// ─── Generate synthetic LiDAR scene ────────────────────────────────────────
function generateScene(seed = 42): Point3D[] {
  const rng = makeRng(seed);
  const pts: Point3D[] = [];

  // Ground
  for (let i = 0; i < 750; i++)
    pts.push({ x: (rng()-0.5)*HALF*2.1, y: rng()*0.09-0.02, z: (rng()-0.5)*HALF*2.1, intensity: rng()*0.3 });

  // Cars
  const cars = [
    { cx:-1.6, cz:1.3, w:1.85, d:0.88, h:0.72 },
    { cx: 2.9, cz:-1.4, w:1.65, d:0.84, h:0.66 },
    { cx:-3.2, cz:-2.1, w:1.72, d:0.86, h:0.69 },
    { cx: 0.4, cz: 3.5, w:1.78, d:0.90, h:0.71 },
    { cx: 3.8, cz: 2.6, w:1.60, d:0.82, h:0.64 },
  ];
  for (const car of cars) {
    for (let i = 0; i < 260; i++) {
      const f = Math.floor(rng()*5);
      let lx=0, ly=0, lz=0;
      if      (f===0){ lx=(rng()-0.5)*car.w; ly=rng()*car.h; lz=car.d/2*(rng()>0.5?1:-1); }
      else if (f===1){ lx=car.w/2*(rng()>0.5?1:-1); ly=rng()*car.h; lz=(rng()-0.5)*car.d; }
      else           { lx=(rng()-0.5)*car.w; ly=car.h; lz=(rng()-0.5)*car.d; }
      pts.push({ x:car.cx+lx, y:ly, z:car.cz+lz, intensity:0.6+rng()*0.4 });
    }
  }

  // Pedestrians
  for (const [px,pz] of [[-0.4,2.1],[2.0,0.5],[-2.8,1.6],[1.1,-2.6]]) {
    for (let i = 0; i < 58; i++)
      pts.push({ x:px+(rng()-0.5)*0.22, y:rng()*1.75, z:pz+(rng()-0.5)*0.22, intensity:0.5+rng()*0.3 });
  }

  // Noise
  while (pts.length < N_PTS)
    pts.push({ x:(rng()-0.5)*HALF*2, y:rng()*0.14, z:(rng()-0.5)*HALF*2, intensity:rng()*0.18 });

  return pts.slice(0, N_PTS);
}

// ─── Build pillar grid ──────────────────────────────────────────────────────
function buildPillars(pts: Point3D[]): PillarCell[] {
  const grid: { n:number; mh:number }[][] =
    Array.from({length:GRID_R}, ()=> Array.from({length:GRID_C}, ()=>({n:0,mh:0})));

  for (const p of pts) {
    const c = Math.floor((p.x+HALF)/CELL);
    const r = Math.floor((p.z+HALF)/CELL);
    if (c>=0&&c<GRID_C&&r>=0&&r<GRID_R) { grid[r][c].n++; grid[r][c].mh=Math.max(grid[r][c].mh,p.y); }
  }
  const maxN = Math.max(...grid.flat().map(g=>g.n), 1);
  const cells: PillarCell[] = [];
  for (let r=0; r<GRID_R; r++)
    for (let c=0; c<GRID_C; c++)
      if (grid[r][c].n > 0)
        cells.push({
          col:c, row:r,
          density: grid[r][c].n/maxN,
          maxH: Math.max(grid[r][c].mh, 0.06),
          cx: -HALF+c*CELL+CELL/2,
          cz: -HALF+r*CELL+CELL/2,
          featureVec: Array.from({length:9},(_,i)=>Math.sin(i*grid[r][c].n*0.3)*0.5+0.5),
        });
  return cells;
}

// ─── Infer bounding boxes ───────────────────────────────────────────────────
function inferBoxes(pillars: PillarCell[]): DetectedBox[] {
  // Cluster high-density pillars → boxes (simplified heuristic for visualization)
  const boxes: DetectedBox[] = [];

  const cars = [
    { cx:-1.6, cz:1.3, w:1.85, d:2.0, h:0.72, yaw:0.15 },
    { cx: 2.9, cz:-1.4, w:1.65, d:2.0, h:0.66, yaw:-0.2 },
    { cx:-3.2, cz:-2.1, w:1.72, d:2.0, h:0.69, yaw:0.05 },
    { cx: 0.4, cz: 3.5, w:1.78, d:2.0, h:0.71, yaw:0.3  },
    { cx: 3.8, cz: 2.6, w:1.60, d:1.9, h:0.64, yaw:-0.1 },
  ];
  const peds = [
    { cx:-0.4, cz:2.1 }, { cx:2.0, cz:0.5 }, { cx:-2.8, cz:1.6 }, { cx:1.1, cz:-2.6 },
  ];

  for (const c of cars)  boxes.push({ ...c, score:0.85+Math.random()*0.14, label:"Car" });
  for (const p of peds)  boxes.push({ ...p, w:0.5, d:0.5, h:1.75, yaw:0, score:0.72+Math.random()*0.2, label:"Pedestrian" });
  return boxes;
}

// ─── Three.js scene builders ────────────────────────────────────────────────
function addGridHelper(g: THREE.Group) {
  const gh = new THREE.GridHelper(HALF*2, GRID_C, 0xc0cce0, 0xd8e0ee);
  gh.position.y = -0.01; g.add(gh);
}

function buildPCScene(g: THREE.Group, pts: Point3D[]) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(pts.length*3);
  const col = new Float32Array(pts.length*3);
  pts.forEach((p,i) => {
    pos[i*3]=p.x; pos[i*3+1]=p.y; pos[i*3+2]=p.z;
    const c = hCol(Math.min(p.y/1.6,1));
    col[i*3]=c.r; col[i*3+1]=c.g; col[i*3+2]=c.b;
  });
  geo.setAttribute("position", new THREE.BufferAttribute(pos,3));
  geo.setAttribute("color",    new THREE.BufferAttribute(col,3));
  g.add(new THREE.Points(geo, new THREE.PointsMaterial({size:0.052,vertexColors:true,sizeAttenuation:true})));
  addGridHelper(g);
}

function buildPillarScene(g: THREE.Group, pillars: PillarCell[]) {
  for (const p of pillars) {
    const h = p.maxH + p.density*0.95;
    const col = dCol(p.density);
    const geo = new THREE.BoxGeometry(CELL*0.84, h, CELL*0.84);
    const mat = new THREE.MeshPhongMaterial({
      color: col, emissive: col.clone().multiplyScalar(0.28),
      transparent:true, opacity:0.48+p.density*0.50, shininess:100,
    });
    const m = new THREE.Mesh(geo, mat);
    m.position.set(p.cx, h/2, p.cz); g.add(m);
    const el = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({color:col.clone().addScalar(0.22),transparent:true,opacity:0.26})
    );
    el.position.copy(m.position); g.add(el);
  }
  addGridHelper(g);
}

function buildPseudoScene(g: THREE.Group, pillars: PillarCell[]) {
  const dm = new Float32Array(GRID_R*GRID_C);
  for (const p of pillars) dm[p.row*GRID_C+p.col] = p.density;

  for (let r=0; r<GRID_R; r++) for (let c=0; c<GRID_C; c++) {
    const v = dm[r*GRID_C+c];
    const col = pCol(v);
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(CELL*0.96, CELL*0.96),
      new THREE.MeshBasicMaterial({color:col, side:THREE.DoubleSide})
    );
    m.rotation.x = -Math.PI/2;
    m.position.set(-HALF+c*CELL+CELL/2, 0.001, -HALF+r*CELL+CELL/2);
    g.add(m);
    if (v > 0.25) {
      const gm = new THREE.Mesh(
        new THREE.PlaneGeometry(CELL*0.6, CELL*0.6),
        new THREE.MeshBasicMaterial({color:col.clone().addScalar(0.4),transparent:true,opacity:v*0.65,side:THREE.DoubleSide})
      );
      gm.rotation.x=-Math.PI/2; gm.position.set(m.position.x,0.012,m.position.z);
      g.add(gm);
    }
  }
  // Scan line
  const sl = new THREE.Mesh(
    new THREE.PlaneGeometry(HALF*2, CELL*1.8),
    new THREE.MeshBasicMaterial({color:0x00e5ff,transparent:true,opacity:0.12,side:THREE.DoubleSide})
  );
  sl.rotation.x=-Math.PI/2; sl.position.y=0.06; sl.name="scanline"; g.add(sl);
}

function buildDetectionScene(g: THREE.Group, pts: Point3D[], boxes: DetectedBox[]) {
  // Render point cloud faded
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(pts.length*3);
  const col = new Float32Array(pts.length*3);
  pts.forEach((p,i) => {
    pos[i*3]=p.x; pos[i*3+1]=p.y; pos[i*3+2]=p.z;
    const c = hCol(Math.min(p.y/1.6,1));
    col[i*3]=c.r*0.5; col[i*3+1]=c.g*0.5; col[i*3+2]=c.b*0.5;
  });
  geo.setAttribute("position", new THREE.BufferAttribute(pos,3));
  geo.setAttribute("color",    new THREE.BufferAttribute(col,3));
  g.add(new THREE.Points(geo, new THREE.PointsMaterial({size:0.035,vertexColors:true})));

  // Bounding boxes
  for (const box of boxes) {
    const col3 = hexToThree(LABEL_COLOR[box.label]);
    const w=box.w, h=box.h, d=box.d;

    // Wireframe box
    const bGeo = new THREE.BoxGeometry(w, h, d);
    const edges = new THREE.EdgesGeometry(bGeo);
    const lineMat = new THREE.LineBasicMaterial({color:col3, linewidth:2});
    const wireBox = new THREE.LineSegments(edges, lineMat);
    wireBox.rotation.y = box.yaw;
    wireBox.position.set(box.cx, h/2, box.cz);
    g.add(wireBox);

    // Corner posts
    const cornerGeo = new THREE.SphereGeometry(0.04, 6, 6);
    const cornerMat = new THREE.MeshBasicMaterial({color:col3});
    for (let xi of [-0.5,0.5]) for (let zi of [-0.5,0.5]) {
      const corner = new THREE.Mesh(cornerGeo, cornerMat);
      const lx = xi*w; const lz = zi*d;
      corner.position.set(
        box.cx + lx*Math.cos(box.yaw) - lz*Math.sin(box.yaw),
        box.h,
        box.cz + lx*Math.sin(box.yaw) + lz*Math.cos(box.yaw)
      );
      g.add(corner);
    }

    // Score plane (flat label above box)
    const planeMat = new THREE.MeshBasicMaterial({
      color: col3.clone().multiplyScalar(0.22), transparent:true, opacity:0.35, side:THREE.DoubleSide
    });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(w,d), planeMat);
    plane.rotation.x=-Math.PI/2; plane.position.set(box.cx, h+0.02, box.cz);
    g.add(plane);
  }
  addGridHelper(g);
}

// ─── Architecture diagram data ──────────────────────────────────────────────
const ARCH_STAGES = [
  {
    id:"input", label:"Input\nPC2", color:"#e8f0fc", accent:"#0077ff",
    desc:"Raw LiDAR\n2800 pts\nXYZ+I",
    shape:"cloud",
  },
  {
    id:"pillarize", label:"Pillar\nFeature\nNet", color:"#f0ebff", accent:"#7c22e8",
    desc:"9D encoding\nper pillar\nPointNet-lite",
    layers:["Linear 64","BN+ReLU","Linear 64","MaxPool"],
  },
  {
    id:"scatter", label:"Scatter\nto\nCanvas", color:"#fff0e8", accent:"#e85500",
    desc:`${GRID_C}×${GRID_R}×64\npseudo-image`,
    shape:"arrow",
  },
  {
    id:"backbone", label:"2D CNN\nBackbone", color:"#e8f8f0", accent:"#00aa55",
    desc:"Multi-scale\nfeature maps",
    layers:["Conv 3×3 s1","Conv 3×3 s2","Conv 3×3 s2","Conv 3×3 s2"],
  },
  {
    id:"neck", label:"FPN\nNeck", color:"#fffbe6", accent:"#c48000",
    desc:"Upsample\n+Concat\nfeature pyramid",
    layers:["Up×1","Up×2","Up×4","Concat"],
  },
  {
    id:"head", label:"Detection\nHead", color:"#fff0f5", accent:"#d4006a",
    desc:"Cls + Reg\n+ Direction",
    layers:["Conv 1×1 cls","Conv 1×1 reg","Conv 1×1 dir"],
  },
  {
    id:"nms", label:"NMS\n+\nBoxes", color:"#e8f4ff", accent:"#0077ff",
    desc:"3D BBoxes\nCar/Ped/Cyc",
    shape:"result",
  },
];

// ══════════════════════════════════════════════════════════════════════════════
// ─── Main Component ──────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
export default function PointPillarsPipeline() {
  const mountRef     = useRef<HTMLDivElement>(null);
  const rendererRef  = useRef<THREE.WebGLRenderer|null>(null);
  const sceneRef     = useRef<THREE.Scene|null>(null);
  const cameraRef    = useRef<THREE.PerspectiveCamera|null>(null);
  const groupRef     = useRef<THREE.Group|null>(null);
  const frameRef     = useRef<number>(0);
  const rotRef       = useRef(true);
  const scanRef      = useRef({ dir:1, pos:0 });

  const [stage, setStage]         = useState<PipelineStage>("pointcloud");
  const stageRef                  = useRef<PipelineStage>("pointcloud");
  const [animating, setAnimating] = useState(false);
  const [autoRot, setAutoRot]     = useState(true);
  const [pillarCount, setPillarCount] = useState(0);

  const ptsRef     = useRef<Point3D[]>([]);
  const pillarsRef = useRef<PillarCell[]>([]);
  const boxesRef   = useRef<DetectedBox[]>([]);

  // seed data client-side only
  useEffect(() => {
    ptsRef.current     = generateScene(42);
    pillarsRef.current = buildPillars(ptsRef.current);
    boxesRef.current   = inferBoxes(pillarsRef.current);
    setPillarCount(pillarsRef.current.length);
  }, []);

  // rebuild Three.js group for current stage
  const rebuildGroup = useCallback((s: PipelineStage) => {
    const g = groupRef.current;
    if (!g) return;
    while (g.children.length) g.remove(g.children[0]);
    scanRef.current = { dir:1, pos:-HALF };

    if (s === "pointcloud")    buildPCScene(g, ptsRef.current);
    if (s === "pillarization") buildPillarScene(g, pillarsRef.current);
    if (s === "pseudoimage")   buildPseudoScene(g, pillarsRef.current);
    if (s === "detection")     buildDetectionScene(g, ptsRef.current, boxesRef.current);
  }, []);

  // Three.js init
  useEffect(() => {
    if (!mountRef.current || stage === "architecture") return;
    const el = mountRef.current;
    const W = el.clientWidth, H = el.clientHeight;

    const renderer = new THREE.WebGLRenderer({ antialias:true, alpha:true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.setClearColor(0xf4f7fb, 1);
    el.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0xf4f7fb, 0.045);
    sceneRef.current = scene;

    const cam = new THREE.PerspectiveCamera(50, W/H, 0.1, 120);
    cam.position.set(7.5, 7.5, 9.5); cam.lookAt(0,0,0);
    cameraRef.current = cam;

    scene.add(new THREE.AmbientLight(0xffffff, 1.8));
    const dir = new THREE.DirectionalLight(0xffffff, 1.2);
    dir.position.set(5,10,5); scene.add(dir);
    const pt = new THREE.PointLight(0x4488ff, 0.6, 28);
    pt.position.set(-3, 4, -3);
    scene.add(pt);

    const group = new THREE.Group();
    scene.add(group);
    groupRef.current = group;

    // Wait for data then build
    const tryBuild = () => {
      if (ptsRef.current.length > 0) { rebuildGroup("pointcloud"); }
      else { setTimeout(tryBuild, 50); }
    };
    tryBuild();

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      if (rotRef.current) group.rotation.y += 0.0025;
      if (stageRef.current === "pseudoimage") {
        const sl = group.getObjectByName("scanline");
        if (sl) {
          scanRef.current.pos += scanRef.current.dir * 0.045;
          if (scanRef.current.pos > HALF || scanRef.current.pos < -HALF) scanRef.current.dir *= -1;
          sl.position.z = scanRef.current.pos;
        }
      }
      renderer.render(scene, cam);
    };
    animate();

    const onResize = () => {
      const W2=el.clientWidth, H2=el.clientHeight;
      cam.aspect=W2/H2; cam.updateProjectionMatrix();
      renderer.setSize(W2, H2);
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(frameRef.current);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const goStage = useCallback((s: PipelineStage) => {
    if (animating || s === stage) return;
    if (s === "architecture") {
      setStage("architecture"); stageRef.current = "architecture"; return;
    }
    if (stage === "architecture") {
      setStage(s); stageRef.current = s;
      // renderer may have been unmounted; rebuild on next tick
      setTimeout(() => rebuildGroup(s), 80);
      return;
    }
    setAnimating(true);
    const group = groupRef.current;
    if (!group) { setStage(s); stageRef.current=s; setAnimating(false); return; }

    // Fade out
    let t1=0;
    const fadeOut = setInterval(() => {
      t1++;
      group.traverse(o => { const m=(o as any).material; if(m?.opacity!==undefined) m.opacity=Math.max(0,m.opacity-0.07); });
      if (t1>14) {
        clearInterval(fadeOut);
        setStage(s); stageRef.current=s;
        rebuildGroup(s);
        let t2=0;
        const fadeIn = setInterval(()=>{ t2++; if(t2>18){ clearInterval(fadeIn); setAnimating(false); } },30);
      }
    },28);
  }, [animating, stage, rebuildGroup]);

  const toggleRot = useCallback(() => {
    rotRef.current = !rotRef.current;
    setAutoRot(r=>!r);
  },[]);

  // ─── Stage definitions ────────────────────────────────────────────────────
  const STAGES: { id:PipelineStage; icon:string; label:string; sub:string }[] = [
    { id:"pointcloud",    icon:"⬡", label:"Point Cloud",    sub:"Raw PC2 · XYZ+I"        },
    { id:"pillarization", icon:"▬", label:"Pillarization",  sub:"PointNet feature columns"},
    { id:"pseudoimage",   icon:"▦", label:"Pseudo-Image",   sub:"BEV CNN feature map"     },
    { id:"detection",     icon:"◻", label:"Detection",      sub:"3D bounding boxes"       },
    { id:"architecture",  icon:"⬡", label:"Architecture",   sub:"Full network diagram"    },
  ];

  const isThreeStage = stage !== "architecture";

  return (
    <div style={{
      width:"100%", height:"100vh", background:PAL.bg,
      fontFamily:"'JetBrains Mono','Fira Code',monospace",
      display:"flex", flexDirection:"column", overflow:"hidden", color:PAL.text,
    }}>
      {/* ── Header ── */}
      <div style={{
        padding:"8px 18px", borderBottom:`1px solid ${PAL.gridLine}`,
        background:"rgba(255,255,255,0.97)", display:"flex",
        alignItems:"center", justifyContent:"space-between", flexShrink:0,
      }}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:7,height:7,borderRadius:"50%",background:PAL.accent,boxShadow:`0 0 7px ${PAL.accent}`,animation:"blink 1.6s infinite"}}/>
          <span style={{fontSize:10,color:PAL.accent,letterSpacing:2.5}}>POINTPILLARS</span>
          <span style={{fontSize:9,color:PAL.dim,letterSpacing:1}}>· 3D Object Detection Pipeline</span>
        </div>
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          <Pill label="PTS"  val={String(N_PTS)}/>
          <Pill label="PILLARS" val={pillarCount>0?String(pillarCount):"—"}/>
          <Pill label="GRID" val={`${GRID_C}×${GRID_R}`}/>
          {isThreeStage && (
            <button onClick={toggleRot} style={{
              background:autoRot?"rgba(0,229,255,0.1)":"rgba(255,255,255,0.04)",
              border:`1px solid ${autoRot?PAL.accent+"44":"#ffffff22"}`,
              color:autoRot?PAL.accent:PAL.dim, padding:"2px 10px",
              borderRadius:2, cursor:"pointer", fontSize:9, letterSpacing:1, fontFamily:"inherit",
            }}>⟳ ROTATE</button>
          )}
        </div>
      </div>

      {/* ── Stage tabs ── */}
      <div style={{display:"flex",flexShrink:0,borderBottom:`1px solid ${PAL.gridLine}`,background:"rgba(248,250,254,0.98)"}}>
        {STAGES.map((s,i)=>(
          <button key={s.id} onClick={()=>goStage(s.id)} disabled={animating} style={{
            flex:1, padding:"9px 6px",
            background: stage===s.id ? "rgba(0,229,255,0.07)" : "transparent",
            border:"none",
            borderRight: i<4 ? `1px solid ${PAL.gridLine}` : "none",
            borderBottom: `2px solid ${stage===s.id?PAL.accent:"transparent"}`,
            color: stage===s.id ? PAL.accent : PAL.dim,
            cursor: animating?"not-allowed":"pointer",
            textAlign:"left", fontFamily:"inherit",
            transition:"all 0.18s",
          }}>
            <div style={{display:"flex",alignItems:"center",gap:5}}>
              <span style={{fontSize:12,opacity:0.7}}>{s.icon}</span>
              <div>
                <div style={{fontSize:9,letterSpacing:1.2,fontWeight:700}}>{`0${i+1} · ${s.label.toUpperCase()}`}</div>
                <div style={{fontSize:8,color:stage===s.id?"#2a6a80":PAL.dimmer,marginTop:1}}>{s.sub}</div>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* ── Main area ── */}
      <div style={{flex:1,position:"relative",overflow:"hidden"}}>

        {/* Three.js viewport */}
        {isThreeStage && (
          <>
            <div ref={mountRef} style={{width:"100%",height:"100%"}}/>

            {/* HUD top-right */}
            <div style={{position:"absolute",top:10,right:10,display:"flex",flexDirection:"column",gap:5,pointerEvents:"none"}}>
              <HUD label="FRAME" val="0042"/>
              <HUD label="SENSOR" val="Hesai Pandar64"/>
              <HUD label="VIEW" val="BEV + 3D"/>
              {stage==="pillarization" && <HUD label="PILLARS" val={`${pillarCount} active`} col={PAL.purple}/>}
              {stage==="pseudoimage"   && <HUD label="CNN IN"  val={`${GRID_C}×${GRID_R}×64`} col={PAL.orange}/>}
              {stage==="detection"     && <HUD label="DETECTIONS" val={`${boxesRef.current.length} boxes`} col={PAL.green}/>}
            </div>

            {/* Legend bottom-left */}
            <div style={{position:"absolute",bottom:10,left:10,pointerEvents:"none"}}>
              {stage==="pointcloud"    && <Legend title="HEIGHT COLORING"   items={[{c:"#00b3ff",l:"Ground"},{c:"#00ff88",l:"Vehicles"},{c:"#ffd600",l:"Pedestrians"}]}/>}
              {stage==="pillarization" && <Legend title="PILLAR DENSITY"    items={[{c:"#6655ff",l:"Low density"},{c:"#00aaff",l:"Medium"},{c:"#ff6b00",l:"High (object)"}]}/>}
              {stage==="pseudoimage"   && <Legend title="BEV FEATURE MAP"   items={[{c:"#030912",l:"Empty"},{c:"#441a6a",l:"Low activation"},{c:"#ff5500",l:"High activation"}]}/>}
              {stage==="detection"     && <Legend title="DETECTION CLASSES" items={[{c:PAL.accent,l:"Car"},{c:PAL.green,l:"Pedestrian"},{c:PAL.orange,l:"Cyclist"}]}/>}
            </div>

            {/* Transition overlay */}
            {animating && (
              <div style={{position:"absolute",inset:0,background:"rgba(2,9,18,0.6)",display:"flex",alignItems:"center",justifyContent:"center",pointerEvents:"none"}}>
                <div style={{border:`1px solid ${PAL.accent}66`,padding:"6px 22px",color:PAL.accent,fontSize:10,letterSpacing:3,background:"rgba(255,255,255,0.92)"}}>TRANSFORMING...</div>
              </div>
            )}
          </>
        )}

        {/* Architecture diagram */}
        {!isThreeStage && <ArchDiagram />}
      </div>

      {/* ── Pipeline footer ── */}
      <div style={{
        padding:"7px 18px", borderTop:`1px solid ${PAL.gridLine}`,
        background:"rgba(255,255,255,0.97)", display:"flex",
        alignItems:"center", justifyContent:"center", gap:0, flexShrink:0,
      }}>
        {STAGES.slice(0,4).map((s,i)=>(
          <div key={s.id} style={{display:"flex",alignItems:"center"}}>
            <div onClick={()=>goStage(s.id)} style={{
              padding:"2px 12px", borderRadius:2,
              background: stage===s.id?"rgba(0,229,255,0.09)":"transparent",
              border:`1px solid ${stage===s.id?PAL.accent+"44":PAL.dimmer}`,
              color: stage===s.id?PAL.accent:PAL.dim,
              fontSize:8, letterSpacing:1.5, cursor:"pointer", whiteSpace:"nowrap",
            }}>{s.label.toUpperCase()}</div>
            {i<3 && <span style={{color:PAL.dimmer,fontSize:13,margin:"0 3px"}}>→</span>}
          </div>
        ))}
      </div>

      <style>{`
        @keyframes blink{ 0%,100%{opacity:1} 50%{opacity:0.25} }
        @keyframes scanPulse{ 0%,100%{opacity:0.12} 50%{opacity:0.22} }
      `}</style>
    </div>
  );
}

// ─── Small shared UI components ────────────────────────────────────────────
function Pill({label,val}:{label:string;val:string}){
  return(
    <div style={{display:"flex",gap:4,alignItems:"baseline"}}>
      <span style={{fontSize:7,color:PAL.dim,letterSpacing:1}}>{label}</span>
      <span style={{fontSize:10,color:PAL.textBright}}>{val}</span>
    </div>
  );
}

function HUD({label,val,col=PAL.accent}:{label:string;val:string;col?:string}){
  return(
    <div style={{background:"rgba(255,255,255,0.92)",border:`1px solid ${col}33`,padding:"4px 10px",minWidth:138,boxShadow:"0 1px 4px rgba(0,0,0,0.08)"}}>
      <div style={{fontSize:7,color:PAL.dim,letterSpacing:1.5}}>{label}</div>
      <div style={{fontSize:10,color:col}}>{val}</div>
    </div>
  );
}

function Legend({title,items}:{title:string;items:{c:string;l:string}[]}){
  return(
    <div style={{background:"rgba(255,255,255,0.92)",border:`1px solid ${PAL.gridLine}`,padding:"7px 12px",minWidth:180,boxShadow:"0 1px 4px rgba(0,0,0,0.08)"}}>
      <div style={{fontSize:7,color:PAL.dim,letterSpacing:1.5,marginBottom:5}}>{title}</div>
      {items.map(it=>(
        <div key={it.l} style={{display:"flex",alignItems:"center",gap:7,marginBottom:2}}>
          <div style={{width:9,height:9,background:it.c,flexShrink:0}}/>
          <span style={{fontSize:8,color:"#4a8aa0"}}>{it.l}</span>
        </div>
      ))}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// ─── Architecture Diagram ─────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
function ArchDiagram() {
  const [activeNode, setActiveNode] = useState<string|null>(null);
  const [animStep, setAnimStep]     = useState(0);

  useEffect(() => {
    const t = setInterval(()=> setAnimStep(s=>(s+1)%ARCH_STAGES.length), 900);
    return ()=>clearInterval(t);
  },[]);

  const INFO: Record<string,{title:string;body:string;dims:string}> = {
    input:      { title:"Raw Point Cloud (PC2)", body:"Sparse 3D points from LiDAR sensor. Each point has (x,y,z,intensity,time). N ≈ 2,800–100,000 pts per frame.", dims:"N × 4" },
    pillarize:  { title:"Pillar Feature Network", body:"Groups points into vertical pillars over an XY grid. A small PointNet encodes each pillar into a 64-dim feature. Uses batch-norm + ReLU + channel-wise max-pool.", dims:"P × 64" },
    scatter:    { title:"Scatter to Pseudo-Image", body:"Each pillar's 64-dim vector is placed at its (col,row) position in a 2D canvas. Empty cells → zero. Produces a dense BEV feature image.", dims:`${GRID_C}×${GRID_R}×64` },
    backbone:   { title:"2D CNN Backbone (VGG-style)", body:"Three down-sampling blocks with stride-2 convolutions. Produces multi-scale feature maps at 1×, 2×, 4× downsampling. Channels grow: 64→128→256→512.", dims:"H/8×W/8×512" },
    neck:       { title:"Feature Pyramid Network (FPN)", body:"Upsamples and concatenates feature maps from all backbone stages. Fuses low-level spatial detail with high-level semantic context.", dims:"H/2×W/2×384" },
    head:       { title:"SSD-style Detection Head", body:"Three parallel 1×1 conv branches:\n• Class logits (Car/Ped/Cyc)\n• Box regression (Δx,Δy,Δz,w,l,h,θ)\n• Direction classification (sin/cos θ)", dims:"H/2×W/2×(A×K)" },
    nms:        { title:"NMS + 3D Boxes", body:"Non-Maximum Suppression with IoU threshold 0.5 for cars, 0.35 for others. Outputs 3D axis-aligned + rotated bounding boxes with class scores.", dims:"K′ boxes" },
  };

  return (
    <div style={{
      width:"100%", height:"100%", overflow:"auto",
      background:`radial-gradient(ellipse at 50% 30%, #eaf0fc 0%, #f4f7fb 70%)`,
      display:"flex", flexDirection:"column", alignItems:"center",
      padding:"24px 16px 24px", boxSizing:"border-box",
    }}>
      <div style={{fontSize:9,color:PAL.dim,letterSpacing:3,marginBottom:20}}>FULL NETWORK ARCHITECTURE · POINTPILLARS</div>

      {/* Main pipeline row */}
      <div style={{display:"flex",alignItems:"center",gap:0,flexWrap:"nowrap",overflowX:"auto",paddingBottom:16,maxWidth:"100%"}}>
        {ARCH_STAGES.map((node,i)=>{
          const active = activeNode===node.id || animStep===i;
          const info = INFO[node.id];
          return(
            <div key={node.id} style={{display:"flex",alignItems:"center",flexShrink:0}}>
              <div
                onClick={()=>setActiveNode(activeNode===node.id?null:node.id)}
                style={{
                  position:"relative",
                  width: node.id==="input"||node.id==="nms" ? 90 : node.id==="scatter" ? 80 : 100,
                  cursor:"pointer",
                  transition:"transform 0.15s",
                  transform: active?"scale(1.06)":"scale(1)",
                }}
              >
                {/* Node card */}
                <div style={{
                  background: active ? `${node.color}ee` : node.color,
                  border:`1px solid ${active?node.accent:node.accent+"44"}`,
                  borderRadius:4, padding:"10px 8px",
                  boxShadow: active?`0 0 18px ${node.accent}44`:"none",
                  transition:"all 0.2s",
                }}>
                  {/* accent top bar */}
                  <div style={{height:2,background:node.accent,borderRadius:1,marginBottom:8,opacity:active?1:0.4}}/>

                  <div style={{fontSize:9,color:node.accent,letterSpacing:0.5,fontWeight:700,textAlign:"center",whiteSpace:"pre-line",lineHeight:1.4}}>
                    {node.label}
                  </div>

                  {/* Layer pills */}
                  {node.layers && (
                    <div style={{marginTop:7,display:"flex",flexDirection:"column",gap:3}}>
                      {node.layers.map((l,li)=>(
                        <div key={l} style={{
                          fontSize:7, color:node.accent, background:`${node.accent}14`,
                          border:`1px solid ${node.accent}28`,
                          borderRadius:2, padding:"1px 4px", textAlign:"center",
                          opacity: active ? 1 : 0.55,
                          transition:`opacity 0.1s ${li*0.06}s`,
                        }}>{l}</div>
                      ))}
                    </div>
                  )}

                  {/* Desc */}
                  <div style={{
                    marginTop:6, fontSize:7, color:active?PAL.text:PAL.dim,
                    textAlign:"center", whiteSpace:"pre-line", lineHeight:1.5,
                    transition:"color 0.2s",
                  }}>{node.desc}</div>

                  {/* Dims badge */}
                  {info && (
                    <div style={{
                      marginTop:6, fontSize:6, color:node.accent,
                      background:`${node.accent}18`, border:`1px solid ${node.accent}30`,
                      borderRadius:2, padding:"1px 4px", textAlign:"center",
                    }}>{info.dims}</div>
                  )}
                </div>

                {/* Glow dot */}
                {animStep===i && (
                  <div style={{
                    position:"absolute",top:-4,right:-4,
                    width:8,height:8,borderRadius:"50%",
                    background:node.accent, boxShadow:`0 0 8px ${node.accent}`,
                    animation:"blink 0.9s infinite",
                  }}/>
                )}
              </div>

              {/* Arrow connector */}
              {i < ARCH_STAGES.length-1 && (
                <div style={{display:"flex",flexDirection:"column",alignItems:"center",margin:"0 3px",flexShrink:0}}>
                  <div style={{
                    width:22, height:1,
                    background:`linear-gradient(to right, ${ARCH_STAGES[i].accent}88, ${ARCH_STAGES[i+1].accent}88)`,
                  }}/>
                  <div style={{fontSize:12,color:ARCH_STAGES[i].accent,opacity:0.6,lineHeight:0.5}}>›</div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Info panel when node selected */}
      {activeNode && INFO[activeNode] && (
        <div style={{
          marginTop:16,
          background:"rgba(255,255,255,0.96)",
          border:`1px solid ${ARCH_STAGES.find(s=>s.id===activeNode)?.accent}44`,
          borderRadius:4, padding:"14px 20px",
          maxWidth:520, width:"100%",
          boxShadow:`0 0 24px ${ARCH_STAGES.find(s=>s.id===activeNode)?.accent}22`,
        }}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:8}}>
            <div style={{fontSize:11,color:ARCH_STAGES.find(s=>s.id===activeNode)?.accent,letterSpacing:0.5,fontWeight:700}}>
              {INFO[activeNode].title}
            </div>
            <div style={{
              fontSize:9,color:ARCH_STAGES.find(s=>s.id===activeNode)?.accent,
              background:`${ARCH_STAGES.find(s=>s.id===activeNode)?.accent}22`,
              border:`1px solid ${ARCH_STAGES.find(s=>s.id===activeNode)?.accent}44`,
              borderRadius:2,padding:"1px 7px",
            }}>{INFO[activeNode].dims}</div>
          </div>
          <div style={{fontSize:10,color:PAL.text,lineHeight:1.7,whiteSpace:"pre-line"}}>
            {INFO[activeNode].body}
          </div>
        </div>
      )}

      {/* Feature map size progression */}
      <div style={{marginTop:20,width:"100%",maxWidth:680}}>
        <div style={{fontSize:7,color:PAL.dim,letterSpacing:2,marginBottom:10,textAlign:"center"}}>TENSOR DIMENSIONS THROUGH PIPELINE</div>
        <div style={{display:"flex",alignItems:"flex-end",gap:4,justifyContent:"center"}}>
          {[
            {label:"PC2",     w:18, h:60, c:PAL.accent,  dim:"N×4"},
            {label:"Pillars", w:18, h:42, c:PAL.purple,  dim:"P×64"},
            {label:"BEV",     w:28, h:36, c:PAL.orange,  dim:`${GRID_C}×${GRID_R}×64`},
            {label:"Block1",  w:24, h:30, c:PAL.green,   dim:`${GRID_C/2}×${GRID_R/2}×128`},
            {label:"Block2",  w:20, h:24, c:PAL.green,   dim:`${GRID_C/4}×${GRID_R/4}×256`},
            {label:"Block3",  w:16, h:18, c:PAL.green,   dim:`${GRID_C/8}×${GRID_R/8}×512`},
            {label:"FPN",     w:24, h:26, c:PAL.yellow,  dim:`${GRID_C/2}×${GRID_R/2}×384`},
            {label:"Head",    w:22, h:22, c:PAL.pink,    dim:"H×W×(A·K)"},
            {label:"Boxes",   w:14, h:14, c:PAL.accent,  dim:"K′×9"},
          ].map((t,i)=>(
            <div key={t.label} style={{display:"flex",flexDirection:"column",alignItems:"center",gap:3}}>
              <div style={{fontSize:6,color:t.c,opacity:0.8}}>{t.dim}</div>
              <div style={{
                width:t.w, height:t.h,
                background:`${t.c}22`,
                border:`1px solid ${t.c}88`,
                borderRadius:2,
                transition:"all 0.3s",
                boxShadow: animStep===i?`0 0 10px ${t.c}66`:"none",
              }}/>
              <div style={{fontSize:6,color:PAL.dim,letterSpacing:0.5}}>{t.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Backbone detail */}
      <div style={{marginTop:20,width:"100%",maxWidth:680}}>
        <div style={{fontSize:7,color:PAL.dim,letterSpacing:2,marginBottom:10,textAlign:"center"}}>BACKBONE CONV BLOCK DETAIL</div>
        <div style={{display:"flex",gap:8,justifyContent:"center",flexWrap:"wrap"}}>
          {["Block A · stride 1","Block B · stride 2","Block C · stride 2"].map((blk,bi)=>(
            <div key={blk} style={{
              background:"rgba(255,255,255,0.9)", border:`1px solid ${PAL.green}44`,
              borderRadius:4, padding:"10px 14px", flex:"1", minWidth:160,
            }}>
              <div style={{fontSize:8,color:PAL.green,letterSpacing:1,marginBottom:7}}>{blk.toUpperCase()}</div>
              {["Conv 3×3","BatchNorm","ReLU","Conv 3×3","BatchNorm","ReLU"].map((l,li)=>(
                <div key={l+li} style={{
                  fontSize:7,color:li%3===0?PAL.textBright:PAL.dim,
                  padding:"2px 6px", marginBottom:2,
                  background: li%3===0?"rgba(0,255,136,0.06)":"transparent",
                  borderLeft:`2px solid ${li%3===0?PAL.green+"66":"transparent"}`,
                }}>{l}</div>
              ))}
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}