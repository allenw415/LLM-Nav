// =====================
// Street View Graph Navigator
// - 4 headings: [330, 60, 150, 240]
// - move front depends on current headingIdx
// - edges.json supports arriveHeadingIdx
// - pitch buttons use mode values: 60/90/120
// - no start pitch/zoom required (use defaults)
// =====================

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const mod = (n, m) => ((n % m) + m) % m;

// 記錄操作到 console
function logAction(action) {
  const node = nodeById.get(state.nodeId);
  const headingDeg = HEADING_DEGS[state.headingIdx];
  const pitchG = toGooglePitch(state.pitchMode);
  console.log(`[Action: ${action}]\nNode: ${state.nodeId} (${node?.name || '-'}) | Heading: ${headingDeg}° (idx=${state.headingIdx}) | Pitch Mode: ${state.pitchMode} (Google: ${pitchG}°) | Zoom: ${state.zoom}`);
}

// 你的 4 個方位角（絕對 heading）
const HEADING_DEGS = [330, 60, 150, 240]; // idx 0..3

// 你的 pitch 模式值（不是 Google pitch）：60/90/120
const PITCH_MODES = [60, 90, 120];

// 將「你的 pitch 模式值」轉成 Google POV pitch（範圍約 -90..90）
// 你定義：90=水平；60=往上；120=往下
// 轉換：googlePitch = 90 - pitchMode
function toGooglePitch(pitchMode) {
  return 90 - pitchMode; // 60->+30, 90->0, 120->-30
}

// 角度距離（0..180）
function angDist(a, b) {
  const d = Math.abs(((a - b) % 360 + 360) % 360);
  return Math.min(d, 360 - d);
}

// 將任意 heading（0..360）量化到最近的 HEADING_DEGS idx
function nearestHeadingIdx(heading) {
  let bestIdx = 0;
  let best = Infinity;
  for (let i = 0; i < HEADING_DEGS.length; i++) {
    const d = angDist(heading, HEADING_DEGS[i]);
    if (d < best) { best = d; bestIdx = i; }
  }
  return bestIdx;
}

// 將任意 pitch 量化到最近的 PITCH_MODES（以你的模式值）
function nearestPitchModeFromGooglePitch(googlePitch) {
  const mode = 90 - googlePitch; // inverse
  let best = PITCH_MODES[0], bestD = Infinity;
  for (const m of PITCH_MODES) {
    const d = Math.abs(m - mode);
    if (d < bestD) { bestD = d; best = m; }
  }
  return best;
}

async function loadJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
  return await r.json();
}

function loadGoogleMaps(apiKey) {
  return new Promise((resolve, reject) => {
    if (window.google?.maps) return resolve();
    const s = document.createElement("script");
    s.async = true;
    s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly`;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load Google Maps JS API"));
    document.head.appendChild(s);
  });
}

// ---------- global ----------
let panorama;
let svService;

let nodes = [];
let nodeById = new Map();

// adjacency: fromId -> Map(dirIdx -> {to, arriveHeadingIdx?})
let adj = new Map();

const state = {
  nodeId: null,
  headingIdx: 0,       // 0..3, 對應 HEADING_DEGS
  pitchMode: 90,       // 60/90/120（你的模式）
  zoom: 1              // Google zoom
};

function flattenMapJson(mapData) {
  const out = [];
  for (const obj of mapData.objects ?? []) {
    const name = obj.name;
    for (const rec of obj.records ?? []) {
      const nodeId = rec.nodeId;
      if (!nodeId) continue;
      out.push({
        nodeId,
        name,
        floor: rec.floor ?? "",
        description: rec.description ?? "",
        lat: rec.lat,
        lng: rec.lng,
        panoId: rec.panoId ?? ""
      });
    }
  }
  return out;
}

function buildAdjacency(edges) {
  adj.clear();
  for (const e of edges ?? []) {
    const from = e.from;
    const to = e.to;
    const dirIdx = Number(e.dirIdx);
    const arriveHeadingIdx = (e.arriveHeadingIdx === undefined) ? undefined : Number(e.arriveHeadingIdx);
    if (!from || !to || Number.isNaN(dirIdx)) continue;

    if (!adj.has(from)) adj.set(from, new Map());
    adj.get(from).set(dirIdx, {
      to,
      arriveHeadingIdx: (arriveHeadingIdx === undefined || Number.isNaN(arriveHeadingIdx))
        ? undefined
        : mod(arriveHeadingIdx, 4)
    });
  }
}

async function ensurePanoId(node) {
  if (node.panoId) return node.panoId;

  const { data } = await svService.getPanorama({
    location: { lat: node.lat, lng: node.lng },
    radius: 50
  });

  const pano = data.location?.pano;
  if (!pano) throw new Error("No pano found for this node (check lat/lng or increase radius).");
  node.panoId = pano;
  return pano;
}

function updateStatus() {
  const node = nodeById.get(state.nodeId);
  const statusEl = document.getElementById("status");
  const headingDeg = HEADING_DEGS[state.headingIdx];
  const pitchG = toGooglePitch(state.pitchMode);
  statusEl.textContent =
    `Node: ${node?.nodeId ?? "-"} | heading: ${headingDeg}°(idx=${state.headingIdx}) | pitchMode: ${state.pitchMode} (googlePitch=${pitchG}) | zoom: ${state.zoom}`;
}

function applyView() {
  // 確保 zoom 在範圍內
  state.zoom = clamp(state.zoom, 0, 3);
  
  panorama.setPov({
    heading: HEADING_DEGS[state.headingIdx],
    pitch: toGooglePitch(state.pitchMode)
  });
  panorama.setZoom(state.zoom);
  updateStatus();
}

// opts: { arriveHeadingIdx?: number, keepHeading?: boolean }
async function goToNode(nodeId, opts = {}) {
  const node = nodeById.get(nodeId);
  if (!node) return;

  state.nodeId = nodeId;

  // 只有在 edge 指定 arriveHeadingIdx 時才改 heading
  if (opts.arriveHeadingIdx !== undefined) {
    state.headingIdx = mod(opts.arriveHeadingIdx, 4);
  }

  const panoId = await ensurePanoId(node);
  panorama.setPano(panoId);
  panorama.setVisible(true);
  applyView();

  // sync select
  const sel = document.getElementById("nodeSelect");
  sel.value = nodeId;
}

async function moveRelative(rel) {
  if (rel !== "front") return;

  const from = state.nodeId;
  const m = adj.get(from);
  if (!m) {
    alert("此節點沒有任何定義的邊（edges.json）");
    logAction('Move Front Failed');
    return;
  }

  const absDirIdx = state.headingIdx; // front = 當前方向
  const edge = m.get(absDirIdx);

  if (!edge) {
    alert(`前方不能走（dirIdx=${absDirIdx}, heading=${HEADING_DEGS[absDirIdx]}°）`);
    logAction('Move Front Failed');
    return;
  }

  // 前進後視角回復正常
  state.pitchMode = 90;
  state.zoom = 1;

  await goToNode(edge.to, { arriveHeadingIdx: edge.arriveHeadingIdx });
  logAction('Move Front');
}

function turn(deltaIdx) {
  state.headingIdx = mod(state.headingIdx + deltaIdx, 4);
  // 轉向後視角回復正常
  state.pitchMode = 90;
  state.zoom = 1;
  applyView();
  logAction(deltaIdx > 0 ? 'Turn Right' : 'Turn Left');
}

function setPitchMode(mode) {
  if (!PITCH_MODES.includes(mode)) return;
  state.pitchMode = mode;
  applyView();
  const modeNames = { 60: 'Up', 90: 'Level', 120: 'Down' };
  logAction(`Pitch ${modeNames[mode] || mode}`);
}

function zoom(delta) {
  state.zoom = clamp(state.zoom + delta, 0, 3);
  applyView();
  logAction(`Zoom ${delta > 0 ? 'In' : 'Out'}`);
}

function bindUI() {
  document.getElementById("moveFront").onclick = () => moveRelative("front");

  document.getElementById("turnLeft").onclick  = () => turn(-1);
  document.getElementById("turnRight").onclick = () => turn(+1);

  document.getElementById("pitchUp").onclick    = () => setPitchMode(60);
  document.getElementById("pitchLevel").onclick = () => setPitchMode(90);
  document.getElementById("pitchDown").onclick  = () => setPitchMode(120);

  document.getElementById("zoomIn").onclick  = () => zoom(+1);
  document.getElementById("zoomOut").onclick = () => zoom(-1);

  const sel = document.getElementById("nodeSelect");
  sel.onchange = (e) => goToNode(e.target.value);

  // keyboard
  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp") moveRelative("front");
    else if (e.key.toLowerCase() === "a") turn(-1);
    else if (e.key.toLowerCase() === "d") turn(+1);
    else if (e.key.toLowerCase() === "w") {
      // 往上：120->90->60
      const idx = PITCH_MODES.indexOf(state.pitchMode);
      const next = PITCH_MODES[Math.max(0, idx - 1)];
      setPitchMode(next);
    }
    else if (e.key.toLowerCase() === "s") {
      // 往下：60->90->120
      const idx = PITCH_MODES.indexOf(state.pitchMode);
      const next = PITCH_MODES[Math.min(PITCH_MODES.length - 1, idx + 1)];
      setPitchMode(next);
    }
    else if (e.key === "+" || e.key === "=") zoom(+1);
    else if (e.key === "-" || e.key === "_") zoom(-1);
  });

  // 使用者用滑鼠拖曳視角時，把狀態量化回你的 4 heading + 3 pitch 模式
  panorama.addListener("pov_changed", () => {
    const pov = panorama.getPov();
    if (typeof pov.heading === "number") {
      state.headingIdx = nearestHeadingIdx(pov.heading);
    }
    if (typeof pov.pitch === "number") {
      state.pitchMode = nearestPitchModeFromGooglePitch(pov.pitch);
    }
    updateStatus();
  });

  panorama.addListener("zoom_changed", () => {
    const rawZoom = panorama.getZoom();
    const clampedZoom = clamp(rawZoom, 0, 3);
    if (Math.abs(rawZoom - clampedZoom) > 0.01) {
      panorama.setZoom(clampedZoom);
    }
    state.zoom = clampedZoom;
    updateStatus();
  });
}

async function initApp() {
  const mapData = await loadJson("./map_with_nodeId.json");
  nodes = flattenMapJson(mapData);
  nodeById = new Map(nodes.map(n => [n.nodeId, n]));

  let edges = [];
  try {
  edges = await loadJson(`./edges.json?v=${Date.now()}`);
  } catch {
    console.warn("edges.json not found; movement will not work until you add it.");
  }
  console.log("edges loaded");
  buildAdjacency(edges);

  svService = new google.maps.StreetViewService();
  panorama = new google.maps.StreetViewPanorama(document.getElementById("pano"), {
    disableDefaultUI: true,
    clickToGo: false,
    showRoadLabels: false
  });

  // select options
  const sel = document.getElementById("nodeSelect");
  sel.innerHTML = "";
  for (const n of nodes) {
    const opt = document.createElement("option");
    opt.value = n.nodeId;
    opt.textContent = `${n.nodeId} (${n.description || n.name})`;
    sel.appendChild(opt);
  }

  bindUI();

  if (nodes.length === 0) {
    alert("map_with_nodeId.json 沒有節點");
    return;
  }

  // 起始：只設 nodeId + headingIdx（不設 pitch/zoom）
  const init = window.SV_INIT || {};
  const startNodeId = init.startNodeId && nodeById.has(init.startNodeId)
    ? init.startNodeId
    : nodes[0].nodeId;

  if (typeof init.startHeadingIdx === "number") {
    state.headingIdx = mod(Math.round(init.startHeadingIdx), 4);
  } else {
    state.headingIdx = 0;
  }

  // pitch/zoom 不設定（保持預設值：pitchMode=90、zoom=1）
  await goToNode(startNodeId);
}

// bootstrap
(async function main() {
  const key = window.GMAPS_API_KEY;
  if (!key) {
    alert("Missing GMAPS_API_KEY. Please set it in env.js");
    return;
  }
  await loadGoogleMaps(key);
  await initApp();
})().catch((err) => {
  console.error(err);
  alert(String(err?.message ?? err));
});