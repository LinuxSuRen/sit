// sit 前端：
//  1. 用 MediaPipe Pose Landmarker 在浏览器本地提取 33 个人体关键点
//  2. 按视角（正视/侧视，或单目 default）将镜像后的关键点通过 WebSocket 发给后端
//  3. 展示后端返回的全局坐姿判定、各视角角度偏差与久坐提醒
import {
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";

const CDN_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const SEND_INTERVAL_MS = 200; // 后端按帧去抖，每视角 5 帧/秒足够
const RECONNECT_MS = 2000;
const CALIB_MS = 3000;
const CALIB_INTERVAL_MS = 200;
const VOICE_REPEAT_MS = 2 * 60 * 1000; // 持续不良时每 2 分钟重复提醒
const VOICE_STORAGE_KEY = "sitcoach-voice-enabled";

const VIEW_FRONT = "front";
const VIEW_SIDE = "side";
const VIEW_DEFAULT = "default";
const VIEW_TEXT = { front: "正视", side: "侧视", default: "摄像头" };

// 后端只返回稳定的 code，用户可见文案统一在这里维护
const ISSUE_TEXT = {
  forward_head: "头部前倾",
  torso_lean_forward: "身体前倾",
  torso_lean_back: "身体后仰",
  head_tilt_left: "头向左歪",
  head_tilt_right: "头向右歪",
  long_sitting: "久坐提醒：该起来活动一下了",
};
const SEVERITY_TEXT = { mild: "轻度", severe: "重度" };
const STATUS_TEXT = {
  good: "坐姿标准",
  warning: "坐姿需要注意",
  bad: "坐姿不良",
  no_person: "未检测到人",
  needs_calibration: "等待标定",
  idle: "未开始",
};
const THRESHOLD_LABELS = [
  ["neck_forward_mild_deg", "颈部前倾 轻度"],
  ["neck_forward_severe_deg", "颈部前倾 重度"],
  ["torso_forward_mild_deg", "身体前倾 轻度"],
  ["torso_back_mild_deg", "身体后仰 轻度"],
  ["torso_severe_deg", "躯干角度 重度"],
  ["head_tilt_mild_deg", "头部侧倾 轻度"],
  ["head_tilt_severe_deg", "头部侧倾 重度"],
  ["long_sitting_minutes", "久坐提醒（分钟）"],
];

const $ = (id) => document.getElementById(id);
const els = {
  viewsRow: $("views-row"),
  units: {
    [VIEW_FRONT]: {
      unit: $("unit-front"),
      video: $("video-front"),
      canvas: $("overlay-front"),
      label: $("label-front"),
      deviceSelect: $("device-front"),
      calibBtn: $("calib-front"),
    },
    [VIEW_SIDE]: {
      unit: $("unit-side"),
      video: $("video-side"),
      canvas: $("overlay-side"),
      label: $("label-side"),
      deviceSelect: $("device-side"),
      calibBtn: $("calib-side"),
    },
  },
  banner: $("banner"),
  startBtn: $("start-btn"),
  modeSwitch: $("mode-switch"),
  connBadge: $("conn-badge"),
  statusDot: $("status-dot"),
  statusText: $("status-text"),
  voiceToggle: $("voice-toggle"),
  viewChips: $("view-chips"),
  issues: $("issues"),
  seatedNow: $("seated-now"),
  seatedBar: $("seated-bar"),
  seatedLimit: $("seated-limit"),
  thresholdList: $("threshold-list"),
  metricLabels: {
    neck_forward: $("label-neck"),
    torso_forward: $("label-torso"),
    head_tilt: $("label-tilt"),
  },
};

const state = {
  ws: null,
  running: false,
  reconnectTimer: null,
  calibrating: null, // 正在标定的视角名
  calibrated: {}, // view -> baseline
  latestMetrics: {}, // view -> metrics
  lastResult: null,
  mode: "single", // single | dual
  views: {}, // view -> 运行时单元（视频流/推理器/绘制状态）
  deviceIds: {}, // view -> 当前使用的 deviceId
  voiceEnabled: false,
  spokenIssues: {}, // code -> { severity, at }：语音播报状态
};

// ---------- 启动 / 停止 ----------

async function start() {
  els.startBtn.disabled = true;
  setBanner("正在加载检测模型…", false);
  let degraded = false;
  try {
    const devices = await listVideoDevices();
    // 枚举结果可能包含虚拟摄像头等不可用条目，第二路打不开时自动降级单摄
    state.mode = devices.length >= 2 ? "dual" : "single";
    if (state.mode === "dual") {
      setupDeviceSelect(VIEW_FRONT, devices, devices[0]?.deviceId);
      await initView(VIEW_FRONT, devices[0]?.deviceId);
      const secondId = devices.find((d) => d.deviceId !== state.deviceIds[VIEW_FRONT])?.deviceId;
      try {
        setupDeviceSelect(VIEW_SIDE, devices, secondId);
        await initView(VIEW_SIDE, secondId);
      } catch (err) {
        console.warn("second camera unusable, degrade to single mode", err);
        state.mode = "single";
        degraded = true;
      }
    } else {
      setupDeviceSelect(VIEW_FRONT, devices, devices[0]?.deviceId);
      await initView(VIEW_FRONT, devices[0]?.deviceId);
    }
  } catch (err) {
    console.error(err);
    setBanner("无法访问摄像头或加载模型，请检查浏览器授权与网络后重试", false);
    els.startBtn.disabled = false;
    return;
  }
  applyModeUI();
  state.running = true;
  els.startBtn.textContent = "停止检测";
  els.startBtn.disabled = false;
  els.startBtn.onclick = stop;
  els.modeSwitch.hidden = false;
  for (const view of Object.keys(state.views)) {
    state.views[view].calibBtnEl.disabled = false;
  }
  const uncalibrated = activeViews().filter((v) => !state.calibrated[v]);
  if (degraded) {
    setBanner("第二个摄像头无法打开，已切换为单摄像头模式", false, 4000);
  } else {
    setBanner(
      uncalibrated.length
        ? "请摆好标准坐姿，然后点击「标定正确坐姿」"
        : "请坐到摄像头前",
      false
    );
  }
  updateMetricLabels();
  connectWS();
  requestAnimationFrame(loop);
  loadConfig();
}

function stop() {
  state.running = false;
  clearTimeout(state.reconnectTimer);
  if (state.ws) state.ws.close();
  for (const unit of Object.values(state.views)) {
    if (unit.stream) unit.stream.getTracks().forEach((t) => t.stop());
    unit.videoEl.srcObject = null;
    unit.lastLandmarks = null;
    unit.lastVideoTime = -1;
    const ctx = unit.canvasEl.getContext("2d");
    ctx.clearRect(0, 0, unit.canvasEl.width, unit.canvasEl.height);
    unit.calibBtnEl.disabled = true;
  }
  els.units[VIEW_SIDE].unit.hidden = true;
  els.viewsRow.classList.remove("dual");
  els.units[VIEW_FRONT].label.textContent = VIEW_TEXT[VIEW_DEFAULT];
  els.units[VIEW_FRONT].calibBtn.textContent = "标定正确坐姿";
  els.modeSwitch.hidden = true;
  setBanner("检测已停止", false);
  els.startBtn.textContent = "开始检测";
  els.startBtn.onclick = start;
  setStatus("idle");
  state.latestMetrics = {};
}

// ---------- 单/双摄模式切换 ----------

// applyModeUI 按当前 state.mode 同步界面：窗格可见性、标签、按钮文案
function applyModeUI() {
  const dual = state.mode === "dual";
  if (dual && !state.views[VIEW_SIDE]) {
    // 尚未初始化侧视流的场合不强行展示
    els.units[VIEW_SIDE].unit.hidden = true;
  } else {
    els.units[VIEW_SIDE].unit.hidden = !dual;
  }
  els.viewsRow.classList.toggle("dual", dual);
  els.units[VIEW_FRONT].label.textContent = dual ? VIEW_TEXT[VIEW_FRONT] : VIEW_TEXT[VIEW_DEFAULT];
  els.units[VIEW_FRONT].calibBtn.textContent = dual ? "标定此视角" : "标定正确坐姿";
  els.units[VIEW_SIDE].calibBtn.textContent = "标定此视角";
  els.modeSwitch.textContent = dual ? "只用一个摄像头" : "添加侧视摄像头";
}

// switchMode 在运行中切换单/双摄；枚举到多个摄像头但实际只有一个可用时，
// 用户可以在这里强制回到单摄
async function switchMode() {
  if (!state.running || state.calibrating) return;
  if (state.mode === "dual") {
    const side = state.views[VIEW_SIDE];
    if (side?.stream) side.stream.getTracks().forEach((t) => t.stop());
    if (side) side.videoEl.srcObject = null;
    delete state.views[VIEW_SIDE];
    state.mode = "single";
    state.latestMetrics = {};
    applyModeUI();
    updateMetricLabels();
    renderViewChips(state.lastResult);
    setBanner("已切换为单摄像头模式，请重新标定正确坐姿", false, 4000);
    return;
  }

  // 升级为双摄
  const devices = await listVideoDevices().catch(() => []);
  const second = devices.find((d) => d.deviceId !== state.deviceIds[VIEW_FRONT]);
  if (!second) {
    setBanner("未发现第二个摄像头，请先连接后再试", false, 3000);
    return;
  }
  try {
    setupDeviceSelect(VIEW_FRONT, devices, state.deviceIds[VIEW_FRONT]);
    setupDeviceSelect(VIEW_SIDE, devices, second.deviceId);
    await initView(VIEW_SIDE, second.deviceId);
    state.views[VIEW_SIDE].calibBtnEl.disabled = false;
  } catch {
    setBanner("第二个摄像头无法打开，仍保持单摄像头模式", false, 4000);
    return;
  }
  state.mode = "dual";
  state.latestMetrics = {};
  applyModeUI();
  updateMetricLabels();
  renderViewChips(state.lastResult);
  setBanner("已切换为双摄模式，请分别标定正视与侧视", false, 4000);
}

async function listVideoDevices() {
  // 先开一路拿到授权，设备标签才可见；这路流会在 initView 中被替换
  const probe = await navigator.mediaDevices.getUserMedia({ video: true });
  probe.getTracks().forEach((t) => t.stop());
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "videoinput");
}

function setupDeviceSelect(view, devices, preferredId) {
  const sel = els.units[view].deviceSelect;
  if (devices.length < 2) {
    sel.hidden = true;
    return;
  }
  sel.hidden = false;
  sel.innerHTML = "";
  for (const d of devices) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `摄像头 ${sel.length + 1}`;
    sel.appendChild(opt);
  }
  if (preferredId) sel.value = preferredId;
  sel.onchange = async () => {
    if (!state.running) return;
    try {
      await initView(view, sel.value);
    } catch {
      setBanner("切换摄像头失败，请重试", false);
    }
  };
}

async function initView(view, deviceId) {
  const ui = els.units[view];
  const stream = await navigator.mediaDevices.getUserMedia({
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: 960, height: 720 }
      : { width: 960, height: 720, facingMode: "user" },
    audio: false,
  });
  const existing = state.views[view];
  if (existing?.stream) existing.stream.getTracks().forEach((t) => t.stop());
  ui.video.srcObject = stream;
  await ui.video.play();
  const landmarker = await initLandmarker("GPU").catch(() => initLandmarker("CPU"));
  state.views[view] = {
    view,
    videoEl: ui.video,
    canvasEl: ui.canvas,
    calibBtnEl: ui.calibBtn,
    stream,
    landmarker,
    lastLandmarks: null,
    lastVideoTime: -1,
    lastSentAt: 0,
  };
  if (deviceId) state.deviceIds[view] = deviceId;
}

async function initLandmarker(delegate) {
  const fileset = await FilesetResolver.forVisionTasks(`${CDN_ROOT}/wasm`);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
  });
}

// activeViews 返回当前生效的视角名列表
function activeViews() {
  return Object.keys(state.views);
}

// wsViewName 将界面视角映射为协议视角：单目模式发 default
function wsViewName(view) {
  return state.mode === "dual" ? view : VIEW_DEFAULT;
}

// ---------- 主循环 ----------

function loop() {
  if (!state.running) return;
  for (const unit of Object.values(state.views)) {
    const video = unit.videoEl;
    if (video.readyState >= 2 && video.currentTime !== unit.lastVideoTime) {
      unit.lastVideoTime = video.currentTime;
      try {
        const result = unit.landmarker.detectForVideo(video, performance.now());
        const landmarks = result.landmarks?.[0] ?? null;
        if (landmarks) unit.lastLandmarks = landmarks;
      } catch {
        // 单帧推理失败不终止整个循环
      }
    }
    if (unit.lastLandmarks) drawOverlay(unit, unit.lastLandmarks);
    sendPose(unit);
  }
  requestAnimationFrame(loop);
}

function mirror(landmarks) {
  // 镜像坐标：x 轴翻转后，画面左右与用户照镜子时一致
  return landmarks.map((l) => ({
    x: 1 - l.x,
    y: l.y,
    z: l.z,
    visibility: l.visibility ?? 1,
  }));
}

function sendPose(unit) {
  if (!unit.lastLandmarks) return;
  const now = performance.now();
  if (now - unit.lastSentAt < SEND_INTERVAL_MS) return;
  unit.lastSentAt = now;
  if (!state.ws || state.ws.readyState !== 1) return;
  state.ws.send(
    JSON.stringify({
      view: wsViewName(unit.view),
      landmarks: mirror(unit.lastLandmarks),
      timestamp_ms: Date.now(),
    })
  );
}

// ---------- WebSocket ----------

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/pose`);
  state.ws = ws;
  ws.onopen = () => {
    els.connBadge.textContent = "已连接";
    els.connBadge.className = "badge online";
  };
  ws.onmessage = (ev) => {
    try {
      renderResult(JSON.parse(ev.data));
    } catch {
      // 忽略无法解析的帧
    }
  };
  ws.onclose = () => {
    if (state.ws !== ws) return; // 已被新连接替换
    els.connBadge.textContent = "连接已断开";
    els.connBadge.className = "badge offline";
    if (state.running) {
      els.connBadge.textContent = "正在重连…";
      state.reconnectTimer = setTimeout(connectWS, RECONNECT_MS);
    }
  };
}

// ---------- 标定 ----------

async function calibrate(view) {
  if (!state.running || state.calibrating) return;
  const unit = state.views[view];
  if (!unit) return;
  state.calibrating = view;
  unit.calibBtnEl.disabled = true;
  setBanner(
    `请在${VIEW_TEXT[state.mode === "dual" ? view : VIEW_DEFAULT]}摄像头前保持标准坐姿…`,
    false
  );

  const frames = [];
  const steps = CALIB_MS / CALIB_INTERVAL_MS;
  for (let i = 0; i < steps; i++) {
    setBanner(
      `请保持标准坐姿，剩余 ${Math.ceil((CALIB_MS - i * CALIB_INTERVAL_MS) / 1000)} 秒…`,
      false
    );
    if (unit.lastLandmarks) {
      frames.push({
        landmarks: mirror(unit.lastLandmarks),
        timestamp_ms: Date.now(),
      });
    }
    await new Promise((res) => setTimeout(res, CALIB_INTERVAL_MS));
  }

  try {
    const resp = await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ view: wsViewName(view), poses: frames }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "failed");
    state.calibrated = {};
    for (const [v, vc] of Object.entries(data.views ?? {})) {
      if (vc.calibrated) state.calibrated[v] = vc.baseline;
    }
    setBanner("标定完成，继续保持就能看到实时判定", false, 2000);
  } catch {
    setBanner("标定失败，请确认画面中能看清上半身后重试", false);
  } finally {
    state.calibrating = null;
    unit.calibBtnEl.disabled = false;
    renderViewChips(state.lastResult);
  }
}

// ---------- 结果展示 ----------

function setBanner(text, hidden, autoHideMs) {
  els.banner.textContent = text;
  els.banner.hidden = !!hidden;
  els.banner.dataset.autohide = autoHideMs ? "1" : "";
  if (autoHideMs) {
    setTimeout(() => {
      if (els.banner.dataset.autohide) els.banner.hidden = true;
    }, autoHideMs);
  }
}

function renderResult(r) {
  state.lastResult = r;

  if (!state.calibrating) {
    if (r.status === "no_person") {
      setBanner("未检测到人，请坐回摄像头前", false);
    } else if (r.status === "needs_calibration") {
      setBanner("请摆好标准坐姿，点击「标定此视角」开始监测", false);
    } else if (!els.banner.dataset.autohide) {
      // 正常判定时清掉引导文案；自动隐藏的提示（如“标定完成”）自行超时
      els.banner.hidden = true;
    }
  }

  setStatus(r.status);
  renderIssues(r.issues ?? []);
  maybeSpeak(r.issues ?? []);
  if (r.view && r.metrics) {
    state.latestMetrics[r.view] = r.metrics;
    renderMetrics();
  }
  renderViewChips(r);
  renderSeated(r.seated_seconds ?? 0);
}

function setStatus(status) {
  const clsMap = {
    good: "status-good",
    warning: "status-warning",
    bad: "status-bad",
    no_person: "status-none",
    needs_calibration: "status-none",
    idle: "status-idle",
  };
  els.statusDot.className = `dot ${clsMap[status] ?? "status-idle"}`;
  els.statusText.textContent = STATUS_TEXT[status] ?? "未开始";
}

function renderIssues(issues) {
  els.issues.hidden = issues.length === 0;
  els.issues.innerHTML = "";
  for (const iss of issues) {
    const li = document.createElement("li");
    li.className = `issue ${iss.severity}`;
    const label = ISSUE_TEXT[iss.code] ?? iss.code;
    const prefix = iss.view && state.mode === "dual" ? `${VIEW_TEXT[iss.view]}：` : "";
    const detail =
      iss.code === "long_sitting"
        ? ""
        : `${SEVERITY_TEXT[iss.severity] ?? ""} ${Math.abs(iss.angle).toFixed(0)}°`;
    li.textContent = detail
      ? `${prefix}${label}（${detail.trim()}）`
      : `${prefix}${label}`;
    els.issues.appendChild(li);
  }
}

// ---------- 语音提示 ----------
//
// 触发策略（用户确认）：问题出现时播报一次；从轻度恶化到重度再报一次；
// 持续不良期间每 2 分钟重复提醒；恢复后重置，再次出现会重新播报。

function initVoiceToggle() {
  if (!els.voiceToggle) return;
  state.voiceEnabled = localStorage.getItem(VOICE_STORAGE_KEY) === "1";
  els.voiceToggle.checked = state.voiceEnabled;
  els.voiceToggle.addEventListener("change", () => {
    state.voiceEnabled = els.voiceToggle.checked;
    localStorage.setItem(VOICE_STORAGE_KEY, state.voiceEnabled ? "1" : "0");
    if (state.voiceEnabled) speak("语音提示已开启");
  });
}

function maybeSpeak(issues) {
  if (!state.voiceEnabled || !("speechSynthesis" in window)) return;
  const now = Date.now();
  const due = [];

  for (const iss of issues) {
    const prev = state.spokenIssues[iss.code];
    const escalated =
      prev && prev.severity === "mild" && iss.severity === "severe";
    // 首次出现、恶化升级、或持续不良超过重复间隔时需要播报
    if (!prev || escalated || now - prev.at >= VOICE_REPEAT_MS) {
      due.push(iss);
      state.spokenIssues[iss.code] = { severity: iss.severity, at: now };
    } else {
      state.spokenIssues[iss.code].severity = iss.severity;
    }
  }

  // 已恢复的问题清除记录，再次出现时重新播报
  const active = new Set(issues.map((i) => i.code));
  for (const code of Object.keys(state.spokenIssues)) {
    if (!active.has(code)) delete state.spokenIssues[code];
  }

  if (due.length === 0) return;

  const parts = due.map((iss) => {
    const label = ISSUE_TEXT[iss.code] ?? iss.code;
    return iss.severity === "severe" ? `${label}明显` : label;
  });
  const onlyLongSitting = due.every((i) => i.code === "long_sitting");
  speak(onlyLongSitting ? parts.join("，") : `请注意坐姿：${parts.join("，")}`);
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  try {
    // 打断上一条尚未播完的内容，避免排队长龙
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.lang = "zh-CN";
    utter.rate = 1;
    const zhVoice = window.speechSynthesis
      .getVoices()
      .find((v) => v.lang && v.lang.toLowerCase().startsWith("zh"));
    if (zhVoice) utter.voice = zhVoice;
    window.speechSynthesis.speak(utter);
  } catch {
    // 语音合成失败不影响检测
  }
}

// 指标条路由：侧倾来自正视（或单目），前倾类来自侧视（或单目）
function metricSource(key) {
  const dual = state.mode === "dual";
  const order =
    key === "head_tilt"
      ? dual
        ? [VIEW_FRONT, VIEW_DEFAULT]
        : [VIEW_DEFAULT, VIEW_FRONT]
      : dual
        ? [VIEW_SIDE, VIEW_DEFAULT]
        : [VIEW_DEFAULT, VIEW_SIDE];
  for (const v of order) {
    if (state.latestMetrics[v]) return state.latestMetrics[v];
  }
  return null;
}

function updateMetricLabels() {
  const dual = state.mode === "dual";
  els.metricLabels.neck_forward.textContent = dual ? "颈部前倾（侧视）" : "颈部前倾";
  els.metricLabels.torso_forward.textContent = dual ? "躯干前倾（侧视）" : "躯干前倾";
  els.metricLabels.head_tilt.textContent = dual ? "头部侧倾（正视）" : "头部侧倾";
}

function renderMetrics() {
  for (const el of document.querySelectorAll(".metric")) {
    const key = el.dataset.key;
    const metrics = metricSource(key);
    const out = el.querySelector("output");
    const fill = el.querySelector(".bar i");

    const raw = metrics?.[key];
    // 指标路由总是取“拥有该指标”的视角，因此 0 是真实测量值而非屏蔽值
    const unavailable =
      typeof raw !== "number" ||
      (key === "torso_forward" && metrics.torso_available === false);
    if (unavailable) {
      out.textContent = "—";
      fill.style.width = "0%";
      fill.className = "";
      continue;
    }
    // 标定后显示相对基准的偏差；未标定时显示原始角度
    const view = metricView(key);
    const bl = state.calibrated[view] ?? null;
    const dev = bl ? raw - (bl[key] ?? 0) : raw;
    out.textContent = bl
      ? `${dev >= 0 ? "+" : ""}${dev.toFixed(0)}°`
      : `${raw.toFixed(0)}°`;
    fill.style.width = `${Math.min(100, (Math.abs(dev) / 45) * 100)}%`;
    fill.className = Math.abs(dev) >= 30 ? "severe" : Math.abs(dev) >= 15 ? "warn" : "";
  }
}

function metricView(key) {
  if (state.mode !== "dual") return VIEW_DEFAULT;
  return key === "head_tilt" ? VIEW_FRONT : VIEW_SIDE;
}

function renderViewChips(r) {
  els.viewChips.innerHTML = "";
  const dual = state.mode === "dual";
  const names = dual ? [VIEW_FRONT, VIEW_SIDE] : [VIEW_DEFAULT];
  const viewStates = r?.views ?? {};
  for (const name of names) {
    const label = dual ? VIEW_TEXT[name] : VIEW_TEXT[VIEW_DEFAULT];
    const calib = !!state.calibrated[name];
    const present = viewStates[name]?.present;
    const chip = document.createElement("span");
    chip.className = `chip ${present ? "on" : "off"}`;
    chip.textContent = `${label} ${calib ? "已标定" : "未标定"}${present === undefined ? "" : present ? " · 在座" : " · 无人"}`;
    els.viewChips.appendChild(chip);
  }
}

function renderSeated(seconds) {
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(Math.floor(seconds % 60)).padStart(2, "0");
  els.seatedNow.textContent = `${mm}:${ss}`;
  const limitMin = Number(els.seatedBar.dataset.limit ?? 45) || 45;
  els.seatedBar.max = limitMin * 60;
  els.seatedBar.value = seconds;
}

async function loadConfig() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) return;
    const cfg = await resp.json();
    els.seatedBar.dataset.limit = cfg.long_sitting_minutes;
    els.seatedLimit.textContent = `提醒目标 ${cfg.long_sitting_minutes} 分钟`;
    els.thresholdList.innerHTML = THRESHOLD_LABELS.map(
      ([key, label]) => `<div><dt>${label}</dt><dd>${cfg[key]}</dd></div>`
    ).join("");
  } catch {
    // 阈值展示是辅助信息，加载失败不影响检测
  }
}

async function loadCalibration() {
  try {
    const resp = await fetch("/api/calibration");
    if (!resp.ok) return;
    const data = await resp.json();
    state.calibrated = {};
    for (const [view, vc] of Object.entries(data.views ?? {})) {
      if (vc.calibrated) state.calibrated[view] = vc.baseline;
    }
    renderViewChips(null);
  } catch {
    // 状态展示失败不影响检测，下次标定成功后会自动纠正
  }
}

// ---------- 骨架绘制 ----------

const POSE_CONNECTIONS = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [12, 14], [14, 16],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [0, 7], [0, 8],
];

function drawOverlay(unit, landmarks) {
  const canvas = unit.canvasEl;
  const ctx = canvas.getContext("2d");
  const w = (canvas.width = canvas.clientWidth);
  const h = (canvas.height = canvas.clientHeight);
  ctx.clearRect(0, 0, w, h);
  const pts = mirror(landmarks);
  const visible = (i) => (pts[i].visibility ?? 1) > 0.5;

  ctx.strokeStyle = "rgba(80, 220, 160, 0.9)";
  ctx.lineWidth = 3;
  for (const [a, b] of POSE_CONNECTIONS) {
    if (!visible(a) || !visible(b)) continue;
    ctx.beginPath();
    ctx.moveTo(pts[a].x * w, pts[a].y * h);
    ctx.lineTo(pts[b].x * w, pts[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  for (let i = 0; i < pts.length; i++) {
    if (!visible(i)) continue;
    ctx.beginPath();
    ctx.arc(pts[i].x * w, pts[i].y * h, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

els.startBtn.onclick = start;
els.modeSwitch.onclick = switchMode;
els.units[VIEW_FRONT].calibBtn.onclick = () => calibrate(VIEW_FRONT);
els.units[VIEW_SIDE].calibBtn.onclick = () => calibrate(VIEW_SIDE);
initVoiceToggle();
loadConfig();
loadCalibration();
