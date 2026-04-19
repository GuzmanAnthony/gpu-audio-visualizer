import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/postprocessing/UnrealBloomPass.js";
import { vertexShader, fragmentShader } from "./shaders.js";

const audioInput = document.getElementById("audio-file");
const featureInput = document.getElementById("feature-file");
const playButton = document.getElementById("play-button");
const sampleButton = document.getElementById("sample-button");
const sensitivitySlider = document.getElementById("sensitivity");
const sensitivityValue = document.getElementById("sensitivity-value");
const dropzone = document.getElementById("dropzone");
const featureSource = document.getElementById("feature-source");

const ui = {
  fileName: document.getElementById("file-name"),
  duration: document.getElementById("duration"),
  sampleRate: document.getElementById("sample-rate"),
  status: document.getElementById("status"),
  rms: document.getElementById("rms"),
  avgFreq: document.getElementById("avg-freq"),
  bass: document.getElementById("bass"),
  treble: document.getElementById("treble"),
};

const state = {
  audioElement: new Audio(),
  audioContext: null,
  mediaSource: null,
  analyser: null,
  frequencyData: null,
  timeDomainData: null,
  featureBundle: null,
  currentObjectUrl: null,
};

state.audioElement.crossOrigin = "anonymous";
state.audioElement.preload = "auto";

function updateUiStats({ fileName, duration, sampleRate, status } = {}) {
  if (typeof fileName === "string") ui.fileName.textContent = fileName;
  if (typeof duration === "number" && Number.isFinite(duration)) ui.duration.textContent = `${duration.toFixed(2)} s`;
  if (typeof sampleRate === "number" && Number.isFinite(sampleRate) && sampleRate > 0) {
    ui.sampleRate.textContent = `${Math.round(sampleRate)} Hz`;
  }
  if (typeof status === "string") ui.status.textContent = status;
}

function setMetricText(metrics) {
  ui.rms.textContent = (metrics.rmsActual ?? 0).toFixed(3);
  ui.avgFreq.textContent = `${(metrics.centroidActual ?? 0).toFixed(1)} Hz`;
  ui.bass.textContent = (metrics.bassActual ?? 0).toFixed(3);
  ui.treble.textContent = (metrics.trebleActual ?? 0).toFixed(3);
}

function ensureAudioGraph() {
  if (state.audioContext) return;

  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  state.audioContext = new AudioContextCtor();
  state.mediaSource = state.audioContext.createMediaElementSource(state.audioElement);
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 2048;
  state.analyser.smoothingTimeConstant = 0.82;
  state.frequencyData = new Uint8Array(state.analyser.frequencyBinCount);
  state.timeDomainData = new Uint8Array(state.analyser.fftSize);

  state.mediaSource.connect(state.analyser);
  state.analyser.connect(state.audioContext.destination);
}

function releaseCurrentObjectUrl() {
  if (state.currentObjectUrl) {
    URL.revokeObjectURL(state.currentObjectUrl);
    state.currentObjectUrl = null;
  }
}

async function loadAudioSource(source, label) {
  releaseCurrentObjectUrl();
  const objectUrl = source instanceof File ? URL.createObjectURL(source) : source;
  if (source instanceof File) state.currentObjectUrl = objectUrl;

  state.audioElement.pause();
  state.audioElement.src = objectUrl;
  state.audioElement.load();

  await new Promise((resolve, reject) => {
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Unable to load the selected audio file."));
    };
    const cleanup = () => {
      state.audioElement.removeEventListener("loadeddata", onLoaded);
      state.audioElement.removeEventListener("error", onError);
    };

    state.audioElement.addEventListener("loadeddata", onLoaded, { once: true });
    state.audioElement.addEventListener("error", onError, { once: true });
  });

  updateUiStats({
    fileName: label,
    duration: Number.isFinite(state.audioElement.duration) ? state.audioElement.duration : 0,
    sampleRate: state.featureBundle?.sampleRate ?? state.audioContext?.sampleRate ?? 0,
    status: state.featureBundle ? "Ready (CUDA feature sync available)" : "Ready",
  });

  playButton.disabled = false;
}

function arrayMax(values, fallback = 1) {
  if (!Array.isArray(values) || values.length === 0) return fallback;
  let maxValue = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric > maxValue) maxValue = numeric;
  }
  return Number.isFinite(maxValue) && maxValue > 0 ? maxValue : fallback;
}

function arrayMin(values, fallback = 0) {
  if (!Array.isArray(values) || values.length === 0) return fallback;
  let minValue = Number.POSITIVE_INFINITY;
  for (const value of values) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && numeric < minValue) minValue = numeric;
  }
  return Number.isFinite(minValue) ? minValue : fallback;
}

function normalizeFeatureBundle(raw) {
  const frames = raw?.frames;
  const waveform = raw?.waveform;
  if (!frames || !waveform) {
    throw new Error("Feature JSON is missing frames or waveform data.");
  }

  const rmsMax = arrayMax(frames.rms, 1);
  const peakMax = arrayMax(frames.peak, 1);
  const bassMax = arrayMax(frames.bass, 1);
  const midMax = arrayMax(frames.mid, 1);
  const trebleMax = arrayMax(frames.treble, 1);
  const centroidMin = arrayMin(frames.centroid, 0);
  const centroidMax = arrayMax(frames.centroid, 1);

  const waveformScale = Math.max(
    Math.abs(arrayMin(waveform.min, -1)),
    Math.abs(arrayMax(waveform.max, 1)),
    1e-6,
  );

  return {
    backend: raw.backend ?? "unknown",
    sampleRate: Number(raw.sample_rate ?? 0),
    duration: Number(raw.duration_seconds ?? 0),
    numFrames: Number(raw.num_frames ?? (frames.times?.length ?? 0)),
    waveformBuckets: Number(raw.waveform_buckets ?? (waveform.times?.length ?? 0)),
    frames: {
      times: Array.isArray(frames.times) ? frames.times.map(Number) : [],
      rms: Array.isArray(frames.rms) ? frames.rms.map(Number) : [],
      peak: Array.isArray(frames.peak) ? frames.peak.map(Number) : [],
      bass: Array.isArray(frames.bass) ? frames.bass.map(Number) : [],
      mid: Array.isArray(frames.mid) ? frames.mid.map(Number) : [],
      treble: Array.isArray(frames.treble) ? frames.treble.map(Number) : [],
      centroid: Array.isArray(frames.centroid) ? frames.centroid.map(Number) : [],
      rmsMax,
      peakMax,
      bassMax,
      midMax,
      trebleMax,
      centroidMin,
      centroidMax,
    },
    waveform: {
      times: Array.isArray(waveform.times) ? waveform.times.map(Number) : [],
      min: Array.isArray(waveform.min) ? waveform.min.map(Number) : [],
      max: Array.isArray(waveform.max) ? waveform.max.map(Number) : [],
      scale: waveformScale,
    },
  };
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function getIndexFromRatio(length, ratio) {
  if (length <= 1) return 0;
  return Math.max(0, Math.min(length - 1, Math.floor(ratio * (length - 1))));
}

const smooth = {
  level: 0,
  bass: 0,
  mid: 0,
  treble: 0,
  rms: 0,
  centroidNorm: 0,
  rmsActual: 0,
  bassActual: 0,
  trebleActual: 0,
  centroidActual: 0,
};

function smoothMetrics(target, lerpFactor) {
  smooth.level = THREE.MathUtils.lerp(smooth.level, target.level, lerpFactor);
  smooth.bass = THREE.MathUtils.lerp(smooth.bass, target.bass, lerpFactor);
  smooth.mid = THREE.MathUtils.lerp(smooth.mid, target.mid, lerpFactor);
  smooth.treble = THREE.MathUtils.lerp(smooth.treble, target.treble, lerpFactor);
  smooth.rms = THREE.MathUtils.lerp(smooth.rms, target.rms, lerpFactor);
  smooth.centroidNorm = THREE.MathUtils.lerp(smooth.centroidNorm, target.centroidNorm, lerpFactor * 0.8);
  smooth.rmsActual = THREE.MathUtils.lerp(smooth.rmsActual, target.rmsActual, lerpFactor);
  smooth.bassActual = THREE.MathUtils.lerp(smooth.bassActual, target.bassActual, lerpFactor);
  smooth.trebleActual = THREE.MathUtils.lerp(smooth.trebleActual, target.trebleActual, lerpFactor);
  smooth.centroidActual = THREE.MathUtils.lerp(smooth.centroidActual, target.centroidActual, lerpFactor * 0.8);
  return smooth;
}

function computeFeatureDrivenMetrics() {
  const bundle = state.featureBundle;
  if (!bundle || bundle.duration <= 0 || bundle.numFrames <= 0) {
    return smoothMetrics({
      level: 0,
      bass: 0,
      mid: 0,
      treble: 0,
      rms: 0,
      centroidNorm: 0,
      rmsActual: 0,
      bassActual: 0,
      trebleActual: 0,
      centroidActual: 0,
    }, 0.08);
  }

  const currentTime = Number.isFinite(state.audioElement.currentTime) ? state.audioElement.currentTime : 0;
  const ratio = clamp01(currentTime / Math.max(bundle.duration, 1e-6));
  const idx = getIndexFromRatio(bundle.frames.times.length, ratio);

  const rmsActual = bundle.frames.rms[idx] ?? 0;
  const peakActual = bundle.frames.peak[idx] ?? 0;
  const bassActual = bundle.frames.bass[idx] ?? 0;
  const midActual = bundle.frames.mid[idx] ?? 0;
  const trebleActual = bundle.frames.treble[idx] ?? 0;
  const centroidActual = bundle.frames.centroid[idx] ?? 0;

  const rmsNorm = clamp01(rmsActual / bundle.frames.rmsMax);
  const peakNorm = clamp01(peakActual / bundle.frames.peakMax);
  const bassNorm = clamp01(bassActual / bundle.frames.bassMax);
  const midNorm = clamp01(midActual / bundle.frames.midMax);
  const trebleNorm = clamp01(trebleActual / bundle.frames.trebleMax);
  const centroidNorm = clamp01((centroidActual - bundle.frames.centroidMin) / Math.max(bundle.frames.centroidMax - bundle.frames.centroidMin, 1e-6));

  return smoothMetrics({
    level: clamp01(rmsNorm * 0.75 + peakNorm * 0.25),
    bass: bassNorm,
    mid: midNorm,
    treble: trebleNorm,
    rms: rmsNorm,
    centroidNorm,
    rmsActual,
    bassActual,
    trebleActual,
    centroidActual,
  }, 0.22);
}

function computeBrowserMetrics() {
  if (!state.analyser || !state.frequencyData || !state.timeDomainData || state.audioElement.paused) {
    return smoothMetrics({
      level: smooth.level * 0.96,
      bass: smooth.bass * 0.95,
      mid: smooth.mid * 0.95,
      treble: smooth.treble * 0.95,
      rms: smooth.rms * 0.96,
      centroidNorm: smooth.centroidNorm * 0.96,
      rmsActual: smooth.rmsActual * 0.96,
      bassActual: smooth.bassActual * 0.96,
      trebleActual: smooth.trebleActual * 0.96,
      centroidActual: smooth.centroidActual * 0.96,
    }, 1.0);
  }

  state.analyser.getByteFrequencyData(state.frequencyData);
  state.analyser.getByteTimeDomainData(state.timeDomainData);

  let totalMagnitude = 0;
  let weightedBins = 0;
  for (let i = 0; i < state.frequencyData.length; i += 1) {
    totalMagnitude += state.frequencyData[i];
    weightedBins += state.frequencyData[i] * i;
  }

  const average = totalMagnitude / Math.max(1, state.frequencyData.length) / 255;
  const bassSlice = state.frequencyData.slice(0, 30);
  const midSlice = state.frequencyData.slice(30, 90);
  const trebleSlice = state.frequencyData.slice(90);

  const sliceAverage = (slice) => slice.reduce((sum, value) => sum + value, 0) / Math.max(1, slice.length) / 255;
  const bass = sliceAverage(bassSlice);
  const mid = sliceAverage(midSlice);
  const treble = sliceAverage(trebleSlice);

  let rms = 0;
  for (let i = 0; i < state.timeDomainData.length; i += 1) {
    const value = (state.timeDomainData[i] - 128) / 128;
    rms += value * value;
  }
  rms = Math.sqrt(rms / state.timeDomainData.length);

  const centroidNorm = clamp01(weightedBins / Math.max(totalMagnitude * state.frequencyData.length, 1e-6));
  const centroidActual = centroidNorm * ((state.audioContext?.sampleRate ?? 48000) / 2);

  return smoothMetrics({
    level: average * 0.65 + rms * 0.9,
    bass,
    mid,
    treble,
    rms,
    centroidNorm,
    rmsActual: rms,
    bassActual: bass,
    trebleActual: treble,
    centroidActual,
  }, 0.18);
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x040816);
scene.fog = new THREE.FogExp2(0x050816, 0.06);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.4, 9.5);

const renderer = new THREE.WebGLRenderer({
  canvas: document.getElementById("scene"),
  antialias: true,
  alpha: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.25, 0.6, 0.2);
bloomPass.threshold = 0.08;
bloomPass.radius = 0.55;
composer.addPass(bloomPass);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 5;
controls.maxDistance = 16;

const ambientLight = new THREE.AmbientLight(0x8ca4ff, 1.2);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0x8f73ff, 5, 30);
pointLight.position.set(4, 5, 6);
scene.add(pointLight);

const backLight = new THREE.PointLight(0x00d8ff, 4, 25);
backLight.position.set(-5, -2, -4);
scene.add(backLight);

const shellUniforms = {
  uTime: { value: 0 },
  uLevel: { value: 0 },
  uBass: { value: 0 },
  uMid: { value: 0 },
  uTreble: { value: 0 },
  uSensitivity: { value: Number(sensitivitySlider.value) },
  uColorA: { value: new THREE.Color("#5b55ff") },
  uColorB: { value: new THREE.Color("#6fe7ff") },
  uOpacity: { value: 0.92 },
};

const coreUniforms = {
  uTime: { value: 0 },
  uLevel: { value: 0 },
  uBass: { value: 0 },
  uMid: { value: 0 },
  uTreble: { value: 0 },
  uSensitivity: { value: Number(sensitivitySlider.value) },
  uColorA: { value: new THREE.Color("#ff4fe3") },
  uColorB: { value: new THREE.Color("#7af7ff") },
  uOpacity: { value: 0.88 },
};

const shellMesh = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1.85, 26),
  new THREE.ShaderMaterial({
    uniforms: shellUniforms,
    vertexShader,
    fragmentShader,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  }),
);
scene.add(shellMesh);

const coreMesh = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1.05, 18),
  new THREE.ShaderMaterial({
    uniforms: coreUniforms,
    vertexShader,
    fragmentShader,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  }),
);
scene.add(coreMesh);

const wireframe = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(2.25, 1)),
  new THREE.LineBasicMaterial({
    color: 0x6f74ff,
    transparent: true,
    opacity: 0.28,
  }),
);
scene.add(wireframe);

const starGeometry = new THREE.BufferGeometry();
const starCount = 1800;
const starPositions = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i += 1) {
  const radius = 14 + Math.random() * 16;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
  starPositions[i * 3 + 1] = radius * Math.cos(phi);
  starPositions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
}
starGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
const stars = new THREE.Points(
  starGeometry,
  new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.03,
    transparent: true,
    opacity: 0.8,
  }),
);
scene.add(stars);

const waveformCanvas = document.createElement("canvas");
waveformCanvas.width = 1024;
waveformCanvas.height = 256;
const waveformContext = waveformCanvas.getContext("2d");
const waveformTexture = new THREE.CanvasTexture(waveformCanvas);
waveformTexture.colorSpace = THREE.SRGBColorSpace;

const waveformPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(7.4, 1.8),
  new THREE.MeshBasicMaterial({
    map: waveformTexture,
    transparent: true,
    opacity: 0.48,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  }),
);
waveformPlane.position.set(0, -3.0, -0.8);
scene.add(waveformPlane);

function drawWaveform() {
  const ctx = waveformContext;
  if (!ctx) return;

  const width = waveformCanvas.width;
  const height = waveformCanvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(4, 8, 22, 0.18)";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(140, 164, 255, 0.14)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height * 0.5);
  ctx.lineTo(width, height * 0.5);
  ctx.stroke();

  ctx.strokeStyle = "rgba(120, 248, 255, 0.95)";
  ctx.lineWidth = 2.2;
  ctx.beginPath();

  if (state.featureBundle?.waveform?.max?.length) {
    const bundle = state.featureBundle;
    const currentTime = Number.isFinite(state.audioElement.currentTime) ? state.audioElement.currentTime : 0;
    const ratio = clamp01(currentTime / Math.max(bundle.duration, 1e-6));
    const bucketIndex = getIndexFromRatio(bundle.waveform.times.length, ratio);
    const span = Math.min(bundle.waveform.max.length, 340);
    const start = Math.max(0, bucketIndex - Math.floor(span / 2));
    const end = Math.min(bundle.waveform.max.length - 1, start + span - 1);
    const denom = Math.max(end - start, 1);

    for (let i = start; i <= end; i += 1) {
      const x = ((i - start) / denom) * width;
      const minY = height * 0.5 - (bundle.waveform.max[i] / bundle.waveform.scale) * height * 0.34;
      const maxY = height * 0.5 - (bundle.waveform.min[i] / bundle.waveform.scale) * height * 0.34;
      if (i === start) ctx.moveTo(x, minY);
      else ctx.lineTo(x, minY);
      ctx.lineTo(x, maxY);
    }

    const playheadX = ((bucketIndex - start) / denom) * width;
    ctx.stroke();
    ctx.strokeStyle = "rgba(255, 94, 228, 0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(playheadX, 0);
    ctx.lineTo(playheadX, height);
    ctx.stroke();
  } else if (state.timeDomainData?.length) {
    for (let i = 0; i < state.timeDomainData.length; i += 1) {
      const x = (i / (state.timeDomainData.length - 1)) * width;
      const normalized = (state.timeDomainData[i] - 128) / 128;
      const y = height * 0.5 - normalized * height * 0.34;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  } else {
    ctx.moveTo(0, height * 0.5);
    ctx.lineTo(width, height * 0.5);
    ctx.stroke();
  }

  waveformTexture.needsUpdate = true;
}

function resetFeatureBundle() {
  state.featureBundle = null;
  featureSource.textContent = "Feature source: Browser AnalyserNode fallback";
  featureSource.dataset.mode = "browser";
  updateUiStats({
    sampleRate: state.audioContext?.sampleRate ?? 0,
    status: state.audioElement.src ? "Ready" : "Idle",
  });
}

async function loadFeatureJsonFile(file) {
  const text = await file.text();
  const raw = JSON.parse(text);
  state.featureBundle = normalizeFeatureBundle(raw);
  featureSource.textContent = `Feature source: ${state.featureBundle.backend.toUpperCase()} feature JSON`;
  featureSource.dataset.mode = "cuda";
  updateUiStats({
    sampleRate: state.featureBundle.sampleRate,
    status: state.audioElement.src ? "Ready (CUDA feature sync available)" : "CUDA features loaded",
  });
}

async function handleFiles(fileList) {
  const file = fileList?.[0];
  if (!file) return;

  try {
    if (file.type === "application/json" || file.name.toLowerCase().endsWith(".json")) {
      await loadFeatureJsonFile(file);
      return;
    }

    await loadAudioSource(file, file.name);
  } catch (error) {
    console.error(error);
    updateUiStats({ status: "Load failed" });
  }
}

audioInput.addEventListener("change", (event) => {
  handleFiles(event.target.files);
});

featureInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    await loadFeatureJsonFile(file);
  } catch (error) {
    console.error(error);
    updateUiStats({ status: "Feature load failed" });
  }
});

sampleButton.addEventListener("click", async () => {
  try {
    await loadAudioSource("/data/audio/french_ballet_class.wav", "french_ballet_class.wav");
  } catch (error) {
    console.error(error);
    updateUiStats({ status: "Sample load failed" });
  }
});

playButton.addEventListener("click", async () => {
  if (!state.audioElement.src) return;
  ensureAudioGraph();
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }

  if (state.audioElement.paused) {
    await state.audioElement.play();
  } else {
    state.audioElement.pause();
  }
});

state.audioElement.addEventListener("play", () => updateUiStats({ status: state.featureBundle ? "Playing with CUDA features" : "Playing" }));
state.audioElement.addEventListener("pause", () => updateUiStats({ status: state.featureBundle ? "Paused with CUDA features" : "Paused" }));
state.audioElement.addEventListener("ended", () => updateUiStats({ status: state.featureBundle ? "Ready (CUDA feature sync available)" : "Ready" }));

sensitivitySlider.addEventListener("input", () => {
  const value = Number(sensitivitySlider.value);
  shellUniforms.uSensitivity.value = value;
  coreUniforms.uSensitivity.value = value;
  sensitivityValue.textContent = `${value.toFixed(2)}x`;
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  handleFiles(event.dataTransfer.files);
});

window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);

  const elapsed = clock.getElapsedTime();
  const metrics = state.featureBundle ? computeFeatureDrivenMetrics() : computeBrowserMetrics();
  const bassPulse = 1 + metrics.bass * 0.34;

  shellUniforms.uTime.value = elapsed;
  shellUniforms.uLevel.value = metrics.level;
  shellUniforms.uBass.value = metrics.bass;
  shellUniforms.uMid.value = metrics.mid;
  shellUniforms.uTreble.value = metrics.treble;

  coreUniforms.uTime.value = elapsed + 0.6;
  coreUniforms.uLevel.value = metrics.level;
  coreUniforms.uBass.value = metrics.bass;
  coreUniforms.uMid.value = metrics.mid;
  coreUniforms.uTreble.value = metrics.treble;

  shellMesh.rotation.y += 0.0016 + metrics.bass * 0.012;
  shellMesh.rotation.x += 0.0004 + metrics.mid * 0.0018;
  shellMesh.scale.setScalar(bassPulse);

  coreMesh.rotation.y -= 0.0012 + metrics.mid * 0.0024;
  coreMesh.rotation.x += 0.0006;
  coreMesh.scale.setScalar(0.92 + metrics.bass * 0.24 + metrics.treble * 0.12);

  wireframe.rotation.y -= 0.0011;
  wireframe.rotation.x += 0.0005;
  stars.rotation.y += 0.00025;

  pointLight.intensity = 4.2 + metrics.bass * 6.5;
  backLight.intensity = 2.6 + metrics.treble * 7.0;
  bloomPass.strength = 1.1 + metrics.level * 1.0 + metrics.bass * 1.15;
  waveformPlane.material.opacity = 0.22 + metrics.rms * 0.45;
  waveformPlane.lookAt(camera.position);

  drawWaveform();
  setMetricText(metrics);

  controls.update();
  composer.render();
}

animate();

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener("resize", onResize);

resetFeatureBundle();
setMetricText({ rmsActual: 0, centroidActual: 0, bassActual: 0, trebleActual: 0 });
drawWaveform();
