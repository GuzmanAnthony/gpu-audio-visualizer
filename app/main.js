import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164.1/examples/jsm/controls/OrbitControls.js";

const audioInput = document.getElementById("audio-file");
const featureInput = document.getElementById("feature-file");
const playButton = document.getElementById("play-button");
const sampleButton = document.getElementById("sample-button");
const sensitivitySlider = document.getElementById("sensitivity");
const sensitivityValue = document.getElementById("sensitivity-value");
const dropzone = document.getElementById("dropzone");
const featureSource = document.getElementById("feature-source");

const fileNameEl = document.getElementById("file-name");
const durationEl = document.getElementById("duration");
const sampleRateEl = document.getElementById("sample-rate");
const statusEl = document.getElementById("status");
const rmsEl = document.getElementById("rms");
const avgFreqEl = document.getElementById("avg-freq");
const bassEl = document.getElementById("bass");
const trebleEl = document.getElementById("treble");

const state = {
  audioElement: new Audio(),
  audioContext: null,
  mediaSource: null,
  analyser: null,
  freqData: null,
  timeData: null,
  isPlaying: false,
  sensitivity: Number(sensitivitySlider.value),
  energy: 0,
  bass: 0,
  mid: 0,
  treble: 0,
  centroid: 0,
  rms: 0,
  featureBundle: null,
  featureScale: {
    bass: 1,
    mid: 1,
    treble: 1,
    rms: 1,
  },
};

state.audioElement.crossOrigin = "anonymous";
state.audioElement.preload = "auto";

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050816);
scene.fog = new THREE.Fog(0x050816, 12, 42);

const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 3.2, 10.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.4;
controls.minDistance = 5;
controls.maxDistance = 18;
controls.target.set(0, 0.6, 0);

const ambientLight = new THREE.AmbientLight(0xdbe6ff, 1.0);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0x8bc6ff, 2.4);
keyLight.position.set(4, 6, 5);
scene.add(keyLight);

const rimLight = new THREE.PointLight(0x7c4dff, 30, 40, 2);
rimLight.position.set(-4, 3, -2);
scene.add(rimLight);

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(18, 96),
  new THREE.MeshStandardMaterial({
    color: 0x081120,
    roughness: 0.95,
    metalness: 0.08,
    transparent: true,
    opacity: 0.92,
  })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -1.6;
scene.add(ground);

const orbGeometry = new THREE.IcosahedronGeometry(1.7, 22);
const orbMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x7c4dff,
  emissive: 0x2f3bff,
  emissiveIntensity: 1.35,
  roughness: 0.16,
  metalness: 0.2,
  transmission: 0.18,
  thickness: 0.8,
  clearcoat: 0.85,
  clearcoatRoughness: 0.18,
});
const orb = new THREE.Mesh(orbGeometry, orbMaterial);
scene.add(orb);

const wireOrb = new THREE.LineSegments(
  new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(2.25, 12)),
  new THREE.LineBasicMaterial({ color: 0x8bc6ff, transparent: true, opacity: 0.24 })
);
scene.add(wireOrb);

const starGeometry = new THREE.BufferGeometry();
const starCount = 1200;
const starPositions = new Float32Array(starCount * 3);
for (let i = 0; i < starCount; i += 1) {
  const radius = 12 + Math.random() * 22;
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(2 * Math.random() - 1);
  const x = radius * Math.sin(phi) * Math.cos(theta);
  const y = radius * Math.cos(phi);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  starPositions.set([x, y, z], i * 3);
}
starGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
const stars = new THREE.Points(
  starGeometry,
  new THREE.PointsMaterial({ color: 0xc4d8ff, size: 0.05, transparent: true, opacity: 0.75 })
);
scene.add(stars);

const waveformCanvas = document.createElement("canvas");
waveformCanvas.width = 768;
waveformCanvas.height = 160;
const waveformContext = waveformCanvas.getContext("2d");
const waveformTexture = new THREE.CanvasTexture(waveformCanvas);
waveformTexture.minFilter = THREE.LinearFilter;
waveformTexture.magFilter = THREE.LinearFilter;

const waveformPlane = new THREE.Mesh(
  new THREE.PlaneGeometry(9, 1.7),
  new THREE.MeshBasicMaterial({ map: waveformTexture, transparent: true, opacity: 0.88, side: THREE.DoubleSide })
);
waveformPlane.position.set(0, -3.15, 0);
scene.add(waveformPlane);

function setStatus(message) {
  statusEl.textContent = message;
}

function setFeatureLabel(message) {
  featureSource.textContent = message;
}

function ensureAudioGraph() {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }

  if (!state.mediaSource) {
    state.mediaSource = state.audioContext.createMediaElementSource(state.audioElement);
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 2048;
    state.analyser.smoothingTimeConstant = 0.78;
    state.mediaSource.connect(state.analyser);
    state.analyser.connect(state.audioContext.destination);
    state.freqData = new Uint8Array(state.analyser.frequencyBinCount);
    state.timeData = new Uint8Array(state.analyser.fftSize);
  }
}

function resetFeatureBundle() {
  state.featureBundle = null;
  state.featureScale = { bass: 1, mid: 1, treble: 1, rms: 1 };
  setFeatureLabel("Feature source: Browser AnalyserNode fallback");
}

function isFeatureBundleValid(bundle) {
  return Boolean(
    bundle &&
      bundle.frames &&
      Array.isArray(bundle.frames.rms) &&
      Array.isArray(bundle.frames.bass) &&
      Array.isArray(bundle.frames.mid) &&
      Array.isArray(bundle.frames.treble) &&
      Array.isArray(bundle.frames.centroid) &&
      bundle.waveform &&
      Array.isArray(bundle.waveform.min) &&
      Array.isArray(bundle.waveform.max)
  );
}

function prepareFeatureBundle(bundle) {
  state.featureBundle = bundle;
  const bassMax = Math.max(1e-6, ...bundle.frames.bass.map((value) => Math.abs(value)));
  const midMax = Math.max(1e-6, ...bundle.frames.mid.map((value) => Math.abs(value)));
  const trebleMax = Math.max(1e-6, ...bundle.frames.treble.map((value) => Math.abs(value)));
  const rmsMax = Math.max(1e-6, ...bundle.frames.rms.map((value) => Math.abs(value)));
  state.featureScale = {
    bass: bassMax,
    mid: midMax,
    treble: trebleMax,
    rms: rmsMax,
  };
  sampleRateEl.textContent = `${bundle.sample_rate} Hz`;
  durationEl.textContent = `${Number(bundle.duration_seconds || 0).toFixed(2)} s`;
  setFeatureLabel(`Feature source: ${bundle.backend || "gpu"} feature JSON`);
}

async function loadFeatureFile(file) {
  const text = await file.text();
  const parsed = JSON.parse(text);
  if (!isFeatureBundleValid(parsed)) {
    throw new Error("This JSON file does not match the exported feature bundle format.");
  }
  prepareFeatureBundle(parsed);
}

function attachAudioSource(src, fileLabel = "Loaded audio") {
  state.audioElement.pause();
  state.audioElement.src = src;
  state.audioElement.load();
  playButton.disabled = false;
  fileNameEl.textContent = fileLabel;
  setStatus("Audio loaded");
}

async function handleAudioFile(file) {
  const url = URL.createObjectURL(file);
  attachAudioSource(url, file.name);
}

function currentFeatureFrame() {
  const bundle = state.featureBundle;
  if (!bundle) {
    return null;
  }
  const frameCount = bundle.frames.rms.length;
  if (!frameCount) {
    return null;
  }
  const hopSize = Number(bundle.hop_size || 1);
  const sampleRate = Number(bundle.sample_rate || 1);
  const frameIndex = Math.max(0, Math.min(frameCount - 1, Math.floor((state.audioElement.currentTime * sampleRate) / hopSize)));
  return {
    index: frameIndex,
    rms: bundle.frames.rms[frameIndex] || 0,
    bass: bundle.frames.bass[frameIndex] || 0,
    mid: bundle.frames.mid[frameIndex] || 0,
    treble: bundle.frames.treble[frameIndex] || 0,
    centroid: bundle.frames.centroid[frameIndex] || 0,
  };
}

function updateFromFeatureBundle() {
  const frame = currentFeatureFrame();
  if (!frame) {
    return false;
  }

  state.rms = frame.rms;
  state.bass = Math.min(1, Math.abs(frame.bass) / state.featureScale.bass);
  state.mid = Math.min(1, Math.abs(frame.mid) / state.featureScale.mid);
  state.treble = Math.min(1, Math.abs(frame.treble) / state.featureScale.treble);
  state.centroid = frame.centroid;
  state.energy = Math.min(1.6, (state.rms / state.featureScale.rms) * 2.0) * state.sensitivity;

  rmsEl.textContent = frame.rms.toFixed(3);
  avgFreqEl.textContent = `${frame.centroid.toFixed(1)} Hz`;
  bassEl.textContent = frame.bass.toFixed(1);
  trebleEl.textContent = frame.treble.toFixed(1);
  sampleRateEl.textContent = `${state.featureBundle.sample_rate} Hz`;
  return true;
}

function updateFromAnalyser() {
  if (!state.analyser || !state.freqData || !state.timeData) {
    return;
  }

  state.analyser.getByteFrequencyData(state.freqData);
  state.analyser.getByteTimeDomainData(state.timeData);

  let bass = 0;
  let mid = 0;
  let treble = 0;
  let weighted = 0;
  let total = 0;
  const length = state.freqData.length;

  for (let i = 0; i < length; i += 1) {
    const value = state.freqData[i] / 255;
    const ratio = i / length;
    if (ratio < 0.15) {
      bass += value;
    } else if (ratio < 0.55) {
      mid += value;
    } else {
      treble += value;
    }
    weighted += value * i;
    total += value;
  }

  let rms = 0;
  for (let i = 0; i < state.timeData.length; i += 1) {
    const centered = (state.timeData[i] - 128) / 128;
    rms += centered * centered;
  }
  rms = Math.sqrt(rms / state.timeData.length);

  state.bass = (bass / Math.max(1, length * 0.15)) * state.sensitivity;
  state.mid = (mid / Math.max(1, length * 0.4)) * state.sensitivity;
  state.treble = (treble / Math.max(1, length * 0.45)) * state.sensitivity;
  state.centroid = total > 0 ? (weighted / total / length) * (state.audioContext?.sampleRate || 48000) * 0.5 : 0;
  state.rms = rms;
  state.energy = (rms * 2.2 + state.bass * 0.7 + state.mid * 0.45 + state.treble * 0.3) * state.sensitivity;

  rmsEl.textContent = rms.toFixed(3);
  avgFreqEl.textContent = `${state.centroid.toFixed(1)} Hz`;
  bassEl.textContent = bass.toFixed(1);
  trebleEl.textContent = treble.toFixed(1);
  if (state.audioContext) {
    sampleRateEl.textContent = `${state.audioContext.sampleRate} Hz`;
  }
}

function drawWaveform() {
  waveformContext.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  waveformContext.fillStyle = "rgba(6, 12, 26, 0.15)";
  waveformContext.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

  waveformContext.strokeStyle = "rgba(139, 198, 255, 0.95)";
  waveformContext.lineWidth = 2;
  waveformContext.beginPath();

  if (state.featureBundle) {
    const minValues = state.featureBundle.waveform.min;
    const maxValues = state.featureBundle.waveform.max;
    const count = Math.min(minValues.length, maxValues.length);
    for (let i = 0; i < count; i += 1) {
      const x = count > 1 ? (i / (count - 1)) * waveformCanvas.width : 0;
      const yMin = waveformCanvas.height * (0.5 - 0.42 * minValues[i]);
      const yMax = waveformCanvas.height * (0.5 - 0.42 * maxValues[i]);
      waveformContext.moveTo(x, yMin);
      waveformContext.lineTo(x, yMax);
    }
  } else if (state.timeData) {
    for (let i = 0; i < state.timeData.length; i += 1) {
      const x = (i / (state.timeData.length - 1)) * waveformCanvas.width;
      const normalized = (state.timeData[i] - 128) / 128;
      const y = waveformCanvas.height * (0.5 - normalized * 0.42);
      if (i === 0) {
        waveformContext.moveTo(x, y);
      } else {
        waveformContext.lineTo(x, y);
      }
    }
  } else {
    waveformContext.moveTo(0, waveformCanvas.height * 0.5);
    waveformContext.lineTo(waveformCanvas.width, waveformCanvas.height * 0.5);
  }

  waveformContext.stroke();
  waveformTexture.needsUpdate = true;
}

function animateOrb(elapsed) {
  const pulse = 1 + Math.min(state.energy, 2.0) * 0.18;
  orb.scale.setScalar(THREE.MathUtils.lerp(orb.scale.x, pulse, 0.08));
  wireOrb.scale.setScalar(THREE.MathUtils.lerp(wireOrb.scale.x, 1 + Math.min(state.energy, 1.8) * 0.22, 0.06));

  orb.rotation.x += 0.004 + state.mid * 0.005;
  orb.rotation.y += 0.006 + state.treble * 0.008;
  wireOrb.rotation.y -= 0.002 + state.bass * 0.01;
  wireOrb.rotation.x += 0.0015;

  const hueShift = THREE.MathUtils.clamp(0.58 + state.bass * 0.06 - state.treble * 0.02, 0.5, 0.72);
  orbMaterial.color.setHSL(hueShift, 0.85, THREE.MathUtils.clamp(0.52 + state.energy * 0.06, 0.45, 0.72));
  orbMaterial.emissive.setHSL(hueShift + 0.08, 0.95, THREE.MathUtils.clamp(0.18 + state.energy * 0.08, 0.15, 0.55));
  orbMaterial.emissiveIntensity = 1.15 + state.energy * 0.6;

  const rimBase = 18 + state.bass * 12 + state.treble * 8;
  rimLight.intensity = THREE.MathUtils.lerp(rimLight.intensity, rimBase, 0.08);
  rimLight.position.x = Math.cos(elapsed * 0.5) * 4.5;
  rimLight.position.z = Math.sin(elapsed * 0.45) * 3.3;

  stars.rotation.y += 0.0003 + state.energy * 0.0005;
}

function render() {
  requestAnimationFrame(render);

  const elapsed = performance.now() * 0.001;
  const usedFeatureBundle = state.featureBundle && state.audioElement.src;

  if (state.isPlaying) {
    const consumed = usedFeatureBundle ? updateFromFeatureBundle() : false;
    if (!consumed) {
      updateFromAnalyser();
    }
  }

  drawWaveform();
  animateOrb(elapsed);
  controls.update();
  renderer.render(scene, camera);
}

async function togglePlayback() {
  if (!state.audioElement.src) {
    return;
  }

  ensureAudioGraph();
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }

  if (!state.isPlaying) {
    await state.audioElement.play();
    state.isPlaying = true;
    playButton.textContent = "Pause";
    setStatus("Playing");
  } else {
    state.audioElement.pause();
    state.isPlaying = false;
    playButton.textContent = "Play";
    setStatus("Paused");
  }
}

function updateSensitivity() {
  state.sensitivity = Number(sensitivitySlider.value);
  sensitivityValue.textContent = `${state.sensitivity.toFixed(2)}x`;
}

async function handleDroppedFiles(files) {
  for (const file of files) {
    if (file.type === "application/json" || file.name.toLowerCase().endsWith(".json")) {
      await loadFeatureFile(file);
      setStatus("CUDA feature JSON loaded");
    } else if (file.type.startsWith("audio/")) {
      await handleAudioFile(file);
    }
  }
}

audioInput.addEventListener("change", async (event) => {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }
  await handleAudioFile(file);
});

featureInput.addEventListener("change", async (event) => {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }
  try {
    await loadFeatureFile(file);
    setStatus("CUDA feature JSON loaded");
  } catch (error) {
    console.error(error);
    setStatus("Feature JSON load failed");
  }
});

sampleButton.addEventListener("click", () => {
  attachAudioSource("../data/audio/french_ballet_class.wav", "french_ballet_class.wav");
});

playButton.addEventListener("click", async () => {
  try {
    await togglePlayback();
  } catch (error) {
    console.error(error);
    setStatus("Playback failed");
  }
});

sensitivitySlider.addEventListener("input", updateSensitivity);

state.audioElement.addEventListener("loadedmetadata", () => {
  if (!Number.isNaN(state.audioElement.duration)) {
    durationEl.textContent = `${state.audioElement.duration.toFixed(2)} s`;
  }
});

state.audioElement.addEventListener("ended", () => {
  state.isPlaying = false;
  playButton.textContent = "Play";
  setStatus("Finished");
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", async (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  try {
    await handleDroppedFiles(event.dataTransfer.files || []);
  } catch (error) {
    console.error(error);
    setStatus("Drop handling failed");
  }
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

updateSensitivity();
resetFeatureBundle();
drawWaveform();
render();
