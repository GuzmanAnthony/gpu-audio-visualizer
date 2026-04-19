import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.162.0/+esm';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/controls/OrbitControls.js/+esm';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/postprocessing/EffectComposer.js/+esm';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/postprocessing/RenderPass.js/+esm';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.162.0/examples/jsm/postprocessing/UnrealBloomPass.js/+esm';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.19/+esm';

import { fragmentShader, vertexShader } from './shaders.js';

const canvas = document.getElementById('scene');
const fileInput = document.getElementById('audio-file');
const sampleButton = document.getElementById('sample-button');
const playButton = document.getElementById('play-button');
const sensitivitySlider = document.getElementById('sensitivity');
const sensitivityValue = document.getElementById('sensitivity-value');
const dropzone = document.getElementById('dropzone');

const ui = {
  fileName: document.getElementById('file-name'),
  duration: document.getElementById('duration'),
  sampleRate: document.getElementById('sample-rate'),
  status: document.getElementById('status'),
  rms: document.getElementById('rms'),
  avgFreq: document.getElementById('avg-freq'),
  bass: document.getElementById('bass'),
  treble: document.getElementById('treble'),
};

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x04060d, 0.08);

const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 0.3, 9.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.enablePan = false;
controls.minDistance = 5.2;
controls.maxDistance = 18;

const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0x9fe8ff, 4, 80);
pointLight.position.set(0, 0, 8);
scene.add(pointLight);

const backLight = new THREE.PointLight(0x7f69ff, 5, 80);
backLight.position.set(-4, 2, -5);
scene.add(backLight);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.3, 0.6, 0.2);
composer.addPass(bloomPass);

const params = {
  colorA: '#9fe8ff',
  colorB: '#7d63ff',
  wireOpacity: 0.96,
  coreOpacity: 0.06,
  bloomStrength: 0.5,
  bloomRadius: 0.3,
  bloomThreshold: 0.35,
  autoRotate: true,
  wireframe: true,
};

const uniforms = {
  uTime: { value: 0 },
  uLevel: { value: 0 },
  uBass: { value: 0 },
  uMid: { value: 0 },
  uTreble: { value: 0 },
  uSensitivity: { value: Number(sensitivitySlider.value) },
  uColorA: { value: new THREE.Color(params.colorA) },
  uColorB: { value: new THREE.Color(params.colorB) },
  uOpacity: { value: params.wireOpacity },
};

const coreUniforms = {
  uTime: uniforms.uTime,
  uLevel: uniforms.uLevel,
  uBass: uniforms.uBass,
  uMid: uniforms.uMid,
  uTreble: uniforms.uTreble,
  uSensitivity: uniforms.uSensitivity,
  uColorA: { value: new THREE.Color('#f5f9ff') },
  uColorB: { value: new THREE.Color(params.colorA) },
  uOpacity: { value: params.coreOpacity },
};

const shellGeometry = new THREE.IcosahedronGeometry(2.65, 32);
const wireMaterial = new THREE.ShaderMaterial({
  uniforms,
  vertexShader,
  fragmentShader,
  transparent: true,
  blending: THREE.AdditiveBlending,
  wireframe: true,
  depthWrite: false,
});

const coreMaterial = new THREE.ShaderMaterial({
  uniforms: coreUniforms,
  vertexShader,
  fragmentShader,
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

const shellMesh = new THREE.Mesh(shellGeometry, wireMaterial);
const coreMesh = new THREE.Mesh(new THREE.IcosahedronGeometry(2.32, 20), coreMaterial);
scene.add(coreMesh, shellMesh);

const starsGeometry = new THREE.BufferGeometry();
const starsCount = 1200;
const starPositions = new Float32Array(starsCount * 3);
for (let i = 0; i < starsCount; i += 1) {
  const radius = THREE.MathUtils.randFloat(16, 50);
  const theta = Math.random() * Math.PI * 2;
  const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
  starPositions[i * 3 + 0] = radius * Math.sin(phi) * Math.cos(theta);
  starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
  starPositions[i * 3 + 2] = radius * Math.cos(phi);
}
starsGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
const starsMaterial = new THREE.PointsMaterial({
  size: 0.07,
  color: 0x9cb9ff,
  transparent: true,
  opacity: 0.5,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});
const stars = new THREE.Points(starsGeometry, starsMaterial);
scene.add(stars);

const gui = new GUI({ title: 'Visualizer Controls' });
gui.addColor(params, 'colorA').name('Primary color').onChange((value) => {
  uniforms.uColorA.value.set(value);
  coreUniforms.uColorB.value.set(value);
});
gui.addColor(params, 'colorB').name('Secondary color').onChange((value) => {
  uniforms.uColorB.value.set(value);
});
gui.add(params, 'wireOpacity', 0.2, 1.0, 0.01).name('Wire opacity').onChange((value) => {
  uniforms.uOpacity.value = value;
});
gui.add(params, 'coreOpacity', 0.05, 0.6, 0.01).name('Core opacity').onChange((value) => {
  coreUniforms.uOpacity.value = value;
});
gui.add(params, 'bloomStrength', 0.1, 3.0, 0.01).name('Bloom strength').onChange((value) => {
  bloomPass.strength = value;
});
gui.add(params, 'bloomRadius', 0.0, 1.2, 0.01).name('Bloom radius').onChange((value) => {
  bloomPass.radius = value;
});
gui.add(params, 'bloomThreshold', 0.0, 1.0, 0.01).name('Bloom threshold').onChange((value) => {
  bloomPass.threshold = value;
});
gui.add(params, 'autoRotate').name('Auto rotate');
gui.add(params, 'wireframe').name('Wireframe').onChange((value) => {
  wireMaterial.wireframe = value;
});

bloomPass.strength = params.bloomStrength;
bloomPass.radius = params.bloomRadius;
bloomPass.threshold = params.bloomThreshold;

const audioElement = new Audio();
audioElement.crossOrigin = 'anonymous';
audioElement.loop = false;
audioElement.preload = 'auto';

audioElement.addEventListener('play', () => {
  ui.status.textContent = 'Playing';
  playButton.textContent = 'Pause';
});

audioElement.addEventListener('pause', () => {
  ui.status.textContent = audioElement.src ? 'Paused' : 'Idle';
  playButton.textContent = 'Play';
});

audioElement.addEventListener('ended', () => {
  playButton.textContent = 'Play';
  ui.status.textContent = 'Ended';
});

audioElement.addEventListener('loadedmetadata', () => {
  const duration = Number.isFinite(audioElement.duration) ? audioElement.duration : 0;
  ui.duration.textContent = `${duration.toFixed(2)} s`;
});

let audioContext = null;
let analyser = null;
let mediaSource = null;
let frequencyData = null;
let timeDomainData = null;
let currentObjectUrl = null;
let precomputedFeatures = null;

async function uploadAndFetchFeatures(file) {
  updateUiStats({ status: 'Processing...' });
  try {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/features', { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    precomputedFeatures = await res.json();
    updateUiStats({ status: 'Ready (precomputed)' });
  } catch (err) {
    console.warn('Feature fetch failed, falling back to live Web Audio:', err);
    precomputedFeatures = null;
    updateUiStats({ status: 'Ready (live)' });
  }
}

function getFrameMetrics(currentTime) {
  const times = precomputedFeatures.times;
  let idx = times.length - 1;
  for (let i = 0; i < times.length; i++) {
    if (times[i] > currentTime) { idx = Math.max(0, i - 1); break; }
  }
  return {
    bass:   precomputedFeatures.bass[idx],
    mid:    precomputedFeatures.mid[idx],
    treble: precomputedFeatures.treble[idx],
    level:  precomputedFeatures.level[idx],
    rms:    precomputedFeatures.level[idx],
    avg:    precomputedFeatures.level[idx],
  };
}

function ensureAudioGraph() {
  if (!audioContext) {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioContextCtor();
  }

  if (!mediaSource) {
    mediaSource = audioContext.createMediaElementSource(audioElement);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.84;
    mediaSource.connect(analyser);
    analyser.connect(audioContext.destination);
    frequencyData = new Uint8Array(analyser.frequencyBinCount);
    timeDomainData = new Uint8Array(analyser.fftSize);
  }
}

function updateUiStats({ fileName, duration, sampleRate, status }) {
  if (fileName !== undefined) ui.fileName.textContent = fileName;
  if (duration !== undefined) ui.duration.textContent = `${duration.toFixed(2)} s`;
  if (sampleRate !== undefined) ui.sampleRate.textContent = `${sampleRate} Hz`;
  if (status !== undefined) ui.status.textContent = status;
}

async function loadAudioSource(source, label) {
  ensureAudioGraph();

  if (currentObjectUrl && currentObjectUrl.startsWith('blob:')) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }

  let url = source;
  if (source instanceof File) {
    url = URL.createObjectURL(source);
    currentObjectUrl = url;
  }

  audioElement.pause();
  audioElement.src = url;
  audioElement.load();

  await new Promise((resolve, reject) => {
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error('Unable to load the selected audio file.'));
    };
    const cleanup = () => {
      audioElement.removeEventListener('loadeddata', onLoaded);
      audioElement.removeEventListener('error', onError);
    };
    audioElement.addEventListener('loadeddata', onLoaded, { once: true });
    audioElement.addEventListener('error', onError, { once: true });
  });

  updateUiStats({
    fileName: label,
    duration: Number.isFinite(audioElement.duration) ? audioElement.duration : 0,
    sampleRate: audioContext.sampleRate,
    status: 'Ready',
  });

  playButton.disabled = false;
}

async function handleFiles(fileList) {
  const file = fileList?.[0];
  if (!file) return;
  try {
    await loadAudioSource(file, file.name);
    await uploadAndFetchFeatures(file);
  } catch (error) {
    console.error(error);
    updateUiStats({ status: 'Load failed' });
  }
}

fileInput.addEventListener('change', (event) => {
  handleFiles(event.target.files);
});

sampleButton.addEventListener('click', async () => {
  try {
    await loadAudioSource('/data/audio/french_ballet_class.wav', 'french_ballet_class.wav');
  } catch (error) {
    console.error(error);
    updateUiStats({ status: 'Sample load failed' });
  }
});

playButton.addEventListener('click', async () => {
  if (!audioElement.src) return;
  ensureAudioGraph();
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  if (audioElement.paused) {
    await audioElement.play();
  } else {
    audioElement.pause();
  }
});

sensitivitySlider.addEventListener('input', () => {
  uniforms.uSensitivity.value = Number(sensitivitySlider.value);
  coreUniforms.uSensitivity.value = Number(sensitivitySlider.value);
  sensitivityValue.textContent = `${Number(sensitivitySlider.value).toFixed(2)}x`;
});

['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (event) => {
  handleFiles(event.dataTransfer.files);
});

window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

const clock = new THREE.Clock();
const smooth = {
  level: 0,
  bass: 0,
  mid: 0,
  treble: 0,
  avg: 0,
  rms: 0,
};

function computeMetrics() {
  if (!analyser || !frequencyData || !timeDomainData || audioElement.paused) {
    smooth.level *= 0.96;
    smooth.bass *= 0.94;
    smooth.mid *= 0.94;
    smooth.treble *= 0.94;
    smooth.avg *= 0.94;
    smooth.rms *= 0.94;
    return smooth;
  }

  analyser.getByteFrequencyData(frequencyData);
  analyser.getByteTimeDomainData(timeDomainData);

  const average = frequencyData.reduce((sum, value) => sum + value, 0) / frequencyData.length / 255;
  const bassSlice = frequencyData.slice(0, 30);
  const midSlice = frequencyData.slice(30, 80);
  const trebleSlice = frequencyData.slice(80);

  const bass = bassSlice.reduce((sum, value) => sum + value, 0) / Math.max(1, bassSlice.length) / 255;
  const mid = midSlice.reduce((sum, value) => sum + value, 0) / Math.max(1, midSlice.length) / 255;
  const treble = trebleSlice.reduce((sum, value) => sum + value, 0) / Math.max(1, trebleSlice.length) / 255;

  let rms = 0;
  for (let i = 0; i < timeDomainData.length; i += 1) {
    const value = (timeDomainData[i] - 128) / 128;
    rms += value * value;
  }
  rms = Math.sqrt(rms / timeDomainData.length);

  smooth.avg = THREE.MathUtils.lerp(smooth.avg, average, 0.18);
  smooth.bass = THREE.MathUtils.lerp(smooth.bass, bass, 0.38);
  smooth.mid = THREE.MathUtils.lerp(smooth.mid, mid, 0.18);
  smooth.treble = THREE.MathUtils.lerp(smooth.treble, treble, 0.16);
  smooth.rms = THREE.MathUtils.lerp(smooth.rms, rms, 0.18);
  smooth.level = THREE.MathUtils.lerp(smooth.level, average * 0.65 + rms * 0.9, 0.2);
  return smooth;
}

function animate() {
  requestAnimationFrame(animate);

  const elapsed = clock.getElapsedTime();
  const metrics = precomputedFeatures ? getFrameMetrics(audioElement.currentTime): computeMetrics();
  const bassPulse = 1 + metrics.bass * 0.35;


  uniforms.uTime.value = elapsed;
  uniforms.uLevel.value = metrics.level;
  uniforms.uBass.value = metrics.bass;
  uniforms.uMid.value = metrics.mid;
  uniforms.uTreble.value = metrics.treble;

  coreUniforms.uTime.value = elapsed;
  coreUniforms.uLevel.value = metrics.level;
  coreUniforms.uBass.value = metrics.bass;
  coreUniforms.uMid.value = metrics.mid;
  coreUniforms.uTreble.value = metrics.treble;

  shellMesh.rotation.y += params.autoRotate ? 0.001 + metrics.bass * 0.012 : 0.001;
  shellMesh.rotation.x += params.autoRotate ? 0.0004 + metrics.bass * 0.004 : 0.0;
  shellMesh.scale.setScalar(bassPulse);
  coreMesh.scale.setScalar(0.9 + metrics.bass * 0.25);
  coreMesh.rotation.y -= 0.0006 + metrics.mid * 0.002;
  coreMesh.rotation.x += 0.0003;
  stars.rotation.y += 0.00035;

  pointLight.intensity = 4 + metrics.bass * 5;
  backLight.intensity = 3 + metrics.bass * 8;
  bloomPass.strength = params.bloomStrength + metrics.level * 0.9 + metrics.bass * 1.2;

  ui.rms.textContent = metrics.rms.toFixed(3);
  ui.avgFreq.textContent = (metrics.avg * 255).toFixed(1);
  ui.bass.textContent = (metrics.bass * 255).toFixed(1);
  ui.treble.textContent = (metrics.treble * 255).toFixed(1);

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

window.addEventListener('resize', onResize);

updateUiStats({ status: 'Idle' });
