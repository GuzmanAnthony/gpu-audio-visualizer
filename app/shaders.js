export const vertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uLevel;
  uniform float uBass;
  uniform float uMid;
  uniform float uTreble;
  uniform float uSensitivity;

  varying float vDisplacement;
  varying vec3 vNormalView;
  varying vec3 vWorldPosition;

  float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.17, 0.23));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);

    float n000 = hash(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash(i + vec3(1.0, 1.0, 1.0));

    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);

    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
  }

  float fbm(vec3 p) {
    float value = 0.0;
    float amplitude = 0.55;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++) {
      value += amplitude * noise(p * frequency);
      frequency *= 2.0;
      amplitude *= 0.5;
    }
    return value;
  }

  void main() {
    vec3 p = position;
    float time = uTime * 0.32;
    float level = clamp(uLevel * uSensitivity, 0.0, 3.0);
    float bass = clamp(uBass * uSensitivity, 0.0, 3.0);
    float mid = clamp(uMid * uSensitivity, 0.0, 3.0);
    float treble = clamp(uTreble * uSensitivity, 0.0, 3.0);

    float baseNoise = fbm(normal * (2.4 + bass * 2.5) + vec3(time));
    float detailNoise = fbm(position * (1.3 + treble * 0.5) - vec3(0.0, time * 1.2, 0.0));
    float ripple = sin((position.y + time * 3.0) * 4.0) * 0.08 * (0.5 + mid);
    float bassSpike = sin(position.x * 6.0 + time * 2.0)
                      * sin (position.z * 6.0 + time * 1.5)
                      * bass * 0.6;

    float displacement = baseNoise * (0.20 + bass * 0.85)
                       + detailNoise * (0.10 + treble * 0.12)
                       + ripple * (0.3 + level * 0.3)
                       + bassSpike;

    vec3 displaced = p + normal * displacement;
    vDisplacement = displacement;
    vNormalView = normalize(normalMatrix * normal);

    vec4 worldPosition = modelMatrix * vec4(displaced, 1.0);
    vWorldPosition = worldPosition.xyz;

    gl_Position = projectionMatrix * viewMatrix * worldPosition;
  }
`;

export const fragmentShader = /* glsl */ `
  uniform vec3 uColorA;
  uniform vec3 uColorB;
  uniform float uOpacity;
  uniform float uLevel;
  uniform float uBass;

  varying float vDisplacement;
  varying vec3 vNormalView;
  varying vec3 vWorldPosition;

  void main() {
    float fresnel = pow(1.0 - abs(dot(normalize(vNormalView), vec3(0.0, 0.0, 1.0))), 1.8);
    float glow = smoothstep(0.02, 0.55, vDisplacement + 0.25) + fresnel * 0.8;
    vec3 color = mix(uColorA, uColorB, clamp(glow + uLevel * 0.2 + uBass * 0.5, 0.0, 1.0));
    gl_FragColor = vec4(color, clamp(uOpacity + fresnel * 0.2, 0.0, 1.0));
  }
`;
