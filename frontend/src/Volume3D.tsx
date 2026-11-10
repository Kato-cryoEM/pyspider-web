import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

interface Props {
  volumeB64: string;
  /** [nz, ny, nx] */
  shape: [number, number, number];
  /** [min, max] of the original (pre-uint8) values. */
  range: [number, number];
}

// Standard MIP volume rendering: box in local [-0.5, 0.5]^3, sample
// sampler3D at (p + 0.5). Requires WebGL2 (default in modern browsers).
const VERT = /* glsl */ `
in vec3 position;
uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 uCameraPos;
out vec3 vOrigin;
out vec3 vDirection;
void main() {
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  vOrigin = vec3(inverse(modelMatrix) * vec4(uCameraPos, 1.0));
  vDirection = position - vOrigin;
}
`;

const FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;
uniform sampler3D uVolume;
uniform float uThreshold;
uniform float uStep;
uniform int uMode;      // 0 = MIP, 1 = ISO shaded
in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;

vec2 hitBox(vec3 orig, vec3 dir) {
  vec3 invDir = 1.0 / dir;
  vec3 tMinTmp = (vec3(-0.5) - orig) * invDir;
  vec3 tMaxTmp = (vec3( 0.5) - orig) * invDir;
  vec3 t1 = min(tMinTmp, tMaxTmp);
  vec3 t2 = max(tMinTmp, tMaxTmp);
  return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

float sampleVol(vec3 p) {
  return texture(uVolume, p + 0.5).r;
}

vec3 gradient(vec3 p) {
  float dx = sampleVol(p + vec3(uStep, 0.0, 0.0)) - sampleVol(p - vec3(uStep, 0.0, 0.0));
  float dy = sampleVol(p + vec3(0.0, uStep, 0.0)) - sampleVol(p - vec3(0.0, uStep, 0.0));
  float dz = sampleVol(p + vec3(0.0, 0.0, uStep)) - sampleVol(p - vec3(0.0, 0.0, uStep));
  return vec3(dx, dy, dz);
}

void main() {
  vec3 rayDir = normalize(vDirection);
  vec2 bounds = hitBox(vOrigin, rayDir);
  if (bounds.x >= bounds.y) discard;
  bounds.x = max(bounds.x, 0.0);
  vec3 p = vOrigin + bounds.x * rayDir;
  vec3 stepVec = rayDir * uStep;

  if (uMode == 1) {
    // Iso-surface: first sample above threshold, then Blinn-Phong shade.
    for (int i = 0; i < 512; i++) {
      float v = sampleVol(p);
      if (v >= uThreshold) {
        vec3 g = gradient(p);
        vec3 n = normalize(-g);
        vec3 lightDir = normalize(vec3(0.6, 0.7, 1.0));
        float diffuse = max(dot(n, lightDir), 0.0);
        vec3 half_ = normalize(lightDir - rayDir);
        float spec = pow(max(dot(n, half_), 0.0), 24.0);
        vec3 base = vec3(0.55, 0.75, 1.0);
        vec3 col = base * (0.25 + 0.75 * diffuse) + vec3(0.5) * spec;
        fragColor = vec4(col, 1.0);
        return;
      }
      p += stepVec;
      if (any(lessThan(p, vec3(-0.5))) || any(greaterThan(p, vec3(0.5)))) break;
    }
    discard;
  } else {
    // MIP: track the running maximum along the ray.
    float mip = 0.0;
    for (int i = 0; i < 512; i++) {
      mip = max(mip, sampleVol(p));
      p += stepVec;
      if (any(lessThan(p, vec3(-0.5))) || any(greaterThan(p, vec3(0.5)))) break;
    }
    if (mip < uThreshold) discard;
    fragColor = vec4(vec3(mip), 1.0);
  }
}
`;

export default function Volume3D({ volumeB64, shape, range }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [mode, setMode] = useState<"mip" | "iso">("iso");
  const uThresholdRef = useRef({ value: 0.5 });
  const uModeRef = useRef({ value: 1 });

  useEffect(() => { uThresholdRef.current.value = threshold; }, [threshold]);
  useEffect(() => { uModeRef.current.value = mode === "iso" ? 1 : 0; }, [mode]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const width = container.clientWidth || 260;
    const height = Math.max(220, Math.floor(width * 0.85));

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0e);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    camera.position.set(1.4, 1.1, 1.4);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;

    // Decode base64 → Uint8Array.
    const raw = atob(volumeB64);
    const data = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) data[i] = raw.charCodeAt(i);

    const [nz, ny, nx] = shape;
    const tex = new THREE.Data3DTexture(data, nx, ny, nz);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.unpackAlignment = 1;
    tex.needsUpdate = true;

    const uStep = 0.5 / Math.max(nx, ny, nz);
    const uniforms = {
      uVolume:   { value: tex },
      uThreshold: uThresholdRef.current,
      uStep:     { value: uStep },
      uMode:     uModeRef.current,
      uCameraPos: { value: new THREE.Vector3() },
    };

    const mat = new THREE.RawShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms,
      vertexShader: VERT,
      fragmentShader: FRAG,
      side: THREE.BackSide,
    });

    // Box scaled so the largest dim = 1; the others shrink accordingly.
    const maxDim = Math.max(nx, ny, nz);
    const scale = new THREE.Vector3(nx / maxDim, ny / maxDim, nz / maxDim);

    const geo = new THREE.BoxGeometry(1, 1, 1);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.scale.copy(scale);
    scene.add(mesh);

    const edges = new THREE.EdgesGeometry(geo);
    const line = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x3b3b45 }),
    );
    line.scale.copy(scale);
    scene.add(line);

    let raf = 0;
    const animate = () => {
      controls.update();
      // camera position in world coords, passed into vertex shader.
      uniforms.uCameraPos.value.copy(camera.position);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      const w = container.clientWidth || width;
      const h = Math.max(220, Math.floor(w * 0.85));
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(container);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      tex.dispose();
      geo.dispose();
      edges.dispose();
      mat.dispose();
      container.innerHTML = "";
    };
  }, [volumeB64, shape[0], shape[1], shape[2]]);

  const [nz, ny, nx] = shape;
  return (
    <div>
      <div ref={containerRef} style={{ width: "100%", background: "#000", borderRadius: 3 }} />
      <div style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 10,
                    color: "var(--muted)", marginTop: 4 }}>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value as "iso" | "mip")}
          style={{ background: "var(--panel2)", color: "var(--text)",
                   border: "1px solid #555", borderRadius: 3, fontSize: 10, padding: "1px 2px" }}
        >
          <option value="iso">isosurface</option>
          <option value="mip">MIP</option>
        </select>
        <span>threshold</span>
        <input
          type="range"
          min={0} max={1} step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ minWidth: 30, fontFamily: "ui-monospace, monospace" }}>
          {threshold.toFixed(2)}
        </span>
      </div>
      <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
        {nz}×{ny}×{nx} · [{range[0].toFixed(3)}, {range[1].toFixed(3)}] · drag to rotate, wheel to zoom
      </div>
    </div>
  );
}
