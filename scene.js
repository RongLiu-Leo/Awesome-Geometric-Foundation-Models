(function () {
  'use strict';

  // An original, synthetic architectural scan. Coordinates are in local scene units.
  const background = '#111a22';
  const defaultYaw = 0.57;
  const defaultPitch = 0.39;
  const surfaces = {
    roof: [179, 205, 197], fascia: [219, 222, 198], stone: [176, 191, 180],
    side: [119, 154, 152], glass: [79, 156, 160], inside: [65, 110, 118],
    frame: [204, 221, 202], paving: [117, 147, 144], leaves: [103, 155, 134],
    bark: [136, 145, 124], gravel: [93, 123, 121]
  };

  function buildPavilion() {
    const points = [];
    const faces = [];
    let seed = 461123;
    function random() {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 4294967296;
    }
    function place(x, y, z, material, intensity, size) {
      const rgb = surfaces[material];
      const value = Math.min(1.2, intensity * (0.64 + random() * 0.42));
      const color = rgb.map(channel => Math.min(255, Math.round(channel * value)));
      points.push([x, y, z, 'rgb(' + color.join(',') + ')', size]);
    }
    function line(a, b, material, gap = 0.067, intensity = 1.07, size = 1) {
      const steps = Math.max(1, Math.ceil(Math.hypot(...a.map((v, i) => v - b[i])) / gap));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        place(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
          a[2] + (b[2] - a[2]) * t, material, intensity, size);
      }
    }
    function face(a, b, c, material, gap, intensity, solid = true) {
      const u = b.map((v, i) => v - a[i]);
      const v = c.map((value, i) => value - a[i]);
      if (solid) faces.push([a, b, b.map((n, i) => n + v[i]), c]);
      const columns = Math.max(1, Math.ceil(Math.hypot(...u) / gap));
      const rows = Math.max(1, Math.ceil(Math.hypot(...v) / gap));
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < columns; i++) {
          const s = (i + 0.16 + random() * 0.68) / columns;
          const t = (j + 0.16 + random() * 0.68) / rows;
          const x = a[0] + s * u[0] + t * v[0];
          const y = a[1] + s * u[1] + t * v[1];
          const z = a[2] + s * u[2] + t * v[2];
          let light = intensity;
          if (material === 'glass') {
            const reflected = Math.sin(2.24 * x + 1.86 * y - 2.6 * z);
            light *= 0.56 + 0.41 * (0.5 + 0.5 * reflected) + 0.26 * t;
          } else if (material === 'roof' || material === 'stone') {
            light *= 0.92 + 0.08 * Math.sin(z * 0.9 + x * 0.7);
          }
          place(x, y, z, material, light, 0.76 + random() * 0.51);
        }
      }
    }
    function slab(x0, x1, y0, y1, z0, z1, top = 'roof', front = 'fascia', gap = 0.096) {
      face([x0,y1,z0], [x1,y1,z0], [x0,y1,z1], top, gap, 1.13);
      face([x0,y0,z1], [x1,y0,z1], [x0,y1,z1], front, gap * 0.76, 1.15);
      face([x0,y0,z0], [x1,y0,z0], [x0,y1,z0], front, gap, 0.72);
      face([x1,y0,z0], [x1,y0,z1], [x1,y1,z0], front, gap * 0.85, 0.81);
      face([x0,y0,z0], [x0,y0,z1], [x0,y1,z0], front, gap, 0.87);
      const corners = [[x0,y1,z1],[x1,y1,z1],[x1,y1,z0],[x0,y1,z0]];
      corners.forEach((corner, i) => line(corner, corners[(i + 1) % 4], front, 0.062, 1.12, 0.97));
    }
    function glassFront(x0, x1, y0, y1, z, bays) {
      face([x0,y0,z], [x1,y0,z], [x0,y1,z], 'glass', 0.093, 1.02);
      for (let i = 0; i <= bays; i++) {
        const x = x0 + (x1 - x0) * i / bays;
        line([x,y0,z+0.014], [x,y1,z+0.014], 'frame', 0.057, 1.02);
      }
      for (const y of [y0, y1, y0 + (y1 - y0) * 0.73]) {
        line([x0,y,z+0.018], [x1,y,z+0.018], 'frame', 0.064, 0.91, 0.92);
      }
    }
    function glassSide(x, y0, y1, z0, z1, bays) {
      face([x,y0,z0], [x,y0,z1], [x,y1,z0], 'glass', 0.094, 0.83);
      for (let i = 0; i <= bays; i++) {
        const z = z0 + (z1 - z0) * i / bays;
        line([x+0.015,y0,z], [x+0.015,y1,z], 'frame', 0.063, 0.91);
      }
      line([x+0.015,y1-0.42,z0], [x+0.015,y1-0.42,z1], 'frame', 0.065, 0.82);
    }
    function column(x, z, y0, y1) {
      slab(x-0.045, x+0.045, y0, y1, z-0.045, z+0.045, 'stone', 'frame', 0.072);
    }

    // Low western gallery and a recessed, column-lined entrance under its canopy.
    slab(-5.0, 1.07, 0.04, 0.34, -1.45, 2.16, 'paving', 'stone');
    glassFront(-4.68, -1.46, 0.34, 1.95, 1.59, 6);
    glassFront(-1.46, 0.68, 0.34, 1.95, 0.51, 3);
    glassSide(-1.46, 0.34, 1.95, 0.51, 1.59, 2);
    face([-4.68,0.34,-1.22], [-4.68,0.34,1.59], [-4.68,1.95,-1.22], 'stone', 0.071, 0.83);
    face([-4.68,0.34,-1.22], [0.68,0.34,-1.22], [-4.68,1.95,-1.22], 'inside', 0.10, 0.79);
    slab(-5.13, 1.11, 1.95, 2.13, -1.49, 2.08);
    for (const x of [-4.82,-2.65,-0.46,0.87]) column(x, 1.91, 0.34, 1.95);
    for (let i = 0; i < 4; i++) {
      const front = 2.18 + i * 0.22;
      slab(-2.0, 0.70, 0.012, 0.30 - i * 0.072, front, front + 0.23, 'paving', 'stone');
    }

    // Raised reading room, continuous glazing and a wide cantilever over the terrace.
    slab(0.33, 4.24, 0.03, 1.02, -1.84, 0.60, 'side', 'side');
    slab(0.03, 4.98, 1.02, 1.20, -2.34, 1.63, 'paving', 'fascia');
    for (const x of [0.22,2.43,4.69]) column(x, 1.42, 0.02, 1.02);
    glassFront(0.31, 4.61, 1.20, 3.15, 1.34, 7);
    glassSide(4.61, 1.2, 3.15, -2.01, 1.34, 5);
    glassSide(0.31, 1.2, 3.15, -2.01, 1.34, 4);
    face([0.31,1.2,-2.01],[4.61,1.2,-2.01],[0.31,3.15,-2.01], 'stone', 0.085, 0.76);
    slab(-0.04, 5.04, 3.15, 3.34, -2.39, 1.73);
    // Raised clerestory and a ribbed roof monitor give the roof a legible silhouette.
    slab(0.79, 3.52, 3.34, 3.69, -1.74, -0.79, 'roof', 'side', 0.081);
    slab(0.67, 3.62, 3.69, 3.79, -1.85, -0.69, 'roof', 'fascia');
    for (let x = 0.86; x < 3.5; x += 0.17) {
      line([x,3.36,-0.776], [x,3.65,-0.776], 'frame', 0.068, 0.83, 0.88);
    }
    // A few sheltered garden trees add organic density without hiding the facade.
    function tree(x, z, height, radius) {
      line([x,0,z], [x,height*0.8,z], 'bark', 0.040, 1, 1.23);
      for (let branch = 0; branch < 8; branch++) {
        const angle = random() * Math.PI * 2;
        const bx = Math.cos(angle) * radius * (0.32 + random() * 0.48);
        const bz = Math.sin(angle) * radius * (0.28 + random() * 0.48);
        const by = height * (0.60 + random() * 0.29);
        line([x,height*0.43,z], [x+bx,by,z+bz], 'bark', 0.052, 0.76, 0.88);
        for (let i = 0; i < 96; i++) {
          const theta = random() * Math.PI * 2;
          const latitude = random() * 2 - 1;
          const distance = Math.cbrt(random()) * radius * 0.61;
          const ring = distance * Math.sqrt(1 - latitude * latitude);
          place(x+bx+ring*Math.cos(theta), by+latitude*distance*0.82, z+bz+ring*Math.sin(theta),
            'leaves', 0.71 + 0.29 * latitude, 0.7 + random() * 0.56);
        }
      }
    }
    tree(-5.53, -0.72, 2.58, 0.86);
    tree(4.25, -3.12, 2.24, 0.71);
    tree(-5.02, -2.58, 1.86, 0.64);
    for (let i = 0; i < 1950; i++) {
      const x = random()*13-6.5, z = random()*8.6-4.3;
      if (Math.abs(x/6.7)**3 + Math.abs(z/4.4)**3 > random()*0.84+0.17) continue;
      place(x,-0.025,z,'gravel',0.35+random()*0.27,0.63+random()*0.40);
    }
    return {points, faces};
  }

  // A low-resolution depth prepass keeps the scan's back wall from shining through the front.
  function coverTriangle(a, b, c, depths, width, height) {
    const denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1]);
    if (Math.abs(denominator) < 0.001) return;
    const left = Math.max(0, Math.floor(Math.min(a[0],b[0],c[0])));
    const right = Math.min(width-1, Math.ceil(Math.max(a[0],b[0],c[0])));
    const top = Math.max(0, Math.floor(Math.min(a[1],b[1],c[1])));
    const bottom = Math.min(height-1, Math.ceil(Math.max(a[1],b[1],c[1])));
    for (let y = top; y <= bottom; y++) {
      for (let x = left; x <= right; x++) {
        const w0 = ((b[1]-c[1])*(x+0.5-c[0]) + (c[0]-b[0])*(y+0.5-c[1]))/denominator;
        const w1 = ((c[1]-a[1])*(x+0.5-c[0]) + (a[0]-c[0])*(y+0.5-c[1]))/denominator;
        if (w0 < 0 || w1 < 0 || w0+w1 > 1) continue;
        const depth = w0*a[2] + w1*b[2] + (1-w0-w1)*c[2];
        const index = y*width+x;
        if (depth > depths[index]) depths[index] = depth;
      }
    }
  }

  let pavilion;
  function mount(canvas) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return function () {};
    if (!pavilion) pavilion = buildPavilion();
    let yaw = defaultYaw, pitch = defaultPitch, pending = 0, pointer = null;
    let width = 1, height = 1;
    function draw() {
      pending = 0;
      const box = canvas.getBoundingClientRect();
      width = Math.round(box.width); height = Math.round(box.height);
      if (!width || !height) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(width*dpr)) canvas.width = Math.round(width*dpr);
      if (canvas.height !== Math.round(height*dpr)) canvas.height = Math.round(height*dpr);
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.fillStyle = background; ctx.fillRect(0,0,width,height);
      const scale = Math.min(width/15.45, height/7.70);
      const sinYaw = Math.sin(yaw), cosYaw = Math.cos(yaw);
      const sinPitch = Math.sin(pitch), cosPitch = Math.cos(pitch);
      function project(x,y,z) {
        const forward = x*sinYaw+z*cosYaw;
        const depth = forward*cosPitch+(y-1.32)*sinPitch;
        const size = scale/(1-depth/58);
        return [width*0.5+(x*cosYaw-z*sinYaw)*size,
          height*0.49+(forward*sinPitch-(y-1.32)*cosPitch)*size, depth];
      }
      function stroke(a,b) {
        const start = project(...a), end = project(...b);
        ctx.beginPath(); ctx.moveTo(start[0],start[1]); ctx.lineTo(end[0],end[1]); ctx.stroke();
      }
      ctx.lineWidth = 0.65;
      for (let i = -12; i <= 12; i++) {
        ctx.strokeStyle = i%4 === 0 ? '#25343a' : '#1b292f';
        stroke([i,-0.07,-11],[i,-0.07,11]); stroke([-12,-0.07,i],[12,-0.07,i]);
      }
      const depthW = Math.ceil(width*0.66), depthH = Math.ceil(height*0.66);
      const depths = new Float32Array(depthW*depthH); depths.fill(-Infinity);
      for (const surface of pavilion.faces) {
        const polygon = surface.map(vertex => project(...vertex));
        ctx.fillStyle = background; ctx.beginPath();
        polygon.forEach((p, i) => { if (i) ctx.lineTo(p[0],p[1]); else ctx.moveTo(p[0],p[1]); });
        ctx.closePath(); ctx.fill();
        const plane = polygon.map(p => [p[0]*0.66, p[1]*0.66, p[2]]);
        coverTriangle(plane[0],plane[1],plane[2],depths,depthW,depthH);
        coverTriangle(plane[0],plane[2],plane[3],depths,depthW,depthH);
      }
      const dotScale = Math.min(1.23, Math.max(0.76,scale/31));
      for (const p of pavilion.points) {
        const [x,y,d] = project(p[0],p[1],p[2]);
        if (x<0 || x>=width || y<0 || y>=height) continue;
        const depth = depths[Math.floor(y*0.66)*depthW+Math.floor(x*0.66)];
        if (d < depth-0.11) continue;
        const size = p[4]*dotScale;
        ctx.fillStyle = p[3]; ctx.fillRect(x-size/2,y-size/2,size,size);
      }
      // Survey cameras point their image planes at the same synthetic specimen.
      const cameras = [[-5.77,1.12,3.12], [4.43,1.13,4.49], [6.49,2.17,-1.52]];
      ctx.lineWidth = 1.12;
      for (const camera of cameras) {
        const [x,y,z] = camera;
        const length = Math.hypot(x,z);
        const dx = -x/length, dz = -z/length;
        const cx = x+dx*0.51, cz = z+dz*0.51;
        const rect = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([h,v]) =>
          [cx+dz*h*0.29,y+v*0.2,cz-dx*h*0.29]);
        ctx.strokeStyle = '#e48655';
        rect.forEach((corner,i) => { stroke(camera,corner); stroke(corner,rect[(i+1)%4]); });
        const pos = project(x,y,z);
        ctx.fillStyle = '#ffc29b'; ctx.fillRect(pos[0]-1.1,pos[1]-1.1,2.2,2.2);
      }
    }
    function queue() { if (!pending) pending = window.requestAnimationFrame(draw); }
    function changePitch(amount) { pitch = Math.max(0.08,Math.min(1.0, pitch+amount)); }
    function reset() { yaw = defaultYaw; pitch = defaultPitch; queue(); }
    function down(event) {
      if (event.button !== 0 || pointer) return;
      pointer = {id:event.pointerId,x:event.clientX,y:event.clientY};
      canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing';
    }
    function move(event) {
      if (!pointer || pointer.id !== event.pointerId) return;
      yaw -= (event.clientX-pointer.x) * 0.008;
      changePitch((event.clientY-pointer.y)*0.005);
      pointer.x = event.clientX; pointer.y = event.clientY; queue();
    }
    function up(event) {
      if (!pointer || pointer.id !== event.pointerId) return;
      pointer = null; canvas.style.cursor = 'grab';
    }
    function key(event) {
      switch (event.key) {
        case 'ArrowLeft': yaw -= 0.14; break;
        case 'ArrowRight': yaw += 0.14; break;
        case 'ArrowUp': changePitch(0.09); break;
        case 'ArrowDown': changePitch(-0.09); break;
        case 'Home': reset(); break;
        default: return;
      }
      event.preventDefault(); queue();
    }
    if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
    if (!canvas.hasAttribute('aria-label')) {
      canvas.setAttribute('aria-label', 'Synthetic pavilion point cloud. Drag or use the arrow keys to orbit. Double-click or press Home to reset.');
    }
    canvas.style.touchAction = 'none'; canvas.style.cursor = 'grab';
    const handlers = {pointerdown:down,pointermove:move,pointerup:up,pointercancel:up,lostpointercapture:up,dblclick:reset,keydown:key};
    Object.entries(handlers).forEach(([name,fn]) => canvas.addEventListener(name,fn));
    const observer = new ResizeObserver(queue); observer.observe(canvas); draw();
    return function cleanup() {
      observer.disconnect(); window.cancelAnimationFrame(pending);
      Object.entries(handlers).forEach(([name,fn]) => canvas.removeEventListener(name,fn));
    };
  }
  window.AtlasScene = {mount};
}());
