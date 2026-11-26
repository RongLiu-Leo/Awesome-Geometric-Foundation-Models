(function () {
  'use strict';

  const green = '#456d5d';
  const blue = '#466a7e';
  const orange = '#a55d39';
  const wire = '#76887f';
  const mint = '#dae9dd';
  const sky = '#d9e7ee';
  const sand = '#f3ddc9';
  const paper = '#fafaf4';
  const mono = 'ui-monospace, Consolas, monospace';

  function escapeText(value) {
    return String(value).replace(/[&<>"']/g, char => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[char]);
  }

  function line(d, color = wire, width = 1.2, extra = '') {
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" ${extra}/>`;
  }

  function shape(d, fill, stroke, width = 1.2) {
    return `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round"/>`;
  }

  function circle(x, y, r, fill, stroke = 'none') {
    return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}"/>`;
  }

  function rect(x, y, w, h, fill, stroke, radius = 2) {
    return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="1.1"/>`;
  }

  function label(x, y, text, color = green, size = 8.4) {
    return `<text x="${x}" y="${y}" fill="${color}" text-anchor="middle" font-family="${mono}" font-size="${size}" font-weight="550">${escapeText(text)}</text>`;
  }

  function at(x, y, content, scale = 1) {
    return `<g transform="translate(${x} ${y}) scale(${scale})">${content}</g>`;
  }

  function arrow(x1, y, x2, color = wire) {
    return line(`M${x1} ${y}H${x2}m-3 -2.5 3 2.5-3 2.5`, color);
  }

  function frames(count = 3) {
    let art = '';
    for (let i = count - 1; i >= 0; i--) {
      const scenery = shape('M0 6 35 0 35 27 0 33Z', i ? mint : paper, green)
        + line('M2 28 10 19 16 22 22 14 33 21', green, 1.1)
        + line('m4 29 8-3 7-2 5 1 7-3', green, .7)
        + circle(26, 8, 2, '#b8cdbf', green);
      art += at(i * 6, -i * 4.5, scenery);
    }
    return art;
  }

  function chip(text, x, y, w, fg = green, bg = mint) {
    return rect(x, y, w, 19, bg, fg, 2.7) + label(x + w / 2, y + 12.5, text, fg);
  }

  function tokens(color, fill, count = 5) {
    return Array.from({length: count}, (_, i) => rect(i * 8, 0, 5, 7, i % 3 ? fill : color, color, .7)).join('');
  }

  function depthMap() {
    return shape('M0 8 38 0 38 29 0 37Z', sand, orange)
      + line('M2 31C6 28 8 32 14 22S22 12 27 15s6-2 10-4M1 26c7 0 6 3 12-7s13-11 17-8 5-3 7-4M1 18c4 3 7 0 11-5s8-8 11-8M17 32c4-6 6-13 12-11s6-2 8-3M25 30c4-5 6-5 12-7', orange, .9);
  }

  function rays() {
    return shape('M28 1 40 5 40 25 28 21Z', sky, blue)
      + line('M0 16 28 1M0 16 40 5M0 16 40 25M0 16 28 21M0 16 34 13M0 16 28 11', blue, .85)
      + circle(0, 16, 2, blue);
  }

  function cloud() {
    let art = line('M1 33 17 39 48 26M17 39V31', '#9aa89e', .7);
    for (let z = 0; z < 7; z++) {
      for (let x = 0; x < 11; x++) {
        const h = 14 * Math.exp(-((x - 5) ** 2 / 12 + (z - 3) ** 2 / 7))
          + 3.5 * Math.sin(x * .8 + z * .4);
        art += circle((3 + x * 3.1 + z * 2.5).toFixed(1),
          (24 - x * .45 + z * 1.8 - h).toFixed(1), .87,
          (x + z) % 4 === 0 ? orange : blue);
      }
    }
    return art;
  }

  function camera() {
    return shape('M1 15 25 1 35 6 35 25 25 20Z', sky, blue)
      + line('M1 15 35 6M1 15 35 25M25 1V20', blue, .9)
      + circle(1, 15, 1.5, blue);
  }

  function tracks() {
    const paths = [
      {d:'M0 29C12 23 17 33 30 15s14-12 22-15', color:orange, marks:[[0,29],[16,26],[30,15],[52,0]]},
      {d:'M0 36C14 28 21 35 34 24s15-11 18-12', color:blue, marks:[[0,36],[17,31],[34,24],[52,12]]},
      {d:'M0 21C10 15 20 19 27 13s15-10 20-9', color:green, marks:[[0,21],[16,17],[27,13],[47,4]]}
    ];
    return paths.map(p => line(p.d, p.color, 1.3) + p.marks.map(([x,y]) => circle(x,y,1.75,paper,p.color)).join('')).join('');
  }

  function dualOutput(top, lower, x = 221) {
    return line(`M${x-12} 50h6V29h6m-6 21v21h6`)
      + chip(top, x + 2, 19, 77, orange, sand)
      + chip(lower, x + 2, 61, 77, blue, sky);
  }

  const plates = {
    dvlt() {
      return at(14, 38, frames()) + arrow(65,51,91)
        + chip('shared block', 95,41,94)
        + line('M176 41V28q0-5-5-5h-58q-5 0-5 5v6m-3-3 3 3 3-3', green)
        + label(142,18,'× K steps', green,8) + line('M189 50h20')
        + dualOutput('depth','ray maps') + label(40,88,'RGB views');
    },
    dust3r() {
      return at(12,12,frames(1),.85) + at(12,62,frames(1),.85)
        + arrow(45,28,66) + arrow(45,77,66)
        + chip('ViT',69,19,39) + chip('ViT',69,67,39)
        + arrow(109,28,129) + arrow(109,77,129)
        + chip('decoder',131,19,62,blue,sky) + chip('decoder',131,67,62,blue,sky)
        + line('M150 40l24 24m-4-1 4 1-1-4M175 40l-24 24m0-4 0 4 4-1',blue,1.05)
        + arrow(195,28,217) + arrow(195,77,217)
        + at(218,9,cloud(),.72) + at(218,57,cloud(),.72)
        + line('M260 17h5v69h-5',orange) + label(290,50,'view 1',orange) + label(290,62,'frame',orange);
    },
    mast3r() {
      return at(13,39,frames(2)) + arrow(60,51,81)
        + chip('stereo',84,42,56) + line('M140 51h10V27h14m-14 24v24h14')
        + chip('DPT',164,18,49,orange,sand) + chip('MLP',164,65,49,blue,sky)
        + arrow(215,27,232) + arrow(215,75,232)
        + at(235,5,cloud(),.75) + label(279,33,'XYZ',orange)
        + at(235,60,tokens(blue,sky,6)) + at(235,84,tokens(blue,sky,6))
        + line('M237 68 245 82m8-14 16 14m-8-14-8 14m25-14v14',blue,.8)
        + label(40,88,'image pair');
    },
    vggt() {
      return at(14,38,frames()) + arrow(65,51,82)
        + rect(85,24,104,55,paper,green,4)
        + chip('frame',91,42,40) + chip('global',139,42,43,blue,sky)
        + arrow(131,51,138) + label(137,35,'attention',green,8)
        + line('M189 51h15M205 20v62M205 20h12m-12 21h12m-12 20h12m-12 21h12')
        + chip('cameras',220,10,70,blue,sky) + chip('depth',220,31,70,orange,sand)
        + chip('pointmaps',220,52,70,orange,sand) + chip('tracks',220,73,70,green,mint)
        + label(40,88,'RGB views');
    },
    da3() {
      return at(14,38,frames()) + arrow(65,51,88)
        + chip('DINOv2 ViT',91,41,85) + line('M177 51h16V28h24m-24 23v22h24')
        + at(225,7,depthMap(),.9) + at(226,58,camera(),.92)
        + label(285,29,'depth',orange) + label(285,78,'cameras',blue)
        + label(40,88,'any views');
    },
    pi3() {
      return at(12,9,frames(1),.6) + at(12,39,frames(1),.6) + at(12,69,frames(1),.6)
        + line('M37 20h12v61H37M37 50h12') + arrow(49,50,63)
        + rect(65,21,95,58,paper,green,4)
        + at(76,29,tokens(green,mint,8)) + at(76,63,tokens(green,mint,8))
        + label(112,54,'shared tokens',green,8.5)
        + line('M160 50h9V29h11m-11 21v23h11')
        + chip('local XYZ',181,20,62,orange,sand) + chip('poses',181,63,62,blue,sky)
        + line('M243 29h9v44h-9') + arrow(252,51,262)
        + at(262,32,cloud(),.9) + label(159,96,'order-equivariant');
    },
    mapanything() {
      return at(14,15,frames(2),.85) + chip('priors',12,68,53,blue,sky)
        + line('M57 29h18v20m-10 28h10V49', wire,1.2,'stroke-dasharray="3 2"')
        + arrow(75,49,91) + chip('fusion',94,40,58)
        + arrow(152,49,165) + rect(168,18,70,61,paper,orange,3)
        + label(203,33,'depth · rays',orange,8) + label(203,49,'poses',blue)
        + label(203,65,'scale',orange) + arrow(238,49,251)
        + at(253,26,cloud(),1.05) + label(274,80,'metric XYZ',blue,8);
    },
    omega() {
      const row = y => at(82,y,tokens(green,mint,7)) + rect(143,y-1,9,9,'#a9c5d2',blue,1);
      return at(13,39,frames()) + arrow(65,51,78)
        + row(22) + row(48) + row(74)
        + line('M151 25h14v52h-13m13-26h-13',blue,1.4)
        + arrow(166,51,182) + chip('global / reg.',184,42,64,blue,sky)
        + line('M248 51h8V28h10m-10 23v23h10') + at(269,14,camera(),.8)
        + at(269,61,depthMap(),.8) + label(40,88,'RGB views');
    },
    pi4() {
      return at(13,36,frames()) + arrow(64,50,86) + chip('encoder',89,41,64)
        + line('M153 51h9V24h12m-12 27v23h12') + chip('geometry',174,15,65,orange,sand)
        + chip('motion',174,65,65,blue,sky) + line('M206 35v24m-3-3 3 3 3-3',orange)
        + arrow(240,25,253) + arrow(240,75,253)
        + at(258,4,cloud(),.85) + at(257,59,tracks(),.85)
        + label(41,88,'RGB / video');
    }
  };

  // Groups separated by an arrow are sequential. Sibling labels inside a group
  // are inputs, alternating operations, or parallel outputs as named above them.
  const diagrams = {
    dvlt: {
      title: 'Déjà View', description: 'A shared attention block refines image tokens K times. Two heads decode depth and camera rays.',
      groups: [['Views',['RGB images']],['Encode',['DINOv2']],['Repeat K times',['Frame attention','Global attention']],['Decode',['Depth','Ray maps']]]
    },
    dust3r: {
      title: 'DUSt3R', description: 'Two images share encoder weights. Their decoder streams exchange features. Both dense outputs use the first camera frame.',
      groups: [['Stereo pair',['Image A','Image B']],['Shared weights',['ViT encoder']],['Exchange features',['Decoder A ↔ B']],['First camera frame',['Pointmap A + confidence','Pointmap B + confidence']]]
    },
    mast3r: {
      title: 'MASt3R', description: 'A stereo backbone feeds parallel geometry and descriptor heads. Reciprocal descriptor matching yields pixel pairs.',
      groups: [['Stereo pair',['Image A','Image B']],['Cross-attention',['DUSt3R backbone']],['Parallel heads',['DPT → metric XYZ','MLP → descriptors']],['Descriptor matching',['Reciprocal pixel pairs']]]
    },
    vggt: {
      title: 'VGGT', description: 'Patch and camera tokens alternate between frame and global attention. Separate heads handle camera parameters, depth, points, and query tracks.',
      groups: [['Inputs',['RGB views']],['Tokenize',['Patch tokens','Camera + register tokens']],['Alternate',['Frame attention','Global attention']],['Task heads',['Cameras','Depth · pointmaps','Query-based tracks']]]
    },
    da3: {
      title: 'Depth Anything 3', description: 'Two DPT branches decode depth and an auxiliary ray field. The default camera head reads camera tokens. The exporter unprojects depth with the recovered cameras to form 3D points.',
      groups: [['Inputs',['RGB views','Optional cameras']],['Shared backbone',['DINOv2 transformer']],['Prediction heads',['DPT → depth + ray grid','Camera tokens → cameras']],['Default outputs',['Depth + confidence','Intrinsics + extrinsics']]]
    },
    pi3: {
      title: 'π³', description: 'Shared per-view tokens preserve view symmetry. Local 3D predictions are transformed to a common frame by their predicted camera poses.',
      groups: [['Unordered views',['Shared DINOv2']],['Alternate',['Per-frame attention','All-view attention']],['Parallel predictions',['Local XYZ','Camera poses','Confidence']],['Transform',['Pose × local XYZ']]]
    },
    mapanything: {
      title: 'MapAnything', description: 'Image features fuse with available geometry priors. Depth, view rays, poses and scale compose a metric scene.',
      groups: [['Available inputs',['RGB views','Optional geometry priors']],['Fuse',['Multi-view transformer']],['Factor geometry',['Depth + rays','Poses + scale']],['Compose',['Metric world points']]]
    },
    omega: {
      title: 'VGGT-Ω', description: 'Register attention carries information between frames in part of the backbone. Camera estimates and dense depth describe the scene.',
      groups: [['Inputs',['RGB views']],['Tokenize',['Patch tokens','Camera + 16 registers']],['Mix across views',['Register attention','Global attention retained']],['Predict',['Camera parameters','Dense depth']]]
    },
    pi4: {
      title: 'π⁴', description: 'The geometry branch returns local points and poses in native units plus a metric scale factor. With tracking queries, target-view geometry lifts image correspondences into metric 3D tracks.',
      groups: [['Inputs',['RGB views / video']],['Scene features',['Shared backbone']],['Condition motion',['Geometry branch','Geometry → motion']],['Read out',['Local XYZ · poses · scale','Optional metric 3D tracks','Optional motion masks']]]
    }
  };

  function thumbnail(id) {
    if (!Object.hasOwn(plates, id)) return '';
    const {title, description} = diagrams[id];
    return `<svg class="atlas-diagram-thumb" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 106" preserveAspectRatio="xMidYMid meet" width="320" height="106" style="display:block;width:100%;height:auto" role="img" aria-label="${escapeText(title + '. ' + description)}"><title>${escapeText(title)} architecture</title><desc>${escapeText(description)}</desc>${plates[id]()}</svg>`;
  }

  const flowStyle = `<style>
    .atlas-flow{margin:0;container-type:inline-size;min-width:0}
    .atlas-flow .af-topology{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1.35rem;padding:1.1rem 0;align-items:center}
    .atlas-flow .af-group{min-width:0;position:relative;text-align:center}
    .atlas-flow .af-group+.af-group:before{position:absolute;content:'→';left:-1.35rem;top:50%;width:1.35rem;color:#687c70;line-height:1}
    .atlas-flow .af-label{display:block;font:600 .66rem/1.4 ui-monospace,Consolas,monospace;letter-spacing:.025em;color:#4a6053;margin:0 0 .4rem}
    .atlas-flow .af-node{display:block;border:1px solid #97b4a3;background:#e5eee5;color:#2d5342;border-radius:4px;padding:.4rem .28rem;font:550 .74rem/1.35 ui-monospace,Consolas,monospace;overflow-wrap:anywhere}
    .atlas-flow .af-node+.af-node{margin-top:.18rem}
    .atlas-flow .af-group:nth-child(3) .af-node{background:#e3edf0;border-color:#a3bbc5;color:#355666}
    .atlas-flow .af-group:last-child .af-node{background:#f6e8d9;border-color:#ceac8d;color:#744529}
    .atlas-flow .af-notes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem 1.3rem;list-style:none;padding:0;margin:1rem 0 0}
    .atlas-flow .af-notes li{margin:0;border-top:1px solid #d9dfd7;padding:.65rem 0 0;min-width:0}
    .atlas-flow .af-notes strong{display:block;font-size:.88rem;line-height:1.4;margin-bottom:.18rem}
    .atlas-flow .af-notes span{font-size:.85rem;line-height:1.5;color:#4b5c52}
    @container (max-width:510px){
      .atlas-flow .af-topology{grid-template-columns:1fr;gap:1.1rem;padding:.75rem 0}
      .atlas-flow .af-group{display:grid;grid-template-columns:minmax(64px,.75fr) minmax(0,1.6fr);align-items:center;gap:.5rem;text-align:left}
      .atlas-flow .af-label{margin:0}
      .atlas-flow .af-nodes{min-width:0}
      .atlas-flow .af-group+.af-group:before{content:'↓';left:61%;top:-1.05rem;width:auto}
      .atlas-flow .af-notes{grid-template-columns:1fr;gap:.5rem}
    }
  </style>`;

  function flow(model) {
    if (!model || !Object.hasOwn(diagrams, model.id)) return '';
    const diagram = diagrams[model.id];
    const groups = diagram.groups.map(([heading, nodes]) => {
      const content = nodes.map(node => `<span class="af-node">${escapeText(node)}</span>`).join('');
      return `<div class="af-group"><span class="af-label">${escapeText(heading)}</span><div class="af-nodes">${content}</div></div>`;
    }).join('');
    const notes = (model.stages || []).map(([heading, explanation]) =>
      `<li><strong>${escapeText(heading)}</strong><span>${escapeText(explanation)}</span></li>`
    ).join('');
    return `${flowStyle}<figure class="atlas-flow" aria-label="${escapeText(diagram.title + ' architecture flow')}"><div class="af-topology">${groups}</div>${notes ? '<ol class="af-notes">' + notes + '</ol>' : ''}</figure>`;
  }

  window.AtlasDrawings = Object.freeze({thumbnail, flow});
})();
