(() => {
  const fields = [
    ["inputs", "Inputs"],
    ["exchange", "How views interact"],
    ["output", "Geometry outputs"],
    ["scale", "Coordinate scale"],
    ["motion", "Tracking and motion"],
    ["strength", "Strengths"],
    ["limit", "Limitations"]
  ];
  const focusedPairs = [
    {
      ids: ["dust3r", "vggt"],
      title: "Pairs or a whole scene?",
      detail: "DUSt3R aligns a graph of overlapping image pairs. VGGT exchanges information across the full view set before decoding geometry."
    },
    {
      ids: ["vggt", "pi4"],
      title: "Where does a track live?",
      detail: "VGGT tracks a query pixel in each image. π⁴ also samples the reconstructed surface at each match to trace a path in 3D."
    },
    {
      ids: ["dvlt", "pi4"],
      title: "What do repeated weights buy?",
      detail: "Déjà View reuses a block to refine geometry with an adjustable step count. π⁴ uses separate loops for geometry and motion."
    }
  ];
  const escapeText = value => String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
  const route = (left, right) => `#compare/${encodeURIComponent(left)}/${encodeURIComponent(right)}`;
  const overviewLabels = ["View fusion", "Geometry", "Scale", "Matches and tracks"];
  const overviewFacts = {
    dvlt: ["Repeated frame and global attention", "Depth and world-space rays", "Relative", "No trajectory head"],
    dust3r: ["Two-view cross-attention", "Two pointmaps in camera one’s frame", "Relative", "Matches derived from geometry"],
    mast3r: ["Two-view cross-attention", "Pointmaps and pixel descriptors", "Learned metric estimate", "Reciprocal descriptor matches"],
    vggt: ["Frame and global attention", "Cameras, depth and pointmaps", "Relative", "2D query tracks"],
    da3: ["Cross-view attention within DINOv2", "Depth and cameras (internal ray grid)", "Relative in Main, metric in Nested", "No trajectory head"],
    pi3: ["Reference-free attention across views", "Local pointmaps and poses", "Relative", "No trajectory head"],
    mapanything: ["Image and geometric inputs, then alternating attention", "Local rays, ray depths and poses", "Learned scene scale, optional metric inputs", "No trajectory head"],
    omega: ["Global and compact register attention", "Cameras and depth", "Normalized scene frame", "Per-frame geometry on dynamic scenes"],
    pi4: ["Separate geometry and motion loops", "Refined local pointmaps and poses", "Learned scene scale", "Dense 2D and metric 3D tracks"]
  };
  let focusSide = "";
  let lastPair = "#compare";

  function modeSwitch(allModels) {
    return `<nav class="comparison-modes" aria-label="Comparison view">
      <a href="${lastPair}"${allModels ? "" : ' aria-current="page"'}>Side by side</a>
      <a href="#compare/all"${allModels ? ' aria-current="page"' : ""}>All models</a>
    </nav>`;
  }

  function overview(models) {
    const headings = overviewLabels.map(label => `<th scope="col" role="columnheader">${label}</th>`).join("");
    const rows = models.map(model => {
      const values = Object.hasOwn(overviewFacts, model.id) ? overviewFacts[model.id] : [model.exchange, model.output, model.scale, model.motion];
      const cells = values.map((value, i) => `<td role="cell"><span class="overview-cell-label" aria-hidden="true">${overviewLabels[i]}</span>${escapeText(value || "Not documented")}</td>`).join("");
      return `<tr role="row"><th scope="row" role="rowheader"><a href="#model/${encodeURIComponent(model.id)}">${escapeText(model.name)} <span aria-hidden="true">→</span></a></th>${cells}</tr>`;
    }).join("");
    return `<section class="container comparison" aria-labelledby="comparison-title">
      <header class="comparison-intro"><h1 id="comparison-title">Compare models</h1>
        <p>See how models combine views, represent geometry and recover motion. Select a model to explore its architecture.</p>
      </header>
      ${modeSwitch(true)}
      <div class="comparison-overview">
        <table role="table" aria-label="Geometric foundation models">
          <thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">Model</th>${headings}</tr></thead>
          <tbody role="rowgroup">${rows}</tbody>
        </table>
      </div>
    </section>`;
  }

  function choosePair(models, leftId, rightId) {
    const left = models.find(model => model.id === leftId)
      || models.find(model => model.id === "dust3r") || models[0];
    const alternatives = models.filter(model => model.id !== left.id);
    const right = alternatives.find(model => model.id === rightId)
      || alternatives.find(model => model.id === "vggt") || alternatives[0];
    return { left, right };
  }

  function column(models, model, side) {
    const label = side === "left" ? "First model" : "Second model";
    const options = models.map(item =>
      `<option value="${escapeText(item.id)}"${item.id === model.id ? " selected" : ""}>${escapeText(item.name)}</option>`
    ).join("");
    return `<th scope="col" role="columnheader" class="comparison-model">
      <label class="comparison-meta" for="compare-${side}">${label}</label>
      <div class="comparison-select-wrap">
        <select id="compare-${side}" data-compare-side="${side}">${options}</select>
        <span aria-hidden="true"></span>
      </div>
      <div class="comparison-source"><span>${escapeText(model.family || model.group)}</span>
        <a href="${escapeText(model.paper)}" aria-label="${escapeText(model.name)} paper">Paper <span aria-hidden="true">↗</span></a>
      </div>
    </th>`;
  }

  function render(models, leftId, rightId) {
    if (leftId === "all") return overview(models);
    if (models.length < 2) return '<section class="container comparison"><h1>Compare models</h1><p>Comparison requires two model profiles.</p><a href="#compare/all">View all models</a></section>';
    const { left, right } = choosePair(models, leftId, rightId);
    lastPair = route(left.id, right.id);
    const rows = fields.filter(([field]) => left[field] || right[field]).map(([field, label]) => `<tr role="row">
      <th scope="row" role="rowheader">${label}</th>
      <td role="cell">${escapeText(left[field] || "Not documented")}</td>
      <td role="cell">${escapeText(right[field] || "Not documented")}</td>
    </tr>`).join("");
    const focusLinks = focusedPairs.filter(pair => pair.ids.every(id => models.some(model => model.id === id))).map(pair => {
      const names = pair.ids.map(id => models.find(model => model.id === id).name);
      return `<article class="comparison-reading">
        <h3>${escapeText(pair.title)}</h3>
        <p>${escapeText(pair.detail)}</p>
        <a href="${route(...pair.ids)}">${names.map(escapeText).join(" / ")} <span aria-hidden="true">→</span></a>
      </article>`;
    }).join("");
    return `<section class="container comparison" data-comparison data-left="${escapeText(left.id)}" data-right="${escapeText(right.id)}" aria-labelledby="comparison-title">
      <header class="comparison-intro">
        <h1 id="comparison-title">Compare models</h1>
        <p>Choose two models to compare how they combine views, recover scale and represent motion.</p>
      </header>
      ${modeSwitch(false)}
      <div class="comparison-matrix">
        <table role="table" aria-label="Architecture and capabilities of ${escapeText(left.name)} and ${escapeText(right.name)}">
          <colgroup><col class="comparison-label-col"><col><col></colgroup>
          <thead role="rowgroup"><tr role="row">
            <th scope="col" role="columnheader" class="comparison-axis"><span class="comparison-meta">Design choice</span></th>
            ${column(models, left, "left")}${column(models, right, "right")}
          </tr></thead>
          <tbody role="rowgroup">${rows}</tbody>
        </table>
      </div>
      ${focusLinks ? `<section class="comparison-readings" aria-labelledby="comparison-reading-title">
        <h2 id="comparison-reading-title">Differences to look for</h2>
        <div class="comparison-reading-grid">${focusLinks}</div>
      </section>` : ""}
    </section>`;
  }

  function bind(models) {
    const panel = document.querySelector("[data-comparison]");
    if (!panel) return;
    panel.onchange = event => {
      const select = event.target;
      if (!select.matches("select[data-compare-side]") || !models.some(model => model.id === select.value)) return;
      let { left, right } = panel.dataset;
      const side = select.dataset.compareSide;
      if (side === "left") {
        if (select.value === right) right = left;
        left = select.value;
      } else {
        if (select.value === left) left = right;
        right = select.value;
      }
      focusSide = side;
      window.location.hash = route(left, right);
    };
    if (focusSide) {
      panel.querySelector(`[data-compare-side="${focusSide}"]`).focus({ preventScroll: true });
      focusSide = "";
    }
  }

  window.AtlasCompare = { render, bind };
})();
