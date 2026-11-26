(function roadmapPage() {
  "use strict";
  const chapters = [
    { years: "2023–2024", title: "Start with an image pair", text: "Two images were enough to predict a dense surface. The next question was whether the model could also identify exactly which pixels corresponded.", until: "2024" },
    { years: "2025", title: "Bring the whole scene into one pass", text: "Four papers explored different choices for a set of views. Which image anchors the scene? Can known measurements help? Should the network predict XYZ directly or recover it from rays and depth?", until: "2025" },
    { years: "2026", title: "Spend compute selectively and follow motion", text: "Two concurrent May papers tackled the cost of cross-view reasoning in different ways. In November, π⁴ used repeated computation to reconstruct geometry and track moving points.", until: "2026" }
  ];
  const milestones = {
    dust3r: {
      title: "Put two images in one coordinate frame",
      text: "DUSt3R predicts an XYZ point for each pixel in both images, expressed in the first camera’s frame. This makes dense reconstruction possible without knowing the cameras beforehand. A separate alignment step joins overlapping pairs into a larger scene."
    },
    mast3r: {
      title: "Make pixel matches explicit",
      text: "MASt3R adds a descriptor head to DUSt3R. Alongside a 3D location, each pixel gets a feature vector that can be matched in the other image. Reciprocal matching turns these features into correspondences for tasks such as camera localization.",
      pair: "dust3r", evidence: "https://arxiv.org/abs/2406.09756v1"
    },
    vggt: {
      title: "Let every view inform the reconstruction",
      text: "VGGT alternates attention within each image and across the entire set. Cameras, depth, pointmaps and queried 2D tracks can now come from the same forward pass. Its pointmaps retain DUSt3R’s convention of expressing coordinates in the first camera’s frame.",
      pair: "dust3r", evidence: "https://arxiv.org/html/2503.11651v1#S3.SS1"
    },
    pi3: {
      title: "Remove the privileged first view",
      text: "π³ initializes parts of its network from VGGT, then treats each input view symmetrically. It predicts points in each camera’s local frame along with the pose that places them together. Reordering a photo collection therefore reorders its predictions consistently.",
      pair: "vggt", evidence: "https://arxiv.org/html/2507.13347v3#A1.SS2"
    },
    mapanything: {
      title: "Use measurements when they are available",
      text: "MapAnything combines image features with optional calibration, camera poses or depth before alternating attention across views. A learned scale factor places the reconstruction in metres. This lets the same model use a sparse sensor measurement or work from RGB alone.",
      pair: "vggt", evidence: "https://arxiv.org/html/2509.13414v1#S3.SS1"
    },
    da3: {
      title: "Describe geometry with depth and rays",
      text: "Depth Anything 3 uses a single vision transformer to handle one image or many. A ray specifies where a pixel looks, and depth locates the surface along it. The Main checkpoints recover relative scale, while the Nested checkpoint adds a metric estimate."
    },
    omega: {
      title: "Send fewer tokens across views",
      text: "VGGT-Ω revises VGGT so that selected layers exchange only camera and register tokens. A streamlined head decodes depth. These changes reduce training memory and support training on more static and dynamic scenes.",
      pair: "vggt", evidence: "https://arxiv.org/html/2605.15195v2#S3.SS1"
    },
    dvlt: {
      title: "Reuse a block and choose how long it runs",
      text: "Déjà View takes another route to efficiency. One frame/global attention block refines the scene repeatedly, and the user chooses the number of inference steps. Like Depth Anything 3, it can recover cameras from predicted rays.",
      pair: "da3", evidence: "https://arxiv.org/html/2605.30215v1#S3.SS1"
    },
    pi4: {
      title: "Follow a surface point as it moves",
      text: "π⁴ brings together π³’s reference-free geometry and Déjà View’s shared-weight loops. A separate motion decoder finds the corresponding pixel in later views. Sampling the reconstructed surface at each match yields a metric 3D trajectory.",
      pair: "pi3", evidence: "https://arxiv.org/html/2611.23144v1#S3"
    }
  };

  function render(models, {dateLabel, out, icon, esc}) {
    function entry(m) {
      const note = milestones[m.id];
      const earlier = models.find(model => model.id === note.pair);
      return `<article class="milestone" id="milestone-${m.id}" tabindex="-1" aria-labelledby="milestone-${m.id}-title">
        <div class="milestone-date"><time datetime="${m.preprint}">${dateLabel(m)}</time></div>
        <div class="milestone-copy"><h3 id="milestone-${m.id}-title"><a href="#model/${m.id}">${esc(m.name)}</a></h3>
          <h4>${esc(note.title)}</h4><p>${esc(note.text)}</p>
          <div class="milestone-links"><a href="#model/${m.id}">Architecture ${icon("arrow")}</a>
            <a href="#model/${m.id}/code">Trace the code ${icon("arrow")}</a>
            ${out(m.paper+"v1","arXiv v1")}
            ${note.evidence && note.evidence !== m.paper+"v1" ? out(note.evidence, "Method") : ""}
            ${earlier ? `<a href="#compare/${earlier.id}/${m.id}">Compare with ${esc(earlier.name)} ${icon("arrow")}</a>` : ""}
          </div></div>
      </article>`;
    }
    let priorYear = "";
    const sections = chapters.map(chapter => {
      const selection = models.filter(m => m.preprint.slice(0,4)>priorYear && m.preprint.slice(0,4)<=chapter.until);
      priorYear = chapter.until;
      return `<section class="road-chapter" aria-labelledby="chapter-${chapter.until}"><header>
        <div class="eyebrow muted"><span class="short-rule"></span>${chapter.years}</div>
        <h2 id="chapter-${chapter.until}">${esc(chapter.title)}</h2><p>${esc(chapter.text)}</p></header>
        <div class="milestones">${selection.map(entry).join("")}</div></section>`;
    }).join("");
    return `<div class="container roadmap"><header class="road-intro">
      <div class="eyebrow"><span class="short-rule"></span>Research roadmap <span class="slash">/</span> 2023–2026</div>
      <h1>From image pairs to 4D motion<span class="blue-dot">.</span></h1>
      <p>Follow nine papers from placing pixels in 3D to following a moving point through time. Along the way, the methods branch on how to combine views, recover scale and spend compute.</p>
      </header><section class="road-overview" aria-labelledby="road-dates"><h2 id="road-dates">First arXiv submissions</h2>
      <p>Select a paper to jump to its contribution.</p>${window.AtlasRoadmapChart.render(models)}</section>${sections}
      <div class="ending-links"><a class="button primary" href="#compare/all">${icon("compare")} Compare all nine</a><a class="text-link" href="#roadmap">Back to timeline ${icon("arrow")}</a></div></div>`;
  }
  window.AtlasRoadmap = {render};
})();
