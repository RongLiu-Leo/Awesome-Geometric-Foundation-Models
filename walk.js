(function codeWalk(){
  "use strict";

  const esc = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const encodePath = path => path.split("/").map(encodeURIComponent).join("/");
  const linkTo = (id,step) => `#model/${id}/code${step ? "/"+step : ""}`;
  // Each node links to the stage that implements the named operation.
  const paths = {
    dust3r: [["RGB pair","view1, view2",1],["View exchange","Twin decoders",3],["Dense readout","DPT heads",4],["First-camera XYZ","Two pointmaps",5]],
    mast3r: [["RGB pair","view1, view2",1],["View exchange","Stereo decoder",2],["Dense readout","XYZ + descriptors",5],["Optional matches","Reciprocal search",6]],
    vggt: [["RGB views","images",1],["View exchange","Frame / global",2],["Parallel heads","Camera, DPT, tracker",3],["Geometry branch","Depth + pointmaps",4]],
    pi3: [["RGB views","imgs",1],["View exchange","Frame / global",2],["Local geometry","local_points",4],["Scene geometry","Pose × local XYZ",5]],
    mapanything: [["RGB and priors","View dictionaries",2],["View exchange","Scale token + views",3],["Factored geometry","Rays, depth, poses",4],["Metric geometry","pts3d_cam, pts3d",6]],
    da3: [["RGB views","Optional cameras",1],["View exchange","DINOv2 layers",2],["Depth + cameras","DPT / camera head",4],["Exported XYZ","Unproject depth",6]],
    omega: [["RGB views","images",1],["View exchange","Global / registers",3],["Dense readout","Depth + confidence",4],["Demo pointmap","Unproject depth",6]],
    dvlt: [["RGB views","DINOv2 tokens",1],["View exchange","K shared steps",2],["Dense readout","Depth + rays",4],["Direct XYZ","Origin + depth × ray",5]],
    pi4: [["RGB frames","images",1],["View exchange","Geometry loop",2],["Local geometry","Refined pointmaps",3],["Optional motion","Metric 3D tracks",6]]
  };
  function sourceUrl(block) {
    return `https://github.com/${block.repo}/blob/${block.commit}/${encodePath(block.file)}#L${block.start}-L${block.end}`;
  }

  function highlight(line) {
    const tokens = /(#.*$)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(\b(?:as|and|assert|class|def|elif|else|for|from|if|import|in|is|not|or|return|with|yield|None|True|False)\b)|(\b\d+(?:\.\d+)?\b)/g;
    let offset = 0;
    let formatted = "";
    for (const match of line.matchAll(tokens)) {
      const color = match[1] ? "comment" : match[2] ? "string" : match[3] ? "keyword" : "number";
      formatted += esc(line.slice(offset, match.index)) + `<span class="tok-${color}">${esc(match[0])}</span>`;
      offset = match.index + match[0].length;
    }
    return formatted + esc(line.slice(offset));
  }

  function excerpt(block) {
    const lines = block.text.split("\n");
    const numbers = lines.map((_,i)=>block.start+i).join("\n");
    const code = lines.map(highlight).join("\n");
    return `<figure class="code-panel walk-snippet">
      <figcaption><span class="walk-symbol">${esc(block.symbol || block.file.split("/").pop())}</span>
      <a href="${esc(sourceUrl(block))}" target="_blank" rel="noopener noreferrer" title="View these lines on GitHub">
      ${esc(block.file)} <b>L${block.start}–${block.end} ↗</b></a></figcaption>
      <div class="walk-source-scroll" role="region" tabindex="0" aria-label="${esc(block.file)} source code">
        <div class="walk-source-inner"><span class="walk-line-numbers" aria-hidden="true">${numbers}</span><pre><code>${code}</code></pre></div>
      </div></figure>`;
  }

  function tensorList(title, items) {
    return `<section class="walk-tensors"><h3>${title}</h3><ul>${items.map(item=>{
      const split = item.indexOf(" — ");
      const signature = split>=0 ? item.slice(0,split) : item;
      const description = split>=0 ? item.slice(split+3) : "";
      return `<li><code>${esc(signature)}</code>${description ? `<span>${esc(description)}</span>` : ""}</li>`;
    }).join("")}</ul></section>`;
  }

  function sourceCredits(walk) {
    const repos = new Set([walk.repo,...(walk.creditedRepos || [])]);
    for(const step of walk.steps) for(const block of step.blocks) repos.add(block.repo);
    return `<details class="walk-credits"><summary>Source credits and licenses</summary>
      ${Array.from(repos,repo=>{
        const info=window.AtlasWalkData.licenses[repo];
        return `<section><h4>${esc(repo)}</h4>
          <a href="${esc(info.url)}" target="_blank" rel="noopener noreferrer">Upstream license ↗</a>
          ${info.notices.map(notice=>`<details><summary>${esc(notice.name)}</summary><pre>${esc(notice.text)}</pre></details>`).join("")}
        </section>`;
      }).join("")}</details>`;
  }

  function stage(walk, index) {
    const item = walk.steps[index];
    const references = (item.references || []).map(({label,url}) =>
      `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(label)} ↗</a>`).join("");
    return `<article class="walk-stage" id="walk-stage-${index+1}" aria-labelledby="walk-stage-heading-${index+1}">
      <div class="walk-step-count">Step ${index+1} of ${walk.steps.length}</div>
      <h4 id="walk-stage-heading-${index+1}" tabindex="-1">${esc(item.title)}</h4><p>${esc(item.note)}</p>
      ${item.blocks.map(excerpt).join("")}
      ${item.credit ? `<p class="walk-component-credit">${esc(item.credit)}</p>` : ""}
      ${references ? `<div class="walk-references">${references}</div>` : ""}</article>`;
  }

  function panel(walk, key) {
    const all = key==="all";
    const index = /^\d+$/.test(key||"") ? Math.max(0, Math.min(walk.steps.length-1, Number(key)-1)) : 0;
    const itemLinks = walk.steps.map((item,i)=>
      `<li><a href="${linkTo(walk.id,i+1)}"${!all && i===index?' aria-current="step"':""}><span>${i+1}</span>${esc(item.title)}</a></li>`
    ).join("");
    const visible = all ? walk.steps.map((_,i)=>stage(walk,i)).join("") : stage(walk,index);
    const paging = !all ? `<div class="walk-paging">
      ${index>0 ? `<a href="${linkTo(walk.id,index)}">← Previous step</a>` : ""}
      ${index+1<walk.steps.length ? `<a class="walk-next" href="${linkTo(walk.id,index+2)}">Next step →</a>` : `<a class="walk-next" href="${linkTo(walk.id,"all")}">Read all steps →</a>`}</div>` : "";
    return `<nav class="walk-steps" aria-label="Code walkthrough steps"><h3>Forward pass</h3><ol>${itemLinks}</ol>
      <a class="walk-show-all" href="${linkTo(walk.id,"all")}"${all?' aria-current="page"':""}>Read all steps</a></nav>
      <div class="walk-stages">${visible}${paging}</div>`;
  }

  function render(model,key) {
    const walk=window.AtlasWalkData.models[model.id];
    return `<section class="content-section walk" id="original-code" aria-labelledby="walk-heading">
      <div class="section-line"><h2 id="walk-heading" tabindex="-1">Inside the original code</h2>
      <a class="out walk-revision" href="https://github.com/${walk.repo}/tree/${walk.commit}" target="_blank" rel="noopener noreferrer">${esc(walk.repo)} <code>${esc(walk.commit.slice(0,8))}</code> ↗</a></div>
      <nav class="walk-path" aria-label="Trace tensors to source code">${paths[model.id].map(([label,tensor,step])=>
        `<a href="${linkTo(model.id,step)}"><span>${esc(label)}</span><strong>${esc(tensor)}</strong></a>`
      ).join("")}</nav>
      <details class="walk-io"><summary>Input and output tensors <span>Shapes, coordinate frames and scale</span></summary>
        <div class="walk-contract">${tensorList("Inputs",walk.inputs)}${tensorList("Outputs",walk.outputs)}</div>
        <p class="walk-shape-key">B is the batch size and H × W is the image grid. The view axis is S, V, T or N depending on the model. In VGGT’s track output, N counts query points.</p>
      </details>
      <div class="walk-layout" id="walk-flow" tabindex="-1">${panel(walk,key)}</div>
      ${sourceCredits(walk)}</section>`;
  }

  function select(model,key) {
    const host=document.getElementById("walk-flow");
    host.innerHTML=panel(window.AtlasWalkData.models[model.id],key);
  }

  window.AtlasWalk = {render,select};
})();
