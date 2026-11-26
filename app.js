(function atlas(){
  "use strict";
  const projectTitle = document.title;
  const models = [...window.AtlasData].sort((a,b)=>a.preprint.localeCompare(b.preprint));
  const dateFormat = new Intl.DateTimeFormat("en-GB",{day:"numeric",month:"short",year:"numeric",timeZone:"UTC"});
  const dateLabel = m => dateFormat.format(new Date(m.preprint+"T00:00:00Z"));
  const byId = Object.fromEntries(models.map(model => [model.id, model]));
  const esc = value => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
  const icons = {
    arrow: '<path d="M4 10h11m-4-4 4 4-4 4"/>',
    down: '<path d="M10 3v13m-5-5 5 5 5-5"/>',
    external: '<path d="M5 15 15 5M6 5h9v9"/>',
    compare: '<rect x="2" y="4" width="6" height="12" rx="1"/><rect x="12" y="4" width="6" height="12" rx="1"/>',
    code: '<path d="m6 6-4 4 4 4m8-8 4 4-4 4M11.5 4l-3 12"/>',
    book: '<path d="M4 2h9l3 3v13H4zM12 2v4h4M7 10h6m-6 3h6"/>',
    globe: '<circle cx="10" cy="10" r="7.5"/><ellipse cx="10" cy="10" rx="3.4" ry="7.5"/><path d="M3 10h14"/>',
    orbit: '<ellipse cx="10" cy="10" rx="8" ry="3.2" transform="rotate(-28 10 10)"/><circle cx="10" cy="10" r="2.2"/>',
    search: '<circle cx="8.5" cy="8.5" r="5.5"/><path d="m13 13 4.5 4.5"/>',
    reset: '<path d="M3.5 8a6.7 6.7 0 1 1-.1 4M3 3v5h5"/>',
    back: '<path d="M16 10H4m5-5-5 5 5 5"/>',
    grid: '<path d="M3 3h5v5H3zM12 3h5v5h-5zM3 12h5v5H3zM12 12h5v5h-5z"/>',
    layers: '<path d="m10 2 8 5-8 5-8-5zM2 10l8 5 8-5m-16 4 8 4 8-4"/>',
    plus: '<path d="M3 10h14m-7-7v14"/>',
    minus: '<path d="M3 10h14"/>'
  };
  function icon(id){return `<svg class="icon" width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[id]}</svg>`;}
  const fileUrl = (m) => m.code.split("/blob/")[0];
  const repoName = (m) => fileUrl(m).replace("https://github.com/","");
  const main = document.querySelector("main");
  let releaseScene = null;
  let activeGroup = "All models";
  let searchTerm = "";
  let previousPage = "";
  let previousModel = "";
  const groups = { "All models": models.map(m=>m.id), "Pairwise": ["dust3r", "mast3r"], "Joint multi-view": ["dvlt", "vggt", "da3", "pi3", "mapanything", "omega"], "4D motion": ["pi4"] };
  const aliases = { dvlt:"dvlt deja view",pi3:"pi3 pi^3",pi4:"pi4 pi^4",omega:"vggt omega",da3:"da3" };
  function out(url,label,extra="",leading=""){
    return `<a class="out ${extra}" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${leading ? icon(leading) : ""}${esc(label)} ${icon("external")}</a>`;
  }
  function card(model){
    return `<a class="model-card ${model.color}" href="#model/${model.id}">
      <div class="card-plate">${AtlasDrawings.thumbnail(model.id)}</div>
      <div class="card-copy"><div class="card-meta"><span><time datetime="${model.preprint}">${dateLabel(model)}</time></span><span>${esc(model.family)}</span></div>
      <div class="card-title"><h3>${esc(model.name)}</h3>${icon("arrow")}</div>
      <p>${esc(model.summary)}</p></div></a>`;
  }
  function cards(){
    const normalized = value => value.normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase();
    const query = normalized(searchTerm.trim());
    const filtered = models.filter(m => groups[activeGroup].includes(m.id) && normalized([m.name,m.family,m.title,m.venue,aliases[m.id]||""].join(" ")).includes(query));
    document.querySelector("#model-cards").innerHTML = filtered.length ? filtered.map(card).join("") : `<div class="empty"><h3>No matching models</h3><p>Try another name or view the full index.</p><button id="clear-search" class="button secondary">Clear filters</button></div>`;
    document.querySelector("#index-count").textContent = `${filtered.length} ${filtered.length === 1 ? "model" : "models"}`;
    document.querySelectorAll(".filter").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.group===activeGroup)));
    document.querySelector("#clear-search")?.addEventListener("click",()=>{activeGroup="All models"; searchTerm="";document.querySelector("#model-search").value=""; cards();document.querySelector("#model-search").focus();});
  }
  function overview(){
    return `<div class="container">
      <section class="hero" aria-labelledby="hero-title">
        <div class="hero-copy"><div class="eyebrow"><span class="short-rule"></span>Computer vision</div>
        <h1 id="hero-title">Geometric<br>foundation models<span class="blue-dot">.</span></h1>
        <p>An illustrated guide to nine models that reconstruct scenes from images. Explore their architectures, trace the original code, and compare their strengths and limitations.</p>
        <div class="hero-actions"><a class="button primary" href="#models">Explore models ${icon("down")}</a><a class="button secondary" href="#compare">${icon("compare")} Compare models</a></div></div>
        <figure class="scene"><div class="scene-area"><canvas id="scene" tabindex="0" role="img" aria-label="Interactive illustrative point cloud of a pavilion. Drag or use arrow keys to orbit. Press Home to reset."></canvas><div class="scene-label"><i></i> Point cloud</div><div class="scene-orbit">${icon("orbit")} Drag to orbit</div></div>
        <figcaption><span class="caption-name">Illustrative reconstruction</span><span class="caption-legend"><i></i> Camera frusta</span><button id="reset-view" title="Reset view" aria-label="Reset point cloud view">${icon("reset")}</button></figcaption></figure>
      </section>
      <section id="model-index" class="index-section" aria-labelledby="index-title"><div class="index-title-row"><div><div class="eyebrow muted"><span class="short-rule"></span>Architecture library</div><h2 id="index-title">Model index</h2></div><span class="count" id="index-count" aria-live="polite"></span></div>
      <p class="index-chronology">Ordered by first arXiv submission. <a href="#roadmap">Explore the research roadmap ${icon("arrow")}</a></p>
      <div class="index-tools"><div class="filters" role="group" aria-label="Filter by architecture">${Object.keys(groups).map(g=>`<button type="button" class="filter" data-group="${g}">${g}</button>`).join("")}</div>
      <label class="search-box">${icon("search")}<input id="model-search" type="search" aria-label="Search models" placeholder="Search models" value="${esc(searchTerm)}"></label></div>
      <div class="cards" id="model-cards"></div></section>
      <section class="shared" aria-labelledby="shared-title"><div class="shared-heading"><div class="eyebrow muted"><span class="short-rule"></span>Across the nine works</div><h2 id="shared-title">What they share</h2><p>Every model starts with RGB images and learns to connect visual features to scene geometry. Their choices diverge at three steps.</p><a class="text-link" href="#compare">Compare architectures ${icon("arrow")}</a></div>
      <ol class="shared-notes">
        <li><h3>How views communicate</h3><p>DUSt3R and MASt3R work on pairs and align them afterward. Joint models such as VGGT exchange information across the full view set. Déjà View reuses the same weights for that exchange.</p><a href="#compare/dust3r/vggt">DUSt3R and VGGT ${icon("arrow")}</a></li>
        <li><h3>Where a pixel lands in 3D</h3><p>A pointmap stores XYZ at each pixel. A depth-and-ray model builds XYZ from a camera ray and a distance. For measurements in metres, the reconstruction also needs a metric scale.</p><a href="#compare/pi3/mapanything">π³ and MapAnything ${icon("arrow")}</a></li>
        <li><h3>What persists through time</h3><p>Camera poses describe the camera’s motion. Following a moving object requires identifying the same surface point in later frames. π⁴ lifts those correspondences into 3D trajectories.</p><a href="#compare/omega/pi4">VGGT-Ω and π⁴ ${icon("arrow")}</a></li>
      </ol></section></div>`;
  }
  function bindOverview(){
    cards();
    document.querySelectorAll(".filter").forEach(button=>button.addEventListener("click",()=>{activeGroup=button.dataset.group;cards();}));
    document.querySelector("#model-search").addEventListener("input",event=>{searchTerm=event.target.value;cards();});
    const canvas = document.querySelector("#scene");
    releaseScene = AtlasScene.mount(canvas);
    document.querySelector("#reset-view").addEventListener("click",()=>canvas.dispatchEvent(new KeyboardEvent("keydown",{key:"Home"})));
  }
  function detail(m, step){
    const other = ({dvlt:"vggt",dust3r:"mast3r",mast3r:"dust3r",vggt:"omega",pi3:"pi4",pi4:"pi3",omega:"vggt",da3:"mapanything",mapanything:"da3"})[m.id];
    const next = models[models.indexOf(m)+1];
    return `<div class="container reading">
      <a class="back" href="#models">${icon("back")} Model index</a>
      <header class="detail-header"><div><div class="eyebrow"><span class="short-rule"></span>${esc(m.family)} <span class="slash">/</span> ${esc(m.venue)}</div><h1>${esc(m.name)}</h1><p>${esc(m.summary)}</p><div class="detail-links"><a href="#model/${m.id}/code" class="text-link">${icon("code")} Trace the code ${icon("down")}</a> ${out(m.paper,"Paper","","book")} ${out(fileUrl(m),"Repository","","code")}</div></div><div class="detail-plate ${m.color}">${AtlasDrawings.thumbnail(m.id)}</div></header>
      <section class="content-section"><div class="section-line"><h2>Model architecture</h2>${m.figure ? out(m.figure,"Authors’ figure") : out(m.paper,"Original paper")}</div>${AtlasDrawings.flow(m)}</section>
      ${AtlasWalk.render(m,step)}
      <div class="two-panels"><section class="note-panel"><span class="note-icon">${icon("plus")}</span><h2>Strengths</h2><p>${esc(m.strength)}</p></section><section class="note-panel limit"><span class="note-icon">${icon("minus")}</span><h2>Limitations</h2><p>${esc(m.limit)}</p></section></div>
      <section class="content-section"><div class="section-line"><h2>Reading the outputs</h2></div><dl class="output-list">${[["Inputs","inputs"],["Geometry","output"],["Scale & frame","scale"],["Motion","motion"]].map(([k,v])=>`<div><dt>${k}</dt><dd>${esc(m[v])}</dd></div>`).join("")}</dl></section>
      <div class="detail-links">${m.moreCode ? out(m.moreCode,m.moreSymbol,"","code") : ""} ${out(m.project,"Project page","","globe")}${m.id==="omega"?out("https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md","Benchmark checkpoint"):""}</div>
      <div class="ending-links"><a class="button primary" href="#compare/${m.id}/${other}">${icon("compare")} Compare with ${esc(byId[other].name)}</a>${next ? `<a href="#model/${next.id}" class="text-link">Next in time · ${esc(next.name)} ${icon("arrow")}</a>` : `<a href="#roadmap" class="text-link">View the research roadmap ${icon("arrow")}</a>`}</div></div>`;
  }
  function sources(){
    return `<div class="container reading source-page"><div class="eyebrow"><span class="short-rule"></span>Research references</div><h1>Source library<span class="blue-dot">.</span></h1><p class="intro">The papers, project pages and code behind each architecture, ordered by first arXiv submission.</p>
      <p class="source-purpose">This guide is for noncommercial research and education. Each code walkthrough includes its upstream licenses, credits and exact source revision.</p>
      <div class="sources-list">${models.map(m=>`<article class="source-item"><div><div class="card-meta"><span>${esc(m.venue)}</span><span>Preprint <time datetime="${m.preprint}">${dateLabel(m)}</time></span></div><h2><a href="#model/${m.id}">${esc(m.name)} ${icon("arrow")}</a></h2><p>${esc(m.title)}</p></div>
      <div class="source-links">${out(m.paper,"Paper","","book")}${out(m.project,"Project","","globe")}${out(m.code,"Model code","","code")}<div class="repo-name">${esc(repoName(m))}</div></div></article>`).join("")}</div></div>`;
  }
  function route(initial=false){
    const [page="overview", id, second, step] = (location.hash.slice(1)||"overview").split("/");
    if(page==="main-content"){main.focus();return;}
    if(page==="models" && previousPage==="overview"){
      document.querySelector("#model-index")?.scrollIntoView();document.querySelector("#index-title")?.focus({preventScroll:true});return;
    }
    if(releaseScene){releaseScene();releaseScene=null;}
    const model=page==="model" && Object.hasOwn(byId,id) ? byId[id] : null;
    const showCode=model && second==="code";
    let current = "overview";
    if(model){
      if(showCode && previousPage==="model" && previousModel===id)AtlasWalk.select(model,step);
      else main.innerHTML=detail(model,showCode?step:null);
      current="model";
    }
    else if(page==="roadmap"){
      if(previousPage!=="roadmap")main.innerHTML=AtlasRoadmap.render(models,{dateLabel,out,icon,esc});
      current="roadmap";
    }
    else if(page==="compare"){main.innerHTML=AtlasCompare.render(models,id,second);AtlasCompare.bind(models);current="compare";}
    else if(page==="sources"){main.innerHTML=sources();current="sources";}
    else {main.innerHTML=overview();bindOverview();document.querySelector("#index-title").tabIndex=-1;}
    const pageTitles={overview:"Geometric foundation models",roadmap:"Research roadmap",sources:"Source library",compare:"Compare models"};
    const label = model ? model.name : pageTitles[current];
    document.title = current === "overview" ? projectTitle : `${label} · ${projectTitle}`;
    document.querySelectorAll("[data-nav]").forEach(link=>{
      if(link.dataset.nav===(current==="model"?"overview":current))link.setAttribute("aria-current","page");else link.removeAttribute("aria-current");
    });
    const milestone = current==="roadmap" && id ? document.getElementById("milestone-"+id) : null;
    if(showCode){
      const target=document.querySelector(step ? ".walk-stage h4" : "#walk-heading");
      target.scrollIntoView({block:"start"});
      target.focus({preventScroll:true});
    } else if(milestone){
      milestone.scrollIntoView({block:"start"});
      milestone.focus({preventScroll:true});
    } else if(page==="models"){
      document.querySelector("#model-index").scrollIntoView();
      if(!initial)document.querySelector("#index-title").focus({preventScroll:true});
    } else if(!(current==="compare"&&previousPage==="compare")){
      window.scrollTo(0,0);
      if(!initial)main.focus({preventScroll:true});
    }
    previousPage=current;
    previousModel=model ? model.id : "";
  }
  document.querySelector(".skip-link").addEventListener("click",e=>{e.preventDefault();main.focus();});
  document.addEventListener("click",event=>{
    const link=event.target.closest("a[href^='#roadmap'],a[href^='#model/'][href*='/code']");
    if(link && link.hash===location.hash && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey){
      event.preventDefault();
      route();
    }
  });
  window.addEventListener("hashchange",()=>route());
  route(true);
})();
