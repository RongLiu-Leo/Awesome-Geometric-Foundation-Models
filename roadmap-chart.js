(() => {
  "use strict";

  const dateFormat = new Intl.DateTimeFormat("en-US", {
    timeZone: "UTC", month: "short", day: "numeric"
  });
  const escape = value => String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);

  const styles = `<style>
    .atlas-chronology {
      --chron-rule: #c1cdcb;
      --chron-blue: #274ed2;
      padding: 21px 26px 20px;
      background: linear-gradient(110deg, #eff3f0, #f4f6f5 65%, #edf1f7);
      border: 1px solid #cdd6d1;
      border-radius: 5px;
      color: #1d2c29;
    }
    .atlas-chronology ol { list-style: none; margin: 0; padding: 0 }
    .atlas-chronology .chron-years { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)) }
    .atlas-chronology .chron-year { min-width: 0; position: relative; padding: 0 16px 0 0 }
    .atlas-chronology .chron-year:last-child { padding-right: 0 }
    .atlas-chronology .chron-year::before {
      content: ""; position: absolute; top: 9px; left: 5px; width: 100%; border-top: 1px solid var(--chron-rule);
    }
    .atlas-chronology .chron-year:last-child::before { width: calc(100% - 5px); border-top-style: dashed }
    .atlas-chronology .chron-heading {
      position: relative; display: block; padding: 25px 0 8px;
      font: 600 25px/1.1 var(--sans,Arial,sans-serif); letter-spacing: -.055em;
    }
    .atlas-chronology .chron-heading::before {
      content: ""; position: absolute; top: 5px; left: 0; width: 7px; height: 7px;
      box-sizing: content-box; border: 1px solid var(--chron-blue); border-radius: 50%; background: #f0f3f1;
    }
    .atlas-chronology .chron-year:last-child .chron-heading::before { background: var(--chron-blue) }
    .atlas-chronology .chron-entry {
      display: grid; grid-template-columns: 44px 1fr; gap: 5px; align-items: baseline;
      width: fit-content; max-width: 100%; padding: 5px 1px; text-decoration: none;
      font: 600 14px/1.25 var(--sans,Arial,sans-serif); color: #263b6e;
    }
    .atlas-chronology .chron-entry time {
      color: #53675f; font: 11px/1.25 var(--mono,Consolas,monospace); white-space: nowrap;
    }
    .atlas-chronology .chron-entry:hover span { text-decoration: underline; text-underline-offset: 3px }
    .atlas-chronology .chron-entry:focus-visible { outline: 2px solid var(--chron-blue); outline-offset: 3px }
    @media (min-width: 561px) and (max-width: 820px) {
      .atlas-chronology .chron-years { grid-template-columns: repeat(2,minmax(0,1fr)); row-gap: 17px }
      .atlas-chronology .chron-year:nth-child(2)::before { width: calc(100% - 5px) }
    }
    @media (max-width: 560px) {
      .atlas-chronology { padding: 18px }
      .atlas-chronology .chron-years { display: block }
      .atlas-chronology .chron-year { display: grid; grid-template-columns: 74px 1fr; gap: 10px; padding: 0 0 12px 0 }
      .atlas-chronology .chron-year:last-child { padding-bottom: 0 }
      .atlas-chronology .chron-year::before { top: 10px; left: 4px; height: 100%; width: 0; border-top: 0; border-left: 1px solid var(--chron-rule) }
      .atlas-chronology .chron-year:last-child::before { display: none }
      .atlas-chronology .chron-heading { padding: 2px 0 0 16px; font-size: 20px }
      .atlas-chronology .chron-heading::before { top: 8px }
      .atlas-chronology .chron-entry { font-size: 14px }
    }
  </style>`;

  function render(models) {
    const years = new Map();
    for (const model of models) {
      const year = model.preprint.slice(0, 4);
      if (!years.has(year)) years.set(year, []);
      years.get(year).push(model);
    }
    const sections = Array.from(years, ([year, entries]) => {
      const links = entries.map(model => {
        const date = new Date(model.preprint + "T00:00:00Z");
        return `<li><a class="chron-entry" href="#roadmap/${encodeURIComponent(model.id)}"><time datetime="${escape(model.preprint)}">${dateFormat.format(date)}</time> <span>${escape(model.name)}</span></a></li>`;
      }).join("");
      return `<li class="chron-year"><span class="chron-heading">${escape(year)}</span><ol aria-label="Papers from ${escape(year)}">${links}</ol></li>`;
    }).join("");
    return `${styles}<nav class="atlas-chronology" aria-label="Jump to a milestone by first arXiv submission date"><ol class="chron-years">${sections}</ol></nav>`;
  }

  window.AtlasRoadmapChart = { render };
})();
