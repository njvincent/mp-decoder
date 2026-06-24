(function () {
  const demo = window.SHEETCOPY_CNOT_DEMO;
  if (!demo || !Array.isArray(demo.frames)) {
    document.body.textContent = "Missing SHEETCOPY_CNOT_DEMO data.";
    return;
  }

  const L = demo.L;
  const WIDTH = 360;
  const MARGIN = 30;
  const SCALE = (WIDTH - 2 * MARGIN) / Math.max(L - 1, 1);
  let index = 0;
  let timer = null;

  const style = document.createElement("style");
  style.textContent = `
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; overflow: hidden; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2933;
      background: #f6f8fa;
    }
    main {
      height: 100vh;
      max-width: 1180px;
      margin: 0 auto;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    h1 { font-size: 18px; margin: 0; }
    .note { color: #5b6775; margin: 0; line-height: 1.25; font-size: 12px; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    button {
      padding: 5px 9px;
      border: 1px solid #b7c0cc;
      border-radius: 6px;
      background: white;
      cursor: pointer;
      font-size: 12px;
    }
    input[type="range"] { flex: 1 1 300px; }
    .frameLabel { font-size: 12px; color: #334e68; min-width: 260px; }
    .plots {
      flex: 1;
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .panel {
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: white;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      padding: 7px;
    }
    .panel h2 {
      font-size: 13px;
      line-height: 1.15;
      margin: 0 0 4px;
      color: #263442;
    }
    .plotWrap {
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    canvas {
      display: block;
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 100%;
      aspect-ratio: 1 / 1;
      border: 1px solid #d8dee4;
      border-radius: 6px;
      background: #fbfcfe;
    }
    .meta {
      min-height: 30px;
      margin-top: 4px;
      color: #52606d;
      font-size: 11px;
      line-height: 1.2;
    }
    .legend {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11px;
      color: #52606d;
    }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 2px;
      margin-right: 4px;
      vertical-align: -1px;
    }
    @media (max-width: 760px) {
      html, body { overflow: auto; }
      main { height: auto; min-height: 100vh; }
      .plots { grid-template-columns: 1fr; grid-template-rows: none; }
      .panel { min-height: 270px; }
    }
  `;
  document.head.appendChild(style);

  document.body.innerHTML = `
    <main>
      <h1>Sheet-copy CNOT X-sector Demo</h1>
      <p class="note">Four views for each frame: the original control sheet, the copied control sheet on the target block, the original target sheet, and the algebraic target merge. The first three panels show independent decoder sheets; only the fourth panel is merged.</p>
      <div class="controls">
        <button id="prev">Prev</button>
        <button id="play">Play</button>
        <button id="next">Next</button>
        <input id="slider" type="range" min="0" max="0" value="0">
        <span id="frameLabel" class="frameLabel"></span>
      </div>
      <section class="plots">
        <article class="panel">
          <h2>Control Sheet</h2>
          <div class="plotWrap"><canvas id="controlCanvas" width="${WIDTH}" height="${WIDTH}"></canvas></div>
          <div id="controlMeta" class="meta"></div>
        </article>
        <article class="panel">
          <h2>Original Target Sheet</h2>
          <div class="plotWrap"><canvas id="targetCanvas" width="${WIDTH}" height="${WIDTH}"></canvas></div>
          <div id="targetMeta" class="meta"></div>
        </article>
        <article class="panel">
          <h2>Copied Control Sheet</h2>
          <div class="plotWrap"><canvas id="copyCanvas" width="${WIDTH}" height="${WIDTH}"></canvas></div>
          <div id="copyMeta" class="meta"></div>
        </article>
        <article class="panel">
          <h2>Final Target Merge</h2>
          <div class="plotWrap"><canvas id="mergeCanvas" width="${WIDTH}" height="${WIDTH}"></canvas></div>
          <div id="mergeMeta" class="meta"></div>
        </article>
      </section>
      <div class="legend">
        <span><span class="swatch" style="background:#e67e22"></span>physical X</span>
        <span><span class="swatch" style="background:#2d7ff9"></span>correction</span>
        <span><span class="swatch" style="background:#111827"></span>decoded residual</span>
        <span><span class="swatch" style="background:#d62828;border-radius:50%"></span>syndrome</span>
        <span><span class="swatch" style="background:#7b2cbf"></span>history count</span>
        <span><span class="swatch" style="background:#2a9d8f"></span>field site</span>
      </div>
    </main>
  `;

  const slider = document.getElementById("slider");
  const frameLabel = document.getElementById("frameLabel");
  const playButton = document.getElementById("play");
  slider.max = Math.max(demo.frames.length - 1, 0);

  function pt(i, j) {
    return [MARGIN + (i - 1) * SCALE, MARGIN + (j - 1) * SCALE];
  }

  function edgeEndpoints(edge) {
    const i = edge[0], j = edge[1], o = edge[2];
    const a = pt(i, j);
    if (o === 1) return [a[0], a[1], MARGIN + (i % L) * SCALE, a[1]];
    return [a[0], a[1], a[0], MARGIN + (j % L) * SCALE];
  }

  function drawEdges(ctx, edges, color, width, offset) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    for (const edge of edges || []) {
      let [x1, y1, x2, y2] = edgeEndpoints(edge);
      if (edge[2] === 1) {
        y1 += offset;
        y2 += offset;
      } else {
        x1 += offset;
        x2 += offset;
      }
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.restore();
  }

  function emptyBlock(message) {
    return {
      empty: true,
      empty_message: message,
      physical: [],
      correction: [],
      decoded: [],
      syndromes: [],
      hist: [],
      fields: [],
      logical_status: "n/a"
    };
  }

  function findSheet(frame, predicate) {
    return frame.sheets.find(predicate) || null;
  }

  function rootControlSheet(frame) {
    return findSheet(frame, sheet => sheet.summary.block === 1 && sheet.summary.parent_lineage_id === null);
  }

  function originalTargetSheet(frame) {
    return findSheet(frame, sheet => sheet.summary.block === 2 && sheet.summary.parent_lineage_id === null);
  }

  function copiedControlSheet(frame) {
    return findSheet(frame, sheet => sheet.summary.block === 2 && sheet.summary.parent_lineage_id !== null);
  }

  function panelForSheet(sheet, missingMessage) {
    if (!sheet) return { block: emptyBlock(missingMessage), summary: null };
    return { block: sheet.block, summary: sheet.summary };
  }

  function panelForMerge(frame) {
    return { block: frame.target, summary: null, merge: true };
  }

  function drawBlock(canvasId, metaId, panel) {
    const block = panel.block;
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#fbfcfe";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#d9e2ec";
    ctx.lineWidth = 1;
    for (let i = 1; i <= L; i++) {
      let [x1, y1] = pt(i, 1), [x2, y2] = pt(i, L);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    for (let j = 1; j <= L; j++) {
      let [x1, y1] = pt(1, j), [x2, y2] = pt(L, j);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }

    if (block.empty) {
      ctx.fillStyle = "#829ab1";
      ctx.font = "15px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(block.empty_message, WIDTH / 2, WIDTH / 2);
      document.getElementById(metaId).textContent = block.empty_message;
      return;
    }

    for (const f of block.fields || []) {
      const [x, y] = pt(f[0], f[1]);
      const alpha = Math.max(0.12, Math.min(0.45, 0.55 / Math.max(f[2], 1)));
      ctx.fillStyle = "rgba(42, 157, 143, " + alpha + ")";
      ctx.fillRect(x - 10, y - 10, 20, 20);
    }

    drawEdges(ctx, block.physical, "#e67e22", 6, -4);
    drawEdges(ctx, block.correction, "#2d7ff9", 4, 4);
    drawEdges(ctx, block.decoded, "#111827", 3, 0);

    for (const h of block.hist || []) {
      const [x, y] = pt(h[0], h[1]);
      ctx.fillStyle = "rgba(123, 44, 191, 0.72)";
      ctx.fillRect(x - 8, y - 8, 16, 16);
      ctx.fillStyle = "white";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(h[2]), x, y);
    }
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";

    for (const s of block.syndromes || []) {
      const [x, y] = pt(s[0], s[1]);
      ctx.fillStyle = "#d62828";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();
    }

    ctx.fillStyle = "#627d98";
    for (let i = 1; i <= L; i++) {
      for (let j = 1; j <= L; j++) {
        const [x, y] = pt(i, j);
        ctx.beginPath();
        ctx.arc(x, y, 1.8, 0, 2 * Math.PI);
        ctx.fill();
      }
    }

    document.getElementById(metaId).textContent = metaText(panel);
  }

  function metaText(panel) {
    const block = panel.block;
    if (panel.merge) {
      const lineages = (block.lineages || []).join(", ");
      return "target lineages [" + lineages + "] | residual " + block.decoded.length +
        " | syndromes " + block.syndromes.length + " | " + block.logical_status;
    }
    const s = panel.summary;
    if (!s) return "";
    const parent = s.parent_lineage_id === null ? "none" : String(s.parent_lineage_id);
    return "lineage " + s.lineage_id + " | parent " + parent +
      " | residual " + s.decoded_count + " | hist " + s.hist_count +
      " | fields " + s.field_count + " | " + panel.block.logical_status;
  }

  function render() {
    const frame = demo.frames[index];
    slider.value = index;
    frameLabel.textContent = "frame " + (index + 1) + " / " + demo.frames.length +
      ": " + frame.label + " | sheets " + frame.sheet_count +
      " | active " + frame.active_sheet_count;

    drawBlock("controlCanvas", "controlMeta", panelForSheet(rootControlSheet(frame), "control sheet missing"));
    drawBlock("copyCanvas", "copyMeta", panelForSheet(copiedControlSheet(frame), "copy not created yet"));
    drawBlock("targetCanvas", "targetMeta", panelForSheet(originalTargetSheet(frame), "target sheet missing"));
    drawBlock("mergeCanvas", "mergeMeta", panelForMerge(frame));
  }

  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    playButton.textContent = "Play";
  }

  document.getElementById("prev").onclick = () => {
    stop();
    index = Math.max(0, index - 1);
    render();
  };
  document.getElementById("next").onclick = () => {
    stop();
    index = Math.min(demo.frames.length - 1, index + 1);
    render();
  };
  slider.oninput = () => {
    stop();
    index = Number(slider.value);
    render();
  };
  playButton.onclick = () => {
    if (timer !== null) {
      stop();
      return;
    }
    playButton.textContent = "Pause";
    timer = setInterval(() => {
      index = (index + 1) % demo.frames.length;
      render();
    }, 650);
  };

  render();
})();
