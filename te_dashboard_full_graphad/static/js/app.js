/******************************************************
 * TE Dashboard – UTAR + GraphAD+ + Selective LLM
 * Propagation graph version based on routing paths
 ******************************************************/
console.log(">>> app.js loaded (QA_PLOT_v3)");

let teData = null;
let selectedIndex = null;
let cy = null;

// === Variable label mapping ===
let VAR_KO = {};
let VAR_PROC = {};  // varId -> process name

// Chart.js instance used for Q&A plots
let qaLineChart = null;

// == Utilities for the correlation heatmap ==
function mean(arr) {
  const n = arr.length || 1;
  return arr.reduce((a, b) => a + b, 0) / n;
}
function std(arr) {
  const m = mean(arr);
  const n = arr.length || 1;
  const v = arr.reduce((a, b) => a + (b - m) * (b - m), 0) / n;
  return Math.sqrt(v);
}
function corr(a, b) {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  if (da === 0 || db === 0) return 0;
  return num / Math.sqrt(da * db);
}


let qaChart = null;


function stripMarkdownArtifacts(text) {
  if (!text) return text;
  return String(text)
    .replace(/```/g, "")
    .replace(/\*\*/g, "")
    .replace(/__/g, "")
    .replace(/`/g, "")
    .replace(/^\s{0,3}#{1,6}\s*/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}


/** Infer a variable name from the question text. */
function resolveVarName(text) {
  const lower = text.toLowerCase();

  // 1) Match xmeas_6 / xmv_1 first.
  const m = lower.match(/x(meas|mv)_[0-9]+/);
  if (m) return m[0];

  // 2) Match by label text.
  const clean = text.replace(/\s+/g, "");
  for (const [id, ko] of Object.entries(VAR_KO)) {
    const koClean = ko.replace(/\s+/g, "");
    const shortClean = getVarKoShort(id).replace(/\s+/g, "");
    if (koClean.includes(clean) || shortClean.includes(clean) || clean.includes(shortClean)) {
      return id;
    }
  }
  return null;
}


/** Extract a variable series from the full uploaded dataset. */
function getVariableSeries(varId) {
  if (!teData || !teData.rows) return null;
  return teData.rows.map(r => r.features[varId]);
}


// Pick Reactor-related variables for plotting.
function getReactorVarIdsForPlot(row) {
  const reactorVars = [];

  // 1) Prefer variables in graphad_topk that belong to the Reactor process.
  if (row && row.graphad_topk) {
    row.graphad_topk.forEach(v => {
      const varId = v.var;
      const proc = VAR_PROC[varId] || v.process || "";
      if (proc.toLowerCase().includes("reactor")) {
        reactorVars.push(varId);
      }
    });
  }

  // 2) If none are found, scan the full process map for Reactor variables.
  if (reactorVars.length === 0) {
    for (const [varId, proc] of Object.entries(VAR_PROC)) {
      if (proc && proc.toLowerCase().includes("reactor")) {
        reactorVars.push(varId);
      }
    }
  }

  // Limit the list if it gets too long.
  return reactorVars.slice(0, 4);
}




function renderVariableLinePlot(varIds) {
  console.log("renderVariableLinePlot called with", varIds);

  if (!teData || !teData.rows || !Array.isArray(varIds) || varIds.length === 0) {
    console.warn("renderVariableLinePlot: no data or varIds");
    return;
  }

  const canvas = document.getElementById("qa-line-plot");
  console.log("canvas in renderVariableLinePlot =", canvas);

  if (!canvas) {
    console.warn("qa-line-plot canvas not found in DOM");
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    console.warn("2D context not available for qa-line-plot");
    return;
  }

  const labels = teData.rows.map((r, i) => r.index ?? i);

  const datasets = [];
  const baseValuesList = [];

  varIds.forEach((varId) => {
    const values = getVariableSeries(varId);
    if (!values) return;
    baseValuesList.push(values);

    datasets.push({
      label: `${varId} (${getVarKoShort(varId)})`,
      data: values,
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.15,
    });
  });

  if (baseValuesList.length === 0) {
    console.warn("renderVariableLinePlot: no valid series");
    return;
  }

  // For a single variable, add a mean ±3σ band.
  if (baseValuesList.length === 1) {
    const values = baseValuesList[0];
    const m = mean(values);
    const s = std(values);
    const upper = values.map(() => m + 3 * s);
    const lower = values.map(() => m - 3 * s);

    datasets.push({
      label: "mean + 3σ",
      data: upper,
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
    });
    datasets.push({
      label: "mean - 3σ",
      data: lower,
      borderWidth: 1,
      borderDash: [4, 4],
      pointRadius: 0,
    });
  }

  // Highlight the currently selected sample.
  let annotations = {};
  if (selectedIndex !== null && baseValuesList.length > 0) {
    const highlightValues = labels.map((_, i) =>
      i === selectedIndex ? baseValuesList[0][i] : null
    );
    datasets.push({
      label: "Selected sample",
      data: highlightValues,
      borderWidth: 0,
      pointRadius: 5,
      pointHoverRadius: 6,
      showLine: false,
    });

    annotations = {
      selectedLine: {
        type: "line",
        xMin: selectedIndex,
        xMax: selectedIndex,
        borderColor: "rgba(255,0,0,0.6)",
        borderWidth: 1,
      },
    };
  }

  if (qaLineChart) qaLineChart.destroy();

  qaLineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          position: "top",
        },
        zoom: {
          pan: {
            enabled: true,
            mode: "x",
          },
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true,
            },
            mode: "xy",
          },
        },
        annotation: {
          annotations,
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Sample index" },
        },
        y: {
          title: { display: true, text: "Value" },
        },
      },
    },
  });
}



function renderTopKCorrelationHeatmap(row) {
  const container = document.getElementById("qa-heatmap-container");
  if (!container) return;

  container.innerHTML = "";

  if (!row || !row.graphad_topk || row.graphad_topk.length === 0) {
    return;
  }

  const varIds = row.graphad_topk.map(v => v.var);
  const seriesMap = {};
  varIds.forEach(id => {
    seriesMap[id] = getVariableSeries(id) || [];
  });

  // Correlation matrix
  const n = varIds.length;
  const corrMat = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = seriesMap[varIds[i]];
      const b = seriesMap[varIds[j]];
      corrMat[i][j] = corr(a, b);
    }
  }

  // Map value -1 ~ 1 to color (blue ~ white ~ red).
  function corrColor(v) {
    const clamped = Math.max(-1, Math.min(1, v));
    if (clamped >= 0) {
      const r = 255;
      const g = Math.round(255 * (1 - clamped));
      const b = Math.round(255 * (1 - clamped));
      return `rgb(${r},${g},${b})`;
    } else {
      const t = -clamped;
      const r = Math.round(255 * (1 - t));
      const g = Math.round(255 * (1 - t));
      const b = 255;
      return `rgb(${r},${g},${b})`;
    }
  }

  let html = `<h3 style="margin-top:8px;font-size:13px;">Top-K Variable Correlation Heatmap</h3>`;
  html += `<table style="border-collapse:collapse;font-size:10px;">`;

  // header
  html += "<thead><tr><th></th>";
  varIds.forEach(id => {
    html += `<th style="border:1px solid #ccc;padding:2px 4px;">${id}</th>`;
  });
  html += "</tr></thead><tbody>";

  // rows
  for (let i = 0; i < n; i++) {
    html += `<tr><th style="border:1px solid #ccc;padding:2px 4px;">${varIds[i]}</th>`;
    for (let j = 0; j < n; j++) {
      const v = corrMat[i][j];
      const color = corrColor(v);
      html += `<td style="border:1px solid #eee;padding:2px 4px;background:${color};text-align:center;">${v.toFixed(2)}</td>`;
    }
    html += "</tr>";
  }

  html += "</tbody></table>";

  container.innerHTML = html;
}





/** Draw a simple Chart.js plot. */
function renderQAPlot(varId) {
  const values = getVariableSeries(varId);
  if (!values) return;

  const ctx = document.getElementById("qa-plot").getContext("2d");

  if (qaChart) qaChart.destroy();

  qaChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: values.map((_, i) => i),
      datasets: [{
        label: `${varId} (${getVarKoShort(varId)})`,
        data: values,
        borderWidth: 2,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Sample index" } },
        y: { title: { display: true, text: varId } }
      }
    }
  });
}



/**
 * Read static/data/TE_variable_process_map.csv and populate
 * VAR_KO as { xmeas_1: "A feed flow (stream 1)", ... }.
 */
async function loadVarMap() {
  try {
    const res = await fetch("/static/data/TE_variable_process_map.csv");
    if (!res.ok) {
      console.error("Failed to load TE_variable_process_map.csv:", res.status);
      return;
    }
    const text = await res.text();

    const lines = text.trim().split(/\r?\n/);
    // Header: var, process, ko
    VAR_KO = {};
    VAR_PROC = {};
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      const parts = line.split(",");
      if (parts.length < 3) continue;

      const varId = parts[0].trim();
      const proc = parts[1].trim();
      const ko = parts.slice(2).join(",").trim();

      VAR_KO[varId] = ko;
      VAR_PROC[varId] = proc;
    }

    console.log("VAR_KO loaded:", VAR_KO);
    console.log("VAR_PROC loaded:", VAR_PROC);
  } catch (e) {
    console.error("Error while parsing TE_variable_process_map.csv:", e);
  }
}

/**
 * Return a short variable label with parenthesized suffixes removed.
 */
function getVarKoShort(id) {
  const ko = VAR_KO[id] || id;
  return ko.replace(/\s*\(.*?\)\s*/g, "");
}


/* ----------------------------------------------------
 *  CSV upload
 * --------------------------------------------------*/
async function uploadCSV(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/api/upload_te", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}


/* ----------------------------------------------------
 *  Render the sample table
 * --------------------------------------------------*/
function renderSampleTable(data) {
  const tbody = document.querySelector("#sample-table tbody");
  tbody.innerHTML = "";

  data.rows.forEach(row => {
    const tr = document.createElement("tr");
    tr.classList.add("sample-row");
    tr.dataset.index = row.index;

    const status = row.predicted_is_anomaly ? "Anomalous" : "Normal";

    tr.innerHTML = `
      <td>${row.index}</td>
      <td>${row.scores.final_prob.toFixed(3)}</td>
      <td>${row.scores.decision_source}</td>
      <td>${status}</td>
    `;

    tr.addEventListener("click", () => {
      selectSample(row.index);
    });

    tbody.appendChild(tr);
  });
}


/* ----------------------------------------------------
 *   Highlight the selected row
 * --------------------------------------------------*/
function highlightSelectedRow(idx) {
  document.querySelectorAll("tr.sample-row").forEach(tr => {
    tr.classList.toggle("selected", parseInt(tr.dataset.index) === idx);
  });
}


/* ----------------------------------------------------
 *   Render the score panel
 * --------------------------------------------------*/
function renderScores(row) {
  const container = document.getElementById("scores-grid");
  container.innerHTML = "";

  const s = row.scores;
  const items = [
    ["UTAR base", s.utar_base_score ?? s.gtar_score],
    ["RF prob", s.rf_prob],
    ["XGB prob", s.xgb_prob],
    ["ModernTCN prob", s.tcn_prob ?? s.lstm_prob],
    ["Final prob", s.final_prob],
  ];

  items.forEach(([label, value]) => {
    const div = document.createElement("div");
    div.classList.add("score-card");
    div.innerHTML = `
      <div class="score-label">${label}</div>
      <div class="score-value">${value.toFixed(3)}</div>
    `;
    container.appendChild(div);
  });

  const info = document.createElement("div");
  info.classList.add("score-card");
  info.innerHTML = `
    <div class="score-label">Decision Path</div>
    <div class="score-value">${s.decision_source}</div>
    <div class="score-badge">${s.selected_model}</div>
  `;
  container.appendChild(info);

  const routing = document.createElement("div");
  routing.classList.add("score-card");
  routing.innerHTML = `
    <div class="score-label">Gray-Zone / LLM</div>
    <div class="score-value">q=${(s.selected_q ?? 0).toFixed(2)} | tau=${(s.tau ?? 0).toFixed(2)}</div>
    <div class="score-badge">${s.gray_zone ? "Gray-Zone" : "Direct"} / ${s.llm_called ? "LLM Called" : "No LLM"}</div>
  `;
  container.appendChild(routing);

  const status = document.createElement("div");
  status.classList.add("score-card");
  status.innerHTML = `
    <div class="score-label">Final Prediction</div>
    <div class="score-value">${row.predicted_is_anomaly ? "Anomalous" : "Normal"}</div>
  `;
  container.appendChild(status);
}


/* ----------------------------------------------------
 *  Top-K variables (table only)
 * --------------------------------------------------*/
/* ----------------------------------------------------
 *  Render the Top-K variable table with process mapping and display labels
 * --------------------------------------------------*/
function renderTopK(row) {
  const tbody = document.querySelector("#topk-table tbody");
  tbody.innerHTML = "";

  if (!row || !row.graphad_topk) return;

  row.graphad_topk.forEach(v => {
    const varId = v.var;
    const varKoShort = getVarKoShort(varId);
    const proc = VAR_PROC[varId] || v.process || "Unknown";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${varId} (${varKoShort})</td>
      <td>${v.score.toFixed(3)}</td>
      <td>${v.direction}</td>
      <td>${proc}</td>
    `;
    tbody.appendChild(tr);
  });
}


/* ----------------------------------------------------
 *   Process graph (knowledge-graph style)
 *   - built from row.routing_paths
 *   - separates process nodes from variable nodes
 *   - uses process→variable and variable→variable edges
 * --------------------------------------------------*/
/* ----------------------------------------------------
 *   Process graph (knowledge-graph style)
 *   - built from row.routing_paths
 *   - separates process nodes from variable nodes
 *   - uses process→variable and variable→variable edges
 * --------------------------------------------------*/
function initGraph(row) {
  if (!row) return;

  const paths =
    row.routing_paths ||
    row.paths ||
    (row.llm_struct && row.llm_struct.paths) ||
    [];

  console.log("routing_paths", row.routing_paths, "paths used", paths);

  if (!Array.isArray(paths) || paths.length === 0) {
    if (cy) {
      cy.elements().remove();
      cy.resize();
    }
    return;
  }

  const elements = [];
  const nodeMap = new Map(); // id -> { label, color, type, role }
  const edgeSet = new Set();

  // Process-specific colors for variable nodes.
  const processColors = {
    "Feed System": "#4CAF50",
    "Recycle/Compressor": "#009688",
    "Compressor": "#607D8B",
    "Purge/Compressor": "#00796B",
    "Reactor": "#FF9800",
    "Reactor Feed Analysis": "#8BC34A",
    "Reactor Cooling": "#3F51B5",
    "Separator": "#3F51B5",
    "Separator Cooling": "#2196F3",
    "Stripper": "#E91E63",
    "Purge Gas Analysis": "#9C27B0",
    "Product Analysis": "#009688",
    "Manipulator/Feed": "#795548",
    "Manipulator/Recycle/Compressor": "#6D4C41",
    "Manipulator/Purge": "#5D4037",
    "Manipulator/Separator": "#8D6E63",
    "Manipulator/Stripper": "#9C27B0",
    "Manipulator/Reactor Cooling": "#3949AB",
    "Manipulator/Condenser Cooling": "#283593",
  };
  const defaultColor = "#9E9E9E";

  // ==== Convert paths to nodes and edges ====
  paths.forEach(path => {
    if (!Array.isArray(path)) return;

    let prevVarNodeId = null;

    for (let i = 0; i < path.length; i++) {
      const hop = path[i];
      if (typeof hop !== "string") continue;
      if (!hop.includes(":")) continue;

      const [procRaw, varIdRaw] = hop.split(":", 2);

      // Use let because the process name may be remapped below.
      let proc = (procRaw || "").trim();
      const varId = (varIdRaw || "").trim();

      if (!varId) continue;

      // If the process is missing or unknown, recover it from the CSV mapping.
      if (!proc || proc.toLowerCase() === "unknown") {
        const mappedProc = VAR_PROC[varId];
        if (mappedProc) {
          proc = mappedProc;
        } else {
          proc = "Unknown";
        }
      }

      const procNodeId = `proc:${proc}`;
      const varNodeId = `var:${varId}`;

      const varKoShort = getVarKoShort(varId);
      const varColor = processColors[proc] || defaultColor;

      // Process node
      if (!nodeMap.has(procNodeId)) {
        nodeMap.set(procNodeId, {
          id: procNodeId,
          label: proc,
          color: "#263238",
          type: "process",
          role: "process",
        });
      }

      // Variable node
      if (!nodeMap.has(varNodeId)) {
        nodeMap.set(varNodeId, {
          id: varNodeId,
          label: varKoShort,
          color: varColor,
          type: "var",
          role: "middle",
        });
      }

      // Process → variable edge
      edgeSet.add(JSON.stringify({
        id: `pv:${procNodeId}->${varNodeId}`,
        source: procNodeId,
        target: varNodeId,
        type: "pv",
      }));

      // Variable → variable edge
      if (prevVarNodeId) {
        edgeSet.add(JSON.stringify({
          id: `vv:${prevVarNodeId}->${varNodeId}`,
          source: prevVarNodeId,
          target: varNodeId,
          type: "vv",
        }));
      }

      prevVarNodeId = varNodeId;
    }
  });

  // ==== Mark the first/last variable nodes for emphasis ====
  paths.forEach(path => {
    if (!Array.isArray(path) || path.length === 0) return;

    const first = path[0];
    const last = path[path.length - 1];

    [first, last].forEach((hop, idx) => {
      if (typeof hop !== "string" || !hop.includes(":")) return;
      const [, varIdRaw] = hop.split(":", 2);
      const varId = (varIdRaw || "").trim();
      const varNodeId = `var:${varId}`;
      const node = nodeMap.get(varNodeId);
      if (!node) return;
      node.role = idx === 0 ? "source" : "sink";
    });
  });

  // ==== Build Cytoscape node / edge elements ====
  for (const node of nodeMap.values()) {
    elements.push({
      data: {
        id: node.id,
        label: node.label,
        color: node.color,
        type: node.type,
        role: node.role,
      },
    });
  }

  edgeSet.forEach(json => {
    const e = JSON.parse(json);
    elements.push({
      data: {
        id: e.id,
        source: e.source,
        target: e.target,
        edge_type: e.type,
      },
    });
  });

  const container = document.getElementById("graph-container");

  if (!cy) {
    cy = cytoscape({
      container,
      elements,
      style: [
        {
          selector: "node[type = 'process']",
          style: {
            "shape": "round-rectangle",
            "background-color": "#263238",
            "label": "data(label)",
            "color": "#FFFFFF",
            "font-size": 10,
            "text-valign": "center",
            "text-halign": "center",
            "width": 70,
            "height": 30,
            "border-width": 1,
            "border-color": "#000000",
          },
        },
        {
          selector: "node[type = 'var']",
          style: {
            "shape": "ellipse",
            "background-color": "data(color)",
            "label": "data(label)",
            "color": "#FFFFFF",
            "font-size": 9,
            "text-wrap": "wrap",
            "text-max-width": 80,
            "text-valign": "center",
            "text-halign": "center",
            "width": 38,
            "height": 38,
            "border-width": 1,
            "border-color": "#333333",
          },
        },
        {
          selector: "node[role = 'source']",
          style: {
            "border-width": 3,
            "border-color": "#4CAF50",
          },
        },
        {
          selector: "node[role = 'sink']",
          style: {
            "border-width": 3,
            "border-color": "#FF5252",
          },
        },
        {
          selector: "edge[edge_type = 'pv']",
          style: {
            "width": 1.5,
            "line-color": "#B0BEC5",
            "target-arrow-color": "#B0BEC5",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
        {
          selector: "edge[edge_type = 'vv']",
          style: {
            "width": 3,
            "line-color": "#FFB74D",
            "target-arrow-color": "#FFB74D",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
          },
        },
      ],
      layout: {
        name: "fcose",
        animate: false,
        randomize: true,
        idealEdgeLength: 150,
        nodeRepulsion: 6000,
        gravity: 0.8,
        numIter: 2500,
      },
    });
  } else {
    cy.json({ elements });
    cy.layout({
      name: "fcose",
      animate: false,
      randomize: true,
      idealEdgeLength: 150,
      nodeRepulsion: 6000,
      gravity: 0.8,
      numIter: 2500,
    }).run();
  }

  cy.fit();
}




/* ----------------------------------------------------
 *   Handle row selection
 * --------------------------------------------------*/
function selectSample(idx) {
  if (!teData) return;

  const row = teData.rows.find(r => r.index === idx);
  if (!row) return;

  selectedIndex = idx;
  highlightSelectedRow(idx);
  renderScores(row);
  renderTopK(row);
  initGraph(row);

  document.getElementById("explanation-output").textContent = "";
  document.getElementById("qa-answer-text").textContent = "";
}



/* ----------------------------------------------------
 *   LLM explanation
 * --------------------------------------------------*/
async function handleExplain() {
  if (!teData || selectedIndex === null) return;

  const row = teData.rows.find(r => r.index === selectedIndex);
  if (!row) return;

  const res = await fetch("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(row.llm_struct),
  });

  const data = await res.json();
  const el = document.getElementById("explanation-output");
  el.textContent = stripMarkdownArtifacts(data.explanation || JSON.stringify(data));

}



async function handleQA() {
  if (!teData || selectedIndex === null) {
    console.warn("QA: teData or selectedIndex is missing");
    return;
  }

  const qEl   = document.getElementById("qa-input");
  const outEl = document.getElementById("qa-answer-text");
  const q     = qEl.value.trim();

  if (!q) return;
  if (!outEl) {
    console.error("QA: qa-output element not found");
    return;
  }

  const qLower = q.toLowerCase();
  const row    = teData.rows.find(r => r.index === selectedIndex);

  if (!row) {
    console.warn("QA: selected row not found");
    return;
  }

  // Clear the textarea after each question.
  qEl.value = "";

  // Reset only the previous plot / heatmap.
  const heatDiv = document.getElementById("qa-heatmap-container");
  if (heatDiv) heatDiv.innerHTML = "";
  if (qaLineChart) {
    qaLineChart.destroy();
    qaLineChart = null;
  }

  // Show progress immediately.
  outEl.textContent = `Q: ${q}\n\nThinking...`;
  outEl.scrollTop = 0;

  // ---------- Step 1: LLM Q&A ----------
  let answerText = "";
  try {
    const res = await fetch("/api/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ struct: row.llm_struct, question: q }),
    });

    const data = await res.json();
    console.log("QA response:", data);

    answerText = stripMarkdownArtifacts(data.answer || "(Empty response.)");

    // Always overwrite the previous answer.
    outEl.textContent = `Q: ${q}\n\nA: ${answerText}`;
    outEl.scrollTop = 0;
    console.log("QA outEl after set:", outEl.textContent);
  } catch (e) {
    console.error("Error during QA fetch/json:", e);
    outEl.textContent = "An error occurred while calling LLM Q&A:\n" + e;
    return;
  }

  // ---------- Step 2: plot / heatmap handling ----------
  let didPlot = false;

  try {
    // 1) Correlation heatmap
    if (qLower.includes("correlation") || qLower.includes("heatmap") || qLower.includes("heat map")) {
      renderTopKCorrelationHeatmap(row);
      didPlot = true;
    }
    // 2) Top-K variable multi-plot
    else if (
      qLower.includes("top-k") ||
      qLower.includes("topk") ||
      qLower.includes("impact") ||
      qLower.includes("important variable") ||
      qLower.includes("important variables")
    ) {
      if (row && row.graphad_topk && row.graphad_topk.length > 0) {
        const varIds = row.graphad_topk.map(v => v.var);
        renderVariableLinePlot(varIds);
        didPlot = true;
      }
    }
    // 3) Reactor-related plot
    else if (
      qLower.includes("reactor") &&
      (qLower.includes("graph") ||
       qLower.includes("visualize") ||
       qLower.includes("show") ||
       qLower.includes("plot") ||
       qLower.includes("chart"))
    ) {
      const reactorVars = getReactorVarIdsForPlot(row);
      console.log("Reactor vars for plot:", reactorVars);
      if (reactorVars.length > 0) {
        renderVariableLinePlot(reactorVars);
        didPlot = true;
      }
    }
    // 4) Variable-specific visualization
    else {
      const varId = resolveVarName(q);
      const isPlotRequest =
        qLower.includes("draw") ||
        qLower.includes("visualize") ||
        qLower.includes("show") ||
        qLower.includes("plot") ||
        qLower.includes("graph") ||
        qLower.includes("chart");

      if (varId && isPlotRequest) {
        renderVariableLinePlot([varId]);
        didPlot = true;
      }
    }
  } catch (e) {
    console.error("Error while rendering plot/heatmap:", e);
  }

  if (didPlot) {
    outEl.textContent += "\n\n(The requested variable visualization has been rendered below.)";
    outEl.scrollTop = 0;
  }
}












/* ----------------------------------------------------
 *   Event binding
 * --------------------------------------------------*/
function attachEventHandlers() {
  const fileInput = document.getElementById("file-input");
  const reloadBtn = document.getElementById("reload-button");
  const explainBtn = document.getElementById("explain-button");
  const qaBtn = document.getElementById("qa-button");
  const btnSample = document.getElementById("btn-sample");


  btnSample.onclick = () => {
    if (selectedIndex !== null && teData) {
      const row = teData.rows.find(r => r.index === selectedIndex);
      if (row) initGraph(row);
    }
  };

  /* ---- File upload ---- */
  fileInput.addEventListener("change", async e => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      teData = await uploadCSV(file);
      renderSampleTable(teData);
      if (teData.rows.length > 0) selectSample(teData.rows[0].index);
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
  });

  /* ---- Reset ---- */
  reloadBtn.addEventListener("click", () => {
    location.reload();
  });

  /* ---- LLM ---- */
  explainBtn.addEventListener("click", handleExplain);
  qaBtn.addEventListener("click", handleQA);
}


/* ----------------------------------------------------
 *  (Deprecated) Top-K chart removed
 * --------------------------------------------------*/
function renderTopKChart() {}

window.addEventListener("load", async () => {
  console.log(">>> All resources loaded.");
  await loadVarMap();
  attachEventHandlers();
});
