"use strict";

const ui = {
  form: document.querySelector("#config-form"),
  connectionError: document.querySelector("#connection-error"),
  status: document.querySelector("#job-status"),
  message: document.querySelector("#job-message"),
  progress: document.querySelector("#job-progress"),
  cancel: document.querySelector("#cancel-job"),
  log: document.querySelector("#job-log"),
  result: document.querySelector("#result-json"),
  summaryStatus: document.querySelector("#summary-status"),
  pairedResults: document.querySelector("#paired-results"),
  familyCard: document.querySelector("#family-results-card"),
  familyTable: document.querySelector("#family-results-table"),
  auditCard: document.querySelector("#audit-results-card"),
  auditTable: document.querySelector("#audit-results-table"),
  artifactGrid: document.querySelector("#artifact-grid"),
  artifactCount: document.querySelector("#artifact-count"),
  environment: document.querySelector("#environment-grid"),
  lossCard: document.querySelector("#loss-card"),
  lossChart: document.querySelector("#loss-chart"),
  toast: document.querySelector("#toast"),
  actionButtons: [...document.querySelectorAll("[data-action]")],
  tabs: [...document.querySelectorAll("[role='tab'][data-tab]")],
  panels: [...document.querySelectorAll("[role='tabpanel'][data-panel]")],
};

const state = {
  token: "",
  defaults: {},
  cursor: 0,
  jobId: null,
  pollTimer: null,
  polling: false,
  lastFinishedId: null,
  toastTimer: null,
  losses: [],
  lastLossStep: null,
  resizeTimer: null,
};


async function api(path, options = {}) {
  const method = options.method || "GET";
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (method !== "GET" && method !== "HEAD") {
    headers.set("Content-Type", "application/json");
    if (!state.token) {
      throw new Error("The local session is not ready yet. Reload the page and try again.");
    }
    headers.set("X-Session-Token", state.token);
  }

  let response;
  try {
    response = await fetch(path, { ...options, method, headers, credentials: "same-origin" });
  } catch (error) {
    throw new Error(`Could not reach the local server: ${error.message}`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`The local server returned an invalid response (${response.status}).`);
  }
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed (${response.status}).`);
  }
  return payload;
}


function clone(value) {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}


function selectTab(name, { focus = false } = {}) {
  const selected = ui.tabs.find((tab) => tab.dataset.tab === name);
  if (!selected) return;
  for (const tab of ui.tabs) {
    const active = tab === selected;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const panel of ui.panels) {
    panel.hidden = panel.dataset.panel !== name;
  }
  if (focus) selected.focus();
}


function bindTabs() {
  for (const tab of ui.tabs) {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab));
    tab.addEventListener("keydown", (event) => {
      const current = ui.tabs.indexOf(tab);
      let next = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (current + 1) % ui.tabs.length;
      if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (current - 1 + ui.tabs.length) % ui.tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = ui.tabs.length - 1;
      if (next !== null) {
        event.preventDefault();
        selectTab(ui.tabs[next].dataset.tab, { focus: true });
      }
    });
  }
}


function populateForm(defaults) {
  const controls = document.querySelectorAll("[data-config-group][data-config-key]");
  for (const control of controls) {
    const group = defaults[control.dataset.configGroup];
    if (!group || !(control.dataset.configKey in group)) continue;
    const value = group[control.dataset.configKey];
    if (control.type === "checkbox") {
      control.checked = Boolean(value);
    } else {
      control.value = value === null || value === undefined ? "" : String(value);
    }
  }
  const safetyValues = [
    ["minimum_families", "#safety-minimum-families"],
    ["minimum_deviations_per_family", "#safety-minimum-deviations"],
    ["recommended_deviations_per_family", "#safety-recommended-deviations"],
  ];
  for (const [key, selector] of safetyValues) {
    const value = defaults.safety?.[key];
    if (value !== undefined) document.querySelector(selector).textContent = String(value);
  }
}


function collectConfig() {
  const config = clone(state.defaults);
  const controls = document.querySelectorAll("[data-config-group][data-config-key]");
  for (const control of controls) {
    const group = control.dataset.configGroup;
    const key = control.dataset.configKey;
    if (!config[group] || typeof config[group] !== "object") config[group] = {};
    if (control.type === "checkbox") {
      config[group][key] = control.checked;
    } else if (control.type === "number") {
      config[group][key] = control.value === "" && control.dataset.nullable === "true"
        ? null
        : Number(control.value);
    } else {
      config[group][key] = control.value.trim();
    }
  }
  // Canvas size has one control; the model must use the same spatial size.
  if (config.model && config.preprocessing) {
    config.model.image_size = config.preprocessing.image_size;
  }
  return config;
}


function failField(control, message) {
  const panel = control.closest("[data-panel]");
  if (panel) selectTab(panel.dataset.panel);
  const details = control.closest("details");
  if (details) details.open = true;
  control.setCustomValidity(message);
  control.reportValidity();
  control.addEventListener("input", () => control.setCustomValidity(""), { once: true });
  showToast(message);
  return false;
}


function reportRelevantValidity(kind) {
  const groups = {
    validate: new Set(["preprocessing", "registration"]),
    train: new Set(["preprocessing", "registration", "model", "training", "novelty", "quality"]),
    generate: new Set(["generation", "novelty", "quality"]),
  }[kind];
  if (!groups) return false;
  const controls = document.querySelectorAll("[data-config-group][data-config-key]");
  for (const control of controls) {
    if (!groups.has(control.dataset.configGroup)) continue;
    if (!control.checkValidity()) {
      control.reportValidity();
      return false;
    }
  }
  return true;
}


function validateConfig(kind) {
  if (!reportRelevantValidity(kind)) return false;
  const config = collectConfig();
  const paths = config.paths || {};
  const requiredPaths = {
    validate: [["data", "Choose the paired-family dataset folder."], ["report", "Choose a validation report folder."]],
    train: [["data", "Choose the paired-family dataset folder on the Dataset tab."], ["run", "Choose a training run folder."]],
    generate: [
      ["checkpoint", "Choose a paired-family model checkpoint."],
      ["base", "Choose the base image to build upon."],
      ["out", "Choose a generation output folder."],
    ],
  }[kind];
  for (const [key, message] of requiredPaths) {
    if (!paths[key]) {
      return failField(document.querySelector(`[data-config-group="paths"][data-config-key="${key}"]`), message);
    }
  }

  if (kind === "validate" || kind === "train") {
    const imageSize = config.preprocessing.image_size;
    if (imageSize % 16 !== 0) return failField(document.querySelector("#pre-image-size"), "Image size must be divisible by 16.");
    if (config.preprocessing.margin >= imageSize / 3) return failField(document.querySelector("#pre-margin"), "Margin must be smaller than one third of the image size.");
  }

  if (kind === "train") {
    if (config.model.min_stroke_width > config.model.max_stroke_width) return failField(document.querySelector("#model-max-stroke"), "Maximum stroke width must be at least the minimum stroke width.");
    if (paths.resume && paths.init_checkpoint) {
      return failField(document.querySelector("#path-init-checkpoint"), "Choose either exact resume or weight initialization, not both.");
    }
    const training = config.training;
    const exampleTotal = training.real_pair_probability
      + training.synthetic_pair_probability
      + training.identity_pair_probability;
    if (Math.abs(exampleTotal - 1) > 0.000001) {
      return failField(document.querySelector("#train-identity-pair"), "Real, synthetic, and identity example probabilities must add up to 1.00.");
    }
  }

  if (kind === "train" || kind === "generate") {
    const novelty = config.novelty;
    if (novelty.review_threshold >= novelty.duplicate_threshold) return failField(document.querySelector("#nov-duplicate"), "Duplicate threshold must be greater than the review threshold.");
    if (novelty.review_threshold > novelty.transformed_review_threshold) return failField(document.querySelector("#nov-transform"), "Rotated / mirrored review threshold must be at least the review threshold.");
    const weightTotal = novelty.skeleton_weight + novelty.rendered_weight + novelty.topology_weight;
    if (Math.abs(weightTotal - 1) > 0.000001) return failField(document.querySelector("#nov-topology-weight"), "Novelty metric weights must add up to 1.00.");
    if (novelty.precise_finalists > novelty.shortlist_maximum) return failField(document.querySelector("#nov-finalists"), "Precise finalists cannot exceed the shortlist maximum.");
  }
  return true;
}


function setBusy(busy) {
  for (const button of ui.actionButtons) button.disabled = busy;
  ui.cancel.disabled = !busy;
}


function findNamedData(sources, names) {
  const wanted = new Set(names.map((name) => name.toLowerCase()));
  const queue = (Array.isArray(sources) ? sources : [sources])
    .filter((value) => value && typeof value === "object")
    .map((value) => ({ value, depth: 0 }));
  const seen = new Set();
  while (queue.length) {
    const { value, depth } = queue.shift();
    if (!value || typeof value !== "object" || seen.has(value) || depth > 5) continue;
    seen.add(value);
    const entries = Array.isArray(value)
      ? value.slice(0, 100).map((item, index) => [String(index), item])
      : Object.entries(value).slice(0, 200);
    for (const [key, item] of entries) {
      if (wanted.has(key.toLowerCase()) && item && typeof item === "object") return item;
    }
    for (const [_key, item] of entries) {
      if (item && typeof item === "object") queue.push({ value: item, depth: depth + 1 });
    }
  }
  return null;
}


function flattenRecord(record) {
  const output = {};
  for (const [key, value] of Object.entries(record || {})) {
    if (value === null || typeof value !== "object") {
      output[key] = value;
    } else if (Array.isArray(value)) {
      output[key] = value.every((item) => item === null || typeof item !== "object")
        ? value.join(", ")
        : `${value.length} item${value.length === 1 ? "" : "s"}`;
    } else {
      const simpleEntries = Object.entries(value)
        .filter(([_nestedKey, item]) => item === null || typeof item !== "object")
        .slice(0, 12);
      if (simpleEntries.length) {
        for (const [nestedKey, item] of simpleEntries) output[`${key}_${nestedKey}`] = item;
      } else {
        output[key] = "Details saved in manifest";
      }
    }
  }
  return output;
}


function normalizeRecords(value, kind) {
  if (!value || typeof value !== "object") return [];
  if (Array.isArray(value)) {
    return value.map((item, index) => (
      item && typeof item === "object"
        ? flattenRecord(item)
        : { [kind === "family" ? "family_id" : "fold"]: item ?? index + 1 }
    ));
  }

  const preferred = kind === "family"
    ? ["families", "family_summaries", "summaries", "items", "results"]
    : ["folds", "fold_metrics", "results", "metrics"];
  for (const key of preferred) {
    if (Array.isArray(value[key])) {
      const records = normalizeRecords(value[key], kind);
      if (kind === "audit" && value.macro_average && typeof value.macro_average === "object") {
        records.push(flattenRecord({ fold: "Macro average", ...value.macro_average }));
      }
      return records;
    }
  }

  const entries = Object.entries(value);
  if (entries.length && entries.every(([_key, item]) => item && typeof item === "object" && !Array.isArray(item))) {
    return entries.map(([key, item]) => flattenRecord({ [kind === "family" ? "family_id" : "fold"]: key, ...item }));
  }
  return [flattenRecord(value)];
}


function displayCell(value) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number" && !Number.isInteger(value)) {
    return Math.abs(value) >= 1000 ? value.toLocaleString() : Number(value.toPrecision(5)).toString();
  }
  return String(value);
}


function renderRecordTable(card, target, value, kind) {
  const records = normalizeRecords(value, kind).slice(0, 200);
  if (!records.length) {
    card.hidden = true;
    target.replaceChildren();
    return false;
  }
  const preferred = kind === "family"
    ? ["family_id", "family", "status", "usable", "valid", "train", "validation", "warnings"]
    : [
      "fold",
      "held_out_families",
      "status",
      "epoch",
      "scores_reconstruction_dice",
      "scores_prior_sample_quality",
      "scores_base_retention",
      "scores_diversity",
      "scores_change_distribution_agreement",
      "scores_eligible_deviations",
      "scores_excluded_cross_family_duplicates",
      "deviation_count",
      "audit_epochs",
    ];
  const found = new Set(records.flatMap((record) => Object.keys(record)));
  const columns = [
    ...preferred.filter((key) => found.has(key)),
    ...[...found].filter((key) => !preferred.includes(key)),
  ].slice(0, 14);
  if (!columns.length) {
    card.hidden = true;
    return false;
  }

  const table = document.createElement("table");
  table.className = "data-table";
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const column of columns) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = titleCase(column.replace(/^scores_/, ""));
    headRow.append(cell);
  }
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const record of records) {
    const row = document.createElement("tr");
    for (const column of columns) {
      const cell = document.createElement("td");
      cell.textContent = displayCell(record[column]);
      cell.title = cell.textContent;
      row.append(cell);
    }
    body.append(row);
  }
  table.append(head, body);
  target.replaceChildren(table);
  card.hidden = false;
  return true;
}


function renderPairedResults(result, payload) {
  const sources = [result, payload];
  const familyData = findNamedData(sources, ["family_summaries", "family_summary", "per_family", "families"]);
  let auditData = findNamedData(sources, ["audit", "audit_metrics", "audit_results", "unseen_base_audit", "fold_metrics", "folds"]);
  if (!auditData && payload?.stage === "audit") {
    auditData = [{
      fold: payload.fold,
      epoch: payload.epoch,
      status: "running",
      metrics: payload.metrics || {},
    }];
  }
  const hasFamilies = renderRecordTable(ui.familyCard, ui.familyTable, familyData, "family");
  const hasAudit = renderRecordTable(ui.auditCard, ui.auditTable, auditData, "audit");
  ui.pairedResults.hidden = !(hasFamilies || hasAudit);
}


function updateJob(job) {
  if (!job) return;
  state.jobId = job.id;
  const status = job.status || "idle";
  const active = status === "running" || status === "cancelling";
  const progress = Math.max(0, Math.min(1, Number(job.progress) || 0));
  captureLiveLoss(job);
  ui.status.textContent = status;
  ui.status.dataset.status = status;
  ui.message.textContent = job.message || "Ready";
  ui.progress.value = progress;
  ui.progress.textContent = `${Math.round(progress * 100)}%`;
  ui.progress.setAttribute("aria-valuetext", `${Math.round(progress * 100)} percent`);
  ui.summaryStatus.textContent = job.id ? `${job.kind} · ${status}` : "No job yet";
  setBusy(active);

  const result = job.result ?? (Object.keys(job.payload || {}).length ? job.payload : null);
  renderPairedResults(job.result, job.payload);
  if (result !== null) {
    ui.result.textContent = JSON.stringify(result, null, 2);
  } else if (job.id) {
    ui.result.textContent = `${job.kind} is ${status}.\n\nProgress: ${Math.round(progress * 100)}%\n${job.message || ""}`;
  }

  if (active) {
    startPolling();
  } else {
    stopPolling();
    if (job.id && state.lastFinishedId !== job.id) {
      state.lastFinishedId = job.id;
      refreshArtifacts();
    }
  }
}


function appendLogs(logs) {
  if (!Array.isArray(logs) || logs.length === 0) return;
  const shouldStick = ui.log.scrollHeight - ui.log.scrollTop - ui.log.clientHeight < 32;
  const fragment = document.createDocumentFragment();
  for (const entry of logs) {
    const row = document.createElement("div");
    row.className = "log-line";
    row.dataset.level = entry.level || "info";
    const time = document.createElement("span");
    time.className = "log-time";
    const date = new Date(entry.time);
    time.textContent = Number.isNaN(date.valueOf()) ? "" : date.toLocaleTimeString([], { hour12: false });
    const text = document.createElement("span");
    text.className = "log-text";
    text.textContent = entry.message || "";
    row.append(time, text);
    fragment.append(row);
  }
  ui.log.append(fragment);
  while (ui.log.childElementCount > 600) ui.log.firstElementChild.remove();
  if (shouldStick) ui.log.scrollTop = ui.log.scrollHeight;
}


function ingestSnapshot(payload) {
  appendLogs(payload.logs);
  if (Number.isInteger(payload.log_cursor)) state.cursor = payload.log_cursor;
  updateJob(payload.job);
}


async function pollJob() {
  if (state.polling) return;
  state.polling = true;
  try {
    const payload = await api(`/api/jobs?after=${state.cursor}`);
    ingestSnapshot(payload);
  } catch (error) {
    showToast(error.message);
  } finally {
    state.polling = false;
  }
}


function startPolling() {
  if (state.pollTimer !== null) return;
  state.pollTimer = window.setInterval(pollJob, 750);
}


function stopPolling() {
  if (state.pollTimer === null) return;
  window.clearInterval(state.pollTimer);
  state.pollTimer = null;
}


async function startJob(kind) {
  if (!validateConfig(kind)) return;
  state.cursor = 0;
  state.lastFinishedId = null;
  state.losses = [];
  state.lastLossStep = null;
  ui.lossCard.hidden = true;
  ui.log.replaceChildren();
  renderPairedResults(null, null);
  ui.result.textContent = "Starting…";
  setBusy(true);
  selectTab("results");
  try {
    const payload = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ kind, config: collectConfig() }),
    });
    updateJob(payload.job);
    await pollJob();
  } catch (error) {
    setBusy(false);
    showToast(error.message);
    ui.result.textContent = error.message;
  }
}


async function cancelJob() {
  ui.cancel.disabled = true;
  try {
    const payload = await api("/api/jobs/cancel", { method: "POST", body: "{}" });
    updateJob(payload.job);
    await pollJob();
  } catch (error) {
    showToast(error.message);
    ui.cancel.disabled = false;
  }
}


async function choosePath(button) {
  const target = document.getElementById(button.dataset.target);
  if (!target) return;
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = "Opening…";
  try {
    const payload = await api("/api/pick", {
      method: "POST",
      body: JSON.stringify({
        kind: button.dataset.picker,
        title: button.dataset.title || "Choose a path",
        initial: target.value,
      }),
    });
    if (payload.path) {
      target.value = payload.path;
      target.dispatchEvent(new Event("input", { bubbles: true }));
    }
  } catch (error) {
    showToast(error.message);
    target.focus();
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}


function humanSize(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size < 10 ? size.toFixed(1) : Math.round(size)} ${units[unit]}`;
}


function artifactCard(file) {
  const item = document.createElement("article");
  item.className = "artifact-item";
  if (file.preview) {
    const previewLink = document.createElement("a");
    previewLink.className = "artifact-preview";
    previewLink.href = file.url;
    previewLink.target = "_blank";
    previewLink.rel = "noopener";
    const image = document.createElement("img");
    image.src = file.url;
    image.alt = `Preview of ${file.name}`;
    image.loading = "lazy";
    previewLink.append(image);
    item.append(previewLink);
  }
  const link = document.createElement("a");
  link.className = "artifact-meta";
  link.href = file.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.title = file.path;
  const name = document.createElement("span");
  name.className = "artifact-name";
  name.textContent = file.name;
  const path = document.createElement("span");
  path.className = "artifact-path";
  path.textContent = `${file.relative} · ${humanSize(file.size)}`;
  link.append(name, path);
  item.append(link);
  return item;
}


async function refreshArtifacts() {
  try {
    const payload = await api("/api/artifacts");
    const files = payload.files || [];
    ui.artifactGrid.replaceChildren();
    if (files.length === 0) {
      const empty = document.createElement("p");
      empty.className = "artifact-empty";
      empty.textContent = "Artifacts will appear after a job writes to its selected report, run, or output folder.";
      ui.artifactGrid.append(empty);
      ui.artifactCount.textContent = "No files found in selected output folders.";
      return;
    }
    const metricsFile = [...files].reverse().find((file) => file.name.toLowerCase() === "metrics.json");
    if (metricsFile) loadLossArtifact(metricsFile.url);
    const visible = files.slice(0, 300);
    const fragment = document.createDocumentFragment();
    for (const file of visible) fragment.append(artifactCard(file));
    ui.artifactGrid.append(fragment);
    const suffix = payload.truncated || files.length > visible.length ? " (display limited)" : "";
    ui.artifactCount.textContent = `${files.length.toLocaleString()} file${files.length === 1 ? "" : "s"}${suffix}`;
  } catch (error) {
    showToast(error.message);
  }
}


function captureLiveLoss(job) {
  if (job.kind !== "train" || !job.payload || typeof job.payload !== "object") return;
  const metrics = job.payload.metrics;
  const loss = Number(metrics?.loss);
  if (!Number.isFinite(loss)) return;
  const step = `${job.payload.epoch ?? ""}:${job.payload.batch ?? ""}`;
  if (step === state.lastLossStep) return;
  state.lastLossStep = step;
  state.losses.push({ train: loss, validation: null });
  if (state.losses.length > 1200) state.losses.splice(0, state.losses.length - 1200);
  drawLossChart();
}


async function loadLossArtifact(url) {
  try {
    const response = await fetch(url, { credentials: "same-origin" });
    if (!response.ok) return;
    const rows = await response.json();
    if (!Array.isArray(rows) || rows.length === 0) return;
    const losses = rows.map((row) => ({
      train: Number(row.train_loss),
      validation: Number(row.validation_loss),
    })).filter((row) => Number.isFinite(row.train) || Number.isFinite(row.validation));
    if (losses.length) {
      state.losses = losses;
      drawLossChart();
    }
  } catch (_error) {
    // A partially-written metrics file should not disrupt the rest of Results.
  }
}


function drawLossChart() {
  if (!state.losses.length) return;
  ui.lossCard.hidden = false;
  const canvas = ui.lossChart;
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(320, Math.round(bounds.width));
  const height = 220;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);

  const values = state.losses.flatMap((row) => [row.train, row.validation]).filter(Number.isFinite);
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    minimum = Math.max(0, minimum - 0.5);
    maximum += 0.5;
  }
  const padding = { top: 18, right: 16, bottom: 26, left: 48 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const x = (index) => padding.left + (state.losses.length === 1 ? innerWidth / 2 : index * innerWidth / (state.losses.length - 1));
  const y = (value) => padding.top + (maximum - value) * innerHeight / (maximum - minimum);

  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#deded7";
  context.fillStyle = "#797970";
  context.font = "10px ui-monospace, monospace";
  context.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const lineY = padding.top + line * innerHeight / 4;
    const value = maximum - line * (maximum - minimum) / 4;
    context.beginPath();
    context.moveTo(padding.left, lineY);
    context.lineTo(width - padding.right, lineY);
    context.stroke();
    context.fillText(value.toPrecision(3), 6, lineY + 3);
  }

  const paint = (key, color) => {
    context.beginPath();
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.lineCap = "round";
    let started = false;
    state.losses.forEach((row, index) => {
      const value = row[key];
      if (!Number.isFinite(value)) {
        started = false;
        return;
      }
      if (!started) context.moveTo(x(index), y(value));
      else context.lineTo(x(index), y(value));
      started = true;
    });
    context.stroke();
  };
  paint("train", "#245e4b");
  paint("validation", "#b66b3d");
  context.fillStyle = "#797970";
  context.fillText("start", padding.left, height - 8);
  const endLabel = String(state.losses.length);
  context.fillText(endLabel, width - padding.right - context.measureText(endLabel).width, height - 8);
}


function titleCase(text) {
  return String(text).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}


function flattenEnvironment(value, prefix = "") {
  const output = [];
  for (const [key, item] of Object.entries(value || {})) {
    const label = prefix ? `${prefix} · ${key}` : key;
    if (item && typeof item === "object" && !Array.isArray(item)) {
      output.push(...flattenEnvironment(item, label));
    } else {
      output.push([label, item]);
    }
  }
  return output;
}


function renderEnvironment(environment) {
  ui.environment.replaceChildren();
  const fragment = document.createDocumentFragment();
  for (const [labelText, value] of flattenEnvironment(environment)) {
    const item = document.createElement("div");
    item.className = "environment-item";
    const label = document.createElement("span");
    label.className = "environment-label";
    label.textContent = titleCase(labelText);
    const displayed = document.createElement("span");
    displayed.className = "environment-value";
    if (typeof value === "boolean") {
      displayed.classList.add(value ? "good" : "bad");
      displayed.textContent = value ? "Available" : "Unavailable";
    } else if (value === null || value === "") {
      displayed.textContent = "None";
    } else {
      displayed.textContent = String(value);
      displayed.title = String(value);
    }
    item.append(label, displayed);
    fragment.append(item);
  }
  ui.environment.append(fragment);
}


function showToast(message) {
  window.clearTimeout(state.toastTimer);
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  ui.toast.dataset.hiding = "false";
  state.toastTimer = window.setTimeout(() => {
    ui.toast.dataset.hiding = "true";
    state.toastTimer = window.setTimeout(() => {
      ui.toast.hidden = true;
    }, 170);
  }, 4200);
}


async function reloadEnvironment() {
  try {
    const payload = await api("/api/bootstrap");
    state.token = payload.token;
    renderEnvironment(payload.environment);
    showToast("Environment refreshed.");
  } catch (error) {
    showToast(error.message);
  }
}


function bindControls() {
  for (const button of ui.actionButtons) {
    button.addEventListener("click", () => startJob(button.dataset.action));
  }
  for (const button of document.querySelectorAll("[data-picker]")) {
    button.addEventListener("click", () => choosePath(button));
  }
  ui.cancel.addEventListener("click", cancelJob);
  document.querySelector("#refresh-results").addEventListener("click", refreshArtifacts);
  document.querySelector("#clear-log").addEventListener("click", () => ui.log.replaceChildren());
  document.querySelector("#reload-environment").addEventListener("click", reloadEnvironment);
  ui.form.addEventListener("invalid", (event) => {
    const panel = event.target.closest("[data-panel]");
    const details = event.target.closest("details");
    if (panel) selectTab(panel.dataset.panel);
    if (details) details.open = true;
  }, true);
  window.addEventListener("resize", () => {
    window.clearTimeout(state.resizeTimer);
    state.resizeTimer = window.setTimeout(() => {
      if (!ui.lossCard.hidden) drawLossChart();
    }, 120);
  });
}


async function bootstrap() {
  bindTabs();
  bindControls();
  try {
    const payload = await api("/api/bootstrap");
    state.token = payload.token;
    state.defaults = payload.defaults || {};
    populateForm(state.defaults);
    renderEnvironment(payload.environment);
    ingestSnapshot(payload);
    await refreshArtifacts();
    ui.message.textContent = payload.job?.message || "Ready";
  } catch (error) {
    ui.connectionError.hidden = false;
    ui.connectionError.textContent = error.message;
    ui.message.textContent = "Server unavailable";
    setBusy(false);
    for (const button of ui.actionButtons) button.disabled = true;
  }
}


bootstrap();
