const $ = (selector) => document.querySelector(selector);
const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;
const formatMetric = (value) => Number(value).toFixed(3);

function toast(message) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2600);
}

function renderMetrics(data) {
  const best = data.metrics[0];
  $("#bestModel").textContent = data.best_model;
  $("#bestAuc").textContent = formatMetric(best.roc_auc);
  $("#datasetRows").textContent = data.dataset_rows.toLocaleString();
  $("#positiveRate").textContent = formatPercent(data.positive_rate);
  $("#trainedAt").textContent = `Trained ${data.trained_at}`;

  $("#metricsTable").innerHTML = data.metrics.map((row, index) => `
    <tr class="${index === 0 ? "best-row" : ""}">
      <td><strong>${row.model}</strong>${index === 0 ? '<span class="best-label">Best</span>' : ""}</td>
      <td>${formatMetric(row.accuracy)}</td><td>${formatMetric(row.precision)}</td>
      <td>${formatMetric(row.recall)}</td><td>${formatMetric(row.f1)}</td><td><strong>${formatMetric(row.roc_auc)}</strong></td>
    </tr>`).join("");

  $("#importanceChart").innerHTML = data.feature_importance.map(item => `
    <div class="importance-row"><span>${item.feature}</span><div class="bar-track"><div class="bar-fill" style="width:${item.value * 100}%"></div></div><strong>${item.value.toFixed(2)}</strong></div>
  `).join("");
}

async function loadMetrics() {
  const response = await fetch("/api/metrics");
  if (!response.ok) throw new Error("Could not load model metrics");
  renderMetrics(await response.json());
}

function renderPrediction(result) {
  $("#emptyResult").classList.add("hidden");
  $("#predictionResult").classList.remove("hidden");
  $("#riskProbability").textContent = formatPercent(result.probability);
  $("#decisionText").textContent = result.decision;
  $("#predictionModel").textContent = result.best_model;
  $("#riskBadge").textContent = `${result.risk_level} risk`;
  const colors = { Low: "#2563eb", Moderate: "#d97706", High: "#dc2626" };
  $(".risk-gauge").style.borderColor = colors[result.risk_level];
  $("#riskBadge").style.color = colors[result.risk_level];
  $("#modelConsensus").innerHTML = Object.entries(result.model_probabilities).map(([name, value]) => `
    <div class="consensus-row"><span>${name}</span><div class="bar-track"><div class="bar-fill" style="width:${value * 100}%"></div></div><strong>${formatPercent(value)}</strong></div>
  `).join("");
  $("#riskDrivers").innerHTML = result.drivers.map(driver => `<span>${driver}</span>`).join("");
}

$("#predictionForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "Calculating score...";
  const payload = Object.fromEntries(new FormData(event.currentTarget).entries());
  try {
    const response = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    renderPrediction(result);
    toast("Risk score generated");
  } catch (error) {
    toast(error.message || "Prediction failed");
  } finally {
    button.disabled = false;
    button.textContent = "Calculate risk score";
  }
});

$("#retrainButton").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Refreshing...";
  try {
    const response = await fetch("/api/retrain", { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    renderMetrics(result);
    toast(`Models refreshed in ${result.trained_in}s`);
  } catch (error) {
    toast(error.message || "Retraining failed");
  } finally {
    button.disabled = false;
    button.textContent = "Refresh models";
  }
});

async function loadDataPreview() {
  const text = await fetch("customer_default_dataset.csv").then(response => response.text());
  const lines = text.trim().split("\n").slice(0, 9).map(line => line.split(","));
  const [headers, ...rows] = lines;
  $("#dataTableWrap").innerHTML = `<table><thead><tr>${headers.map(cell => `<th>${cell.replaceAll("_", " ")}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

document.querySelectorAll("nav a").forEach(link => link.addEventListener("click", () => {
  document.querySelectorAll("nav a").forEach(item => item.classList.remove("active"));
  link.classList.add("active");
}));

Promise.all([loadMetrics(), loadDataPreview()]).catch(error => toast(error.message));
