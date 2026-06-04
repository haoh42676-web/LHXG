const runBtn = document.querySelector("#runBtn");
const message = document.querySelector("#message");
const gate = document.querySelector("#gate");
const metricsEl = document.querySelector("#metrics");
const targetTable = document.querySelector("#targetTable");
const rankTable = document.querySelector("#rankTable");
const logoutBtn = document.querySelector("#logoutBtn");

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function renderMetrics(metrics, targetCount) {
  const items = [
    ["目标股票", targetCount],
    ["全市场股票", metrics.stock_pool],
    ["有效样本", metrics.feature_rows],
    ["1日成功率", pct(metrics.success_1d)],
    ["5日成功率", pct(metrics.success_5d)],
    ["1个月成功率", pct(metrics.success_20d)],
    ["优化轮数", metrics.optimization_attempts],
    ["最佳模型", metrics.best_model],
    ["失败数据源", metrics.data_source_failures],
    ["新闻条数", metrics.news_rows],
  ];
  metricsEl.innerHTML = items.map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value ?? "-"}</strong></div>`).join("");
  const pass = metrics.accuracy_gate_95 === 1;
  gate.textContent = pass ? "误差0.5个百分点验证：通过95%门槛" : "误差0.5个百分点验证：未达95%，已保留当前最优模型";
  gate.className = `gate ${pass ? "pass" : "fail"}`;
}

function renderTable(table, rows, targetOnly = false) {
  if (!rows || rows.length === 0) {
    table.innerHTML = `<tbody><tr><td class="empty">${targetOnly ? "没有股票同时满足目标条件。" : "暂无数据。"}</td></tr></tbody>`;
    return;
  }
  const columns = [
    ["name", "股票", (v) => v],
    ["code", "股票代码", (v) => v],
    ["pred_1d", "预计1日涨幅", pct],
    ["pred_5d", "预计5日涨幅", pct],
    ["pred_20d", "预计1个月涨幅", pct],
    ["risk_rate", "风险率", pct],
    ["basis", "依据", (v) => v || ""],
  ];
  const thead = `<thead><tr>${columns.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map((row) => `<tr>${columns.map(([key, , format]) => `<td>${format(row[key])}</td>`).join("")}</tr>`).join("")}</tbody>`;
  table.innerHTML = thead + tbody;
}

async function poll(taskId) {
  const response = await fetch(`/api/status/${taskId}`);
  const data = await response.json();
  message.textContent = data.message || data.status;
  if (data.status === "done") {
    runBtn.disabled = false;
    runBtn.textContent = "一键开始选股";
    renderMetrics(data.result.metrics, data.result.target_count);
    renderTable(targetTable, data.result.target, true);
    renderTable(rankTable, data.result.ranking, false);
    message.textContent = `完成，结果已保存：${data.result.output_csv}`;
    return;
  }
  if (data.status === "error") {
    runBtn.disabled = false;
    runBtn.textContent = "一键开始选股";
    message.textContent = `运行失败：${data.message}`;
    return;
  }
  setTimeout(() => poll(taskId), 2500);
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  runBtn.textContent = "运行中";
  message.textContent = "任务启动中";
  targetTable.innerHTML = "";
  rankTable.innerHTML = "";
  metricsEl.innerHTML = "";
  gate.textContent = "误差0.5个百分点验证：验证中";
  gate.className = "gate";
  const topNValue = document.querySelector("#topN").value;
  const payload = {
    top_n: topNValue ? Number(topNValue) : null,
    lookback_days: Number(document.querySelector("#lookbackDays").value || 1825),
    adjust: document.querySelector("#adjust").value,
    fetch_news: document.querySelector("#fetchNews").checked,
    refresh_spot: false,
    refresh_history: document.querySelector("#refreshHistory").checked,
    refresh_news: document.querySelector("#refreshNews").checked,
    deepseek_model: "deepseek-chat",
  };
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "启动失败");
    poll(data.task_id);
  } catch (error) {
    runBtn.disabled = false;
    runBtn.textContent = "一键开始选股";
    message.textContent = `启动失败：${error}`;
  }
});

logoutBtn.addEventListener("click", async () => {
  await fetch("/api/logout", {method: "POST"});
  window.location.href = "/";
});
