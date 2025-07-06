// static/js/app.js

console.log("[DEBUG] app.js loaded");

// 拉取库存并渲染
async function fetchInventory() {
  console.log("[DEBUG] fetchInventory");
  const res = await fetch('/api/inventory');
  const data = await res.json();
  const container = document.getElementById('inventory');
  container.innerHTML = '';
  data.forEach(item => {
    const div = document.createElement('div');
    div.innerText = `${item.name}: ${item.count} 包`;
    if (item.missing > 0) {
      const span = document.createElement('span');
      span.style.color = 'red';
      span.innerText = ` 缺货 ${item.missing} 包`;
      div.appendChild(span);
    }
    container.appendChild(div);
  });
}

// 暂停/补货按钮
async function control(action) {
  console.log("[DEBUG] control:", action);
  await fetch('/api/control', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ action })
  });
}

// 生成 AI 分析报告
async function fetchAnalysis() {
  console.log("[DEBUG] fetchAnalysis called");
  const res = await fetch('/api/analysis');
  console.log("[DEBUG] /api/analysis status:", res.status);
  const { chart_url, analysis } = await res.json();
  console.log("[DEBUG] analysis data:", chart_url, analysis);
  document.getElementById('analysis_img').src = chart_url + '?t=' + Date.now();
  document.getElementById('analysis_text').innerText = analysis;
}

// 每秒更新一次库存
setInterval(fetchInventory, 1000);
fetchInventory();
