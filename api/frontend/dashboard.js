'use strict';

const $ = id => document.getElementById(id);
const format = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : '—';
let controller;
let reports = [];
let benchmarkTimer;

async function refreshStatus() {
  try {
    const result = await fetch('/api/status');
    if (!result.ok) throw Error();
    const status = await result.json();
    $('device').textContent = status.busy ? 'CUDA · Busy' : status.loaded ? 'CUDA · Ready' : 'CUDA · On demand';
    $('model').textContent = status.model;
  } catch {
    $('device').textContent = 'Backend unavailable';
  }
}

$('generation').addEventListener('submit', async event => {
  event.preventDefault();
  controller = new AbortController();
  $('send').disabled = true;
  $('stop').hidden = false;
  for (const id of ['ttft', 'decode', 'generated']) $(id).textContent = '—';
  $('text').textContent = '';
  $('generation-state').textContent = 'Starting';
  const utf8 = new TextDecoder();
  const tokens = new TextDecoder();
  let pending = '';
  let ended = false;
  try {
    const response = await fetch('/api/generate', {
      method: 'POST', headers: { 'content-type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({prompt: $('prompt').value, max_tokens: Number($('limit').value)}),
    });
    if (!response.ok) {
      const error = await response.json();
      throw Error(typeof error.detail === 'string' ? error.detail : 'Invalid generation request.');
    }
    const reader = response.body.getReader();
    const handle = line => {
      const message = JSON.parse(line);
      if (message.event === 'loading') $('generation-state').textContent = 'Preparing CUDA';
      if (message.event === 'token') {
        $('generation-state').textContent = 'Generating';
        const chunks = message.bytes.match(/../g) || [];
        $('text').append(tokens.decode(Uint8Array.from(chunks, chunk => parseInt(chunk, 16)), {stream: true}));
      }
      if (message.event === 'error') throw Error(message.error);
      if (message.event === 'done') {
        ended = true;
        $('text').append(tokens.decode());
        $('ttft').textContent = message.generated_tokens ? format(message.first_token_ms, 0) + ' ms' : '—';
        $('decode').textContent = message.generated_tokens > 1 ? format(message.decode_tokens_per_sec) + ' t/s' : '—';
        $('generated').textContent = String(message.generated_tokens);
        $('generation-state').textContent = 'Complete';
      }
    };
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += utf8.decode(value, {stream: true});
      const lines = pending.split('\n');
      pending = lines.pop();
      for (const line of lines) if (line) handle(line);
    }
    if (pending.trim()) handle(pending);
    if (!ended) throw Error('Generation stream ended early.');
  } catch (error) {
    $('generation-state').textContent = error.name === 'AbortError' ? 'Stopped' : error.message;
  } finally {
    controller.abort();
    controller = undefined;
    $('send').disabled = false;
    $('stop').hidden = true;
    refreshStatus();
  }
});

$('stop').addEventListener('click', () => controller?.abort());

function benchmarkLabel(report) {
  if (report.runtime === 'spark-qwen-graph-bucket') return 'Bucketed CUDA Graph';
  if (report.runtime === 'spark-qwen-v0') return 'SPARK baseline';
  return report.runtime;
}

function renderChart(rows) {
  const latest = new Map();
  for (const row of rows) if (row.report && !latest.has(row.report.runtime))
    latest.set(row.report.runtime, row.report);
  const values = [...latest.values()].filter(r => Number.isFinite(r.metrics.decode_tokens_per_sec.median));
  const max = Math.max(1, ...values.map(r => r.metrics.decode_tokens_per_sec.median)) * 1.08;
  $('chart').replaceChildren();
  for (const report of values) {
    const value = report.metrics.decode_tokens_per_sec.median;
    const row = document.createElement('div');
    row.className = 'chart-row';
    const label = document.createElement('div');
    label.className = 'chart-label';
    const number = document.createElement('span');
    number.textContent = format(value);
    label.append(benchmarkLabel(report), number);
    const track = document.createElement('div');
    track.className = 'track';
    const bar = document.createElement('div');
    bar.className = 'bar' + (report.runtime === 'spark-qwen-v0' ? ' baseline' : '');
    bar.style.width = (value / max * 100) + '%';
    track.append(bar);
    row.append(label, track);
    $('chart').append(row);
  }
}

function renderDetails() {
  const selected = reports.find(row => row.id === $('trials').value);
  const container = $('details');
  container.replaceChildren();
  if (!selected?.report) {
    if (selected) container.textContent = selected.error || selected.state;
    return;
  }
  const metrics = selected.report.metrics;
  const table = document.createElement('table');
  const header = table.createTHead().insertRow();
  for (const heading of ['Metric', 'Median', 'IQR', 'P95']) {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = heading;
    header.append(cell);
  }
  const body = table.createTBody();
  for (const [key, label] of [
    ['decode_tokens_per_sec', 'Decode t/s'], ['prefill_tokens_per_sec', 'Prefill t/s'],
    ['first_token_ms', 'First token (ms)'], ['decode_ms', 'Decode (ms)'],
  ]) {
    const row = body.insertRow();
    for (const value of [label, ...['median', 'iqr', 'p95'].map(summary => format(metrics[key]?.[summary]))])
      row.insertCell().textContent = value;
  }
  const raw = document.createElement('a');
  raw.href = '/runs/' + encodeURIComponent(selected.id);
  raw.textContent = 'Raw trial data';
  raw.target = '_blank';
  raw.rel = 'noopener';
  raw.className = 'raw';
  container.append(table, raw);
}

async function refreshRuns() {
  const response = await fetch('/runs');
  if (!response.ok) throw Error('Benchmark runs unavailable.');
  reports = await response.json();
  const selected = $('trials').value;
  $('trials').replaceChildren();
  for (const row of reports) {
    const title = row.report ? benchmarkLabel(row.report) : row.name + ' · ' + row.state;
    $('trials').add(new Option(title, row.id));
  }
  if (reports.some(row => row.id === selected)) $('trials').value = selected;
  renderChart(reports);
  renderDetails();
}

$('trials').addEventListener('change', renderDetails);
$('benchmark').addEventListener('submit', async event => {
  event.preventDefault();
  $('run').disabled = true;
  $('benchmark-state').textContent = 'Queuing benchmark';
  try {
    const response = await fetch('/runs', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({mode: $('mode').value}),
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.detail || 'Could not queue benchmark.');
    const check = async () => {
      try {
        const resultResponse = await fetch('/runs/' + encodeURIComponent(result.id));
        if (!resultResponse.ok) throw Error('Could not retrieve the benchmark.');
        const row = await resultResponse.json();
        $('benchmark-state').textContent = row.state === 'running'
          ? 'Running five warm-ups and ten trials' : row.error || row.state;
        if (row.state === 'queued' || row.state === 'running') return;
        clearInterval(benchmarkTimer);
        await refreshRuns();
        $('trials').value = row.id;
        renderDetails();
        $('run').disabled = false;
      } catch (error) {
        clearInterval(benchmarkTimer);
        $('benchmark-state').textContent = error.message;
        $('run').disabled = false;
      }
    };
    benchmarkTimer = setInterval(check, 2500);
    await check();
  } catch (error) {
    $('benchmark-state').textContent = error.message;
    $('run').disabled = false;
  }
});

refreshStatus();
refreshRuns().catch(error => { $('benchmark-state').textContent = error.message; });
