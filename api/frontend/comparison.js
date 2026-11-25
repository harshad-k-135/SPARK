(() => {
  'use strict';

  const byId = id => document.getElementById(id);
  const displays = {
    decode_tokens_per_sec: { label: 'Decode speed', unit: 'tok/s', higher: true },
    first_token_ms: { label: 'First token', unit: 'ms', higher: false },
    prefill_tokens_per_sec: { label: 'Prefill speed', unit: 'tok/s', higher: true },
    peak_vram_mib: { label: 'Peak VRAM', unit: 'MiB', higher: false },
    decode_ms: { label: 'Decode latency', unit: 'ms', higher: false },
    total_latency_ms: { label: 'Total latency', unit: 'ms', higher: false },
    peak_ram_mib: { label: 'Peak RAM', unit: 'MiB', higher: false },
    cpu_utilization_pct: { label: 'CPU use', unit: '%', higher: null },
    board_tokens_per_joule: { label: 'Board efficiency', unit: 'tok/J', higher: true },
    gpu_utilization_pct: { label: 'GPU use', unit: '%', higher: null },
  };
  const engines = [['spark', 'SPARK'], ['llama_cpp', 'llama.cpp']];
  const number = value => Number.isFinite(value) ? value.toLocaleString(undefined,
    { maximumFractionDigits: 2 }) : '—';
  let comparison;
  let history = [];
  let activeTrial = 0;
  let monitorRevision = 0;
  let historyRevision = 0;
  let promptLength = 512;
  let metric = 'decode_tokens_per_sec';

  function button(text, pressed, action) {
    const element = document.createElement('button');
    element.type = 'button';
    element.textContent = text;
    element.setAttribute('aria-pressed', String(pressed));
    element.addEventListener('click', action);
    return element;
  }

  function chart(canvas, samples, key, scale, unit) {
    const context = canvas.getContext('2d');
    const {width, height} = canvas;
    const left = 60, right = width - 16, top = 12, bottom = height - 27;
    context.clearRect(0, 0, width, height);
    const points = Object.values(samples).flat().filter(point => Number.isFinite(point[key]));
    if (!points.length) return false;
    const highest = scale || Math.ceil(Math.max(...points.map(point => point[key])) / 128) * 128 || 128;
    const duration = Math.max(1, ...points.map(point => point.elapsed_ms)) / 1000;
    context.font = '13px system-ui';
    context.fillStyle = '#b9c9dd';
    for (let i = 0; i <= 2; i++) {
      const y = bottom - (bottom - top) * i / 2;
      context.fillText(number(highest * i / 2) + unit, 0, y + 4);
      context.strokeStyle = '#334262';
      context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke();
    }
    context.fillText('0 s', left, height - 6);
    context.textAlign = 'right';
    context.fillText(number(duration) + ' s', right, height - 6);
    context.textAlign = 'left';
    for (const [index, [id]] of engines.entries()) {
      context.strokeStyle = index ? '#c297ff' : '#65f2cd';
      context.fillStyle = context.strokeStyle;
      context.lineWidth = 2;
      context.beginPath();
      let penDown = false;
      for (const point of samples[id] || []) {
        if (!Number.isFinite(point[key])) { penDown = false; continue; }
        const x = left + (right - left) * point.elapsed_ms / (duration * 1000);
        const y = bottom - (bottom - top) * point[key] / highest;
        if (penDown) context.lineTo(x, y); else context.moveTo(x, y);
        context.fillRect(x - 1, y - 1, 2, 2);
        penDown = true;
      }
      context.stroke();
    }
    return true;
  }

  async function refreshMonitor() {
    if (!byId('monitor').open || !comparison) return;
    const version = ++monitorRevision;
    const status = byId('monitor-state');
    status.textContent = 'Loading sampled telemetry';
    try {
      const params = new URLSearchParams({run: comparison.id, prompt: promptLength, trial: activeTrial + 1});
      const result = await fetch('/api/comparison?' + params);
      if (!result.ok) throw Error('Telemetry request failed.');
      const entries = await result.json();
      if (version !== monitorRevision) return;
      const utilization = chart(byId('monitor-util'), entries, 'utilization_pct', 100, '%');
      const memory = chart(byId('monitor-memory'), entries, 'memory_mib', null, ' MiB');
      status.textContent = utilization || memory
        ? 'Trial ' + (activeTrial + 1) + ' · SPARK in mint · llama.cpp in violet · sampled every 100 ms'
        : 'No board samples were captured for this trial.';
    } catch (error) { if (version === monitorRevision) status.textContent = error.message; }
  }

  byId('monitor').addEventListener('toggle', refreshMonitor);

  function draw() {
    const report = comparison.report;
    const workload = report.workloads.find(group => group.prompt_tokens === promptLength) ||
      report.workloads[0];
    promptLength = workload.prompt_tokens;
    const visibleMetrics = Object.entries(displays).filter(([key]) =>
      engines.some(([id]) => Number.isFinite(workload.engines[id].metrics[key]?.median)));
    if (!visibleMetrics.some(([key]) => key === metric)) metric = 'decode_tokens_per_sec';
    const selected = displays[metric];
    const stats = engines.map(([id, label]) =>
      ({ id, label, ...workload.engines[id].metrics[metric] }));
    byId('context-options').replaceChildren(...report.workloads.map(group =>
      button(group.prompt_tokens + ' tokens', promptLength === group.prompt_tokens, () => {
        promptLength = group.prompt_tokens; draw();
        byId('context-options').querySelector('[aria-pressed="true"]').focus();
      })));
    byId('metric-options').replaceChildren(...visibleMetrics.map(([id, details]) =>
      button(details.label, id === metric, () => {
        metric = id; draw();
        byId('metric-options').querySelector('[aria-pressed="true"]').focus();
      })));

    const [spark, llama] = stats;
    const winner = byId('winner');
    winner.className = 'winner';
    if (!stats.every(side => Number.isFinite(side.median))) {
      winner.textContent = 'Comparison unavailable for this metric';
    } else if (selected.higher === null) {
      winner.textContent = 'Median utilization across measured trials';
    } else if (spark.median === llama.median) {
      winner.textContent = 'Same median on this workload';
    } else {
      const aWins = selected.higher ? spark.median > llama.median : spark.median < llama.median;
      const a = aWins ? spark : llama;
      const b = aWins ? llama : spark;
      const delta = selected.higher ? (a.median / b.median - 1) * 100 : (1 - a.median / b.median) * 100;
      winner.classList.add(a.id);
      winner.textContent = a.label + ' leads by ' + number(delta) + '%' +
        (selected.higher ? ' on median throughput' : ' on the median');
    }
    const largest = Math.max(1, ...stats.flatMap(s => Number.isFinite(s.median) ? [s.median] : []));
    const bars = byId('compare-bars');
    bars.replaceChildren();
    for (const side of stats) {
      const wrapper = document.createElement('div');
      wrapper.className = 'match-bar ' + side.id;
      const label = document.createElement('div');
      label.className = 'bar-heading';
      const name = document.createElement('span');
      name.textContent = side.label;
      const median = document.createElement('strong');
      median.textContent = number(side.median) + ' ' + selected.unit;
      label.append(name, median);
      const rail = document.createElement('div');
      rail.className = 'track';
      const fill = document.createElement('div');
      fill.className = 'compare-fill';
      fill.style.width = (Number.isFinite(side.median) ? side.median * 100 / largest : 0) + '%';
      rail.append(fill);
      const summary = document.createElement('p');
      summary.textContent = 'IQR ' + number(side.iqr) + ' · P95 ' + number(side.p95) + ' ' + selected.unit;
      wrapper.append(label, rail, summary);
      bars.append(wrapper);
    }

    const trialPairs = engines.map(([id]) => workload.engines[id].trials.filter(t => !t.warmup));
    const finite = trialPairs.flatMap(trials => trials.map(t => t[metric]).filter(Number.isFinite));
    const height = Math.max(1, ...finite);
    const chart = byId('trial-chart');
    chart.replaceChildren();
    chart.style.gridTemplateColumns = 'repeat(' + report.measured + ', minmax(0,1fr))';
    byId('trial-unit').textContent = '0–' + number(height) + ' ' + selected.unit;
    const showTrial = (i, controls) => {
      activeTrial = i;
      refreshMonitor();
      for (const group of chart.querySelectorAll('button'))
        group.setAttribute('aria-pressed', String(group === controls));
      byId('trial-detail').textContent = 'Trial ' + (i + 1) + ' · ' + engines
        .map(([, label], engine) => label + ' ' + number(trialPairs[engine][i]?.[metric]) + ' ' + selected.unit)
        .join(' · ');
    };
    for (let i = 0; i < report.measured; i++) {
      const group = button('', false, () => showTrial(i, group));
      group.className = 'trial-pair';
      group.setAttribute('aria-label', 'Inspect trial ' + (i + 1));
      for (const [j, [id]] of engines.entries()) {
        const bar = document.createElement('span');
        bar.className = id;
        const value = trialPairs[j][i]?.[metric];
        bar.style.height = (Number.isFinite(value) ? value * 100 / height : 0) + '%';
        bar.setAttribute('aria-hidden', 'true');
        group.append(bar);
      }
      chart.append(group);
    }
    activeTrial = Math.min(activeTrial, report.measured - 1);
    if (chart.children[activeTrial]) showTrial(activeTrial, chart.children[activeTrial]);
    const prefix = workload.greedy_prefix_match;
    byId('quality').textContent = workload.matching_trials === report.measured
      ? 'Identical ' + report.output_tokens + '-token greedy output in all ' + report.measured + ' measured trials.'
      : workload.matching_trials + '/' + report.measured + ' measured pairs matched exactly. The first pair shared ' +
        prefix + ' of ' + report.output_tokens + ' leading token IDs.';
    byId('protocol').textContent = report.warmups + ' warm-ups · ' + report.measured +
      ' measured per runtime · ' + report.output_tokens + ' output tokens · ' + report.context + ' context';
    byId('runtime-config').textContent = workload.engines.spark.configuration;
    byId('compare-evidence').href = '/runs/' + encodeURIComponent(comparison.id) + '?download=true';
    byId('delete-comparison').hidden = history.find(run => run.id === comparison.id)?.saved !== false;
    byId('compare-body').hidden = false;
    byId('context-options').querySelector('[aria-pressed="true"]')?.setAttribute('title', 'Selected prompt length');
  }

  async function refresh(run = '') {
    const version = ++historyRevision;
    const [response, index] = await Promise.all([fetch(run ? '/api/comparison?run=' + encodeURIComponent(run) : '/api/comparison'),
      fetch('/api/comparisons')]);
    if (!index.ok) throw Error('Comparison history is unavailable.');
    const nextHistory = await index.json();
    if (!response.ok) throw Error('Comparison results are unavailable.');
    const next = await response.json();
    if (version !== historyRevision) return;
    history = nextHistory;
    comparison = next;
    byId('comparison-history').replaceChildren(...history.map(row => new Option(
      row.recorded_at ? row.id + ' · ' + new Date(row.recorded_at).toLocaleString()
        : row.id + ' · ' + (row.commit || row.name), row.id)));
    if (comparison?.report.workloads.length) {
      byId('comparison-history').value = comparison.id;
      draw();
      byId('compare-state').textContent = '';
    } else {
      byId('compare-body').hidden = true;
      byId('compare-state').textContent = 'Run the comparison to see both CUDA engines side by side.';
    }
  }

  const showError = error => { byId('compare-state').textContent = error.message; };
  byId('comparison-history').addEventListener('change', event => {
    refresh(event.target.value).catch(showError);
  });
  byId('delete-comparison').addEventListener('click', async () => {
    if (!comparison || !confirm('Delete this saved benchmark run?')) return;
    try {
      const response = await fetch('/runs/' + encodeURIComponent(comparison.id), {method: 'DELETE'});
      if (!response.ok) throw Error((await response.json()).detail || 'Could not delete the comparison.');
      await refresh();
    } catch (error) { showError(error); }
  });

  byId('compare-run').addEventListener('click', async () => {
    const run = byId('compare-run');
    run.disabled = true;
    const status = byId('compare-state');
    status.textContent = 'Queuing comparison';
    try {
      const response = await fetch('/runs', { method: 'POST',
        headers: { 'content-type': 'application/json' }, body: JSON.stringify({ mode: 'head-to-head' }) });
      const job = await response.json();
      if (!response.ok) throw Error(job.detail || 'Could not queue the comparison.');
      for (;;) {
        const result = await fetch('/runs/' + encodeURIComponent(job.id) + '?status_only=true');
        if (!result.ok) throw Error('Could not read the comparison status.');
        const row = await result.json();
        if (row.state === 'completed') { await refresh(); break; }
        if (row.state === 'failed') throw Error(row.error || 'Comparison failed.');
        status.textContent = row.state === 'queued'
          ? 'Waiting for the GPU' : 'Running 128-, 512-, and 1,024-token workloads';
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    } catch (error) { status.textContent = error.message; }
    finally { run.disabled = false; }
  });

  refresh().catch(showError);
})();
