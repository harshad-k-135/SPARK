(() => {
  'use strict';

  const byId = id => document.getElementById(id);
  const displays = {
    decode_tokens_per_sec: { label: 'Decode speed', unit: 'tok/s', higher: true },
    first_token_ms: { label: 'First token', unit: 'ms', higher: false },
    prefill_tokens_per_sec: { label: 'Prefill speed', unit: 'tok/s', higher: true },
    peak_vram_mib: { label: 'Peak VRAM', unit: 'MiB', higher: false },
    decode_ms: { label: 'Decode latency', unit: 'ms', higher: false },
    gpu_utilization_pct: { label: 'GPU use', unit: '%', higher: null },
  };
  const engines = [['spark', 'SPARK'], ['llama_cpp', 'llama.cpp']];
  const number = value => Number.isFinite(value) ? value.toLocaleString(undefined,
    { maximumFractionDigits: 2 }) : '—';
  let comparison;
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

  function draw() {
    const report = comparison.report;
    const workload = report.workloads.find(group => group.prompt_tokens === promptLength) ||
      report.workloads[0];
    promptLength = workload.prompt_tokens;
    const selected = displays[metric];
    const stats = engines.map(([id, label]) =>
      ({ id, label, ...workload.engines[id].metrics[metric] }));
    byId('context-options').replaceChildren(...report.workloads.map(group =>
      button(group.prompt_tokens + ' tokens', promptLength === group.prompt_tokens, () => {
        promptLength = group.prompt_tokens; draw();
        byId('context-options').querySelector('[aria-pressed="true"]').focus();
      })));
    byId('metric-options').replaceChildren(...Object.entries(displays).map(([id, details]) =>
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
      winner.textContent = 'Median utilization during measured trials';
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
    if (chart.firstChild) showTrial(0, chart.firstChild);
    const prefix = workload.greedy_prefix_match;
    byId('quality').textContent = prefix === report.output_tokens
      ? 'Identical ' + prefix + '-token greedy sequence on the first measured trial.'
      : 'First measured trial shared ' + prefix + ' of ' + report.output_tokens +
        ' leading token IDs. Output diverged, so the speed comparison is workload-specific.';
    byId('protocol').textContent = report.warmups + ' warm-ups · ' + report.measured +
      ' measured per runtime · ' + report.output_tokens + ' output tokens · ' + report.context + ' context';
    byId('compare-evidence').href = '/runs/' + encodeURIComponent(comparison.id);
    byId('compare-body').hidden = false;
    byId('context-options').querySelector('[aria-pressed="true"]')?.setAttribute('title', 'Selected prompt length');
  }

  async function refresh() {
    const response = await fetch('/api/comparison');
    if (!response.ok) throw Error('Comparison results are unavailable.');
    comparison = await response.json();
    if (comparison?.report.workloads.length) {
      draw();
      byId('compare-state').textContent = '';
    } else {
      byId('compare-state').textContent = 'Run the comparison to see both CUDA engines side by side.';
    }
  }

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
        const result = await fetch('/runs/' + encodeURIComponent(job.id));
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

  refresh().catch(error => { byId('compare-state').textContent = error.message; });
})();
