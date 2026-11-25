const fs = require('node:fs');
const report = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const heading = process.argv[3] || 'CUDA benchmark';
const labels = {
    decode_tokens_per_sec: 'Decode (tok/s)', first_token_ms: 'TTFT (ms)',
    prefill_tokens_per_sec: 'Prefill (tok/s)', decode_ms: 'Decode latency (ms)',
    total_latency_ms: 'Total latency (ms)', peak_vram_mib: 'Peak board VRAM (MiB)',
    peak_ram_mib: 'Peak process RAM (MiB)', gpu_utilization_pct: 'Board GPU utilization (%)',
    cpu_utilization_pct: 'Engine CPU utilization (%)', board_tokens_per_joule: 'Board-only tok/J',
};
const names = {spark: 'SPARK', llama_cpp: 'llama.cpp'};
const f = value => Number.isFinite(value) ? value.toFixed(2) : 'Unavailable';
console.log('# ' + heading);
console.log('');
console.log('Measured on ' + report.hardware + '.');
console.log('Host ' + report.host.cpu + ', Windows ' + report.host.os +
    ', ' + (report.host.ram_bytes / 2 ** 30).toFixed(2) + ' GiB exposed RAM.');
console.log('SPARK commit ' + report.spark_commit + '. llama.cpp ' + report.llama_version.split(/\r?\n/)[0] + '.');
console.log('GGUF SHA-256 ' + report.model_sha256 + '.');
console.log('');
console.log(report.warmups + ' warm-ups, ' + report.measured + ' measured runs per runtime and workload.');
console.log('Each trial uses a single sequence, ' + report.output_tokens + ' forced greedy outputs and a ' +
    report.context + '-token context. See [methodology](../docs/BENCHMARKING.md).');
for (const group of report.workloads) {
    console.log('\n## ' + group.prompt_tokens + ' prompt tokens\n');
    console.log('Exact measured output matches ' + group.matching_trials + '/' + report.measured + '.\n');
    for (const [name, engine] of Object.entries(group.engines)) {
        console.log('**' + names[name] + '** cold process-ready latency was ' +
            f(engine.load_ms) + ' ms (one observation). ' + engine.configuration + '.\n');
        console.log('| Metric | Median | Mean | Sample SD | IQR | P95 | n |');
        console.log('| --- | ---: | ---: | ---: | ---: | ---: | ---: |');
        for (const [key, label] of Object.entries(labels)) {
            const row = engine.metrics[key];
            console.log('| ' + [label, ...['median', 'mean', 'stdev', 'iqr', 'p95'].map(k => f(row[k])), row.n].join(' | ') + ' |');
        }
        console.log('');
    }
}
