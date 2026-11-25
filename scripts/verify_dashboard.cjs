const assert = require('node:assert/strict');
const {setTimeout: delay} = require('node:timers/promises');

async function main() {
    const [, , target = 'http://127.0.0.1:8765/'] = process.argv;
    const request = (path, options) => fetch(new URL(path, target), options);
    const json = async (path, options) => {
        const response = await request(path, options);
        if (!response.ok) throw new Error(path + ' HTTP ' + response.status + ' ' + await response.text());
        return response.json();
    };
    const status = await json('/api/status');
    assert.equal(status.device, 'cuda');
    const history = await json('/api/comparisons');
    assert.ok(history.length >= 2);
    const comparison = await json('/api/comparison');
    const samples = await json('/api/comparison?run=' + encodeURIComponent(comparison.id) + '&prompt=512&trial=1');
    assert.ok(samples.spark.length && samples.llama_cpp.length);
    const experiments = await json('/api/experiments');
    assert.ok(experiments.length >= 3);
    const job = await json('/runs', {method: 'POST', headers: {'content-type': 'application/json'},
        body: JSON.stringify({mode: 'optimized'})});
    const path = '/runs/' + encodeURIComponent(job.id);
    let current;
    for (let attempt = 0; attempt < 240; attempt++) {
        current = await json(path + '?status_only=true');
        if (current.state === 'completed') break;
        if (current.state === 'failed') throw new Error(current.error);
        await delay(1000);
    }
    assert.equal(current?.state, 'completed');
    const exportResponse = await request(path + '?download=true');
    assert.equal(exportResponse.status, 200);
    assert.match(exportResponse.headers.get('content-disposition'), /attachment/);
    const {report} = await exportResponse.json();
    assert.equal(report.measured, 10);
    assert.equal(report.runtime, 'spark-qwen-tuned');
    assert.equal((await request('/runs/mission-baseline', {method: 'DELETE'})).status, 409);
    assert.equal((await request(path, {method: 'DELETE'})).status, 204);
    assert.equal((await request(path)).status, 404);
    console.log(JSON.stringify({gpu: status.device, comparison: comparison.id, archived_experiments: experiments.length,
        completed_trials: report.measured, tuned_median_decode_tps: report.metrics.decode_tokens_per_sec.median,
        export_verified: true, archived_run_protected: true, test_run_deleted: true}));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
