const assert = require('node:assert/strict');
const {execFile} = require('node:child_process');
const {promisify} = require('node:util');
const {performance} = require('node:perf_hooks');

const execute = promisify(execFile);
async function main() {
    const [, , spark, model, ...variant] = process.argv;
    if (!spark || !model || variant.some(flag => !['--q6-block-gemv', '--q4-block-gemv', '--bucket-graph', '--fp16-prefill'].includes(flag)))
        throw new Error('usage: node scripts/stress_qwen_cuda.cjs SPARK GGUF [--bucket-graph] [--q6-block-gemv] [--q4-block-gemv] [--fp16-prefill]');
    const passage = 'The observatory opens at sunset. The team checks the telescope, weather, and calibration. ';
    const {stdout} = await execute(spark, ['tokenize', '--model', model, '--prompt', passage.repeat(8)]);
    const ids = JSON.parse(stdout);
    const outputTokens = 16;
    const promptLength = 4096 - outputTokens;
    const prompt = Array.from({length: promptLength}, (_, i) => ids[i % ids.length]);
    const run = async (input, settings = {}) => {
        const args = ['generate', '--model', model, '--prompt-ids', input.join(','), '--device', 'cuda',
            '--cuda-weights', 'packed', '--batch-prefill', '--context', '4096',
            '--ids', '--ignore-eos', '--max-new-tokens', String(outputTokens)];
        if (variant.includes('--bucket-graph')) args.push('--cuda-graph', 'bucket');
        if (variant.includes('--q6-block-gemv')) args.push('--q6-block-gemv');
        if (variant.includes('--q4-block-gemv')) args.push('--q4-block-gemv');
        if (variant.includes('--fp16-prefill')) args.push('--fp16-prefill');
        return execute(spark, args, {windowsHide: true, timeout: 240000, ...settings});
    };
    await assert.rejects(run([...prompt, ids[0]]), /prompt and output exceed the context/);
    await assert.rejects(run([999999]), /invalid CUDA prompt tokens/);
    const trials = [];
    for (let i = 0; i < 2; i++) {
        const start = performance.now();
        const result = await run(prompt);
        trials.push({generated_ids: JSON.parse(result.stdout), elapsed_ms: performance.now() - start});
    }
    assert.equal(trials[0].generated_ids.length, outputTokens);
    assert.deepEqual(trials[0].generated_ids, trials[1].generated_ids);
    console.log(JSON.stringify({device: 'cuda', prompt_tokens: promptLength,
        output_tokens: outputTokens, overflow_rejected: true, invalid_token_rejected: true,
        stable_outputs: true, configuration: variant, trials}));
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
