const { execFile } = require('node:child_process');
const { createHash } = require('node:crypto');
const { createReadStream } = require('node:fs');
const { writeFile } = require('node:fs/promises');
const { promisify } = require('node:util');
const execute = promisify(execFile);

const passage = 'The observatory log records a clear evening and a steady breeze from the west. ' +
    'The team set up the small telescope before sunset and checked the alignment. ' +
    'Summarize the observations and explain what the team should record next.\n';

function quantile(data, q) {
    const sorted = [...data].sort((a, b) => a - b);
    const position = (sorted.length - 1) * q;
    const index = Math.floor(position);
    return sorted[index] + (sorted[Math.ceil(position)] - sorted[index]) * (position - index);
}

async function main() {
    const [, , spark, model, output, device = 'cuda'] = process.argv;
    if (!spark || !model || !output || device !== 'cuda')
        throw new Error('usage: node measure_native.cjs SPARK_EXE MODEL OUTPUT [cuda]');
    const prefillTokens = 512, generatedTokens = 128, warmups = 5, measured = 10;
    const { stdout: input } = await execute(spark, [
        'tokenize', '--model', model, '--prompt', passage.repeat(32),
    ]);
    const ids = JSON.parse(input).slice(0, prefillTokens);
    if (ids.length !== prefillTokens) throw new Error('incorrect primary prompt length');

    const modelHash = createHash('sha256');
    for await (const chunk of createReadStream(model)) modelHash.update(chunk);
    const { stdout: commit } = await execute('git', ['rev-parse', 'HEAD']);
    const { stdout: hardware } = await execute('nvidia-smi', [
        '--query-gpu=name,driver_version,pstate,memory.total', '--format=csv,noheader',
    ]);
    const graph = process.env.SPARK_CUDA_GRAPH || 'off';
    const { stderr } = await execute(spark, [
        'generate', '--model', model, '--prompt-ids', ids.join(','), '--device', device,
        '--max-new-tokens', String(generatedTokens), '--ignore-eos', '--ids', '--timings',
        '--warmups', String(warmups), '--trials', String(measured), '--cuda-graph', graph,
    ], { timeout: 1800000 });
    const events = stderr.trim().split(/\r?\n/).filter(line => line.startsWith('{')).map(JSON.parse);
    const trials = events.filter(row => row.event === 'generation_timings');
    const results = trials.filter(row => !row.warmup);
    if (trials.length !== warmups + measured || results.length !== measured ||
        trials.some(row => row.generated_tokens !== generatedTokens || row.prompt_tokens !== prefillTokens))
        throw new Error('invalid native measurement output');
    const fields = ['prefill_tokens_per_sec', 'decode_tokens_per_sec', 'first_token_ms', 'decode_ms'];
    const metrics = Object.fromEntries(fields.map(field => {
        const data = results.map(trial => trial[field]);
        return [field, { median: quantile(data, 0.5), iqr: quantile(data, 0.75) - quantile(data, 0.25),
            p95: quantile(data, 0.95) }];
    }));
    const report = { runtime: graph === 'off' ? 'spark-qwen-v0' : 'spark-qwen-graph-' + graph,
        commit: commit.trim(), hardware: hardware.trim(),
        model_sha256: modelHash.digest('hex'), context: 2048, prompt_token_ids: ids, warmups, measured,
        time_boundaries: 'First token starts at prefill on supplied token IDs. Load is measured separately.',
        cuda_plan: events.find(row => row.event === 'cuda_plan'), metrics, trials };
    await writeFile(output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(metrics));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
