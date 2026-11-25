const { execFile, spawn } = require('node:child_process');
const { createHash } = require('node:crypto');
const { createReadStream } = require('node:fs');
const { writeFile, rename } = require('node:fs/promises');
const { createServer } = require('node:net');
const { createInterface } = require('node:readline');
const { setTimeout: delay } = require('node:timers/promises');
const { promisify } = require('node:util');

const exec = promisify(execFile);
const contexts = [128, 512, 1024];
const outputTokens = 128, warmups = 5, measured = 10;
const passage = 'The observatory log records a clear evening and a steady breeze from the west. ' +
    'The team set up the small telescope before sunset and checked the alignment. ' +
    'Summarize the observations and explain what the team should record next.\n';
const metrics = ['decode_tokens_per_sec', 'first_token_ms', 'prefill_tokens_per_sec',
    'decode_ms', 'peak_vram_mib', 'gpu_utilization_pct'];

function quantile(values, fraction) {
    if (!values.length) return null;
    const data = [...values].sort((a, b) => a - b);
    const index = (data.length - 1) * fraction;
    const lower = Math.floor(index);
    return data[lower] + (data[Math.ceil(index)] - data[lower]) * (index - lower);
}
function summarize(trials) {
    return Object.fromEntries(metrics.map(key => {
        const data = trials.filter(t => !t.warmup).map(t => t[key]).filter(Number.isFinite);
        return [key, { median: quantile(data, 0.5),
            iqr: data.length ? quantile(data, 0.75) - quantile(data, 0.25) : null,
            p95: quantile(data, 0.95) }];
    }));
}

function managed(executable, flags, options) {
    const child = spawn(executable, flags, { windowsHide: true, ...options });
    let logs = '', failure;
    child.stderr?.on('data', chunk => { logs = (logs + chunk.toString()).slice(-75000); });
    child.on('error', error => { failure = error; });
    const closed = new Promise(resolve => child.once('close', resolve));
    return { child, get logs() { return logs; }, get failure() { return failure; },
        async stop() {
            if (child.exitCode === null && !child.killed) child.kill();
            await closed;
        } };
}

function telemetry() {
    const params = ['utilization.gpu', 'memory.used', 'power.draw', 'temperature.gpu', 'clocks.gr', 'clocks.mem'];
    const device = managed('nvidia-smi', ['--query-gpu=' + params.join(','),
        '--format=csv,noheader,nounits', '--loop-ms=100'], { stdio: ['ignore', 'pipe', 'pipe'] });
    const samples = [];
    const lines = createInterface({ input: device.child.stdout });
    lines.on('line', line => {
        const values = line.split(',').map(item => item.trim()).map(item =>
            item && Number.isFinite(Number(item)) ? Number(item) : null);
        if (values.length === params.length && values.some(Number.isFinite))
            samples.push({ time_ms: performance.now(), utilization_pct: values[0], memory_mib: values[1],
                power_w: values[2], temperature_c: values[3], graphics_mhz: values[4], memory_mhz: values[5] });
    });
    return { samples, async stop() { await device.stop(); lines.close(); } };
}

function finish(start, stamps, ids, prefill, sensor) {
    if (ids.length !== outputTokens || stamps.length !== outputTokens ||
        !Number.isFinite(prefill) || prefill <= 0) throw new Error('incomplete token or timing output: ' + ids.length);
    const end = stamps.at(-1);
    const decode = end - stamps[0];
    if (decode <= 0) throw new Error('stream did not expose distinct token arrival times');
    const samples = sensor.samples.filter(s => s.time_ms >= start && s.time_ms <= end);
    const vram = samples.map(s => s.memory_mib).filter(Number.isFinite);
    const utilization = samples.map(s => s.utilization_pct).filter(Number.isFinite);
    return { first_token_ms: stamps[0] - start, decode_ms: decode,
        decode_tokens_per_sec: (outputTokens - 1) * 1000 / decode, prefill_ms: prefill,
        prefill_tokens_per_sec: null, generated_token_ids: ids,
        peak_vram_mib: vram.length ? Math.max(...vram) : null,
        gpu_utilization_pct: utilization.length ? utilization.reduce((a, b) => a + b, 0) / utilization.length : null,
        telemetry: samples.map(({time_ms, ...rest}) => ({elapsed_ms: time_ms - start, ...rest})) };
}

async function sparkSession(exe, model) {
    const launch = performance.now();
    const proc = managed(exe, ['serve', '--model', model], { stdio: ['pipe', 'pipe', 'pipe'] });
    proc.child.stdin.on('error', () => {});
    const reader = createInterface({ input: proc.child.stdout })[Symbol.asyncIterator]();
    try {
        const ready = await reader.next();
        if (ready.done || JSON.parse(ready.value).event !== 'ready') throw new Error(proc.logs || 'SPARK failed to start');
    } catch (error) { await proc.stop(); throw error; }
    return {
        load_ms: performance.now() - launch,
        stop: proc.stop,
        async trial(prompt, sensor) {
            const payload = prompt.join(',');
            const start = performance.now();
            proc.child.stdin.write('ids ' + Buffer.byteLength(payload) + ' ' + outputTokens + '\n' + payload);
            const ids = [], times = [];
            for (;;) {
                const line = await reader.next();
                if (line.done) throw new Error(proc.logs || 'SPARK worker closed');
                const event = JSON.parse(line.value);
                if (event.event === 'error') throw new Error(event.error);
                if (event.event === 'token') { ids.push(event.id); times.push(performance.now()); }
                if (event.event === 'done') return finish(start, times, ids, event.prefill_ms, sensor);
            }
        },
    };
}

async function portForSession() {
    const listener = createServer();
    await new Promise((resolve, reject) => listener.once('error', reject).listen(0, '127.0.0.1', resolve));
    const { port } = listener.address();
    await new Promise((resolve, reject) => listener.close(error => error ? reject(error) : resolve()));
    return port;
}

async function llamaSession(exe, model, device) {
    const port = await portForSession();
    const flags = ['--model', model, '--host', '127.0.0.1', '--port', String(port),
        '--device', device, '--n-gpu-layers', '99', '--ctx-size', '2048', '--parallel', '1',
        '--batch-size', '1024', '--ubatch-size', '512', '--flash-attn', 'auto',
        '--cache-type-k', 'f32', '--cache-type-v', 'f32', '--log-verbosity', '4'];
    const url = 'http://127.0.0.1:' + port;
    const launch = performance.now();
    const proc = managed(exe, flags, { stdio: ['ignore', 'ignore', 'pipe'] });
    let ready = false;
    try {
        for (let i = 0; i < 240; i++) {
            if (proc.failure || proc.child.exitCode !== null) throw new Error(proc.logs || 'llama.cpp failed to start');
            try {
                const response = await fetch(url + '/health', {signal: AbortSignal.timeout(1000)});
                if (response.ok && (await response.json()).status === 'ok') { ready = true; break; }
            } catch {}
            await delay(250);
        }
        if (!ready || !/offloaded (\d+)\/\1 layers to GPU/.test(proc.logs))
            throw new Error('full llama.cpp CUDA offload was not confirmed\n' + proc.logs.slice(-4000));
    } catch (error) { await proc.stop(); throw error; }
    const offloaded = Number(proc.logs.match(/offloaded (\d+)\/\1 layers to GPU/)[1]);
    return {
        load_ms: performance.now() - launch,
        configuration: flags.slice(flags.indexOf('--device')).join(' '),
        offloaded_layers: offloaded,
        stop: proc.stop,
        async trial(prompt, sensor) {
            const start = performance.now();
            const response = await fetch(url + '/completion', {
                method: 'POST', headers: {'content-type': 'application/json'},
                body: JSON.stringify({prompt, n_predict: outputTokens, temperature: 0, seed: 0,
                    stream: true, return_tokens: true, ignore_eos: true, cache_prompt: false}),
                signal: AbortSignal.timeout(240000),
            });
            if (!response.ok) throw new Error('llama.cpp HTTP ' + response.status + ' ' + await response.text());
            const ids = [], times = [];
            let timing, pending = '';
            const text = new TextDecoder();
            const consume = line => {
                if (!line.startsWith('data:')) return;
                const data = line.slice(5).trim();
                if (data === '[DONE]') return;
                const event = JSON.parse(data);
                if (event.error) throw new Error(JSON.stringify(event.error));
                if (event.timings) timing = event.timings;
                if (event.stop) return;
                const stamp = performance.now();
                for (const id of event.tokens || []) { ids.push(id); times.push(stamp); }
            };
            for await (const chunk of response.body) {
                pending += text.decode(chunk, {stream: true});
                const lines = pending.split(/\r?\n/);
                pending = lines.pop();
                for (const line of lines) consume(line);
            }
            consume(pending + text.decode());
            if (!timing || timing.prompt_n !== prompt.length || (timing.cache_n ?? 0) !== 0)
                throw new Error('llama.cpp reported unexpected prefill counts: ' + JSON.stringify(timing));
            return finish(start, times, ids, timing.prompt_ms, sensor);
        },
    };
}

async function main() {
    const [, , spark, llama, model, out, ...options] = process.argv;
    if (!spark || !llama || !model || !out || options.some(x => x !== '--smoke'))
        throw new Error('usage: node compare_cuda.cjs SPARK_EXE LLAMA_SERVER_EXE GGUF OUT [--smoke]');
    const smoke = options.includes('--smoke');
    const { stdout: idsText } = await exec(spark, ['tokenize', '--model', model, '--prompt', passage.repeat(32)]);
    const ids = JSON.parse(idsText);
    if (ids.length < contexts.at(-1)) throw new Error('primary prompt is too short');
    const { stdout: devices } = await exec(llama, ['--list-devices']);
    const match = devices.match(/^\s*(CUDA\d+):.*RTX 3050/m);
    if (!match) throw new Error('llama.cpp did not enumerate the RTX 3050 CUDA backend');
    const { stdout: head } = await exec('git', ['rev-parse', 'HEAD']);
    const { stdout: version, stderr: versionError } = await exec(llama, ['--version']);
    const { stdout: gpu } = await exec('nvidia-smi', ['--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']);
    let powerMode = null;
    try { powerMode = (await exec('powercfg', ['/getactivescheme'])).stdout.trim(); } catch {}
    const hash = createHash('sha256');
    for await (const chunk of createReadStream(model)) hash.update(chunk);
    const report = { kind: 'head-to-head', runtime: 'SPARK vs llama.cpp', smoke,
        model_sha256: hash.digest('hex'), hardware: gpu.trim(), power_mode: powerMode,
        context: 2048, warmups: smoke ? 0 : warmups, measured: smoke ? 1 : measured,
        output_tokens: outputTokens, spark_commit: head.trim(), llama_version: (version + versionError).trim(),
        timing: 'Client-observed first-to-last token delivery through local SPARK IPC and llama.cpp loopback SSE. First token starts before each request. Prefill uses each engine’s timer.',
        workloads: [] };
    // Run one GPU process at a time. Reverse order for the middle context.
    for (const length of smoke ? [128] : contexts) {
        const prompt = ids.slice(0, length);
        const group = { prompt_tokens: length, prompt_token_ids: prompt, engines: {} };
        const order = length === 512 ? ['llama_cpp', 'spark'] : ['spark', 'llama_cpp'];
        for (const name of order) {
            const session = await (name === 'spark' ? sparkSession(spark, model) : llamaSession(llama, model, match[1]));
            let sensor;
            try {
                sensor = telemetry();
                const trials = [];
                await delay(160);
                for (let i = 0; i < report.warmups + report.measured; i++) {
                    const result = await session.trial(prompt, sensor);
                    trials.push({...result, run: i + 1, warmup: i < report.warmups,
                        prefill_tokens_per_sec: length * 1000 / result.prefill_ms});
                    console.log(JSON.stringify({ engine: name, prompt_tokens: length, run: i + 1,
                        total_runs: report.warmups + report.measured }));
                }
                group.engines[name] = {load_ms: session.load_ms, trials, metrics: summarize(trials),
                    offloaded_layers: session.offloaded_layers,
                    configuration: session.configuration || 'Packed Q4_K_M CUDA, FP32 KV, graphs off'};
            } finally {
                if (sensor) await sensor.stop();
                await session.stop();
            }
        }
        const a = group.engines.spark.trials[report.warmups].generated_token_ids;
        const b = group.engines.llama_cpp.trials[report.warmups].generated_token_ids;
        const conflict = a.findIndex((id, index) => id !== b[index]);
        group.greedy_prefix_match = conflict === -1 ? outputTokens : conflict;
        report.workloads.push(group);
        await writeFile(out + '.tmp', JSON.stringify(report) + '\n');
        await rename(out + '.tmp', out);
    }
    console.log(JSON.stringify({completed: true, output: out}));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
