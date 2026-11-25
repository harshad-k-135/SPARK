const assert = require('node:assert/strict');
const {execFile} = require('node:child_process');
const {mkdir, mkdtemp, readFile, rm, rmdir, writeFile} = require('node:fs/promises');
const {join, resolve} = require('node:path');
const {promisify} = require('node:util');

const execute = promisify(execFile);
async function main() {
    const [, , spark, model, ...variant] = process.argv;
    if (!spark || !model || variant.some(arg => !['--fp16-prefill', '--q6-block-gemv', '--q4-block-gemv', '--bucket-graph'].includes(arg)))
        throw new Error('usage: node scripts/verify_native_parity.cjs SPARK GGUF [worker tuning flags]');
    await mkdir(resolve('build'), {recursive: true});
    const directory = await mkdtemp(resolve('build/native-parity-'));
    try {
        const invalid = join(directory, 'truncated.gguf');
        await writeFile(invalid, 'GGUF');
        await assert.rejects(execute(spark, ['inspect', '--model', invalid]), /GGUF|header|truncated/i);
        const prompt = 'A GPU processes many independent operations in parallel. List two benefits and give an example.';
        const flags = ['generate', '--model', model, '--prompt', prompt, '--chat', '--ids',
            '--max-new-tokens', '16', '--ignore-eos', '--first-logits'];
        const options = {env: {...process.env, OMP_NUM_THREADS: '6'}, windowsHide: true, timeout: 180000};
        const cpuFile = join(directory, 'cpu.bin');
        const gpuFile = join(directory, 'gpu.bin');
        const cpu = JSON.parse((await execute(spark, [...flags, cpuFile], options)).stdout);
        const cuda = ['--device', 'cuda', '--cuda-weights', 'packed', '--batch-prefill'];
        for (const flag of variant) {
            if (flag === '--bucket-graph') cuda.push('--cuda-graph', 'bucket');
            else cuda.push(flag);
        }
        const gpu = JSON.parse((await execute(spark, [...flags, gpuFile, ...cuda], options)).stdout);
        assert.equal(gpu.length, 16);
        assert.deepEqual(gpu, cpu);
        const a = await readFile(cpuFile), b = await readFile(gpuFile);
        assert.equal(a.length, b.length);
        assert.equal(a.length, 248320 * 4);
        let maxLogitError = 0;
        for (let i = 0; i < a.length; i += 4)
            maxLogitError = Math.max(maxLogitError, Math.abs(a.readFloatLE(i) - b.readFloatLE(i)));
        assert.ok(maxLogitError < 0.05, 'max absolute logit error ' + maxLogitError);
        console.log(JSON.stringify({truncated_gguf_rejected: true, cpu_cuda_match: true,
            max_absolute_logit_error: maxLogitError, logits_compared: a.length / 4, generated_ids: gpu}));
    } finally {
        for (const name of ['truncated.gguf', 'cpu.bin', 'gpu.bin'])
            await rm(join(directory, name), {force: true});
        await rmdir(directory);
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
