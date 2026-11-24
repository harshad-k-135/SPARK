const { execFile } = require('node:child_process');
const { mkdtemp, readFile, rm, rmdir } = require('node:fs/promises');
const { join } = require('node:path');
const { tmpdir } = require('node:os');
const { promisify } = require('node:util');
const execute = promisify(execFile);

const smokePrompts = [
    ['arithmetic', 'What is 2 + 2? Answer with just the digit.'],
    ['science', 'In one sentence, explain why the sky appears blue.'],
    ['multilingual', '用一句话解释什么是彩虹。'],
];

async function main() {
    const [, , spark, model, server = 'http://127.0.0.1:18081', mode] = process.argv;
    if (!spark || !model || (mode && mode !== '--full'))
        throw new Error('usage: node verify_qwen_generation.cjs SPARK_EXE MODEL [URL [--full]]');
    const full = mode === '--full';
    const prompts = full
        ? JSON.parse(await readFile(join(__dirname, '../../tests/golden/prompts.json'), 'utf8'))
            .map(({id, text, tokens}) => [id, text, tokens])
        : smokePrompts;
    const count = full ? 64 : 16;
    if (full && new Set(prompts.map(([id]) => id)).size < 20)
        throw new Error('full parity requires at least 20 distinct prompts');
    const post = async (route, body) => {
        const response = await fetch(new URL(route, server), {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(240000),
        });
        if (!response.ok) throw new Error('comparison server HTTP ' + response.status);
        return response.json();
    };
    const temp = await mkdtemp(join(tmpdir(), 'spark-qwen-'));
    const logitsFile = join(temp, 'first-logits.bin');
    const lengths = [];
    try {
        for (const [id, prompt, length] of prompts) {
            const rendered = '<|im_start|>user\n' + prompt +
                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
            const tokenize = async content => (await post('/tokenize', {
                content, add_special: false, parse_special: !full,
            })).tokens;
            let tokens = await tokenize(full ? prompt : rendered);
            for (let repeats = 2; length && tokens.length < length; repeats *= 2)
                tokens = await tokenize((prompt + '\n\n').repeat(repeats));
            const promptTokens = length ? tokens.slice(0, length) : tokens;
            if (full && (promptTokens.length < 16 || promptTokens.length > 1500))
                throw new Error('prompt length outside the gate range for ' + id);
            lengths.push(promptTokens.length);
            const nativeArgs = ['generate', '--model', model, '--prompt-ids', promptTokens.join(','),
                '--max-new-tokens', String(count), '--ids', '--first-logits', logitsFile];
            if (full) nativeArgs.push('--ignore-eos');
            const request = { prompt: promptTokens, temperature: 0, n_predict: count,
                return_tokens: true, cache_prompt: false, ignore_eos: full, n_probs: full ? 0 : 8 };
            const [native, reference] = await Promise.all([
                execute(spark, nativeArgs, {
                    env: { ...process.env, OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || '6' },
                }),
                post('/completion', request),
            ]);
            const ids = JSON.parse(native.stdout);
            const floats = await readFile(logitsFile);
            const first = full ? await post('/completion', {
                ...request, n_predict: 1, ignore_eos: false, n_probs: floats.length / 4,
            }) : reference;
            const scores = first.completion_probabilities[0].top_logprobs;
            if (full && (scores.length !== floats.length / 4 || ids.length !== count ||
                reference.tokens.length !== count)) throw new Error('incomplete gate output for ' + id);
            const baseId = scores[0].id;
            const baseNative = floats.readFloatLE(baseId * 4);
            let error = 0;
            for (const {id, logprob} of scores) {
                if (typeof logprob !== 'number') throw new Error('non-finite reference logprob');
                const gap = floats.readFloatLE(id * 4) - baseNative - (logprob - scores[0].logprob);
                error = Math.max(error, Math.abs(gap));
            }
            // SPARK's public generation excludes the terminal token.
            const referenceIds = reference.stop_type === 'eos'
                ? reference.tokens.slice(0, -1) : reference.tokens;
            const match = JSON.stringify(ids) === JSON.stringify(referenceIds);
            console.log(JSON.stringify({ id, prompt_tokens: promptTokens.length, native_ids: ids,
                reference_ids: referenceIds, sequence_match: match,
                first_token_match: ids[0] === referenceIds[0], logits_compared: scores.length,
                max_logit_gap_error: error }));
            if (!match || (full && (!Number.isFinite(error) || error >= 0.05))) process.exitCode = 1;
        }
    } finally {
        await rm(logitsFile, { force: true });
        await rmdir(temp);
    }
    if (full && (Math.min(...lengths) !== 16 || Math.max(...lengths) !== 1500))
        throw new Error('full parity must cover prompt lengths from 16 to 1,500');
}

main().catch(error => { console.error(error); process.exitCode = 2; });
