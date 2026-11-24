const { execFile } = require('node:child_process');
const { mkdtemp, readFile, rm } = require('node:fs/promises');
const { join } = require('node:path');
const { tmpdir } = require('node:os');
const { promisify } = require('node:util');
const execute = promisify(execFile);

const prompts = [
    ['arithmetic', 'What is 2 + 2? Answer with just the digit.'],
    ['science', 'In one sentence, explain why the sky appears blue.'],
    ['multilingual', '用一句话解释什么是彩虹。'],
];

async function main() {
    const [, , spark, model, server = 'http://127.0.0.1:18081'] = process.argv;
    if (!spark || !model) throw new Error('usage: node verify_qwen_generation.cjs SPARK_EXE MODEL [URL]');
    const post = async (route, body) => {
        const response = await fetch(new URL(route, server), {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(body),
            signal: AbortSignal.timeout(60000),
        });
        if (!response.ok) throw new Error('comparison server HTTP ' + response.status);
        return response.json();
    };
    const temp = await mkdtemp(join(tmpdir(), 'spark-qwen-'));
    try {
        const logitsFile = join(temp, 'first-logits.bin');
        for (const [id, prompt] of prompts) {
            const rendered = '<|im_start|>user\n' + prompt +
                '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
            const { tokens: promptTokens } = await post('/tokenize', {
                content: rendered, add_special: false, parse_special: true,
            });
            const [native, reference] = await Promise.all([
                execute(spark, ['generate', '--model', model, '--prompt-ids', promptTokens.join(','),
                    '--max-new-tokens', '16', '--ids', '--first-logits', logitsFile], {
                    env: { ...process.env, OMP_NUM_THREADS: process.env.OMP_NUM_THREADS || '6' },
                }),
                post('/completion', { prompt: promptTokens, temperature: 0, n_predict: 16,
                    return_tokens: true, cache_prompt: false, n_probs: 8 }),
            ]);
            const ids = JSON.parse(native.stdout);
            const floats = await readFile(logitsFile);
            const scores = reference.completion_probabilities[0].top_logprobs;
            const baseId = scores[0].id;
            const baseNative = floats.readFloatLE(baseId * 4);
            const error = Math.max(...scores.map(({id, logprob}) =>
                Math.abs(floats.readFloatLE(id * 4) - baseNative - (logprob - scores[0].logprob))));
            // SPARK's public generation excludes the terminal token.
            const referenceIds = reference.stop_type === 'eos'
                ? reference.tokens.slice(0, -1) : reference.tokens;
            const match = JSON.stringify(ids) === JSON.stringify(referenceIds);
            console.log(JSON.stringify({ id, prompt_tokens: promptTokens.length, native_ids: ids,
                reference_ids: referenceIds, sequence_match: match,
                first_token_match: ids[0] === referenceIds[0], max_top8_logit_gap_error: error }));
            if (!match) process.exitCode = 1;
        }
    } finally {
        await rm(temp, { recursive: true, force: true });
    }
}

main().catch(error => { console.error(error); process.exitCode = 2; });
