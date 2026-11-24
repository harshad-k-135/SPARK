const { execFileSync } = require('node:child_process');
const assert = require('node:assert/strict');
const cases = [
    ['arithmetic', 'What is 2 + 2?', false],
    ['numbers', '123456 07:45 -1.25', false],
    ['unicode', '你好，東京！ नमस्ते दुनिया。 café cafe\u0301 🌦️', false],
    ['spaces', '  hello\t\tworld\r\n\n  okay  ', false],
    ['contractions', "Don't ask. WE'RE here; I'll explain.", false],
    ['leading-mark', '\u0301.', false],
    ['template', '<|im_start|>user\nHello<|im_end|>\n', true],
    ['literal-template', '<|im_start|>user', false],
];
(async () => {
    const [, , spark, model, reference = 'http://127.0.0.1:18081'] = process.argv;
    for (const [name, content, parse_special] of cases) {
        const args = ['tokenize', '--model', model, '--prompt', content];
        if (parse_special) args.push('--special');
        const ids = JSON.parse(execFileSync(spark, args, { encoding: 'utf8' }));
        const request = await fetch(new URL('/tokenize', reference), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, add_special: false, parse_special }),
        });
        if (!request.ok) throw new Error('external tokenize HTTP ' + request.status);
        const { tokens } = await request.json();
        assert.deepEqual(ids, tokens, name);
        console.log(JSON.stringify({name, token_count: ids.length, tokens: ids, matches: true}));
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
