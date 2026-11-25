const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const {mkdir, mkdtemp, writeFile} = require('node:fs/promises');
const {join, resolve} = require('node:path');
const {createServer} = require('node:net');
const {setTimeout: delay} = require('node:timers/promises');

async function waitFor(action, description, attempts = 80) {
    for (let i = 0; i < attempts; i++) {
        try {
            const result = await action();
            if (result) return result;
        } catch {}
        await delay(200);
    }
    throw new Error('Timed out waiting for ' + description);
}

async function main() {
    const [, , browser, target = 'http://127.0.0.1:8765/', destination = 'docs/showcase', ...flags] = process.argv;
    if (!browser || flags.some(flag => !['--sandbox-workaround', '--charts-only'].includes(flag)))
        throw new Error('usage: node scripts/capture_showcase.cjs CHROMIUM [URL [OUTPUT]] [--sandbox-workaround] [--charts-only]');
    await mkdir(destination, {recursive: true});
    const build = resolve('build');
    await mkdir(build, {recursive: true});
    const profile = await mkdtemp(join(build, 'showcase-browser-'));
    const listener = createServer();
    await new Promise((ok, fail) => listener.once('error', fail).listen(0, '127.0.0.1', ok));
    const port = listener.address().port;
    await new Promise(ok => listener.close(ok));
    const options = ['--headless', '--disable-background-networking', '--no-first-run',
        '--remote-debugging-port=' + port, '--user-data-dir=' + profile, 'about:blank'];
    if (flags.includes('--sandbox-workaround'))
        options.unshift('--disable-gpu', '--in-process-gpu', '--single-process', '--no-sandbox');
    const child = spawn(browser, options, {windowsHide: true, stdio: ['ignore', 'ignore', 'pipe']});
    let startupError, log = '', socket;
    child.on('error', error => { startupError = error; });
    child.stderr.on('data', chunk => { log = (log + chunk.toString()).slice(-2500); });
    try {
        const page = await waitFor(async () => {
            const response = await fetch('http://127.0.0.1:' + port + '/json/list');
            return (await response.json()).find(tab => tab.type === 'page');
        }, 'Chromium debugging endpoint', 35);
        socket = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((ok, fail) => {
            socket.addEventListener('open', ok, {once: true});
            socket.addEventListener('error', fail, {once: true});
        });
        let serial = 0;
        const pending = new Map();
        socket.addEventListener('message', ({data}) => {
            const event = JSON.parse(data);
            const callback = pending.get(event.id);
            if (callback) {
                pending.delete(event.id);
                clearTimeout(callback.timer);
                if (event.error) callback.fail(new Error(event.error.message));
                else callback.ok(event.result);
            }
        });
        const command = (method, params = {}) => new Promise((ok, fail) => {
            const id = ++serial;
            const timer = setTimeout(() => {
                pending.delete(id);
                fail(new Error(method + ' timed out'));
            }, 10000);
            pending.set(id, {ok, fail, timer});
            socket.send(JSON.stringify({id, method, params}));
        });
        const evaluate = async expression => {
            const result = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
            if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
            return result.result.value;
        };
        const scroll = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView()`);
        const pick = label => evaluate(`[...document.querySelectorAll('#metric-options button')]
            .find(button=>button.textContent===${JSON.stringify(label)}).click()`);
        const screenshot = async name => {
            // Let pending CSS transitions settle before capture.
            await delay(300);
            const frame = await command('Page.captureScreenshot', {format: 'png'});
            await writeFile(join(destination, name + '.png'), Buffer.from(frame.data, 'base64'));
        };
        await command('Emulation.setDeviceMetricsOverride', {width: 1440, height: 1080, deviceScaleFactor: 1, mobile: false});
        await command('Page.navigate', {url: target});
        await waitFor(() => evaluate("document.querySelectorAll('.trial-pair').length === 10"), 'ten measured trial pairs');
        assert.match(await evaluate("document.querySelector('#quality').textContent"), /Identical 128-token/);
        await pick('First token');
        await screenshot('dashboard');
        await pick('Decode speed');
        await screenshot('benchmark_comparison');
        await pick('Prefill speed');
        await screenshot('prefill_throughput');
        await pick('Total latency');
        await screenshot('performance_summary');
        await evaluate("document.querySelector('#monitor').open = true; document.querySelectorAll('.trial-pair')[2].click()");
        await waitFor(() => evaluate("document.querySelector('#monitor-state').textContent.startsWith('Trial 3')"), 'board samples');
        await scroll('#monitor');
        await screenshot('gpu_monitoring');
        await evaluate("document.querySelector('#monitor').open = false");
        await waitFor(() => evaluate("document.querySelectorAll('#experiment-rows tr').length >= 2"), 'archived experiment series');
        await scroll('#experiments');
        await evaluate('window.scrollBy(0, -190)');
        await screenshot('optimization_history');
        await scroll('#pipeline');
        await screenshot('architecture');
        if (!flags.includes('--charts-only')) {
            await scroll('#live');
            await evaluate(`
                document.querySelector('#prompt').value = 'Describe, in three detailed paragraphs, how a CUDA kernel processes a prompt on a GPU.';
                document.querySelector('#limit').value = 240;
                document.querySelector('#generation').requestSubmit();
            `);
            await waitFor(() => evaluate("document.querySelector('#generation-state').textContent === 'Generating' && document.querySelector('#text').textContent.trim().length > 24"), 'native token stream');
            await screenshot('live_inference');
            await waitFor(() => evaluate("document.querySelector('#generation-state').textContent === 'Complete'"), 'native completion', 200);
            assert.ok(Number(await evaluate("document.querySelector('#generated').textContent")) > 0);
            assert.match(await evaluate("document.querySelector('#prefill').textContent"), /t\/s/);
            await screenshot('live_inference_complete');
        }
        console.log('Captured the measured comparison, board telemetry, history, execution pipeline' +
            (flags.includes('--charts-only') ? '.' : ' and live CUDA inference.'));
    } catch (error) {
        if (startupError || log) console.error(startupError || log);
        throw error;
    } finally {
        socket?.close();
        child.kill();
    }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
