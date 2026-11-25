"""Local control plane for the native SPARK CUDA worker."""

import asyncio
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "api" / "frontend"
DATA = ROOT / "build" / "dashboard"
DB = DATA / "runs.sqlite"
RESULTS = ROOT / "benchmark" / "results"
TUNED_WORKER_ARGS = ("--batch-prefill", "--bucket-graph", "--q6-block-gemv", "--q4-block-gemv")
SNAPSHOTS = {
    "baseline-v0": RESULTS / "spark-qwen-v0.json",
    "graph-bucket": RESULTS / "spark-qwen-graph-bucket.json",
    "graph-single": RESULTS / "spark-qwen-graph-single.json",
    "packed": RESULTS / "spark-qwen-packed.json",
    "cuda-head-to-head": RESULTS / "spark-vs-llama-cuda.json",
    "mission-baseline": ROOT / "benchmarks" / "baseline.json",
    "batch-prefill": ROOT / "benchmarks" / "experiments" / "batch-prefill-primary.json",
    "q4-block-gemv": ROOT / "benchmarks" / "experiments" / "q4-block-primary.json",
    "packed-graph": ROOT / "benchmarks" / "experiments" / "packed-graph-primary.json",
    "q6-block-gemv": ROOT / "benchmarks" / "experiments" / "q6-block-primary.json",
    "fp16-prefill": ROOT / "benchmarks" / "experiments" / "fp16-prefill-primary.json",
    "mission-optimized": ROOT / "benchmarks" / "optimized.json",
    "mission-repeat": ROOT / "benchmarks" / "optimized-repeat-primary.json",
    "optimized-q4": ROOT / "benchmarks" / "optimized-q4.json",
    "repeat-q4": ROOT / "benchmarks" / "optimized-q4-repeat-primary.json",
}



class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=128, ge=1, le=512)


class BenchmarkRequest(BaseModel):
    mode: str = Field(pattern="^(baseline|bucket|packed|optimized|head-to-head)$")


def connection():
    return sqlite3.connect(DB)


def save_run(identifier, state, name, report=None, error=None):
    with connection() as db:
        db.execute(
            "INSERT INTO runs (id,state,name,report,error) VALUES (?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET state=excluded.state, "
            "name=excluded.name, report=excluded.report, error=excluded.error",
            (identifier, state, name, json.dumps(report) if report else None, error),
        )


def row_data(row):
    identifier, state, name, report, error = row
    return {"id": identifier, "state": state, "name": name,
            "report": json.loads(report) if report else None, "error": error}


class Worker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.process = None
        self.log = None
        self.tasks = set()
        self.spark = Path(os.environ.get("SPARK_EXE", ROOT / "build" / "cpu" / "spark.exe"))
        self.llama = Path(os.environ.get(
            "SPARK_LLAMA_SERVER", ROOT / "build" / "llama-b10159-cuda13" / "llama-server.exe"
        ))
        self.model = Path(os.environ.get("SPARK_MODEL", ""))
        if not self.spark.is_file() or not self.model.is_file():
            raise RuntimeError("Set SPARK_EXE and SPARK_MODEL to existing files.")

    async def stop(self):
        process, self.process = self.process, None
        try:
            if process and process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                await process.wait()
        finally:
            if self.log:
                self.log.close()
                self.log = None

    async def ensure_started(self):
        if self.process and self.process.returncode is None:
            return
        await self.stop()
        self.log = (DATA / "worker.log").open("ab")
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(self.spark), "serve", "--model", str(self.model), *TUNED_WORKER_ARGS, cwd=ROOT,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=self.log,
            )
            ready = await asyncio.wait_for(self.process.stdout.readline(), 120)
            if not ready or json.loads(ready).get("event") != "ready":
                raise RuntimeError("Native CUDA worker startup failed. See build/dashboard/worker.log.")
        except BaseException:
            await self.stop()
            raise

    async def stream(self, text, max_tokens):
        async with self.lock:
            complete = False
            try:
                yield b'{"event":"loading"}\n'
                await self.ensure_started()
                self.process.stdin.write(f"{len(text)} {max_tokens}\n".encode("ascii") + text)
                await self.process.stdin.drain()
                while line := await self.process.stdout.readline():
                    event = json.loads(line).get("event")
                    if event in {"done", "error"}:
                        complete = True
                    yield line
                    if complete:
                        return
                raise RuntimeError("The native CUDA worker stopped. See build/dashboard/worker.log.")
            except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as error:
                yield (json.dumps({"event": "error", "error": str(error)}) + "\n").encode()
            finally:
                if not complete:
                    await asyncio.shield(self.stop())

    async def benchmark(self, identifier, mode):
        process = None
        async with self.lock:
            save_run(identifier, "running", mode)
            try:
                await self.stop()
                output = DATA / f"{identifier}.json"
                tuned = TUNED_WORKER_ARGS if mode == "optimized" else ()
                env = {
                    **os.environ,
                    "SPARK_CUDA_GRAPH": "bucket" if mode == "bucket" or "--bucket-graph" in tuned else "off",
                    "SPARK_CUDA_WEIGHTS": "packed" if mode in {"packed", "optimized"} else "fp32",
                    "SPARK_BATCH_PREFILL": str(int("--batch-prefill" in tuned)),
                    "SPARK_Q4_BLOCK_GEMV": str(int("--q4-block-gemv" in tuned)),
                    "SPARK_Q6_BLOCK_GEMV": str(int("--q6-block-gemv" in tuned)),
                    "SPARK_FP16_PREFILL": str(int("--fp16-prefill" in tuned)),
                }
                if mode == "head-to-head":
                    command = ("benchmark/comparison/compare_cuda.cjs", str(self.spark),
                               str(self.llama), str(self.model), str(output), *TUNED_WORKER_ARGS)
                else:
                    command = ("benchmark/runner/measure_native.cjs", str(self.spark),
                               str(self.model), str(output))
                process = await asyncio.create_subprocess_exec(
                    "node", *command, cwd=ROOT, env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, errors = await process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(errors.decode(errors="replace")[-1200:])
                save_run(identifier, "completed", mode, json.loads(output.read_text(encoding="utf-8")))
            except (OSError, RuntimeError, ValueError) as error:
                save_run(identifier, "failed", mode, error=str(error))
            finally:
                if process and process.returncode is None:
                    with suppress(ProcessLookupError):
                        process.terminate()
                    await process.wait()


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS runs "
                   "(id TEXT PRIMARY KEY, state TEXT NOT NULL, name TEXT NOT NULL, report TEXT, error TEXT)")
        db.execute("UPDATE runs SET state='failed', error='The dashboard shut down before completion.' "
                   "WHERE state IN ('running','queued')")
    for identifier, path in SNAPSHOTS.items():
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            save_run(identifier, "completed", report["runtime"], report)
    app.state.worker = Worker()
    try:
        yield
    finally:
        tasks = tuple(app.state.worker.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await app.state.worker.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=UI), name="dashboard-assets")


@app.get("/")
async def dashboard():
    return FileResponse(UI / "index.html")


@app.get("/api/status")
async def status():
    worker = app.state.worker
    active = worker.process is not None and worker.process.returncode is None
    return {"device": "cuda", "weights": "packed", "model": worker.model.name, "loaded": active,
            "busy": worker.lock.locked(), "cuda_flags": TUNED_WORKER_ARGS}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    text = request.prompt.encode("utf-8")
    if not request.prompt.strip() or len(text) > 32768:
        raise HTTPException(422, "Enter a nonempty UTF-8 prompt of at most 32 KiB.")
    return StreamingResponse(app.state.worker.stream(text, request.max_tokens),
                             media_type="application/x-ndjson")


@app.get("/runs")
async def runs():
    with connection() as db:
        rows = db.execute(
            "SELECT id,state,name,report,error FROM runs "
            "WHERE name NOT IN ('head-to-head','SPARK vs llama.cpp') ORDER BY rowid DESC"
        ).fetchall()
    return [row_data(row) for row in rows]


@app.get("/api/comparison")
async def comparison(run: str | None = None, trial: int | None = None, prompt: int | None = None):
    if trial is not None and (trial < 1 or prompt is None):
        raise HTTPException(422, "Select a prompt length and a one-based measured trial.")
    with connection() as db:
        row = db.execute(
            "SELECT id,state,name,report,error FROM runs "
            "WHERE name IN ('head-to-head','SPARK vs llama.cpp') AND state = 'completed' "
            "AND (? IS NULL OR id = ?) ORDER BY json_array_length(report, '$.workloads') = 3 DESC, rowid DESC LIMIT 1", (run, run)
        ).fetchone()
    if not row:
        return None
    result = row_data(row)
    report = result["report"]
    if trial is not None:
        workload = next((entry for entry in report["workloads"] if entry["prompt_tokens"] == prompt), None)
        if workload is None or trial > report["measured"]:
            raise HTTPException(404, "Measured trial not found.")
        return {
            name: [item for item in engine["trials"] if not item["warmup"]][trial - 1].get("telemetry", [])
            for name, engine in workload["engines"].items()
        }
    for workload in report["workloads"]:
        spark_trials = [trial for trial in workload["engines"]["spark"]["trials"] if not trial["warmup"]]
        llama_trials = [trial for trial in workload["engines"]["llama_cpp"]["trials"] if not trial["warmup"]]
        workload["matching_trials"] = sum(
            left["generated_token_ids"] == right["generated_token_ids"]
            for left, right in zip(spark_trials, llama_trials)
        )
        for engine in workload["engines"].values():
            for trial in engine["trials"]:
                trial.pop("telemetry", None)
                trial.pop("host_telemetry", None)
                trial.pop("generated_token_ids", None)
        workload.pop("prompt_token_ids", None)
    return result


@app.get("/api/comparisons")
async def comparison_history():
    with connection() as db:
        rows = db.execute(
            "SELECT id, name, report FROM runs WHERE name IN ('head-to-head','SPARK vs llama.cpp') "
            "AND state = 'completed' ORDER BY rowid DESC"
        ).fetchall()
    entries = []
    for identifier, name, raw in rows:
        report = json.loads(raw)
        entries.append({"id": identifier, "name": name, "recorded_at": report.get("recorded_at"),
                        "commit": report.get("spark_commit"), "saved": identifier in SNAPSHOTS})
    return entries


@app.get("/api/experiments")
async def experiment_history():
    stages = {
        "mission-baseline": "Serial packed CUDA",
        "batch-prefill": "Batched prefill",
        "packed-graph": "Packed decode graph",
        "q6-block-gemv": "Q6_K block GEMV",
        "fp16-prefill": "FP16 prefill GEMM",
        "q4-block-gemv": "Q4_K block GEMV",
    }
    result = []
    with connection() as db:
        for identifier, label in stages.items():
            row = db.execute("SELECT report FROM runs WHERE id = ? AND state = 'completed'", (identifier,)).fetchone()
            if row is None:
                continue
            report = json.loads(row[0])
            primary = next((item for item in report["workloads"] if item["prompt_tokens"] == 512), None)
            if primary is not None:
                result.append({"id": identifier, "label": label, "metrics": primary["engines"]["spark"]["metrics"]})
    return result


@app.get("/runs/{identifier}")
async def run_details(identifier: str, download: bool = False, status_only: bool = False):
    with connection() as db:
        row = db.execute(
            "SELECT id,state,name,report,error FROM runs WHERE id = ?", (identifier,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Run not found.")
    if status_only:
        return {"id": row[0], "state": row[1], "name": row[2], "error": row[4]}
    result = row_data(row)
    if download:
        return JSONResponse(result, headers={"Content-Disposition": f'attachment; filename="spark-{identifier}.json"'})
    return result


@app.delete("/runs/{identifier}", status_code=204)
async def delete_run(identifier: str):
    if identifier in SNAPSHOTS:
        raise HTTPException(409, "Repository snapshots are preserved.")
    with connection() as db:
        removed = db.execute("DELETE FROM runs WHERE id = ? AND state IN ('completed','failed')", (identifier,))
        if not removed.rowcount:
            if db.execute("SELECT 1 FROM runs WHERE id = ?", (identifier,)).fetchone():
                raise HTTPException(409, "Wait until the run finishes.")
            raise HTTPException(404, "Run not found.")
    # Only UUID-named dashboard jobs have a spool file.
    if identifier.isalnum() and len(identifier) == 32:
        (DATA / f"{identifier}.json").unlink(missing_ok=True)


@app.post("/runs", status_code=202)
async def run_benchmark(request: BenchmarkRequest):
    worker = app.state.worker
    if worker.tasks:
        raise HTTPException(409, "A benchmark is already queued or running.")
    if request.mode == "head-to-head" and not worker.llama.is_file():
        raise HTTPException(503, "The CUDA llama-server binary is unavailable.")
    identifier = uuid.uuid4().hex
    save_run(identifier, "queued", request.mode)
    task = asyncio.create_task(worker.benchmark(identifier, request.mode))
    worker.tasks.add(task)
    task.add_done_callback(worker.tasks.discard)
    return {"id": identifier, "state": "queued"}
