"""Local control plane for the native SPARK CUDA worker."""

import asyncio
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "api" / "frontend"
DATA = ROOT / "build" / "dashboard"
DB = DATA / "runs.sqlite"
RESULTS = ROOT / "benchmark" / "results"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=128, ge=1, le=512)


class BenchmarkRequest(BaseModel):
    mode: str = Field(pattern="^(baseline|bucket|packed)$")


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
                str(self.spark), "serve", "--model", str(self.model), cwd=ROOT,
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
                env = {**os.environ, "SPARK_CUDA_GRAPH": "bucket" if mode == "bucket" else "off",
                       "SPARK_CUDA_WEIGHTS": "packed" if mode == "packed" else "fp32"}
                process = await asyncio.create_subprocess_exec(
                    "node", str(ROOT / "benchmark" / "runner" / "measure_native.cjs"),
                    str(self.spark), str(self.model), str(output), cwd=ROOT, env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                _, errors = await process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(errors.decode(errors="replace")[-1200:])
                save_run(identifier, "completed", mode, json.loads(output.read_text()))
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
    for identifier, filename in (
        ("baseline-v0", "spark-qwen-v0.json"),
        ("graph-bucket", "spark-qwen-graph-bucket.json"),
        ("graph-single", "spark-qwen-graph-single.json"),
        ("packed", "spark-qwen-packed.json"),
    ):
        path = RESULTS / filename
        if path.is_file():
            report = json.loads(path.read_text())
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
            "busy": worker.lock.locked()}


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
        rows = db.execute("SELECT id,state,name,report,error FROM runs ORDER BY rowid DESC").fetchall()
    return [row_data(row) for row in rows]


@app.get("/runs/{identifier}")
async def run_details(identifier: str):
    with connection() as db:
        row = db.execute(
            "SELECT id,state,name,report,error FROM runs WHERE id = ?", (identifier,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Run not found.")
    return row_data(row)


@app.post("/runs", status_code=202)
async def run_benchmark(request: BenchmarkRequest):
    worker = app.state.worker
    if worker.tasks:
        raise HTTPException(409, "A benchmark is already queued or running.")
    identifier = uuid.uuid4().hex
    save_run(identifier, "queued", request.mode)
    task = asyncio.create_task(worker.benchmark(identifier, request.mode))
    worker.tasks.add(task)
    task.add_done_callback(worker.tasks.discard)
    return {"id": identifier, "state": "queued"}
