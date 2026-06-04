from __future__ import annotations

import hashlib
import importlib
import math
import os
import secrets
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quant_platform.pipeline import run_pipeline
from quant_platform.settings import OUTPUT_DIR, ROOT_DIR


STATIC_DIR = ROOT_DIR / "static"
app = FastAPI(title="A股量化选股平台 API")
executor = ThreadPoolExecutor(max_workers=1)
tasks: dict[str, dict[str, Any]] = {}
sessions: set[str] = set()

ADMIN_USER = os.getenv("ADMIN_USER", "13246429006")
ADMIN_PASSWORD_SHA256 = os.getenv("ADMIN_PASSWORD_SHA256", "2e5ebb44a6829536bb6219a98f937127e4ece90a4eab75a2c42e22991051da16")


REQUIRED_IMPORTS = {
    "akshare": "akshare",
    "baostock": "baostock",
    "efinance": "efinance",
    "fastapi": "fastapi",
    "mootdx": "mootdx",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "tushare": "tushare",
    "uvicorn": "uvicorn",
}


class RunRequest(BaseModel):
    top_n: int | None = Field(default=None, ge=20)
    lookback_days: int = Field(default_factory=lambda: int(os.getenv("LOOKBACK_DAYS", "1825")), ge=260, le=1825)
    adjust: str = "qfq"
    fetch_news: bool = True
    refresh_spot: bool = False
    refresh_history: bool = False
    refresh_news: bool = False
    deepseek_api_key: str | None = None
    deepseek_model: str = "deepseek-chat"


class LoginRequest(BaseModel):
    username: str
    password: str


def ensure_environment() -> dict[str, Any]:
    missing: list[str] = []
    for import_name, package_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(package_name)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    return {"installed": missing, "ok": True}


def verify_password(password: str) -> bool:
    digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return secrets.compare_digest(digest, ADMIN_PASSWORD_SHA256)


def require_auth(request: Request) -> bool:
    token = request.cookies.get("aq_session")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="请先登录")
    return True


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return clean_value(value.item())
    return value


def dataframe_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    records = safe.replace({pd.NA: None}).to_dict(orient="records")
    return [{key: clean_value(value) for key, value in record.items()} for record in records]


def run_task(task_id: str, request: RunRequest) -> None:
    tasks[task_id]["status"] = "running"
    tasks[task_id]["message"] = "正在检查环境并抓取全市场数据"
    try:
        env_result = ensure_environment()
        tasks[task_id]["environment"] = env_result
        tasks[task_id]["message"] = "正在抓取沪深全市场行情、新闻、公告、政策、资金与热点数据"
        args = request.model_dump()
        args["deepseek_api_key"] = args.get("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")
        result = run_pipeline(**args)
        predictions = result.predictions.copy()
        target = predictions[predictions["meets_target"] == True].copy()
        payload = {
            "generated_at": result.generated_at.isoformat(),
            "metrics": {key: clean_value(value) for key, value in result.metrics.items()},
            "target_count": int(len(target)),
            "target": dataframe_records(target, limit=300),
            "ranking": dataframe_records(predictions, limit=1000),
            "validation": dataframe_records(result.validation, limit=1000),
            "optimization_log": dataframe_records(result.optimization_log.sort_values("attempt"), limit=600),
            "output_csv": str((OUTPUT_DIR / "latest_predictions.csv").resolve()),
            "validation_csv": str((OUTPUT_DIR / "latest_validation.csv").resolve()),
            "optimization_csv": str((OUTPUT_DIR / "latest_optimization_log.csv").resolve()),
        }
        tasks[task_id].update({"status": "done", "message": "完成", "result": payload})
    except Exception as exc:
        tasks[task_id].update({"status": "error", "message": str(exc)})


@app.get("/")
def index(request: Request) -> FileResponse:
    token = request.cookies.get("aq_session")
    if not token or token not in sessions:
        return FileResponse(STATIC_DIR / "login.html")
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/login")
def login(request: LoginRequest) -> JSONResponse:
    if request.username != ADMIN_USER or not verify_password(request.password):
        raise HTTPException(status_code=401, detail="账号或密码错误")
    token = secrets.token_urlsafe(32)
    sessions.add(token)
    response = JSONResponse({"ok": True})
    response.set_cookie("aq_session", token, httponly=True, samesite="strict", secure=False, max_age=60 * 60 * 12)
    return response


@app.post("/api/logout")
def logout(request: Request) -> JSONResponse:
    token = request.cookies.get("aq_session")
    if token:
        sessions.discard(token)
    response = JSONResponse({"ok": True})
    response.delete_cookie("aq_session")
    return response


@app.get("/api/me")
def me(_: bool = Depends(require_auth)) -> dict[str, Any]:
    return {"authenticated": True, "username": ADMIN_USER}


@app.post("/api/run")
def start_run(request: RunRequest, _: bool = Depends(require_auth)) -> dict[str, str]:
    task_id = uuid.uuid4().hex
    tasks[task_id] = {"status": "queued", "message": "任务已排队"}
    executor.submit(run_task, task_id, request)
    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
def get_status(task_id: str, _: bool = Depends(require_auth)) -> dict[str, Any]:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("status") == "done":
        result = task.get("result", {})
        return {
            "status": "done",
            "message": task.get("message"),
            "result": {
                "generated_at": result.get("generated_at"),
                "metrics": result.get("metrics", {}),
                "target_count": result.get("target_count", 0),
                "target": result.get("target", []),
                "ranking": result.get("ranking", []),
                "output_csv": result.get("output_csv"),
                "validation_csv": result.get("validation_csv"),
                "optimization_csv": result.get("optimization_csv"),
            },
        }
    return task


@app.get("/api/result/{task_id}")
def get_result(task_id: str, _: bool = Depends(require_auth)) -> dict[str, Any]:
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.get("status") != "done":
        raise HTTPException(status_code=409, detail="任务尚未完成")
    return task["result"]


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8501, reload=False)
