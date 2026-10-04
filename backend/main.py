"""FastAPI app: REST APIs for dataset, DSP, leaf analysis and plant health."""
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

import config
from dsp import run_dsp
from health import assess
from image_analysis import analyze_leaf

app = FastAPI(title="Plant Health & Chlorophyll Monitoring using DSP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_df():
    if not config.CSV_PATH.exists():
        raise HTTPException(404, f"Dataset not found at {config.CSV_PATH}")
    df = pd.read_csv(config.CSV_PATH)
    return df[[c for c in config.SENSOR_COLUMNS if c in df.columns]]


def list_images():
    if not config.IMAGE_DIR.exists():
        return []
    return sorted(p.name for p in config.IMAGE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def decode(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "File is not a valid image.")
    return img


def safe_leaf(img):
    try:
        return analyze_leaf(img)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/status")
def status():
    return {"ok": True, "csv_found": config.CSV_PATH.exists(), "images": len(list_images())}


@app.get("/api/dataset/info")
def dataset_info():
    df = load_df()
    return {
        "rows": len(df), "columns": list(df.columns), "units": config.UNITS,
        "stats": df.describe().round(3).to_dict(),
        "missing_values": int(df.isna().sum().sum()),
        "not_in_dataset": ["soil moisture", "light intensity"],
        "defaults": config.DSP_DEFAULTS,
    }


@app.get("/api/images")
def images(offset: int = 0, limit: int = 24):
    names = list_images()
    return {"total": len(names), "items": names[offset:offset + limit]}


@app.get("/api/images/{name}")
def get_image(name: str):
    p = (config.IMAGE_DIR / name).resolve()
    if config.IMAGE_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(p)


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    return safe_leaf(decode(await file.read()))


@app.get("/api/analyze/sample")
def analyze_sample(name: str):
    p = (config.IMAGE_DIR / name).resolve()
    if config.IMAGE_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "Sample image not found")
    return safe_leaf(decode(p.read_bytes()))


@app.get("/api/dsp")
def dsp(column: str = "temperature", start: int = 0, n: int = 256,
        fs: float = config.DSP_DEFAULTS["fs"], cutoff: float = config.DSP_DEFAULTS["cutoff"],
        order: int = Query(config.DSP_DEFAULTS["order"], ge=1, le=10)):
    df = load_df()
    if column not in df.columns:
        raise HTTPException(400, f"Column '{column}' not in dataset. Available: {list(df.columns)}")
    if start < 0 or start >= len(df) - 16:
        raise HTTPException(400, f"start must be between 0 and {len(df) - 17}")
    seg = df.iloc[start:start + n]
    try:
        out = run_dsp(seg[column].to_numpy(), fs, cutoff, order)
    except ValueError as e:
        raise HTTPException(422, str(e))
    # "current readings" = last sample of the filtered window for each real column
    readings = {}
    for c in df.columns:
        try:
            readings[c] = round(float(run_dsp(seg[c].to_numpy(), fs, cutoff, order)["filtered"][-1]), 2)
        except ValueError:
            readings[c] = None
    out.update(column=column, unit=config.UNITS.get(column, ""), start=start, readings=readings,
               time_axis_note="No timestamps in CSV: row index = n; assumed 1 reading/hour.")
    return out


class HealthIn(BaseModel):
    greenness_level: str
    readings: dict


@app.post("/api/health")
def health(body: HealthIn):
    if body.greenness_level not in ("Low", "Moderate", "High"):
        raise HTTPException(400, "greenness_level must be Low, Moderate or High")
    return assess(body.greenness_level, body.readings)


@app.get("/api/config")
def get_config():
    return {"greenness": config.GREENNESS, "env_rules": config.ENV_RULES,
            "green_points": config.GREEN_POINTS, "health_levels": config.HEALTH_LEVELS}
