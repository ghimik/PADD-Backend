from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Tuple, Optional
import os
import cv2
import numpy as np

from core.Neuretus_XElite.api import NeuretusXElite


app = FastAPI(title="Worker Service")

models_dir = Path(__file__).parent.parent / "core" / "Neuretus_XElite" / "models"

elite = NeuretusXElite(
    models_dir=str(models_dir),
    output_dir="/app/storage"
)


class CornersRequest(BaseModel):
    id: str
    path: str


class WarpRequest(BaseModel):
    id: str
    original_path: str
    corners: Dict[str, Tuple[int, int]]
    output_path: str


class OCRRequest(BaseModel):
    id: str
    warped_path: str
    output_json: str
    output_pdf: str


@app.post("/internal/corners")
async def detect_corners(req: CornersRequest):
    """Детекция углов и bbox"""
    try:
        image = cv2.imread(req.path)
        if image is None:
            raise HTTPException(400, "Cannot read image")
        
        corners, bbox = elite.find_corners_and_bbox(image, doc_id=req.id)
        
        return {
            "corners": corners,
            "bbox": bbox
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/internal/warp")
async def warp_perspective(req: WarpRequest):
    """Выравнивание перспективы"""
    try:
        image = cv2.imread(req.original_path)
        if image is None:
            raise HTTPException(400, "Cannot read image")
        
        warped = elite.warp_perspective(image, req.corners, doc_id=req.id)
        cv2.imwrite(req.output_path, warped)
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/internal/ocr")
async def do_ocr(req: OCRRequest):
    """OCR и генерация PDF"""
    try:
        image = cv2.imread(req.warped_path)
        if image is None:
            raise HTTPException(400, "Cannot read image")
        
        # Сохраняем пути для PDFEngine
        doc_dir = os.path.dirname(req.warped_path)
        elite.do_ocr(image, doc_id=req.id)
        
        # PDF уже должен быть создан внутри do_ocr
        if not os.path.exists(req.output_pdf):
            raise HTTPException(500, "PDF generation failed")
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/internal/health")
async def health():
    return {"status": "ok"}