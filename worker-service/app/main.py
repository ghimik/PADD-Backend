import os
import io
import json
from pathlib import Path
import uuid
import tempfile
from typing import Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response

from core.Neuretus_XElite.core.detectors import (
        MalboroDetector, 
        ComputantisDetector, 
        CornerBaneRefiner
    )
from core.Neuretus_XElite.core.geometry import HomographyCorrector
from core.Neuretus_XElite.core.ocr import OCRProcessor, RotationDetector
from core.Neuretus_XElite.core.pdfyer import PDFEngine


models_dir = Path(__file__).parent.parent / "core" / "Neuretus_XElite" / "models"

MODELS_CONFIG = {
    "malboro_path": models_dir / "sychok_bygarety.pt",
    "computantis_path": models_dir / "computantis.pt",
    "refiner_path": models_dir / "corner_bane.pth",
    "font_path": models_dir / "fonts" / "DejaVuSans.ttf"
}

BASE_DIR = "app/storage"
os.makedirs(BASE_DIR, exist_ok=True)

app = FastAPI(title="Document Processing API")

models = {}

@app.on_event("startup")
async def startup_event():
    print("Загрузка моделей...")
    try:
        models["rotation"] = RotationDetector(output_dir=BASE_DIR)
        
        models["malboro"] = MalboroDetector(
            model_path=MODELS_CONFIG["malboro_path"], 
            output_dir=BASE_DIR
        )
        
        models["computantis"] = ComputantisDetector(
            model_path=MODELS_CONFIG["computantis_path"], 
            output_dir=BASE_DIR
        )
        
        models["refiner"] = CornerBaneRefiner(
            model_path=MODELS_CONFIG["refiner_path"], 
            output_dir=BASE_DIR
        )
        
        models["pdf_engine"] = PDFEngine(font_path=MODELS_CONFIG["font_path"])
        
        print("Модели успешно загружены.")
    except Exception as e:
        print(f"ОШИБКА ЗАГРУЗКИ МОДЕЛЕЙ: {e}")
        exit(1)



def load_image_from_upload(file: UploadFile) -> np.ndarray:
    """Читает загруженный файл в numpy array (BGR для OpenCV)."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
    return img

def get_request_dir():
    """Создает уникальную папку для запроса."""
    request_id = str(uuid.uuid4())
    path = os.path.join(BASE_DIR, request_id)
    os.makedirs(path, exist_ok=True)
    return path


@app.post("/define_rotation_angle")
async def api_define_rotation_angle(file: UploadFile = File(...)):
    """
    1) define_rotation_angle - принимает файл, возвращает угол.
    """
    img = load_image_from_upload(file)
    
    angle, score = models["rotation"].detect_angle(img)
    
    return {
        "angle": int(angle),
        "score": float(score)
    }


@app.post("/find_corners_and_bbox")
async def api_find_corners_and_bbox(file: UploadFile = File(...)):
    """
    2) find_corners_and_bbox - двухэтапная логика с фоллбэком.
    """
    img = load_image_from_upload(file)
    
    corners, bbox = None, None
    detector_used = None
    
    try:
        corners, bbox = models["malboro"].detect(img)
        detector_used = "malboro"
    except Exception as e:
        print(f"Malboro failed: {e}. Trying Computantis...")
        try:
            corners, bbox = models["computantis"].detect(img)
            detector_used = "computantis"
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Detection failed on both models: {str(e2)}")
    
    clean_corners = {
        k: (int(v[0]), int(v[1])) for k, v in corners.items()
    }
    
    clean_bbox = [int(x) for x in bbox]
    
    return {
        "detector": detector_used,
        "corners": clean_corners,
        "bbox": clean_bbox
    }


@app.post("/refine_corners")
async def api_refine_corners(
    file: UploadFile = File(...),
    corners: str = Form(...), 
    bbox: Optional[str] = Form(None) 
):
    """
    3) refine_corners - уточнение углов. 
    """
    img = load_image_from_upload(file)
    
    try:
        corners_dict = json.loads(corners)
        corners_dict = {k: tuple(v) for k, v in corners_dict.items()}
    except:
        raise HTTPException(status_code=400, detail="Invalid corners JSON format")

    bbox_tuple = None
    if bbox:
        try:
            bbox_list = json.loads(bbox)
            bbox_tuple = tuple(bbox_list)
        except:
            pass
            
    if bbox_tuple is None:
        xs = [c[0] for c in corners_dict.values()]
        ys = [c[1] for c in corners_dict.values()]
        bbox_tuple = (min(xs), min(ys), max(xs), max(ys))

    try:
        refined_corners = models["refiner"].refine(
            img, 
            coarse_corners=corners_dict, 
            bbox=bbox_tuple
        )
        
        clean_refined = {k: (int(v[0]), int(v[1])) for k, v in refined_corners.items()}
        
        return {"refined_corners": clean_refined}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")


@app.post("/warp_perspective")
async def api_warp_perspective(
    file: UploadFile = File(...),
    corners: str = Form(...)
):
    """
    4) warp_perspective - выпрямление документа.
    """
    img = load_image_from_upload(file)
    
    try:
        corners_dict = json.loads(corners)
        corners_dict = {k: tuple(v) for k, v in corners_dict.items()}
    except:
        raise HTTPException(status_code=400, detail="Invalid corners JSON format")

    out_dir = get_request_dir()
    corrector = HomographyCorrector(output_dir=out_dir)
    
    try:
        warped_img = corrector.correct(img, corners_dict)
        
        success, buffer = cv2.imencode('.jpg', warped_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode output image")
            
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warping failed: {str(e)}")


@app.post("/do_ocr")
async def api_do_ocr(file: UploadFile = File(...)):
    """
    5) do_ocr - делает OCR, возвращает PDF файл.
    """
    img = load_image_from_upload(file)
    
    doc_id = str(uuid.uuid4())
    doc_dir = os.path.join(BASE_DIR, doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    
    ocr = OCRProcessor(output_dir=doc_dir)
    
    try:
        json_path = ocr.recognize(img)
        
        pdf_path = os.path.join(doc_dir, "output.pdf")
        img_dir = os.path.join(doc_dir, "ocr_output", "imgs")
        
        if os.path.exists(json_path):
            if os.path.exists(img_dir):
                models["pdf_engine"].reconstruct(json_path, pdf_path, image_dir=img_dir)
            else:
                models["pdf_engine"].reconstruct(json_path, pdf_path)
        else:
            raise HTTPException(status_code=500, detail="OCR result JSON not found")

        if not os.path.exists(pdf_path):
             raise HTTPException(status_code=500, detail="PDF was not generated")
             
        return FileResponse(
            pdf_path, 
            media_type="application/pdf", 
            filename=f"{doc_id}.pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Pipeline failed: {str(e)}")

@app.post("/stretch_to_aspect")
async def api_stretch_to_aspect(
    file: UploadFile = File(...),
    target_width: int = Form(...),
    target_height: int = Form(...)
):
    """
    Растягивает изображение под заданное соотношение сторон.
    """
    
    img = load_image_from_upload(file)

    try:
        stretched = cv2.resize(
            img,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

        success, buffer = cv2.imencode(".jpg", stretched)

        if not success:
            raise HTTPException(status_code=500, detail="Encoding failed")

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))