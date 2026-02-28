import os
import uuid
import shutil
import json
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Dispatcher Service")

# Конфиг
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/app/storage")
WORKER_URL = os.environ.get("WORKER_URL", "http://worker:8001")

os.makedirs(STORAGE_DIR, exist_ok=True)


class WarpRequest(BaseModel):
    corners: dict


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Загружает изображение, возвращает ID"""
    # Генерируем ID
    doc_id = str(uuid.uuid4())[:8]
    
    # Создаем папку сессии
    session_dir = Path(STORAGE_DIR) / doc_id
    session_dir.mkdir(exist_ok=True)
    
    # Сохраняем оригинал
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    original_path = session_dir / f"original{ext}"
    
    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Отправляем задачу воркеру (fire-and-forget)
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{WORKER_URL}/internal/corners",
                json={
                    "id": doc_id,
                    "path": str(original_path)
                },
                timeout=5.0
            )
        except:
            # Если воркер недоступен - ничего страшного, клиент все равно 
            # получит 425 при попытке получить углы
            pass
    
    return {"id": doc_id}


@app.get("/{doc_id}/corners")
async def get_corners(doc_id: str):
    """Возвращает углы, если они уже посчитаны"""
    session_dir = Path(STORAGE_DIR) / doc_id
    corners_file = session_dir / "corners.json"
    
    if not session_dir.exists():
        raise HTTPException(404, "not_found")
    
    if not corners_file.exists():
        # Проверяем, может оригинал вообще не загружен
        if not list(session_dir.glob("original.*")):
            raise HTTPException(404, "not_found")
        
        # Если оригинал есть, а углов нет - значит еще считаются
        raise HTTPException(425, "processing")
    
    with open(corners_file) as f:
        data = json.load(f)
    
    return data


@app.post("/{doc_id}/warp")
async def warp_document(doc_id: str, request: WarpRequest):
    """Выравнивает по переданным углам"""
    session_dir = Path(STORAGE_DIR) / doc_id
    
    # Ищем оригинал (любое расширение)
    originals = list(session_dir.glob("original.*"))
    if not originals:
        raise HTTPException(404, "not_found")
    
    original_path = originals[0]
    warped_path = session_dir / "warped.jpg"
    
    # Отправляем задачу воркеру
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{WORKER_URL}/internal/warp",
            json={
                "id": doc_id,
                "original_path": str(original_path),
                "corners": request.corners,
                "output_path": str(warped_path)
            },
            timeout=30.0
        )
    
    if resp.status_code != 200:
        raise HTTPException(500, "warp_failed")
    
    return FileResponse(
        warped_path,
        media_type="image/jpeg",
        filename=f"{doc_id}_warped.jpg"
    )


@app.post("/{doc_id}/ocr")
async def ocr_document(doc_id: str):
    """Делает OCR и возвращает PDF"""
    session_dir = Path(STORAGE_DIR) / doc_id
    warped_path = session_dir / "warped.jpg"
    pdf_path = session_dir / "document.pdf"
    
    if not warped_path.exists():
        raise HTTPException(409, "warp_first")
    
    # Если PDF уже есть - сразу отдаем
    if pdf_path.exists():
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{doc_id}.pdf"
        )
    
    # Иначе запускаем OCR
    ocr_json_path = session_dir / "ocr.json"
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{WORKER_URL}/internal/ocr",
            json={
                "id": doc_id,
                "warped_path": str(warped_path),
                "output_json": str(ocr_json_path),
                "output_pdf": str(pdf_path)
            },
            timeout=60.0
        )
    
    if resp.status_code != 200:
        raise HTTPException(500, "ocr_failed")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{doc_id}.pdf"
    )


# Прямой доступ к файлам (для отладки)
@app.get("/{doc_id}/original")
async def get_original(doc_id: str):
    session_dir = Path(STORAGE_DIR) / doc_id
    originals = list(session_dir.glob("original.*"))
    if not originals:
        raise HTTPException(404, "not_found")
    return FileResponse(originals[0])


@app.get("/{doc_id}/warped")
async def get_warped(doc_id: str):
    session_dir = Path(STORAGE_DIR) / doc_id
    warped = session_dir / "warped.jpg"
    if not warped.exists():
        raise HTTPException(404, "not_found")
    return FileResponse(warped)


@app.get("/{doc_id}/pdf")
async def get_pdf(doc_id: str):
    session_dir = Path(STORAGE_DIR) / doc_id
    pdf = session_dir / "document.pdf"
    if not pdf.exists():
        raise HTTPException(404, "not_found")
    return FileResponse(pdf)