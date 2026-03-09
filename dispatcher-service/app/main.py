import os
import uuid
import shutil
import json
import asyncio
from pathlib import Path
from typing import Optional
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Dispatcher Service")

# Конфиг
STORAGE_DIR = os.environ.get("STORAGE_DIR", "/app/storage")
WORKER_URL = os.environ.get("WORKER_URL", "http://worker:8041")

os.makedirs(STORAGE_DIR, exist_ok=True)

# HTTP клиент для воркера
http_client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


class WarpRequest(BaseModel):
    corners: dict


class ProcessingStatus:
    PENDING = "pending"        # только загружено
    CORNERS_DETECTING = "detecting"  # считаются углы
    CORNERS_DONE = "corners_done"    # углы посчитаны
    WARP_DONE = "warp_done"          # выравнивание готово
    OCR_PROCESSING = "ocr_processing"  # делается OCR
    OCR_DONE = "ocr_done"            # PDF готов
    ERROR = "error"                   # ошибка


@app.on_event("shutdown")
async def shutdown():
    await http_client.aclose()


def get_session_dir(doc_id: str) -> Path:
    return Path(STORAGE_DIR) / doc_id


def get_status_file(doc_id: str) -> Path:
    return get_session_dir(doc_id) / "status.json"


def get_corners_file(doc_id: str) -> Path:
    return get_session_dir(doc_id) / "corners.json"


def get_original_path(doc_id: str) -> Optional[Path]:
    session_dir = get_session_dir(doc_id)
    originals = list(session_dir.glob("original.*"))
    return originals[0] if originals else None


def get_warped_path(doc_id: str) -> Path:
    return get_session_dir(doc_id) / "warped.jpg"


def get_pdf_path(doc_id: str) -> Path:
    return get_session_dir(doc_id) / "document.pdf"


def update_status(doc_id: str, status: str, error: str = None):
    """Обновляет статус документа"""
    status_file = get_status_file(doc_id)
    status_data = {"status": status}
    if error:
        status_data["error"] = error
    with open(status_file, "w") as f:
        json.dump(status_data, f)


@app.post("/upload")
async def upload_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Загружает изображение и запускает детекцию углов в фоне"""
    # Генерируем ID
    doc_id = str(uuid.uuid4())[:8]
    
    # Создаем папку сессии
    session_dir = get_session_dir(doc_id)
    session_dir.mkdir(exist_ok=True)
    
    # Сохраняем оригинал
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    original_path = session_dir / f"original{ext}"
    
    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Устанавливаем статус
    update_status(doc_id, ProcessingStatus.PENDING)
    
    # Запускаем детекцию углов в фоне
    background_tasks.add_task(detect_corners_background, doc_id, str(original_path))
    
    return {"id": doc_id, "status": ProcessingStatus.PENDING}


async def detect_corners_background(doc_id: str, image_path: str):
    """Фоновая задача для детекции углов"""
    try:
        update_status(doc_id, ProcessingStatus.CORNERS_DETECTING)
        
        # Отправляем запрос воркеру
        response = await http_client.post(
            f"{WORKER_URL}/internal/corners",
            json={"id": doc_id, "path": image_path}
        )
        
        if response.status_code != 200:
            error_msg = f"Worker error: {response.text}"
            update_status(doc_id, ProcessingStatus.ERROR, error_msg)
            return
        
        # Сохраняем результат
        result = response.json()
        corners_file = get_corners_file(doc_id)
        with open(corners_file, "w") as f:
            json.dump(result, f)
        
        update_status(doc_id, ProcessingStatus.CORNERS_DONE)
        
    except Exception as e:
        update_status(doc_id, ProcessingStatus.ERROR, str(e))


@app.get("/{doc_id}/status")
async def get_status(doc_id: str):
    """Возвращает статус обработки документа"""
    session_dir = get_session_dir(doc_id)
    if not session_dir.exists():
        raise HTTPException(404, "Document not found")
    
    status_file = get_status_file(doc_id)
    if not status_file.exists():
        # Для обратной совместимости - если нет статуса, но есть оригинал
        original = get_original_path(doc_id)
        if original:
            corners_file = get_corners_file(doc_id)
            if corners_file.exists():
                update_status(doc_id, ProcessingStatus.CORNERS_DONE)
            else:
                update_status(doc_id, ProcessingStatus.PENDING)
        else:
            raise HTTPException(404, "Document not found")
    
    with open(status_file) as f:
        status_data = json.load(f)
    
    # Добавляем информацию о наличии файлов
    status_data["files"] = {
        "original": get_original_path(doc_id) is not None,
        "corners": get_corners_file(doc_id).exists(),
        "warped": get_warped_path(doc_id).exists(),
        "pdf": get_pdf_path(doc_id).exists()
    }
    
    return status_data


@app.get("/{doc_id}/corners")
async def get_corners(doc_id: str):
    """Возвращает углы, если они уже посчитаны"""
    session_dir = get_session_dir(doc_id)
    if not session_dir.exists():
        raise HTTPException(404, "Document not found")
    
    corners_file = get_corners_file(doc_id)
    if not corners_file.exists():
        # Проверяем статус
        status_file = get_status_file(doc_id)
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f).get("status")
            if status == ProcessingStatus.CORNERS_DETECTING:
                raise HTTPException(425, "Corners are being detected")
            elif status == ProcessingStatus.ERROR:
                raise HTTPException(500, "Processing failed")
        
        # Если нет статуса и нет углов - запускаем детекцию
        original = get_original_path(doc_id)
        if original:
            # Запускаем в фоне и возвращаем 202
            asyncio.create_task(detect_corners_background(doc_id, str(original)))
            raise HTTPException(202, "Detection started")
        else:
            raise HTTPException(404, "Original image not found")
    
    with open(corners_file) as f:
        data = json.load(f)
    
    return data


@app.post("/{doc_id}/warp")
async def warp_document(doc_id: str, request: WarpRequest, background_tasks: BackgroundTasks):
    """Выравнивает по переданным углам"""
    session_dir = get_session_dir(doc_id)
    if not session_dir.exists():
        raise HTTPException(404, "Document not found")
    
    # Проверяем наличие оригинала
    original_path = get_original_path(doc_id)
    if not original_path:
        raise HTTPException(404, "Original image not found")
    
    warped_path = get_warped_path(doc_id)
    
    # Отправляем задачу воркеру
    try:
        response = await http_client.post(
            f"{WORKER_URL}/internal/warp",
            json={
                "id": doc_id,
                "original_path": str(original_path),
                "corners": request.corners,
                "output_path": str(warped_path)
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(500, f"Warp failed: {response.text}")
        
        update_status(doc_id, ProcessingStatus.WARP_DONE)
        
        # Возвращаем выровненное изображение
        return FileResponse(
            warped_path,
            media_type="image/jpeg",
            filename=f"{doc_id}_warped.jpg"
        )
        
    except httpx.RequestError as e:
        raise HTTPException(503, f"Worker unavailable: {str(e)}")


@app.post("/{doc_id}/ocr")
async def ocr_document(doc_id: str, background_tasks: BackgroundTasks):
    """Запускает OCR и возвращает PDF"""
    session_dir = get_session_dir(doc_id)
    if not session_dir.exists():
        raise HTTPException(404, "Document not found")
    
    warped_path = get_warped_path(doc_id)
    if not warped_path.exists():
        raise HTTPException(409, "Warp image not found. Call /warp first")
    
    pdf_path = get_pdf_path(doc_id)
    
    # Если PDF уже есть - сразу отдаем
    if pdf_path.exists():
        update_status(doc_id, ProcessingStatus.OCR_DONE)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{doc_id}.pdf"
        )
    
    # Обновляем статус
    update_status(doc_id, ProcessingStatus.OCR_PROCESSING)
    
    # Запускаем OCR
    try:
        response = await http_client.post(
            f"{WORKER_URL}/internal/ocr",
            json={
                "id": doc_id,
                "warped_path": str(warped_path),
                "output_json": str(session_dir / "ocr_result.json"),
                "output_pdf": str(pdf_path)
            }
        )
        
        if response.status_code != 200:
            update_status(doc_id, ProcessingStatus.ERROR, f"OCR failed: {response.text}")
            raise HTTPException(500, "OCR failed")
        
        update_status(doc_id, ProcessingStatus.OCR_DONE)
        
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{doc_id}.pdf"
        )
        
    except httpx.RequestError as e:
        update_status(doc_id, ProcessingStatus.ERROR, str(e))
        raise HTTPException(503, f"Worker unavailable: {str(e)}")


# Прямой доступ к файлам (для отладки)
@app.get("/{doc_id}/original")
async def get_original(doc_id: str):
    original = get_original_path(doc_id)
    if not original:
        raise HTTPException(404, "Original image not found")
    return FileResponse(original)


@app.get("/{doc_id}/warped")
async def get_warped(doc_id: str):
    warped = get_warped_path(doc_id)
    if not warped.exists():
        raise HTTPException(404, "Warped image not found")
    return FileResponse(warped)


@app.get("/{doc_id}/pdf")
async def get_pdf(doc_id: str):
    pdf = get_pdf_path(doc_id)
    if not pdf.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(pdf)


@app.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Удаляет все файлы документа"""
    session_dir = get_session_dir(doc_id)
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return {"status": "deleted"}