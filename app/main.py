# app/main.py

import logging
import time
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from . import crud, models, schemas
from .db import get_db, engine
from .cache import embedding_cache
from .config import settings
from .ai_processing import (
    detect_and_recognize_faces,
    process_employee_images
)

# --- App Initialization ---
logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Face Recognition API")
templates = Jinja2Templates(directory="templates")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    # This is a good place to create DB tables if they don't exist
    async with engine.begin() as conn:
        # await conn.run_sync(models.Base.metadata.drop_all) # Use for dropping
        await conn.run_sync(models.Base.metadata.create_all)
    
    # Pre-load embeddings into the cache
    logging.info("Loading embeddings into cache on startup...")
    async for db in get_db():
        names, embeddings, ids = await crud.load_all_embeddings(db)
        embedding_cache.update(names, embeddings, ids)
        break # Since get_db is a generator, we just need one iteration
    logging.info("Startup complete.")

# --- Helper for API Responses ---
def make_response(status, code, flag, message, data=None):
    return {
        "STATUS": status, "CODE": code, "FLAG": flag,
        "MESSAGE": message, "DATA": data
    }

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/hi")
def read_hi():
    return "TechV1z0r !"

@app.post("/upload", response_model=schemas.StandardResponse)
async def upload_images(
    name: str = Form(...),
    id: str = Form(...),
    pictures: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not all([name, id, pictures]):
        raise HTTPException(status_code=400, detail="Missing required parameters.")

    existing_employee = await crud.get_employee_by_id(db, id)
    if existing_employee:
        return JSONResponse(
            status_code=200,
            content=make_response(0, 2, False, f"Employee with ID {id} already exists.")
        )

    try:
        # Step 1: Read all file contents asynchronously in the main thread
        files_data = []
        for file in pictures:
            contents = await file.read()
            files_data.append((file.filename, contents))

        # Step 2: Run the synchronous, CPU-bound processing in a thread pool
        avg_embedding, rep_img_path = await run_in_threadpool(
            process_employee_images, employee_name=name, employee_id=id, files_data=files_data
        )

        if avg_embedding is None:
            return JSONResponse(
                status_code=200,
                content=make_response(0, 2, False, "Failed to generate embeddings. No faces found or invalid images.")
            )

        # Step 3: DB operation is async and runs in the main thread
        await crud.create_employee(db, emp_id=id, name=name, embedding=avg_embedding, image_path=rep_img_path)

        # Step 4: Update the live cache
        embedding_cache.add_employee(name, avg_embedding, id)

        return JSONResponse(
            status_code=200,
            content=make_response(1, 1, True, f"{name} is stored successfully.")
        )
    except Exception as e:
        logging.error(f"Error during upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save images to database.")


@app.post("/recognize", response_model=schemas.RecognitionResponse)
async def recognize(file: UploadFile = File(...)):
    start_total = time.time()
    if not file:
        return {"faces": []}

    try:
        # Read image bytes
        contents = await file.read()
        
        # Decode image in memory
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            logging.error("cv2.imdecode failed, image is None.")
            return {"faces": []}

        # Get cache data
        cache_data = embedding_cache.get_all()
        if embedding_cache.is_empty():
            logging.warning("Recognition attempted but embedding cache is empty.")
            return {"faces": []}
            
        # Run the CPU-bound detection and recognition in a thread pool
        start_proc = time.time()
        recognized_faces = await run_in_threadpool(
            detect_and_recognize_faces, image_bgr=image_bgr, cache_data=cache_data
        )
        proc_time = time.time() - start_proc
        total_time = time.time() - start_total
        
        logging.info(f"Recognition result: {len(recognized_faces)} faces (proc={proc_time:.3f}s total={total_time:.3f}s)")
        
        return {"faces": recognized_faces}
    except Exception as e:
        logging.exception("Error processing recognition request: %s", e)
        return {"faces": []}