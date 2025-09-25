# app/main.py

import logging
import time
import numpy as np
import cv2
from fastapi import FastAPI, File, Form, UploadFile, Depends, HTTPException, Request, BackgroundTasks
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
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    logging.info("Loading embeddings into cache on startup...")
    async for db in get_db():
        names, embeddings, ids ,member_code= await crud.load_all_embeddings(db)
        embedding_cache.update(names, embeddings, ids, member_code)
        break
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
    member_code: str = Form(...),
    pictures: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db)
):
    if not all([name, id, pictures]):
        raise HTTPException(status_code=400, detail="Missing required parameters.")

    try:
        files_data = []
        for file in pictures:
            contents = await file.read()
            files_data.append((file.filename, contents))

        avg_embedding, rep_img_path = await run_in_threadpool(
            process_employee_images, employee_name=name, employee_id=id, files_data=files_data
        )

        if avg_embedding is None:
            return JSONResponse(
                status_code=200,
                content=make_response(0, 2, False, "Failed to generate embeddings. No faces found or invalid images.")
            )

        existing_employee = await crud.get_employee_by_id(db, id)
        
        message = ""
        if existing_employee:
            await crud.update_employee(
                db, emp_id=id, name=name, member_code=member_code, 
                embedding=avg_embedding, image_path=rep_img_path
            )
            message = f"Employee {name} (ID: {id}) was successfully updated."
        else:
            await crud.create_employee(
                db, emp_id=id, name=name, member_code=member_code, 
                embedding=avg_embedding, image_path=rep_img_path
            )
            message = f"{name} is stored successfully."
        
        embedding_cache.update_or_add_employee(id, name, member_code, avg_embedding)
        
        return JSONResponse(
            status_code=200,
            content=make_response(1, 1, True, message)
        )

    except Exception as e:
        logging.error(f"Error during upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save images to database.")

@app.post("/recognize", response_model=schemas.RecognitionResponse)
async def recognize(background_tasks: BackgroundTasks, # Add this
    file: UploadFile = File(...), 
    db: AsyncSession = Depends(get_db)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            logging.error("cv2.imdecode failed, image is None.")
            return {"faces": []}

        cache_data = embedding_cache.get_all()
        if embedding_cache.is_empty():
            logging.warning("Recognition attempted but embedding cache is empty.")
            return {"faces": []}
            
        recognized_faces = await run_in_threadpool(
            detect_and_recognize_faces, image_bgr=image_bgr, cache_data=cache_data
        )

        if recognized_faces:
            best_face = max(
                (f for f in recognized_faces if f.get("name") != "Unknown"), 
                key=lambda f: f.get("score", 0), 
                default=None
            )
            if best_face:
                try:
                    names, _, ids, member_codes = embedding_cache.get_all()
                    idx = names.index(best_face["name"])
                    emp_id_to_log = ids[idx]
                    member_code_to_log = member_codes[idx]
                    
                    background_tasks.add_task(
                        crud.create_recognition_log,
                        db=db,
                        emp_id=emp_id_to_log,
                        name=best_face["name"],
                        member_code=member_code_to_log
                    )
                except Exception as log_error:
                    logging.error(f"Failed to save recognition log: {log_error}")
        
        return {"faces": recognized_faces}
    except Exception as e:
        logging.exception("Error processing recognition request: %s", e)
        return {"faces": []}

@app.delete("/employees/{employee_id}", response_model=schemas.StandardResponse)
async def delete_employee(employee_id: str, db: AsyncSession = Depends(get_db)):
    """
    Deletes an employee record from the database and the live cache.
    """
    deleted_employee = await crud.delete_employee_by_id(db, employee_id)

    if not deleted_employee:
        raise HTTPException(status_code=404, detail=f"Employee with ID '{employee_id}' not found.")

    embedding_cache.remove_employee(employee_id)

    message = f"Successfully deleted employee {deleted_employee.name} (ID: {employee_id})."
    return JSONResponse(
        status_code=200,
        content=make_response(1, 1, True, message)
    )

@app.get("/employees", response_model=schemas.EmployeeListResponse)
async def list_employees(db: AsyncSession = Depends(get_db)):
    """
    Returns a list of all registered employees with their ID, name, and member code.
    """
    employees_from_db = await crud.get_all_employees(db)
    return {"employees": employees_from_db}