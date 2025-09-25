# app/crud.py

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Tuple, Optional

from . import models, schemas

async def get_employee_by_id(db: AsyncSession, emp_id: str) -> Optional[models.Employee]:
    """Fetch a single employee by their ID."""
    result = await db.execute(select(models.Employee).filter(models.Employee.id == emp_id))
    return result.scalars().first()

async def create_employee(
    db: AsyncSession, 
    emp_id: str, 
    name: str, 
    member_code: str, 
    embedding: np.ndarray, 
    image_path: str
):
    """Create a new employee record in the database."""
    db_employee = models.Employee(
        id=emp_id,
        name=name,
        member_code=member_code, # And also here
        embedding=embedding.tobytes(),
        image_path=image_path
    )
    db.add(db_employee)
    await db.commit()
    await db.refresh(db_employee)
    return db_employee

async def update_employee(
    db: AsyncSession,
    emp_id: str,
    name: str,
    member_code: Optional[str],
    embedding: np.ndarray,
    image_path: str
):
    """Fetches an employee by ID and updates their details."""
    db_employee = await get_employee_by_id(db, emp_id)
    if db_employee:
        db_employee.name = name
        db_employee.member_code = member_code
        db_employee.embedding = embedding.tobytes()
        db_employee.image_path = image_path
        await db.commit()
        await db.refresh(db_employee)
    return db_employee

async def delete_employee_by_id(db: AsyncSession, emp_id: str) -> Optional[models.Employee]:
    """Deletes an employee from the database by their ID."""
    db_employee = await get_employee_by_id(db, emp_id)
    if db_employee:
        await db.delete(db_employee)
        await db.commit()
    return db_employee

async def load_all_embeddings(db: AsyncSession) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    """Load all employee names, IDs, and embeddings from the database."""
    result = await db.execute(select(
        models.Employee.name, 
        models.Employee.embedding, 
        models.Employee.id, 
        models.Employee.member_code
    ))
    
    names, embeddings, ids, member_codes = [], [], [], []
    for name, emb_bytes, emp_id, member_code in result.all():
        if len(emb_bytes) % 4 == 0:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape[0] == 512:
                embeddings.append(emb)
                names.append(name)
                ids.append(emp_id)
                member_codes.append(member_code)

    return names, np.array(embeddings), ids, member_codes

async def get_all_employees(db: AsyncSession) -> List[models.Employee]:
    """Fetches all employee records from the database."""
    result = await db.execute(select(models.Employee))
    return result.scalars().all()

async def create_recognition_log(
    db: AsyncSession, 
    emp_id: str, 
    name: str, 
    member_code: str
):
    """Creates a new entry in the recognition_log table."""
    db_log = models.RecognitionLog(
        employee_id=emp_id,
        name=name,
        member_code=member_code
    )
    db.add(db_log)
    await db.commit()
    return db_log