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

async def create_employee(db: AsyncSession, emp_id: str, name: str, embedding: np.ndarray, image_path: str):
    """Create a new employee record in the database."""
    db_employee = models.Employee(
        id=emp_id,
        name=name,
        embedding=embedding.tobytes(),
        image_path=image_path
    )
    db.add(db_employee)
    await db.commit()
    await db.refresh(db_employee)
    return db_employee

async def load_all_embeddings(db: AsyncSession) -> Tuple[List[str], np.ndarray, List[str]]:
    """Load all employee names, IDs, and embeddings from the database."""
    result = await db.execute(select(models.Employee.name, models.Employee.embedding, models.Employee.id))
    
    names, embeddings, ids = [], [], []
    for name, emb_bytes, emp_id in result.all():
        if len(emb_bytes) % 4 == 0:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            if emb.shape[0] == 512:
                embeddings.append(emb)
                names.append(name)
                ids.append(emp_id)

    return names, np.array(embeddings), ids