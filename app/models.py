# app/models.py

from sqlalchemy import Column, String, LargeBinary
from .db import Base

class Employee(Base):
    __tablename__ = "employees"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    embedding = Column(LargeBinary, nullable=False)
    image_path = Column(String, nullable=True)