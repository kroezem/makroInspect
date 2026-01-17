import json
from datetime import datetime
from typing import Optional, List
from sqlmodel import Field, SQLModel, Session, text, Relationship


# --- Schema Definitions ---

class ConfigState(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    station_id: str = Field(index=True)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    segmented_at: Optional[datetime] = Field(default=None, index=True)

    # Technical fields last
    extension: str
    hash: str = Field(index=True, unique=True)

    instances: List["Instance"] = Relationship(back_populates="image")


class Instance(SQLModel, table=True):
    # Primary Identifiers
    id: str = Field(primary_key=True)  # Format: 000000012_01
    label: str = Field(default="unlabeled", index=True)
    station_id: str = Field(index=True)

    # Foreign Key
    image_id: int = Field(foreign_key="image.id", index=True)
    image: Optional[Image] = Relationship(back_populates="instances")

    # Metrics
    score: Optional[float] = Field(default=None, index=True)
    crop_iou: Optional[float] = Field(default=None)

    # State Flags
    in_bank: bool = Field(default=False, index=True)

    # Data Payload
    obb_json: str = Field(default="{}")

    # Timestamps
    cropped_at: Optional[datetime] = Field(default=None)
    evaluated_at: Optional[datetime] = Field(default=None)

    @property
    def obb(self) -> dict:
        return json.loads(self.obb_json)


# --- Initialization Helper ---

def init_db(engine):
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.exec(text("PRAGMA journal_mode=WAL;"))
        session.exec(text("PRAGMA synchronous=NORMAL;"))
        session.commit()