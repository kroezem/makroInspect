import json
from datetime import datetime
from typing import Optional, List
from sqlmodel import Field, SQLModel, text, Relationship


# --- Schema Definitions ---

class ConfigState(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Image(SQLModel, table=True):
    # Primary Identifiers
    id: Optional[int] = Field(default=None, primary_key=True)  # The source of truth
    station_id: str = Field(index=True)

    # Technical Metadata (Critical for rebuilding filename)
    extension: str
    hash: str = Field(index=True, unique=True)

    # Scheduling & Logic
    priority: int = Field(default=5, index=True)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    segmented_at: Optional[datetime] = Field(default=None, index=True)

    # Relationships
    instances: List["Instance"] = Relationship(back_populates="image")


class Instance(SQLModel, table=True):
    id: str = Field(primary_key=True)
    station_id: str = Field(index=True)

    # Labels: "unlabeled" (default), "normal" (used for bank), "anomaly", "exclude" (ignored)
    label: str = Field(default="unlabeled", index=True)
    score: Optional[float] = Field(default=None, index=True)

    priority: int = Field(default=5, index=True)
    # bank_index >= 0: in bank at that slot, -1: considered but not selected, None: not yet analyzed
    bank_index: Optional[int] = Field(default=None, index=True)
    # bank_id tracks which bank version was used for scoring
    bank_id: Optional[str] = Field(default=None, index=True)

    image_id: int = Field(foreign_key="image.id", index=True)
    image: Optional[Image] = Relationship(back_populates="instances")

    crop_iou: Optional[float] = Field(default=None)
    obb_json: str = Field(default="{}")

    cropped_at: Optional[datetime] = Field(default=None)
    evaluated_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def obb(self) -> dict:
        return json.loads(self.obb_json)


def init_db(engine):
    SQLModel.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA synchronous=NORMAL;"))
        conn.execute(text("PRAGMA cache_size=-64000;"))
        conn.commit()