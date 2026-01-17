import hashlib
import shutil
import time
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from sqlmodel import select

from core.project import Project
from core.database import Image
from workers.base import BaseWorker

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class IngestWorker(BaseWorker):
    def load_model(self):
        pass

    def process(self, project: Project) -> int:
        inbox_dir = project.root / "inbox"
        if not inbox_dir.exists():
            return 0

        candidates = self._scan_inbox(inbox_dir)
        if not candidates:
            return 0

        count = 0

        # Process individually to isolate failures and manage transactions safely
        for station_id, file_path in candidates:
            with project.get_session() as session:
                try:
                    # 1. Deduplicate
                    file_hash = self._sha256(file_path)
                    existing = session.exec(select(Image).where(Image.hash == file_hash)).first()

                    if existing:
                        try:
                            file_path.unlink()
                        except OSError:
                            pass
                        continue

                    # 2. Stage DB Record
                    ext = self._normalize_ext(file_path.suffix)
                    img = Image(
                        hash=file_hash,
                        station_id=station_id,
                        extension=ext.lstrip("."),
                        ingested_at=datetime.utcnow()
                    )
                    session.add(img)

                    # Flush to generate ID, but do not commit yet
                    session.flush()
                    session.refresh(img)

                    # 3. Move File
                    dst_path = project.get_raw_path(img)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(dst_path))

                    # 4. Finalize
                    session.commit()
                    count += 1

                except Exception as e:
                    print(f"[Ingest] Failed {file_path.name}: {e}")
                    # Session automatically rolls back on exit if commit() wasn't reached
                    continue

        return count

    def _scan_inbox(self, inbox_dir: Path) -> List[Tuple[str, Path]]:
        items = []

        def is_valid(p: Path):
            return (p.is_file() and
                    not p.name.startswith(".") and
                    self._normalize_ext(p.suffix) in ALLOWED_EXT)

        # Root files -> 'root'
        for p in inbox_dir.iterdir():
            if is_valid(p) and self._is_stable(p):
                items.append(("root", p))

        # Subfolders -> 'station_id'
        for d in inbox_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                station = d.name
                for p in d.iterdir():
                    if is_valid(p) and self._is_stable(p):
                        items.append((station, p))

        items.sort(key=lambda x: x[1].stat().st_mtime)
        return items

    @staticmethod
    def _is_stable(p: Path, min_age: float = 0.75) -> bool:
        try:
            st = p.stat()
            age = time.time() - st.st_mtime
            return age > min_age and st.st_size > 0
        except FileNotFoundError:
            return False

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        ext = ext.lower()
        if ext == ".jpeg": return ".jpg"
        return ext