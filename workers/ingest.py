import hashlib
import shutil
import time
from pathlib import Path
from sqlmodel import select

from core.database import Image
from core.project import Project
from core.telemetry import IngestStats


class IngestWorker:
    """
    Ingest worker - scans inbox and imports new images.

    Returns IngestStats, no print statements.
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def process(self, project: Project) -> IngestStats:
        """
        Process a single project's inbox.
        Returns IngestStats with results.
        """
        start_time = time.perf_counter()
        inbox = project.inbox_path
        raw_dir = project.root / "artifacts" / "raw"

        stats = IngestStats(project_name=project.root.name)

        files = self._scan_inbox(inbox)
        if not files:
            return stats

        ready_files = self._filter_locked(files)
        if not ready_files:
            return stats

        stats.is_sprint = len(ready_files) <= 8
        priority_val = 10 if stats.is_sprint else 5

        for src_path in ready_files:
            result = self._ingest_file(project, src_path, raw_dir, priority_val)
            if result == "success":
                stats.items_processed += 1
            elif result == "duplicate":
                stats.duplicates += 1
            elif result == "error":
                stats.errors += 1

        stats.duration_ms = (time.perf_counter() - start_time) * 1000
        return stats

    def _scan_inbox(self, inbox: Path) -> list:
        """Recursively scan inbox for valid image files."""
        files = []
        for p in inbox.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.VALID_EXTENSIONS and not p.name.startswith('.'):
                files.append(p)
        return files

    def _filter_locked(self, files: list) -> list:
        """Filter out files that are still being written."""
        ready = []
        for p in files:
            try:
                p.rename(p)
                ready.append(p)
            except OSError:
                continue
        return ready

    def _calculate_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _ingest_file(self, project: Project, src_path: Path, raw_dir: Path, priority: int) -> str:
        """
        Ingest a single file.
        Returns: "success", "duplicate", or "error"
        """
        try:
            file_hash = self._calculate_hash(src_path)
        except Exception:
            return "error"

        with project.get_session() as session:
            stmt = select(Image).where(Image.hash == file_hash)
            existing = session.execute(stmt).scalars().first()

            if existing:
                try:
                    src_path.unlink()
                except:
                    pass
                return "duplicate"

            station_id = src_path.parent.name if src_path.parent != project.inbox_path else "unknown"
            extension = src_path.suffix.lower().lstrip(".")

            img = Image(
                hash=file_hash,
                extension=extension,
                station_id=station_id,
                priority=priority
            )
            session.add(img)
            session.flush()

            safe_name = f"{img.id:09d}.{extension}"
            dst_path = raw_dir / safe_name

            try:
                shutil.move(str(src_path), str(dst_path))

                if station_id != "unknown":
                    try:
                        src_path.parent.rmdir()
                    except:
                        pass

                session.commit()
                return "success"

            except Exception:
                session.rollback()
                return "error"
