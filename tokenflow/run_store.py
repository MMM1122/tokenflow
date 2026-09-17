"""Local journal: lock the run and durably record intent before each paid attempt."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_text(path: Path, text: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=True, indent=2) + "\n")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError(f"Missing or invalid run artifact: {path.name}") from None


@contextmanager
def run_lock(directory: Path):
    """OS locks release on process exit; never delete an inode another writer may hold."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                stream.write(b"0")
                stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Another process is using this evaluation directory") from None
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RunStore:
    def __init__(self, directory: Path):
        self.directory = directory / "attempts"
        self.directory.mkdir(exist_ok=True)

    def path(self, case_id: str, arm: str, suffix: str) -> Path:
        return self.directory / f"{digest([case_id, arm])}.{suffix}.json"

    def begin(self, case_id: str, arm: str, messages):
        atomic_json(
            self.path(case_id, arm, "started"),
            {
                "id": case_id,
                "arm": arm,
                "messages_sha256": digest(messages),
            },
        )

    def finish(self, row: dict):
        atomic_json(
            self.path(row["id"], row["arm"], "completed"),
            {
                "record": row,
                "sha256": digest(row),
            },
        )

    def validate(self, planned: list[dict]) -> dict[tuple[str, str], dict]:
        """Validate all recorded attempts before the runner can issue a new call."""
        rows, expected_files = {}, set()
        for case in planned:
            for arm in case["arms"]:
                key = (case["id"], arm)
                started = self.path(*key, "started")
                completed = self.path(*key, "completed")
                expected_files.update((started.name, completed.name))
                if started.exists():
                    marker = read_json(started)
                    expected = {
                        "id": case["id"],
                        "arm": arm,
                        "messages_sha256": digest(case["messages"][arm]),
                    }
                    if marker != expected:
                        raise ValueError("Journal attempt does not match the saved plan")
                if completed.exists():
                    envelope = read_json(completed)
                    if not isinstance(envelope, dict) or envelope.get("sha256") != digest(
                        envelope.get("record")
                    ):
                        raise ValueError("Journal result checksum is invalid")
                    row = envelope["record"]
                    if not isinstance(row, dict) or (row.get("id"), row.get("arm")) != key:
                        raise ValueError("Journal result identity is invalid")
                    if row.get("status") not in {
                        "ok",
                        "provider_error",
                        "uncertain",
                        "budget_exceeded",
                    }:
                        raise ValueError("Journal result status is invalid")
                    if row["status"] != "budget_exceeded" and not started.exists():
                        raise ValueError("Journal result has no recorded attempt")
                    if row.get("family") != case["family"]:
                        raise ValueError("Journal result family is invalid")
                    rows[key] = row
        unexpected = {path.name for path in self.directory.glob("*.json")} - expected_files
        if unexpected:
            raise ValueError("Journal contains attempts outside the saved plan")
        return rows
