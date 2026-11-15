"""
Audit Logger — Structured JSON audit trail with Trace IDs.

Uses orjson for ultra-fast serialization.
Appends to daily files: logs/audit/YYYY-MM-DD.jsonl
Immutable — append-only, no overwrites.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import orjson

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class AuditLogger:
    """
    Immutable structured audit trail logger.

    Every event includes:
    - trace_id: Unique ID for tracing related events
    - timestamp: IST timestamp
    - event_type: Category of event
    - payload: Event-specific data
    - source: Module that generated the event
    """

    def __init__(self, log_dir: str = "logs/audit") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: Optional[str] = None
        self._current_file: Optional[Path] = None
        self._write_lock = threading.Lock()

    def log_event(
        self,
        event_type: str,
        trace_id: str = "",
        payload: Optional[dict[str, Any]] = None,
        source: str = "",
    ) -> None:
        """
        Log a structured audit event.

        Args:
            event_type: Category (ORDER_PLACED, RISK_OVERRIDE, etc.)
            trace_id: Unique trace ID for event correlation
            payload: Event-specific data dict
            source: Module name generating the event
        """
        if not trace_id:
            trace_id = str(uuid.uuid4())

        now = datetime.now(IST)
        event = {
            "trace_id": trace_id,
            "timestamp": now.isoformat(),
            "event_type": event_type,
            "source": source,
            "payload": payload or {},
        }

        # Serialize with orjson (fast, handles datetime natively)
        line = orjson.dumps(event) + b"\n"

        # Write to daily JSONL file
        file_path = self._get_file_path(now)
        try:
            with self._write_lock, open(file_path, "ab") as f:
                f.write(line)
        except Exception as exc:
            logger.error("Failed to write audit event: %s", exc)

    def _get_file_path(self, now: datetime) -> Path:
        """Get the log file path for the current date."""
        date_str = now.strftime("%Y-%m-%d")
        if date_str != self._current_date:
            self._current_date = date_str
            self._current_file = self._log_dir / f"{date_str}.jsonl"
        assert self._current_file is not None
        return self._current_file

    def read_events(
        self, date_str: Optional[str] = None,
        event_type: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Read audit events with optional filtering.

        Args:
            date_str: Date string YYYY-MM-DD (default: today)
            event_type: Filter by event type
            trace_id: Filter by trace ID
        """
        if date_str is None:
            date_str = datetime.now(IST).strftime("%Y-%m-%d")

        file_path = self._log_dir / f"{date_str}.jsonl"
        if not file_path.exists():
            return []

        events: list[dict[str, Any]] = []
        try:
            with open(file_path, "rb") as f:
                for line in f:
                    event = orjson.loads(line)
                    if event_type and event.get("event_type") != event_type:
                        continue
                    if trace_id and event.get("trace_id") != trace_id:
                        continue
                    events.append(event)
        except Exception as exc:
            logger.error("Failed to read audit log: %s", exc)

        return events
