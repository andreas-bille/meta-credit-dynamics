from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json
import uuid
from pathlib import Path


@dataclass
class TelemetryEvent:
    t: int
    event_type: str
    subject_id: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    corr_id: Optional[str] = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "event_id": self.event_id,
            "t": int(self.t),
            "event_type": self.event_type,
            "subject_id": self.subject_id,
            "attrs": dict(self.attrs),
        }
        if self.corr_id is not None:
            d["corr_id"] = self.corr_id
        return d


@dataclass
class TelemetryLogger:
    """Minimal structured telemetry logger (JSONL) for Phase D.

    Designed to be optional and lightweight; can be disabled by passing path=None.
    """
    path: Optional[Path] = None
    _fh: Any = field(init=False, default=None)

    def __post_init__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w", encoding="utf-8")

    def log(self, t: int, event_type: str, subject_id: str, **attrs):
        if self._fh is None:
            return
        ev = TelemetryEvent(t=t, event_type=event_type, subject_id=str(subject_id), attrs=attrs)
        self._fh.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
