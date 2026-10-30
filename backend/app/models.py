"""SQLModel tables. JSON blobs are stored as TEXT for simplicity."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Clip(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    original_name: str
    path: str
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_count: int = 0
    duration: float = 0.0
    # uploaded | queued | processing | done | error
    status: str = "uploaded"
    progress: float = 0.0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    gt_in: Optional[int] = None
    gt_out: Optional[int] = None
    gt_note: Optional[str] = None

    slot_a: Optional[str] = None  # weights filename used for slot A
    slot_b: Optional[str] = None
    roi_json: Optional[str] = None  # current ROI config for this clip
    preset_id: Optional[int] = None
    summary_json: Optional[str] = None  # per-slot counts for the list view
    render_version: int = 0
    render_status: str = "none"  # none | rendering | done | error

    def roi(self) -> Optional[dict[str, Any]]:
        return json.loads(self.roi_json) if self.roi_json else None

    def summary(self) -> Optional[dict[str, Any]]:
        return json.loads(self.summary_json) if self.summary_json else None


class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    clip_ids_json: str
    slot_a: str
    slot_b: str
    preset_id: Optional[int] = None
    settings_json: str = "{}"
    # queued | running | done | error
    status: str = "queued"
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    def clip_ids(self) -> list[int]:
        return json.loads(self.clip_ids_json)


class Setting(SQLModel, table=True):
    """Key/value app settings (e.g. the active ROI applied to every clip)."""

    key: str = Field(primary_key=True)
    value: str


class RoiPreset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    config_json: str
    created_at: datetime = Field(default_factory=_now)

    def config(self) -> dict[str, Any]:
        return json.loads(self.config_json)
