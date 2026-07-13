from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for public contracts: unknown fields are configuration errors."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
