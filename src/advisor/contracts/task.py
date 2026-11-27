from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    ticker: str | None = Field(default=None, pattern=r"^[0-9]{6}$")
    ticker_source: str | None = Field(default=None, pattern=r"^(explicit|code|name|model|loose)$")
    as_of: date | None = None
    route: Literal["auto", "quick", "standard", "deep"] = "auto"

    @field_validator("as_of")
    @classmethod
    def _no_future_as_of(cls, v: date | None) -> date | None:
        # 未来时点会写出 current_dossier 永远不可见的档案（回填研究只能是过去）
        if v is not None and v > date.today():
            raise ValueError(f"研究时点 {v} 晚于今天，回填研究只能使用过去的日期")
        return v
