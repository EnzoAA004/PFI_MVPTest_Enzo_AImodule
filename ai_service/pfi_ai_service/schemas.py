from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QualitySummary:
    foreground_ratio: float
    n_components: int
    present_classes: List[int] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    mean_confidence: Optional[float] = None
    mean_fg_confidence: Optional[float] = None


@dataclass
class InferenceItem:
    agent_item_id: str
    plane: str
    model_key: str
    case_ref: str
    figure_path: Optional[str] = None
    quality: Optional[QualitySummary] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDecision:
    agent_item_id: str
    agent_status: str
    review_priority: str
    agent_reasons: List[str]
    recommended_action: str
    human_review_required: bool = True
