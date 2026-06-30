from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json


def write_json(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_case_report(path: Path, row: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    md = f"""# Reporte de caso IA - {row.get('agent_item_id')}

## Identificación

- Plano: {row.get('plane')}
- Caso: {row.get('case_ref')}
- Modelo: {row.get('model_key')}

## Calidad

- Foreground ratio: {row.get('foreground_ratio')}
- Componentes: {row.get('n_components')}
- Confianza media: {row.get('mean_confidence')}
- Confianza foreground: {row.get('mean_fg_confidence')}
- Flags: {row.get('flags')}

## Decisión del agente

- Estado: {row.get('agent_status')}
- Prioridad: {row.get('review_priority')}
- Razones: {row.get('agent_reasons')}
- Acción recomendada: {row.get('recommended_action')}

## Política

Este resultado es asistido por IA y requiere revisión profesional.
"""
    path.write_text(md, encoding="utf-8")
    return path


def build_markdown_summary(summary: Dict[str, Any]) -> str:
    return f"""# Resumen del agente IA

Total de ítems: {summary.get('total_items')}

Distribución por plano: {summary.get('plane_distribution')}

Distribución por prioridad: {summary.get('priority_distribution')}

Distribución por estado: {summary.get('status_distribution')}

Confianza foreground media: {summary.get('mean_fg_confidence')}

Dice útil medio: {summary.get('mean_dice_macro_useful_classes')}

El agente funciona como apoyo y no reemplaza la revisión profesional.
"""
