"""
Generador simple de backlog/Kanban para la tesis PFI.
No requiere librerías externas. Crea:
- backlog_kanban.csv
- kanban_board.md

Uso:
python generar_kanban_backlog.py
"""

import csv
from pathlib import Path

TASKS = [
    {"id":"TODO-01","area":"Axial","task":"E11 mapeo de clases axiales","status":"Por hacer","priority":"Crítica","type":"Análisis","acceptance":"Tabla valor raw → clase → nombre anatómico → uso"},
    {"id":"TODO-02","area":"Axial","task":"Decidir tratamiento de raw_0","status":"Por hacer","priority":"Crítica","type":"Análisis","acceptance":"Decisión documentada: excluir, fusionar, mantener o reponderar"},
    {"id":"TODO-03","area":"Sagital","task":"E12 entrenamiento sagital final limpio","status":"Por hacer","priority":"Alta","type":"Modelo","acceptance":"Checkpoint final + reportes comparables"},
    {"id":"TODO-04","area":"Multiplanar","task":"E13 pipeline común de inferencia","status":"Por hacer","priority":"Alta","type":"Pipeline","acceptance":"API común para ambos modelos"},
    {"id":"TODO-05","area":"Agente IA","task":"E14 orquestador IA en Colab","status":"Por hacer","priority":"Alta","type":"Agente","acceptance":"Prototipo de agente funcional en notebook"},
    {"id":"TODO-06","area":"3D","task":"E15 spike de reconstrucción 3D","status":"Por hacer","priority":"Media","type":"3D","acceptance":"Volumen/malla preliminar sobre un caso compatible"},
    {"id":"TODO-07","area":"MVP","task":"Integración backend Python service","status":"Por hacer","priority":"Alta","type":"Producto","acceptance":"Endpoint de inferencia para sagital y axial"},
    {"id":"TODO-08","area":"MVP","task":"UI de overlays y revisión profesional","status":"Por hacer","priority":"Media","type":"Producto","acceptance":"Flujo human-in-the-loop visible"},
    {"id":"BLK-01","area":"3D","task":"Caso con sagital + axial del mismo paciente","status":"Bloqueado","priority":"Crítica","type":"Datos","acceptance":"Conseguir o usar estudio DICOM con ambas series"},
    {"id":"BLK-02","area":"Clases","task":"Mapeo anatómico oficial de labels Al-Kafri","status":"Bloqueado","priority":"Alta","type":"Documentación","acceptance":"Nombre anatómico validado para cada clase"},
    {"id":"FUT-01","area":"Axial","task":"Probar entrenamiento axial excluyendo raw_0","status":"Backlog","priority":"Media","type":"Mejora","acceptance":"Comparativa E10 vs variante sin raw_0"},
    {"id":"FUT-02","area":"Axial","task":"Augmentations y pérdidas alternativas","status":"Backlog","priority":"Media","type":"Mejora","acceptance":"Evaluar impacto en test"},
]

STATUSES = ["Por hacer", "En progreso", "Bloqueado", "Backlog", "Hecho"]

def write_csv(path: Path) -> None:
    fieldnames = ["id", "area", "task", "status", "priority", "type", "acceptance"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(TASKS)

def write_markdown(path: Path) -> None:
    lines = ["# Kanban — IA multiplanar RM lumbar", ""]
    for status in STATUSES:
        lines.append(f"## {status}")
        lines.append("")
        items = [t for t in TASKS if t["status"] == status]
        if not items:
            lines.append("_Sin tareas._")
            lines.append("")
            continue
        for t in items:
            lines.append(f"- **{t['id']}** [{t['priority']}] {t['area']} — {t['task']}")
            lines.append(f"  - Tipo: {t['type']}")
            lines.append(f"  - Criterio de aceptación: {t['acceptance']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    write_csv(Path("backlog_kanban.csv"))
    write_markdown(Path("kanban_board.md"))
    print("Archivos generados: backlog_kanban.csv, kanban_board.md")
