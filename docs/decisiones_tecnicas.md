# Decisiones técnicas

## 2026-06-25 — Separación entre desarrollo y ejecución

Se decide usar GitHub como fuente principal del código y Google Colab como entorno de ejecución/experimentación.

Motivo:

- Evitar que el notebook se vuelva monolítico.
- Permitir trabajo local con Codex/PyCharm.
- Mantener historial de cambios.
- Ejecutar en Colab usando datos en Drive.

## 2026-06-25 — Datos fuera del repositorio

Se decide no versionar datasets, checkpoints ni outputs pesados.

Motivo:

- Evitar repositorios pesados.
- Mantener separación entre código y datos.
- Facilitar uso de Drive para SPIDER y resultados.

## 2026-06-25 — U-Net 2D como baseline inicial

Se agrega una U-Net 2D compacta como modelo inicial.

Motivo:

- Arquitectura entendible.
- Adecuada para un primer MVP.
- Ejecutable en Colab.
- Útil como baseline antes de evaluar alternativas más complejas.

## 2026-06-25 — Mediciones geométricas no diagnósticas

Las mediciones se documentan como valores geométricos derivados de máscaras.

Motivo:

- Mantener el alcance académico.
- Evitar inferencias clínicas no validadas.
- Permitir revisión profesional.
