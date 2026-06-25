# Validación

## Validación técnica

La validación técnica se realizará comparando máscaras predichas contra máscaras de referencia del dataset.

Métricas iniciales:

- Dice por clase.
- IoU por clase.
- Dice promedio.

Métricas futuras:

- HD95.
- Normalized Surface Dice.
- Métricas por paciente, estructura y secuencia.

## Validación cualitativa

Se prevé una revisión cualitativa complementaria por un profesional del área. La revisión debe enfocarse en:

- plausibilidad anatómica de las segmentaciones;
- claridad de visualización;
- utilidad de mediciones;
- legibilidad de salida estructurada;
- confianza/desconfianza sobre resultados automáticos.

## Criterio importante

Un buen valor de Dice no implica utilidad clínica automática. Las métricas técnicas deben complementarse con revisión visual y profesional.
