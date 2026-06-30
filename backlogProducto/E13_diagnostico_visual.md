# E13 - Diagnostico visual y correccion requerida

Estado: requiere ajuste antes de cierre definitivo.

## Resultado axial

El pipeline axial T2 funciono correctamente. Los overlays son coherentes, las mascaras aparecen alineadas y las metricas por clase son consistentes con E10/E11.

Quality axial observada:
- 6 ejemplos generados.
- mean_confidence aproximada: 0.989.
- mean_fg_confidence aproximada: 0.94 a 0.97.
- Un ejemplo marco muchos_componentes, pero visualmente el resultado es aceptable para prototipo.

## Resultado sagital

Los ejemplos sagitales aparecen visualmente raros o estirados. El problema probable no es el checkpoint final E12, sino la logica de seleccion/visualizacion del corte sagital dentro de E13.

Evidencia:
- SPIDER usa arrays con forma z, y, x.
- El modelo sagital trabaja con sagittal_axis = 2.
- Algunos ejemplos reportaron indices de slice como 149, 157 o 221, imposibles para axis=2 en casos con x cercano a 15, 17 o 24.

## Decision

Antes de cerrar E13 hay que corregir la celda sagital para que:
- seleccione slices sobre axis=2,
- use indices entre 0 y shape[2]-1,
- preserve orientacion/aspecto visual,
- genere nuevamente overlays sagitales.

## Proximo paso

Aplicar patch E13 v2 para inferencia sagital corregida. Luego regenerar reportes y figuras.
