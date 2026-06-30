# E13 - Patch requerido: deteccion dinamica de eje sagital

Estado: requiere ajuste menor.

## Resultado del patch axis=2

El patch axis=2 corrigio 5 de 6 ejemplos sagitales. Los primeros cinco casos se ven anatomica y visualmente correctos:

- 101_t1: axis2_slice 8, shape (298, 320, 17)
- 116_t1: axis2_slice 6, shape (320, 320, 15)
- 117_t1: axis2_slice 12, shape (448, 448, 24)
- 12_t1: axis2_slice 8, shape (320, 320, 15)
- 131_t1: axis2_slice 12, shape (463, 448, 24)

El caso 136_t1 quedo mal porque su shape es (17, 512, 512). Para este tipo de volumen el eje de stack sagital no es axis=2 sino probablemente axis=0. Extraer axis=2 genera una tira 17x512 que luego se deforma al resize.

## Decision

E13 no debe asumir axis=2 fijo para todos los casos SPIDER. Debe elegir dinamicamente el eje de stack sagital por caso. Regla propuesta:

- Probar los tres ejes.
- Descartar extracciones cuyo corte 2D tenga una dimension menor a 128 px.
- Priorizar el eje con menor cantidad de slices, porque en SPIDER las series sagitales suelen tener 15-24 cortes.
- Elegir el slice con mayor foreground agrupado dentro de ese eje.

## Estado del pipeline

- Axial T2: valido.
- Sagital: valido con patch para casos estandar, pero requiere deteccion dinamica de eje para casos como 136_t1.

## Proximo paso

Aplicar E13 patch v3: inferencia sagital con eje dinamico por caso y regenerar ejemplos sagitales.
