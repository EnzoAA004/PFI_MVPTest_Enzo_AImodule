"""Que nivel lumbar esta mirando un corte axial.

Un corte axial de una RM lumbar se adquiere a la altura de un disco: tiene nivel, y el
informe se lee nivel por nivel. Pero la corrida axial sola no puede nombrarlo. El modelo
axial (Al-Kafri) segmenta disco, elemento posterior, saco tecal y region
anteroposterior, ninguno de los cuales es numerable: para decir "L4-L5" hay que contar
espacios discales desde la union lumbosacra, y en un corte axial se ve uno solo.

El resultado era que **todas** las mediciones axiales salian con `level=None` y caian en
"sin nivel asignado", que es el cajon de las medidas que la corrida no pudo ubicar. Se
leian como una falla del sistema cuando en realidad nunca se habia intentado ubicarlas.

El nivel esta en el sagital, que si ve la serie completa de discos. Este modulo cruza
los dos planos: toma la extension craneocaudal de cada disco que publico el sagital
-`discLevels`, en coordenadas del paciente- y pregunta cual de esos discos corta el
plano del corte axial.

**La pregunta se hace en coordenadas del paciente y no en indices de corte.** Una serie
axial lumbar se adquiere en bloques angulados, uno por disco -en el estudio de
referencia los bloques van a 3,5, 5,9 y 23 grados-, asi que "el corte 7" no esta a una
altura que se pueda deducir del sagital sin la geometria. Proyectar sobre la normal del
propio axial es lo unico que respeta esa angulacion.

Cuando el corte no cae dentro de ningun disco no se asigna nivel. Es deliberado: un
axial puede estar tomado a la altura de un cuerpo vertebral y no de un espacio discal, y
ahi la respuesta correcta es que no hay nivel discal que informar. Adjudicarle el mas
cercano pondria un numero de estenosis bajo un nivel que no es, que es peor que no
ponerlo.
"""
from __future__ import annotations

from typing import Any, Dict


def _dot(a, b) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _is_point(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(item, (int, float)) for item in value
    )


def axial_slice_level(sagittal_quality: Dict[str, Any] | None, axial_quality: Dict[str, Any] | None) -> str | None:
    """Nivel del corte axial que se infirio, o None si no cae en un espacio discal."""
    if not isinstance(sagittal_quality, dict) or not isinstance(axial_quality, dict):
        return None
    levels = sagittal_quality.get("discLevels")
    geometry = axial_quality.get("volumeGeometry")
    if not isinstance(levels, list) or not levels or not isinstance(geometry, dict):
        return None
    plane = geometry.get("slicePlane")
    if not isinstance(plane, dict):
        return None
    normal = plane.get("normal")
    position = plane.get("position")
    if not _is_point(normal) or not _is_point(position):
        return None

    # El plano del corte axial es {p : normal·p = offset}. Un disco lo cruza cuando sus
    # dos extremos caen de lados distintos, que es exactamente que `offset` este entre
    # las proyecciones de esos extremos.
    offset = _dot(normal, position)
    best_level: str | None = None
    best_distance = float("inf")
    for item in levels:
        if not isinstance(item, dict):
            continue
        level = item.get("level")
        hull = item.get("worldHull")
        if not isinstance(level, str) or not level or not isinstance(hull, list):
            continue
        projections = [_dot(normal, point) for point in hull if _is_point(point)]
        if not projections:
            continue
        # El disco es una componente conexa, asi que sus proyecciones sobre la normal
        # llenan un intervalo: el plano lo cruza exactamente cuando `offset` cae dentro.
        # Es la misma pregunta que mirar por donde pasa la linea de referencia sobre el
        # sagital, resuelta sobre el disco entero y no sobre una columna suya.
        low, high = min(projections), max(projections)
        if not low <= offset <= high:
            continue
        # Un plano angulado cruza mas de un disco, y hay que elegir.
        #
        # La linea de referencia entra al sagital inclinada, asi que a lo largo de la
        # profundidad del disco sube o baja varios milimetros y alcanza a tocar el borde
        # del disco vecino. Quedarse con el primero de la lista elegiria por orden
        # anatomico, que no es un criterio: se elige el que el corte atraviesa mas cerca
        # del medio, que es el disco sobre el que la serie fue planificada.
        #
        # La distancia se mide en unidades de la propia extension del disco, para que
        # uno mas alto no gane solo por ser mas grande.
        half = (high - low) / 2
        centre = (high + low) / 2
        distance = abs(offset - centre) / half if half > 0 else 0.0
        if distance < best_distance:
            best_level, best_distance = level, distance
    return best_level
