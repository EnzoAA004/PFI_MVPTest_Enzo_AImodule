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


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
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
    return _level_at(levels, normal, position)


def axial_slice_levels(
    sagittal_quality: Dict[str, Any] | None,
    axial_quality: Dict[str, Any] | None,
) -> Dict[int, str]:
    """Nivel de **cada** corte axial, por indice.

    `axial_slice_level` responde por un corte: el que eligio el modelo. Eso alcanza para
    ponerle nivel a las mediciones de esa corrida, pero no para el visor, donde el medico
    recorre la serie entera y marca sobre el corte que quiere.

    Y no se puede propagar el nivel de un corte a los demas. Una serie axial lumbar se
    adquiere en bloques angulados, uno por disco -en el estudio de referencia, cortes 1-5
    a 3,5 grados, 6-10 a 5,9 y 11-15 a 23-, asi que cada bloque mira un disco distinto.
    Darle a los quince cortes el nivel del que analizo el modelo pondria dos tercios de
    las marcas bajo un nivel que no es. Es el mismo error que la linea de referencia ya
    corrigio del lado del visor cuando dejo de usar la direccion global del volumen.

    Por eso se resuelve corte por corte, con la posicion y la orientacion propias de cada
    uno, que la corrida ya publica en `volumeGeometry`. Los cortes que no caen en ningun
    espacio discal simplemente no aparecen en el resultado.
    """
    if not isinstance(sagittal_quality, dict) or not isinstance(axial_quality, dict):
        return {}
    levels = sagittal_quality.get("discLevels")
    geometry = axial_quality.get("volumeGeometry")
    if not isinstance(levels, list) or not levels or not isinstance(geometry, dict):
        return {}
    positions = geometry.get("slicePositions")
    orientations = geometry.get("sliceOrientations")
    if not isinstance(positions, list) or not positions:
        return {}

    # Sin orientacion por corte se usa la normal del plano analizado: es menos exacto,
    # pero la posicion propia de cada corte ya corrige la mayor parte del error.
    fallback_normal = None
    plane = geometry.get("slicePlane")
    if isinstance(plane, dict) and _is_point(plane.get("normal")):
        fallback_normal = plane["normal"]

    result: Dict[int, str] = {}
    for index, position in enumerate(positions):
        if not _is_point(position):
            continue
        normal = fallback_normal
        if isinstance(orientations, list) and index < len(orientations):
            item = orientations[index]
            # DICOM 0020|0037 son seis numeros: el vector de la fila y el de la columna.
            # La normal del corte es su producto vectorial.
            if isinstance(item, (list, tuple)) and len(item) == 6 and all(isinstance(v, (int, float)) for v in item):
                normal = _cross(item[0:3], item[3:6])
        if not _is_point(normal):
            continue
        level = _level_at(levels, normal, position)
        if level:
            result[index] = level
    return result


def _level_at(levels, normal, position) -> str | None:
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
