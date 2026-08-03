"""Angulo segmentario y listesis, derivados de los ejes de dos cuerpos vecinos.

No salen de una mascara propia sino de la geometria de otras dos mediciones, y por eso
viajan marcadas como experimentales. Estas pruebas fijan lo que la derivacion tiene que
cumplir para que el numero signifique algo, y sobre todo el caso en que no significa
nada: dos cuerpos paralelos no tienen angulo, y dos alineados no tienen listesis.
"""

import numpy as np
import pytest

from pfi_ai_service.real_inference_runtime import (
    AxisMeasure,
    segmental_angle,
    vertebral_listhesis,
)


def endplate(x0: float, y0: float, x1: float, y1: float) -> AxisMeasure:
    """Segmento de ancho de un cuerpo, que es la direccion de sus platillos."""
    return AxisMeasure(float(np.hypot(x1 - x0, y1 - y0)), (x0, y0), (x1, y1))


def test_dos_cuerpos_paralelos_no_forman_angulo():
    degrees, points = segmental_angle(endplate(10, 20, 50, 20), endplate(10, 60, 50, 60))
    assert degrees == pytest.approx(0)
    # Se dibujan los dos segmentos que produjeron el numero, no una figura aparte.
    assert len(points) == 4


def test_el_angulo_es_el_que_separa_los_dos_platillos():
    degrees, _ = segmental_angle(endplate(10, 20, 50, 20), endplate(10, 60, 50, 100))
    assert degrees == pytest.approx(45)


def test_el_angulo_no_depende_del_orden_de_los_extremos():
    """Una recta no tiene sentido: el mismo platillo no puede dar 8 grados o 172."""
    directo, _ = segmental_angle(endplate(10, 20, 50, 20), endplate(10, 60, 50, 80))
    invertido, _ = segmental_angle(endplate(50, 20, 10, 20), endplate(50, 80, 10, 60))
    assert directo == pytest.approx(invertido)


def test_dos_cuerpos_alineados_no_tienen_listesis():
    slip, grade, points = vertebral_listhesis(endplate(10, 20, 50, 20), endplate(10, 60, 50, 60), 1.0, 1.0)
    assert slip == pytest.approx(0)
    assert grade == "grado I"
    # Tres puntos: el platillo inferior y la esquina posterior de la que se deslizo.
    assert len(points) == 3


def test_la_listesis_mide_el_corrimiento_sobre_el_platillo():
    # Cuerpo superior corrido 10 mm hacia atras sobre un platillo de 40 mm.
    slip, grade, _ = vertebral_listhesis(endplate(20, 20, 60, 20), endplate(10, 60, 50, 60), 1.0, 1.0)
    assert slip == pytest.approx(10)
    assert grade == "grado I"


def test_el_grado_sale_de_la_proporcion_y_no_del_milimetraje():
    """El mismo corrimiento sobre un cuerpo mas chico es una listesis mas grave."""
    # 10 mm sobre un cuerpo de 40: un cuarto, grado I.
    _, sobre_cuerpo_grande, _ = vertebral_listhesis(endplate(20, 20, 60, 20), endplate(10, 60, 50, 60), 1.0, 1.0)
    # Los mismos 10 mm sobre un cuerpo de 20: la mitad, grado II.
    _, sobre_cuerpo_chico, _ = vertebral_listhesis(endplate(20, 20, 40, 20), endplate(10, 60, 30, 60), 1.0, 1.0)
    assert sobre_cuerpo_grande == "grado I"
    assert sobre_cuerpo_chico == "grado II"


def test_el_corrimiento_se_mide_con_el_spacing_de_cada_eje():
    slip, _, _ = vertebral_listhesis(endplate(20, 20, 60, 20), endplate(10, 60, 50, 60), 0.5, 1.0)
    assert slip == pytest.approx(5)


def test_un_cuerpo_sin_platillo_medible_no_produce_listesis():
    assert vertebral_listhesis(endplate(10, 20, 50, 20), endplate(10, 60, 10, 60), 1.0, 1.0) is None
