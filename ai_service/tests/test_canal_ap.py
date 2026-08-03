"""Diametro anteroposterior del canal a la altura de cada disco.

Es la medicion con la que se describe una estenosis. Lo que estas pruebas fijan no es
tanto el calculo -el ancho de la mascara en las filas del disco- como los dos casos en
que no hay que publicarlo.

El primero es el que aparecio sobre el estudio real: la mascara del canal terminaba en
la ultima fila de L5-S1 adelgazandose fila a fila hasta un pixel, y el minimo daba
2.19 mm. Leido como diametro AP eso describe un bloqueo completo donde solo hay un
encuadre que se acaba.
"""

import numpy as np
import pytest

from pfi_ai_service.real_inference_runtime import canal_ap_diameter


def canal_band(top: int, bottom: int, left: int, width: int, size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for row in range(top, bottom):
        mask[row, left:left + width] = True
    return mask


def disc_band(top: int, bottom: int, size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[top:bottom, 40:80] = True
    return mask


def test_mide_el_ancho_del_canal_en_las_filas_del_disco():
    canal = canal_band(0, 200, 100, 12)
    assert canal_ap_diameter(canal, disc_band(90, 100), 0.5).length == 6.0


def test_toma_el_punto_mas_estrecho_y_no_el_promedio():
    """Una estenosis se define por donde mas se cierra; promediar la diluiria."""
    canal = canal_band(0, 200, 100, 12)
    canal[95, 106:112] = False  # una fila de 6 en vez de 12
    assert canal_ap_diameter(canal, disc_band(90, 100), 1.0).length == 6.0


def test_no_informa_nada_si_el_canal_termina_dentro_del_disco():
    """El caso real: la mascara se acaba y la caida se leeria como estenosis critica."""
    canal = np.zeros((200, 200), dtype=bool)
    for row in range(0, 96):
        canal[row, 100:112] = True
    for offset, row in enumerate(range(96, 100)):  # se adelgaza y termina
        canal[row, 100:110 - offset * 2] = True
    assert canal_ap_diameter(canal, disc_band(90, 100), 1.0) is None


def test_no_informa_nada_si_el_canal_empieza_dentro_del_disco():
    assert canal_ap_diameter(canal_band(95, 200, 100, 12), disc_band(90, 100), 1.0) is None


def test_un_canal_que_atraviesa_el_disco_por_completo_si_se_informa():
    """El guard tiene que dejar pasar el caso normal, no solo bloquear el roto."""
    assert canal_ap_diameter(canal_band(50, 150, 100, 12), disc_band(90, 100), 1.0).length == 12.0


def test_sin_canal_no_se_inventa_un_diametro():
    assert canal_ap_diameter(np.zeros((200, 200), dtype=bool), disc_band(90, 100), 1.0) is None


def test_sin_disco_no_hay_nivel_al_que_referir_la_medida():
    assert canal_ap_diameter(canal_band(0, 200, 100, 12), np.zeros((200, 200), dtype=bool), 1.0) is None


def test_el_diametro_se_mide_con_el_spacing_de_las_columnas():
    """El AP es horizontal en un sagital: sale del spacing de columnas, no del de filas."""
    canal = canal_band(0, 200, 100, 10)
    assert canal_ap_diameter(canal, disc_band(90, 100), 0.729).length == 7.29


def test_el_segmento_es_la_fila_mas_estrecha_y_mide_lo_que_dice():
    """La linea que se dibuja tiene que ser la que produjo el numero.

    Se ancla en la fila del estrechamiento -no en el medio del disco- porque es ahi
    donde se tomo la medida, y su largo es exactamente el diametro informado.
    """
    canal = canal_band(0, 200, 100, 12)
    canal[95, 106:112] = False  # la fila estrecha
    medida = canal_ap_diameter(canal, disc_band(90, 100), 0.729)
    assert medida.start[1] == 95 and medida.end[1] == 95
    assert (medida.end[0] - medida.start[0]) * 0.729 == pytest.approx(medida.length)
