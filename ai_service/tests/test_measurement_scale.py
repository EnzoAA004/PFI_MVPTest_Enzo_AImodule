"""Escala de las mediciones: grilla de la prediccion contra grilla nativa.

La red recibe el corte reescalado a su tamano de entrada, asi que un pixel de la
prediccion no mide lo mismo que un pixel de la imagen original. Contar pixeles de
la prediccion y multiplicar por el spacing nativo mezclaba las dos grillas: sobre un
estudio real de 384x384 con entrada de 256x256 todas las distancias salian un 33%
cortas -un disco de 28 mm se informaba como 19- y las areas a menos de la mitad,
porque el factor entra al cuadrado.

Estas pruebas fijan el factor y, sobre todo, los casos donde NO hay que corregir.
"""

import numpy as np
import pytest

from pfi_ai_service.real_inference_runtime import build_measurements, prediction_grid_spacing


def test_el_spacing_se_escala_por_el_factor_de_resize():
    spacing = prediction_grid_spacing((0.729, 0.729), (384, 384, 15), 2, (256, 256))
    assert spacing == pytest.approx((1.0935, 1.0935), rel=1e-6)


def test_sin_resize_el_spacing_no_se_toca():
    """Si la prediccion ya esta en la grilla nativa, corregir seria introducir el error."""
    spacing = prediction_grid_spacing((0.5, 0.5), (256, 256, 20), 2, (256, 256))
    assert spacing == pytest.approx((0.5, 0.5))


def test_ejes_con_factores_distintos_se_corrigen_por_separado():
    spacing = prediction_grid_spacing((1.0, 1.0), (384, 512, 10), 2, (256, 256))
    assert spacing == pytest.approx((1.5, 2.0))


def test_sin_spacing_no_se_inventa_una_escala():
    assert prediction_grid_spacing(None, (384, 384, 15), 2, (256, 256)) is None


def test_un_volumen_2d_conserva_el_spacing_cuando_no_hay_dos_ejes_restantes():
    assert prediction_grid_spacing((0.7, 0.7), (384,), 0, (256, 256)) == (0.7, 0.7)


def test_la_medicion_final_refleja_el_tamano_real_del_disco():
    """Verificacion de punta a punta sobre la geometria que motivo el arreglo.

    Un disco que ocupa 26 pixeles de ancho en la prediccion de 256 equivale a 39
    pixeles nativos, que a 0.729 mm/pixel son 28.4 mm. Con el spacing nativo sin
    corregir daban 19 mm, que es el numero que aparecia en la tabla.
    """
    prediction = np.zeros((256, 256), dtype=np.uint8)
    prediction[100:110, 100:126] = 3  # disc_group
    confidence = np.ones((256, 256), dtype=np.float32)

    nativo = build_measurements("sagittal_spider", "sagittal", prediction, confidence, (0.729, 0.729), 7)
    corregido = build_measurements(
        "sagittal_spider",
        "sagittal",
        prediction,
        confidence,
        prediction_grid_spacing((0.729, 0.729), (384, 384, 15), 2, (256, 256)),
        7,
    )

    def ancho(values):
        return next(item["value"] for item in values if item["id"].endswith("-width"))

    assert ancho(nativo) == pytest.approx(18.95, abs=0.05)
    assert ancho(corregido) == pytest.approx(28.43, abs=0.05)


def test_las_areas_se_corrigen_al_cuadrado():
    prediction = np.zeros((256, 256), dtype=np.uint8)
    prediction[100:110, 100:126] = 3
    confidence = np.ones((256, 256), dtype=np.float32)

    def area(spacing):
        values = build_measurements("sagittal_spider", "sagittal", prediction, confidence, spacing, 7)
        return next(item["value"] for item in values if item["id"].endswith("-area"))

    nativo = area((0.729, 0.729))
    corregido = area(prediction_grid_spacing((0.729, 0.729), (384, 384, 15), 2, (256, 256)))
    assert corregido / nativo == pytest.approx(2.25, rel=1e-3)
