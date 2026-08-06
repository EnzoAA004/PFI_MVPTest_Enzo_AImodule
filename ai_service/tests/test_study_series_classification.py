"""Clasificacion de las series de un estudio: plano, capturas y que se puede analizar.

El caso que motiva estos tests es un estudio real de 7 series donde el localizer y dos
capturas de consola salian etiquetados como series sagitales normales, porque el plano
se leia del encabezado de un solo corte.
"""
from types import SimpleNamespace

from pfi_ai_service.study_ingestion import _is_analyzable, _is_derived, _series_plane

SAGITTAL = [0, 1, 0, 0, 0, -1]
AXIAL = [1, 0, 0, 0, 1, 0]
CORONAL = [1, 0, 0, 0, 0, -1]


def test_serie_de_un_plano_no_es_multiplano():
    plane, multiplanar = _series_plane([AXIAL] * 15)
    assert (plane, multiplanar) == ("axial", False)


def test_localizer_con_los_tres_planos_queda_marcado():
    # Un localizer trae cortes de los tres planos en una sola serie.
    plane, multiplanar = _series_plane([SAGITTAL] * 4 + [CORONAL] * 3 + [AXIAL])
    assert multiplanar is True
    assert plane == "sagittal"


def test_el_plano_de_un_empate_no_cambia_entre_corridas():
    # Sin desempate estable, el orden del `set` hace que el mismo estudio se informe
    # distinto en dos ingestas.
    empate = [SAGITTAL] * 3 + [CORONAL] * 3
    assert {_series_plane(empate)[0] for _ in range(20)} == {"coronal"}


def test_localizer_desbalanceado_tambien_queda_marcado():
    # Con mayoria sagital clara: igual es multiplano, no una serie sagital.
    _plane, multiplanar = _series_plane([SAGITTAL] * 5 + [CORONAL] * 2 + [AXIAL])
    assert multiplanar is True


def test_el_primer_corte_no_decide_el_plano():
    # El corte que GDCM ordena primero es axial, pero la serie es sagital.
    plane, _multiplanar = _series_plane([AXIAL] + [SAGITTAL] * 14)
    assert plane == "sagittal"


def test_serie_sin_orientacion_no_tiene_plano():
    assert _series_plane([None, None]) == (None, False)


def test_captura_de_consola_es_derivada():
    assert _is_derived(SimpleNamespace(ImageType=["DERIVED", "SECONDARY", "POSDISP"])) is True


def test_adquisicion_original_no_es_derivada():
    assert _is_derived(SimpleNamespace(ImageType=["ORIGINAL", "PRIMARY", "M", "NORM"])) is False


def test_serie_sin_image_type_no_es_derivada():
    assert _is_derived(SimpleNamespace()) is False


def _series(**overrides):
    base = {"plane": "axial", "multiplanar": False, "derived": False, "sliceCount": 15}
    return {**base, **overrides}


def test_serie_normal_es_analizable():
    assert _is_analyzable(_series()) is True


def test_localizer_no_es_analizable():
    assert _is_analyzable(_series(multiplanar=True)) is False


def test_captura_no_es_analizable():
    # Las dos capturas `PosDisp` del estudio de referencia: 2 cortes, derivadas.
    assert _is_analyzable(_series(derived=True, sliceCount=2)) is False


def test_corte_unico_no_es_analizable():
    # Un solo corte no es un volumen: el chequeo sagital pide 3 dimensiones.
    assert _is_analyzable(_series(sliceCount=1)) is False
