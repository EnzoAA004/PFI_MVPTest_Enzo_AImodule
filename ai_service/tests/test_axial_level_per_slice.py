"""Nivel lumbar de cada corte axial, no solo del que analizo el modelo.

Lo que se protege aca es lo mismo que en test_axial_level_assignment, pero sobre la serie
entera: que un corte no herede el nivel de otro. Una serie axial lumbar se adquiere en
bloques angulados, uno por disco, asi que propagar el nivel del corte analizado a los
quince pondria dos tercios de las marcas bajo un nivel que no es.

Importa para el ROI subarticular: el nivel viaja en el pedido de clasificacion, y una
estenosis clasificada bajo L4-L5 cuando el corte era de L3-L4 se lee igual de convincente
que una correcta.
"""
import math

from pfi_ai_service.axial_level import axial_slice_levels

from test_axial_level_assignment import DISCOS, sagital


def axial_por_corte(positions, orientations=None, plane_normal=(0.0, 0.0, 1.0)):
    geometry = {
        "slicePlane": {"normal": list(plane_normal), "position": [0.0, 0.0, 0.0]},
        "slicePositions": [list(p) for p in positions],
    }
    if orientations is not None:
        geometry["sliceOrientations"] = [list(o) for o in orientations]
    return {"volumeGeometry": geometry}


# Orientacion axial neutra: la fila recorre el eje x (izquierda del paciente) y la
# columna el eje y (posterior). Su producto vectorial da la normal craneocaudal.
AXIAL_NEUTRO = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_cada_corte_toma_el_nivel_del_disco_que_el_atraviesa():
    # Un corte por disco, de craneal a caudal.
    positions = [(0.0, 0.0, 36.0), (0.0, 0.0, 24.0), (0.0, 0.0, 12.0), (0.0, 0.0, 0.0), (0.0, 0.0, -12.0)]
    niveles = axial_slice_levels(sagital(), axial_por_corte(positions, [AXIAL_NEUTRO] * 5))

    assert niveles == {0: "L1-L2", 1: "L2-L3", 2: "L3-L4", 3: "L4-L5", 4: "L5-S1"}


def test_un_corte_que_no_cae_en_ningun_disco_no_aparece():
    # z=20 es el borde inferior de L2-L3; z=18 esta entre discos.
    positions = [(0.0, 0.0, 24.0), (0.0, 0.0, 18.0), (0.0, 0.0, 12.0)]
    niveles = axial_slice_levels(sagital(), axial_por_corte(positions, [AXIAL_NEUTRO] * 3))

    assert 1 not in niveles
    assert niveles == {0: "L2-L3", 2: "L3-L4"}


def test_ningun_corte_hereda_el_nivel_de_otro():
    """El corte analizado tiene nivel; los que no caen en un disco siguen sin tenerlo."""
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 6.0), (0.0, 0.0, 12.0)]
    niveles = axial_slice_levels(sagital(), axial_por_corte(positions, [AXIAL_NEUTRO] * 3))

    assert niveles[0] == "L4-L5"
    assert 1 not in niveles  # z=6 cae entre L4-L5 (4.0) y L3-L4 (8.0)
    assert niveles[2] == "L3-L4"


def test_bloques_angulados_no_comparten_normal():
    """El caso que motiva el modulo: cada bloque mira su disco con su propia angulacion.

    Con una normal unica para toda la serie, un corte inclinado 23 grados se proyecta a
    una altura que no es la suya. Se verifica que la orientacion propia de cada corte se
    usa, comparando contra el nivel que da su geometria real.
    """
    ang = math.radians(23.0)
    # Bloque inclinado: la columna se inclina en el plano yz, asi que la normal se aparta
    # de z. La posicion es la misma; lo unico que cambia es la orientacion.
    inclinado = [1.0, 0.0, 0.0, 0.0, math.cos(ang), math.sin(ang)]

    position = (0.0, 30.0, 0.0)
    recto = axial_slice_levels(sagital(), axial_por_corte([position], [AXIAL_NEUTRO]))
    torcido = axial_slice_levels(sagital(), axial_por_corte([position], [inclinado]))

    # Con normal craneocaudal el punto proyecta a z=0 -> L4-L5.
    assert recto == {0: "L4-L5"}
    # Con el bloque inclinado el mismo punto proyecta a otra altura, y el nivel cambia.
    # Lo que importa no es cual da, sino que no de el mismo: si diera igual, la
    # orientacion por corte no se estaria usando.
    assert torcido != recto


def test_sin_orientaciones_usa_la_normal_del_plano_analizado():
    """Una corrida vieja no publica sliceOrientations. La posicion propia ya corrige mucho."""
    positions = [(0.0, 0.0, 24.0), (0.0, 0.0, 12.0)]
    niveles = axial_slice_levels(sagital(), axial_por_corte(positions, orientations=None))

    assert niveles == {0: "L2-L3", 1: "L3-L4"}


def test_sin_sagital_no_hay_niveles():
    positions = [(0.0, 0.0, 24.0)]
    assert axial_slice_levels(None, axial_por_corte(positions)) == {}
    assert axial_slice_levels({}, axial_por_corte(positions)) == {}
    assert axial_slice_levels({"discLevels": []}, axial_por_corte(positions)) == {}


def test_sin_posiciones_por_corte_no_se_inventa_ninguna():
    """Sin slicePositions no se extrapola `origen + N x espaciado`: estas series tienen huecos."""
    axial = {"volumeGeometry": {"slicePlane": {"normal": [0.0, 0.0, 1.0], "position": [0.0, 0.0, 0.0]}}}
    assert axial_slice_levels(sagital(), axial) == {}
