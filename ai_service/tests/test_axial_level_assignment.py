"""Que nivel lumbar se le asigna a un corte axial.

Lo que mas se protege aca no es que el nivel se asigne, sino que **no se asigne** el
equivocado. Un diametro de saco tecal publicado bajo L4-L5 cuando el corte era de L3-L4
describe una estenosis en el nivel que no es, y se lee igual de convincente que uno
correcto.
"""
import pytest

from pfi_ai_service.axial_level import axial_slice_level

# Sagital: los discos ordenados de arriba hacia abajo. En LPS el eje z crece hacia
# craneal, asi que L5-S1 -el mas inferior- es el de z mas chico.
#
# Cada disco se publica como el casco convexo de su mascara, en coordenadas del
# paciente. Aca se lo modela como un rectangulo de 40 mm de profundidad
# anteroposterior (el eje y), que es lo que mide un disco lumbar: sin profundidad los
# tests no distinguirian una implementacion que evalua el disco entero de una que lo
# colapsa a una columna.
def _disco(level, z_top, z_bottom, y_from=-20.0, y_to=20.0):
    return {"level": level, "worldHull": [
        [0.0, y_from, z_top], [0.0, y_to, z_top],
        [0.0, y_to, z_bottom], [0.0, y_from, z_bottom],
    ]}


DISCOS = [
    _disco("L1-L2", 40.0, 32.0),
    _disco("L2-L3", 28.0, 20.0),
    _disco("L3-L4", 16.0, 8.0),
    _disco("L4-L5", 4.0, -4.0),
    _disco("L5-S1", -8.0, -16.0),
]


def sagital(discos=DISCOS):
    return {"discLevels": discos}


def axial(position, normal=(0.0, 0.0, 1.0)):
    return {"volumeGeometry": {"slicePlane": {"normal": list(normal), "position": list(position)}}}


@pytest.mark.parametrize("z,esperado", [
    (36.0, "L1-L2"),
    (24.0, "L2-L3"),
    (12.0, "L3-L4"),
    (0.0, "L4-L5"),
    (-12.0, "L5-S1"),
])
def test_el_corte_toma_el_nivel_del_disco_que_atraviesa(z, esperado):
    assert axial_slice_level(sagital(), axial((0.0, 0.0, z))) == esperado


def test_un_corte_entre_dos_discos_no_recibe_nivel():
    # z=6 cae entre L3-L4 (termina en 8) y L4-L5 (empieza en 4): es un cuerpo
    # vertebral. Adjudicarle el disco mas cercano pondria la medicion en un nivel
    # que el corte no muestra.
    assert axial_slice_level(sagital(), axial((0.0, 0.0, 6.0))) is None


def test_un_corte_fuera_del_encuadre_sagital_no_recibe_nivel():
    assert axial_slice_level(sagital(), axial((0.0, 0.0, 200.0))) is None


def test_el_borde_del_disco_cuenta_como_dentro():
    assert axial_slice_level(sagital(), axial((0.0, 0.0, 8.0))) == "L3-L4"


def test_un_axial_angulado_se_resuelve_por_proyeccion():
    """La serie axial se adquiere en bloques angulados, uno por disco.

    Con la normal inclinada el plano ya no es "z constante": la altura a la que corta
    depende de la profundidad. Comparar z contra z daria el nivel de al lado.
    """
    import math
    angulo = math.radians(23)
    normal = (0.0, -math.sin(angulo), math.cos(angulo))
    assert axial_slice_level(sagital(), axial((0.0, 0.0, 0.0), normal)) == "L4-L5"
    # El mismo z, corrido en profundidad: con la angulacion ya no es el mismo nivel.
    assert axial_slice_level(sagital(), axial((0.0, 40.0, 0.0), normal)) != "L4-L5"


def test_el_disco_se_evalua_entero_y_no_por_su_columna_central():
    """Regresion: colapsar el disco a su centroide erra por mas de lo que mide un disco.

    Con 23 grados y 40 mm de profundidad, la altura a la que el plano cruza el disco
    varia unos 17 mm entre el borde anterior y el posterior — mas que la altura de un
    disco entero. Este corte toca el borde anterior de L3-L4 pero pasa lejos de su
    centroide: evaluando una sola columna se le asignaba L4-L5, el nivel de al lado.
    """
    import math
    angulo = math.radians(23)
    normal = (0.0, -math.sin(angulo), math.cos(angulo))
    # Offset elegido para caer sobre el borde anterior (y=-20) de L3-L4, en z=8.
    posicion = (0.0, -20.0, 8.0)
    assert axial_slice_level(sagital(), axial(posicion, normal)) == "L3-L4"

    # Y la comprobacion de que el caso es real: el centroide del disco (y=0) proyecta
    # a otro valor, asi que la version por columna central no lo habria encontrado.
    centro = _dot_test(normal, (0.0, 0.0, 8.0))
    borde = _dot_test(normal, posicion)
    assert abs(centro - borde) > 7, "el caso tiene que separar las dos implementaciones"


def test_entre_dos_discos_cruzados_gana_el_que_se_atraviesa_mas_centrado():
    """Un plano angulado cruza mas de un disco: el orden de la lista no puede decidir.

    Con estos discos y esta angulacion, el corte cae dentro del intervalo de L2-L3 y
    tambien del de L3-L4. Quedarse con el primero daria L2-L3 por estar mas arriba.
    """
    import math
    normal = (0.0, -math.sin(math.radians(23)), math.cos(math.radians(23)))
    offset = _dot_test(normal, (0.0, -20.0, 8.0))
    dentro = []
    for disco in DISCOS:
        proyecciones = [_dot_test(normal, punto) for punto in disco["worldHull"]]
        if min(proyecciones) <= offset <= max(proyecciones):
            dentro.append(disco["level"])
    assert dentro == ["L2-L3", "L3-L4"], f"el caso tiene que ser ambiguo, dio {dentro}"
    # L3-L4 lo atraviesa mas cerca de su medio, asi que es el nivel del corte.
    assert axial_slice_level(sagital(), axial((0.0, -20.0, 8.0), normal)) == "L3-L4"


def _dot_test(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def test_sin_sagital_no_hay_nivel():
    assert axial_slice_level(None, axial((0.0, 0.0, 0.0))) is None
    assert axial_slice_level({"discLevels": []}, axial((0.0, 0.0, 0.0))) is None


def test_geometria_incompleta_no_inventa_nivel():
    assert axial_slice_level(sagital(), {"volumeGeometry": {"slicePlane": {"normal": [0, 0, 1]}}}) is None
    assert axial_slice_level(sagital(), {"volumeGeometry": None}) is None


def test_un_disco_mal_formado_se_saltea_sin_romper():
    discos = [{"level": "L1-L2", "worldHull": None}, {"level": "L2-L3", "worldHull": []}] + DISCOS
    assert axial_slice_level(sagital(discos), axial((0.0, 0.0, 12.0))) == "L3-L4"


def test_el_casco_convexo_conserva_los_extremos_en_toda_direccion():
    """Lo unico que se le pide al casco: que proyecte igual que la nube entera.

    Es la propiedad de la que depende la asignacion de nivel. Si el casco recortara un
    extremo, el intervalo del disco saldria mas corto y un corte que si lo cruza
    quedaria sin nivel.
    """
    import random
    from pfi_ai_service.real_inference_runtime import convex_hull

    random.seed(7)
    nube = [(random.randint(0, 60), random.randint(0, 40)) for _ in range(400)]
    casco = convex_hull(nube)
    assert 3 <= len(casco) <= len(set(nube))
    for angulo in range(0, 360, 15):
        import math
        direccion = (math.cos(math.radians(angulo)), math.sin(math.radians(angulo)))
        proyectar = lambda puntos: [p[0] * direccion[0] + p[1] * direccion[1] for p in puntos]
        assert min(proyectar(casco)) == pytest.approx(min(proyectar(nube)))
        assert max(proyectar(casco)) == pytest.approx(max(proyectar(nube)))


def test_el_casco_de_pocos_puntos_no_rompe():
    from pfi_ai_service.real_inference_runtime import convex_hull

    assert convex_hull([]) == []
    assert convex_hull([(1, 1)]) == [(1, 1)]
    assert convex_hull([(1, 1), (2, 2)]) == [(1, 1), (2, 2)]
    # Todos colineales: no hay area, y el casco se reduce a los dos extremos.
    assert convex_hull([(0, 0), (1, 1), (2, 2), (3, 3)]) == [(0, 0), (3, 3)]
