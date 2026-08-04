"""Ancho y alto medidos sobre los ejes propios de la estructura.

La caja alineada a la imagen no mide el disco: mide la caja que lo contiene. Sobre
el estudio real, L5-S1 -el nivel mas angulado de la columna- daba 25 mm de alto para
un disco que anda por 8-12. Con el error de escala encima nadie lo notaba, porque
todo salia corto; corregida la escala, el defecto quedo a la vista.

La prueba que fija la regla es la invariancia: un mismo rectangulo rotado tiene que
medir lo mismo en cualquier angulo. La version por caja falla eso por construccion.
"""

import math

import numpy as np
import pytest

from pfi_ai_service.real_inference_runtime import oriented_extent


def rotated_rectangle(
    width: float,
    height: float,
    degrees: float,
    row_spacing: float = 1.0,
    col_spacing: float = 1.0,
    size: int = 200,
) -> np.ndarray:
    """Rectangulo de `width` x `height` MILIMETROS rotado `degrees` sobre el centro.

    Se construye en milimetros y no en pixeles a proposito: rotar y despues estirar
    un eje no da un rectangulo rotado, da un paralelogramo mas largo. Con spacing
    anisotropico esa diferencia es justo lo que se quiere medir, asi que la figura
    tiene que ser un rectangulo en el espacio donde se la mide.

    Los intervalos son semiabiertos para que a cero grados el rectangulo ocupe
    exactamente `width` x `height` milimetros y las cuentas no arrastren un pixel de
    mas.
    """
    angle = math.radians(degrees)
    center = size / 2
    ys, xs = np.mgrid[0:size, 0:size]
    dx = (xs - center) * col_spacing
    dy = (ys - center) * row_spacing
    along = dx * math.cos(angle) + dy * math.sin(angle)
    across = -dx * math.sin(angle) + dy * math.cos(angle)
    return (-width / 2 <= along) & (along < width / 2) & (-height / 2 <= across) & (across < height / 2)


@pytest.mark.parametrize("degrees", [0, 10, 25, 45, 70, 90])
def test_un_disco_mide_lo_mismo_este_como_este_inclinado(degrees):
    """La medicion describe la estructura, no como quedo apoyada en la imagen."""
    width, height = oriented_extent(rotated_rectangle(40, 10, degrees), 1.0, 1.0)
    largo, corto = max(width.length, height.length), min(width.length, height.length)
    assert largo == pytest.approx(40, abs=1.5)
    assert corto == pytest.approx(10, abs=1.5)


def test_la_caja_alineada_a_la_imagen_no_media_el_disco():
    """Documenta el defecto que motivo el cambio, con el numero que lo delataba."""
    mask = rotated_rectangle(40, 10, 35)
    ys, xs = np.where(mask)
    caja_alto = float(ys.max() - ys.min() + 1)
    assert caja_alto > 30  # un disco de 10 mm de espesor informado como 31

    _, alto = oriented_extent(mask, 1.0, 1.0)
    assert alto.length == pytest.approx(10, abs=1.5)


def test_a_cuarenta_y_cinco_grados_los_dos_ejes_son_igual_de_horizontales():
    """El limite del criterio, escrito para que nadie lo descubra en produccion.

    Cual de los dos ejes es "el ancho" se decide por cual esta mas cerca de la
    horizontal. A 45 grados exactos los dos estan a la misma distancia y la eleccion
    deja de significar algo: las dos magnitudes son correctas, pero cual se rotula
    ancho y cual alto es arbitrario.

    Se fija lo que si vale en ese caso -que las magnitudes son las del rectangulo- y
    se deja constancia de que el rotulo no.
    """
    ancho, alto = oriented_extent(rotated_rectangle(40, 10, 45), 1.0, 1.0)
    assert max(ancho.length, alto.length) == pytest.approx(40, abs=1.5)
    assert min(ancho.length, alto.length) == pytest.approx(10, abs=1.5)


def test_a_cero_grados_da_exactamente_lo_mismo_que_la_caja():
    """Las medidas que ya estaban bien no se mueven."""
    mask = rotated_rectangle(40, 10, 0)
    ys, xs = np.where(mask)
    ancho, alto = oriented_extent(mask, 1.0, 1.0)
    assert ancho.length == pytest.approx(float(xs.max() - xs.min() + 1))
    assert alto.length == pytest.approx(float(ys.max() - ys.min() + 1))


def test_el_ancho_sigue_siendo_el_horizontal_en_una_vertebra_mas_alta_que_ancha():
    """Ordenar por eje mayor y menor intercambiaria ancho y alto justo aca.

    En un sagital el ancho es el anteroposterior y el alto el craneocaudal: eso lo
    fija la anatomia, no cual de los dos resulte mas largo en este estudio.
    """
    ancho, alto = oriented_extent(rotated_rectangle(10, 40, 0), 1.0, 1.0)
    assert ancho.length == pytest.approx(10, abs=1.5)
    assert alto.length == pytest.approx(40, abs=1.5)


def test_el_spacing_anisotropico_no_inclina_los_ejes_por_si_solo():
    """Los ejes se buscan en milimetros, no en pixeles.

    Con pixeles el doble de altos que anchos, la grilla esta estirada en vertical.
    Buscar los ejes sobre esa grilla los inclinaria por el estiramiento y no por la
    estructura, y el disco mediria de mas en las dos direcciones.

    La tolerancia es mas ancha que en el resto porque la grilla es mas gruesa: un
    disco de 10 mm son 5 pixeles de 2 mm, y esa incertidumbre es real, no del metodo.
    """
    mask = rotated_rectangle(40, 10, 30, row_spacing=2.0, col_spacing=1.0)
    ancho, alto = oriented_extent(mask, 2.0, 1.0)
    assert ancho.length == pytest.approx(40, abs=3.0)
    assert alto.length == pytest.approx(10, abs=3.0)


def test_una_estructura_de_un_pixel_de_espesor_no_mide_cero():
    """La huella del pixel es lo que hacia el `+1` de la caja."""
    mask = np.zeros((50, 50), dtype=bool)
    mask[25, 10:30] = True
    ancho, alto = oriented_extent(mask, 0.5, 0.5)
    assert ancho.length == pytest.approx(10.0)
    assert alto.length == pytest.approx(0.5)


def test_una_mascara_vacia_no_devuelve_una_medida():
    ancho, alto = oriented_extent(np.zeros((10, 10), dtype=bool), 1.0, 1.0)
    assert (ancho.length, alto.length) == (0.0, 0.0)


def test_un_solo_pixel_mide_un_pixel():
    """Con un punto no hay covarianza, y no se inventa una orientacion."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    ancho, alto = oriented_extent(mask, 0.8, 0.6)
    assert (ancho.length, alto.length) == pytest.approx((0.6, 0.8))


def distance(segment, row_spacing=1.0, col_spacing=1.0):
    """Largo del segmento dibujado, en milimetros."""
    dx = (segment.end[0] - segment.start[0]) * col_spacing
    dy = (segment.end[1] - segment.start[1]) * row_spacing
    return math.hypot(dx, dy)


@pytest.mark.parametrize("degrees", [0, 20, 35, 60, 90])
def test_la_linea_dibujada_mide_lo_que_dice_el_numero(degrees):
    """La garantia que hace honesto mostrar la medicion sobre la imagen.

    El visor dibuja el segmento para que el medico vea de donde a donde se midio. Si
    la linea y el valor se calcularan por separado podrian discrepar, y la pantalla
    mostraria una medicion que no es la que dice la tabla.
    """
    ancho, alto = oriented_extent(rotated_rectangle(40, 10, degrees), 1.0, 1.0)
    # La tolerancia es el redondeo de los extremos a dos decimales de pixel, que sobre
    # una grilla de 256 son centesimas de milimetro.
    assert distance(ancho) == pytest.approx(ancho.length, abs=0.05)
    assert distance(alto) == pytest.approx(alto.length, abs=0.05)


def test_la_linea_sigue_midiendo_lo_mismo_con_spacing_anisotropico():
    mask = rotated_rectangle(40, 10, 30, row_spacing=2.0, col_spacing=1.0)
    ancho, alto = oriented_extent(mask, 2.0, 1.0)
    assert distance(ancho, 2.0, 1.0) == pytest.approx(ancho.length, abs=0.05)
    assert distance(alto, 2.0, 1.0) == pytest.approx(alto.length, abs=0.05)


def test_el_segmento_del_ancho_es_horizontal_en_una_estructura_sin_inclinar():
    ancho, alto = oriented_extent(rotated_rectangle(40, 10, 0), 1.0, 1.0)
    assert ancho.start[1] == pytest.approx(ancho.end[1])
    assert alto.start[0] == pytest.approx(alto.end[0])


def test_el_segmento_se_inclina_con_la_estructura():
    """A 35 grados el ancho ya no puede ser una linea horizontal."""
    ancho, _ = oriented_extent(rotated_rectangle(40, 10, 35), 1.0, 1.0)
    assert abs(ancho.end[1] - ancho.start[1]) > 10


def test_el_segmento_pasa_por_el_centro_de_la_estructura():
    """Su punto medio es el centroide: la linea cruza la estructura, no la bordea."""
    mask = rotated_rectangle(40, 10, 25)
    ys, xs = np.where(mask)
    ancho, _ = oriented_extent(mask, 1.0, 1.0)
    assert (ancho.start[0] + ancho.end[0]) / 2 == pytest.approx(float(xs.mean()), abs=0.5)
    assert (ancho.start[1] + ancho.end[1]) / 2 == pytest.approx(float(ys.mean()), abs=0.5)


def test_cada_distancia_publicada_lleva_los_puntos_de_donde_salio():
    """Regresion: el ancho y el alto viajaban sin figura y la cota no se dibujaba.

    Las pruebas de `oriented_extent` seguian pasando porque el calculo estaba bien;
    lo que se habia perdido era el paso de adjuntarlo a la medicion publicada. Por eso
    se verifica la salida de `build_measurements` y no solo el calculador: es el
    contrato que consume el visor.

    El area no lleva puntos a proposito: no tiene dos extremos.
    """
    from pfi_ai_service.real_inference_runtime import build_measurements

    prediction = np.zeros((200, 200), dtype=np.uint8)
    rows = [(20, 45), (55, 80), (90, 115), (125, 150), (155, 180)]
    for top, bottom in rows:
        prediction[top:bottom, 40:80] = 1  # vertebra_group
    for index, (_, bottom) in enumerate(rows[:-1]):
        prediction[bottom:rows[index + 1][0], 40:80] = 3  # disc_group
    confidence = np.ones((200, 200), dtype=np.float32)

    values = build_measurements("sagittal_spider", "sagittal", prediction, confidence, (0.7, 0.7), 7)
    distancias = [item for item in values if item["id"].endswith(("-width", "-height"))]
    areas = [item for item in values if item["id"].endswith("-area")]

    assert distancias, "la geometria sintetica tiene que producir anchos y altos"
    for item in distancias:
        assert len(item["points"]) == 2, f"{item['id']} sin figura"
    for item in areas:
        assert item["points"] == [], f"{item['id']} no deberia llevar figura"
