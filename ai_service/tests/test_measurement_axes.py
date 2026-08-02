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
    largo, corto = max(width, height), min(width, height)
    assert largo == pytest.approx(40, abs=1.5)
    assert corto == pytest.approx(10, abs=1.5)


def test_la_caja_alineada_a_la_imagen_no_media_el_disco():
    """Documenta el defecto que motivo el cambio, con el numero que lo delataba."""
    mask = rotated_rectangle(40, 10, 35)
    ys, xs = np.where(mask)
    caja_alto = float(ys.max() - ys.min() + 1)
    assert caja_alto > 30  # un disco de 10 mm de espesor informado como 31

    _, alto = oriented_extent(mask, 1.0, 1.0)
    assert alto == pytest.approx(10, abs=1.5)


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
    assert max(ancho, alto) == pytest.approx(40, abs=1.5)
    assert min(ancho, alto) == pytest.approx(10, abs=1.5)


def test_a_cero_grados_da_exactamente_lo_mismo_que_la_caja():
    """Las medidas que ya estaban bien no se mueven."""
    mask = rotated_rectangle(40, 10, 0)
    ys, xs = np.where(mask)
    ancho, alto = oriented_extent(mask, 1.0, 1.0)
    assert ancho == pytest.approx(float(xs.max() - xs.min() + 1))
    assert alto == pytest.approx(float(ys.max() - ys.min() + 1))


def test_el_ancho_sigue_siendo_el_horizontal_en_una_vertebra_mas_alta_que_ancha():
    """Ordenar por eje mayor y menor intercambiaria ancho y alto justo aca.

    En un sagital el ancho es el anteroposterior y el alto el craneocaudal: eso lo
    fija la anatomia, no cual de los dos resulte mas largo en este estudio.
    """
    ancho, alto = oriented_extent(rotated_rectangle(10, 40, 0), 1.0, 1.0)
    assert ancho == pytest.approx(10, abs=1.5)
    assert alto == pytest.approx(40, abs=1.5)


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
    assert ancho == pytest.approx(40, abs=3.0)
    assert alto == pytest.approx(10, abs=3.0)


def test_una_estructura_de_un_pixel_de_espesor_no_mide_cero():
    """La huella del pixel es lo que hacia el `+1` de la caja."""
    mask = np.zeros((50, 50), dtype=bool)
    mask[25, 10:30] = True
    ancho, alto = oriented_extent(mask, 0.5, 0.5)
    assert ancho == pytest.approx(10.0)
    assert alto == pytest.approx(0.5)


def test_una_mascara_vacia_no_devuelve_una_medida():
    assert oriented_extent(np.zeros((10, 10), dtype=bool), 1.0, 1.0) == (0.0, 0.0)


def test_un_solo_pixel_mide_un_pixel():
    """Con un punto no hay covarianza, y no se inventa una orientacion."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    assert oriented_extent(mask, 0.8, 0.6) == pytest.approx((0.6, 0.8))
