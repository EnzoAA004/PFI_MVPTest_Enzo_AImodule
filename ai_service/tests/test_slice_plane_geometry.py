"""Ubicacion del corte mostrado en el espacio del paciente.

Es lo que permite cruzar el plano sagital con el axial y trazar una linea de
referencia real. Todo depende de una correspondencia de ejes que es facil de invertir:
ITK entrega la matriz de direccion por filas y sus columnas son los cosenos de i, j, k,
mientras que el arreglo de numpy viene en orden (k, j, i). Tomado al reves, el plano
sale perpendicular a donde tiene que estar y la linea se dibuja cruzada.

Por eso las pruebas usan una geometria elegida para que cada eje tenga una direccion
distinta y reconocible: si dos se intercambian, el valor esperado cambia.
"""

import numpy as np
import pytest

from pfi_ai_service.real_inference_runtime import (
    LoadedInput,
    native_axis_directions,
    slice_plane_geometry,
)


# Matriz identidad de ITK: i -> x, j -> y, k -> z.
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def volume(shape, transform="none", direction=IDENTITY, origin=(10.0, 20.0, 30.0), spacing=None):
    """Volumen sintetico con la metadata que produce la canonicalizacion."""
    array = np.zeros(shape, dtype=np.float32)
    # `arrayAxisSpacingNative` va en el orden del arreglo: (z, y, x).
    native_spacing = spacing or [5.0, 2.0, 1.0]
    return LoadedInput(
        array=array,
        path="/tmp/x",
        suffix=".dcm",
        spacing_xyz=(1.0, 2.0, 5.0),
        metadata={
            "origin": list(origin),
            "direction": list(direction),
            "arrayAxisSpacingNative": native_spacing,
            "inputOrientationTransform": transform,
        },
    )


def test_el_eje_cero_del_arreglo_es_la_tercera_columna_de_la_matriz():
    """El orden (k, j, i) de numpy contra el (i, j, k) de ITK.

    Con una matriz que manda cada eje de indice a una direccion distinta, invertir la
    correspondencia da otra respuesta, que es justo lo que la prueba tiene que notar.
    """
    # i -> +x, j -> +z, k -> -y
    direction = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    ejes = native_axis_directions(direction)
    assert ejes[0] == [0.0, -1.0, 0.0], "el eje 0 del arreglo es k"
    assert ejes[1] == [0.0, 0.0, 1.0], "el eje 1 del arreglo es j"
    assert ejes[2] == [1.0, 0.0, 0.0], "el eje 2 del arreglo es i"


def test_una_matriz_incompleta_no_produce_ejes():
    assert native_axis_directions(None) is None
    assert native_axis_directions([1.0, 0.0, 0.0]) is None


def test_el_corte_cero_arranca_en_el_origen_del_volumen():
    plane = slice_plane_geometry(volume((12, 320, 320)), selected_axis=0, slice_index=0)
    assert plane["position"] == [10.0, 20.0, 30.0]


def test_cada_corte_avanza_un_espaciado_sobre_la_normal():
    """Es lo que ubica el corte N: sin esto todos caerian en el mismo lugar."""
    plane = slice_plane_geometry(volume((12, 320, 320)), selected_axis=0, slice_index=3)
    # Eje 0 del arreglo = k = +z con la identidad; su espaciado es 5 mm.
    assert plane["position"] == [10.0, 20.0, 45.0]
    assert plane["normal"] == [0.0, 0.0, 1.0]
    assert plane["sliceSpacing"] == 5.0


def test_las_direcciones_de_fila_y_columna_son_las_de_los_ejes_que_quedan():
    plane = slice_plane_geometry(volume((12, 320, 320)), selected_axis=0, slice_index=0)
    # Quedan los ejes 1 (j -> +y) y 2 (i -> +x), en ese orden.
    assert plane["rowDirection"] == [0.0, 1.0, 0.0]
    assert plane["colDirection"] == [1.0, 0.0, 0.0]
    assert (plane["rowSpacing"], plane["colSpacing"]) == (2.0, 1.0)


def test_la_transformacion_sagital_reordena_los_ejes():
    """El sagital se canonicaliza moviendo el eje 0 al final.

    El eje canonico 2 -que es el de cortes en un sagital- es el nativo 0, asi que la
    normal del corte es la del nativo 0 y no la del nativo 2. Leerlo sin deshacer la
    transformacion da un plano girado noventa grados.
    """
    loaded = volume((384, 384, 15), transform="move_axis_0_to_last")
    plane = slice_plane_geometry(loaded, selected_axis=2, slice_index=2)
    assert plane["normal"] == [0.0, 0.0, 1.0], "la normal es la del eje nativo 0"
    assert plane["sliceSpacing"] == 5.0
    assert plane["position"] == [10.0, 20.0, 40.0]
    # Canonicos 0 y 1 son nativos 1 y 2.
    assert plane["rowDirection"] == [0.0, 1.0, 0.0]
    assert plane["colDirection"] == [1.0, 0.0, 0.0]


def test_el_plano_declara_su_tamano_en_pixeles():
    plane = slice_plane_geometry(volume((12, 320, 256)), selected_axis=0, slice_index=0)
    assert (plane["rowCount"], plane["colCount"]) == (320, 256)


def test_las_direcciones_del_plano_son_perpendiculares_a_su_normal():
    """Invariante geometrica: si falla, algun eje quedo mal asignado."""
    direction = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
    plane = slice_plane_geometry(volume((12, 320, 320), direction=direction), selected_axis=0, slice_index=1)
    row = np.array(plane["rowDirection"])
    col = np.array(plane["colDirection"])
    normal = np.array(plane["normal"])
    assert float(np.dot(row, normal)) == pytest.approx(0, abs=1e-9)
    assert float(np.dot(col, normal)) == pytest.approx(0, abs=1e-9)
    # Y la normal es efectivamente el producto de las otras dos, salvo el signo.
    assert abs(float(np.dot(np.cross(row, col), normal))) == pytest.approx(1, abs=1e-9)


def test_sin_geometria_no_se_inventa_un_plano():
    loaded = volume((12, 320, 320))
    loaded.metadata["origin"] = None
    assert slice_plane_geometry(loaded, 0, 0) is None
    loaded = volume((12, 320, 320))
    loaded.metadata["direction"] = None
    assert slice_plane_geometry(loaded, 0, 0) is None


def test_un_volumen_2d_no_tiene_plano_de_corte():
    assert slice_plane_geometry(volume((320, 320)), 0, 0) is None


def test_una_transformacion_desconocida_no_se_adivina():
    """Antes que suponer una correspondencia de ejes, no se devuelve nada."""
    loaded = volume((12, 320, 320), transform="flip_something")
    assert slice_plane_geometry(loaded, 0, 0) is None


def test_se_usa_la_posicion_declarada_por_el_corte_y_no_la_calculada():
    """Es lo que hace exacta la ubicacion en una serie con huecos.

    En este dataset las series axiales saltan decenas de milimetros entre dos cortes.
    Con el modelo del espaciado unico, el corte posterior al salto quedaria ubicado
    donde no esta, y la linea de referencia apuntaria a otra altura sin avisarlo.
    """
    loaded = volume((4, 320, 320))
    loaded.metadata["slicePositions"] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 75.0],  # el hueco
        [0.0, 0.0, 80.0],
    ]
    despues_del_hueco = slice_plane_geometry(loaded, selected_axis=0, slice_index=2)
    assert despues_del_hueco["position"] == [0.0, 0.0, 75.0]
    assert despues_del_hueco["positionSource"] == "declared"
    # El modelo del espaciado lo habria puesto en 10, a 65 mm de donde esta.
    assert despues_del_hueco["position"] != [10.0, 20.0, 40.0]


def test_sin_posiciones_declaradas_se_cae_al_espaciado_y_se_dice():
    """Un .mha no trae posicion por corte; el modelo unico vale y se declara cual es."""
    plane = slice_plane_geometry(volume((12, 320, 320)), selected_axis=0, slice_index=2)
    assert plane["position"] == [10.0, 20.0, 40.0]
    assert plane["positionSource"] == "uniform_spacing"


def test_una_lista_de_posiciones_incompleta_no_se_usa_a_medias():
    loaded = volume((12, 320, 320))
    loaded.metadata["slicePositions"] = [[0.0, 0.0, 0.0]]
    plane = slice_plane_geometry(loaded, selected_axis=0, slice_index=5)
    assert plane["positionSource"] == "uniform_spacing"


def test_normal_sale_de_la_orientacion_del_propio_corte():
    """La normal describe al corte analizado, no al primero de la serie.

    Una serie axial lumbar no es un plano unico repetido: se adquiere en bloques
    angulados, uno por disco. La matriz de direccion del volumen es una sola y
    corresponde al primer corte, asi que usarla para el corte 6 lo proyecta sobre un
    plano que esta a 26 grados del suyo. Sobre el estudio de referencia eso corria la
    proyeccion 42 mm -tres espacios discales- y las mediciones axiales de un corte
    L4-L5 se informaban bajo L2-L3.
    """
    loaded = volume((12, 8, 8))
    # El corte 6 mira casi de frente en z; la direccion del volumen sigue siendo la
    # identidad, que es la del corte 0.
    loaded.metadata["sliceOrientations"] = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]] * 12
    loaded.metadata["sliceOrientations"][6] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    geometry = slice_plane_geometry(loaded, 0, 6)

    assert geometry["normalSource"] == "declared"
    # Producto vectorial de fila y columna del corte 6: (1,0,0) x (0,0,1) = (0,-1,0).
    assert [round(value, 6) for value in geometry["normal"]] == [0.0, -1.0, 0.0]


def test_sin_orientaciones_declaradas_cae_a_la_direccion_del_volumen():
    """Sin el dato por corte se usa lo unico que hay, y se deja dicho cual se uso."""
    geometry = slice_plane_geometry(volume((12, 8, 8)), 0, 6)

    assert geometry["normalSource"] == "volume_direction"
    assert [round(value, 6) for value in geometry["normal"]] == [0.0, 0.0, 1.0]
