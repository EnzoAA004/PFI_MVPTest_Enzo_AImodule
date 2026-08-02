"""Nombrado de cuerpos vertebrales y arcos posteriores.

La clase `vertebra_group` del modelo no distingue el cuerpo del arco: sobre un
estudio real devuelve 14 componentes para 7 vertebras. Estas pruebas fijan las dos
reglas que separan y nombran esos componentes, con geometria sintetica para que no
dependan de fixtures ni del artefacto.

Se verifica tanto lo que se nombra como lo que deliberadamente no: un componente
que el encuadre corto no recibe nivel, porque inventarlo seria afirmar una vertebra
que la imagen no muestra.
"""

import numpy as np

from pfi_ai_service.real_inference_runtime import (
    build_segmentation,
    lumbar_disc_levels,
    name_posterior_elements,
    name_vertebral_bodies,
    split_vertebral_bodies,
)


def block(top: int, bottom: int, left: int, right: int, size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def spine():
    """Columna sintetica: cuerpos adelante, arcos atras, discos entre cuerpos.

    En un corte sagital el eje horizontal es el anteroposterior, asi que "adelante"
    son columnas bajas. Se arman 4 cuerpos con sus 4 arcos y los 3 discos que los
    separan, con las alturas alineadas como en una columna real.
    """
    rows = [(20, 50), (60, 90), (100, 130), (140, 170)]
    bodies = [block(top, bottom, 40, 80) for top, bottom in rows]
    posterior = [block(top, bottom, 110, 140) for top, bottom in rows]
    discs = [block(bottom, rows[index + 1][0], 40, 80) for index, (_, bottom) in enumerate(rows[:-1])]
    return bodies, posterior, discs


def test_split_separa_cuerpos_de_arcos_usando_los_discos_como_ancla():
    bodies, posterior, discs = spine()
    split_bodies, split_posterior = split_vertebral_bodies(bodies + posterior, discs)
    assert len(split_bodies) == 4
    assert len(split_posterior) == 4
    # Los cuerpos son los que caen delante del borde posterior de los discos.
    for mask in split_bodies:
        assert float(np.where(mask.any(axis=0))[0].mean()) < 100


def test_sin_discos_no_hay_ancla_y_no_se_inventa_una_separacion():
    bodies, posterior, _ = spine()
    split_bodies, split_posterior = split_vertebral_bodies(bodies + posterior, [])
    assert len(split_bodies) == 8
    assert split_posterior == []


def test_el_arco_hereda_el_nivel_del_cuerpo_que_tiene_enfrente():
    bodies, posterior, _ = spine()
    body_names = ["L2", "L3", "L4", "L5"]
    assert name_posterior_elements(posterior, bodies, body_names) == body_names


def test_un_arco_sin_cuerpo_nombrado_enfrente_queda_sin_nivel():
    bodies, posterior, _ = spine()
    names = name_posterior_elements(posterior, bodies, ["L2", None, "L4", "L5"])
    assert names == ["L2", None, "L4", "L5"]


def test_un_arco_que_no_solapa_con_ningun_cuerpo_queda_sin_nivel():
    bodies, _, _ = spine()
    huerfano = [block(180, 195, 110, 140)]
    assert name_posterior_elements(huerfano, bodies, ["L2", "L3", "L4", "L5"]) == [None]


def test_los_cuerpos_se_nombran_desde_los_discos_ya_identificados():
    bodies, _, discs = spine()
    names = name_vertebral_bodies(bodies, discs, lumbar_disc_levels(len(discs)))
    # Con 3 discos no se alcanza la tabla lumbar completa, asi que no se asigna
    # nivel: contar desde abajo con un encuadre incompleto es exactamente el error
    # que se quiere evitar.
    assert names == [None, None, None, None]


def test_los_ids_de_instancia_son_unicos_aunque_dos_arcos_compartan_nivel():
    """El id no puede derivar del nivel.

    Sobre el estudio real la segmentacion parte el arco de una vertebra en dos
    componentes: los dos son legitimamente "L4" y, cuando el id se derivaba del
    nombre, colisionaban. El front usa ese id como clave de lista, asi que dos
    instancias distintas se pisaban entre si.
    """
    prediction = np.zeros((200, 200), dtype=np.uint8)
    rows = [(10, 35), (45, 70), (80, 105), (115, 140), (150, 175), (185, 199)]
    for top, bottom in rows:
        prediction[top:bottom, 40:80] = 1  # vertebra_group: cuerpo
    for index, (_, bottom) in enumerate(rows[:-1]):
        prediction[bottom:rows[index + 1][0], 40:80] = 3  # disc_group
    # Arco de una vertebra partido en dos componentes separados por una fila vacia,
    # que es como llega de la red cuando el arco queda seccionado.
    prediction[115:125, 110:140] = 1
    prediction[130:140, 110:140] = 1

    segmentation = build_segmentation("sagittal_spider", "sagittal", prediction)
    ids = [item["id"] for item in segmentation["instances"]]
    assert len(ids) == len(set(ids)), f"ids repetidos: {ids}"

    arcos = [item for item in segmentation["instances"] if item["label"] == "posterior_element"]
    assert len(arcos) == 2
    # Los dos fragmentos pertenecen a la misma vertebra, y eso se afirma: no se
    # inventa que uno sea de otro nivel solo para que los nombres no se repitan.
    assert {item["level"] for item in arcos} == {"L4"}


def test_con_los_cinco_discos_lumbares_los_cuerpos_reciben_su_nivel():
    rows = [(10, 35), (45, 70), (80, 105), (115, 140), (150, 175), (185, 199)]
    bodies = [block(top, bottom, 40, 80) for top, bottom in rows]
    discs = [block(bottom, rows[index + 1][0], 40, 80) for index, (_, bottom) in enumerate(rows[:-1])]
    names = name_vertebral_bodies(bodies, discs, lumbar_disc_levels(len(discs)))
    assert names == ["L1", "L2", "L3", "L4", "L5", "S1"]
