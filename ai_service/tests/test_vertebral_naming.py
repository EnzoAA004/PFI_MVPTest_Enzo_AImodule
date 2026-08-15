"""Nombrado de cuerpos vertebrales y arcos posteriores.

La clase `vertebra_group` del modelo no distingue el cuerpo del arco. Estas pruebas
fijan las reglas que separan y nombran esos componentes, incluyendo la elongacion
caudal y fragmentacion observadas en el estudio real.

Se verifica tanto lo que se nombra como lo que deliberadamente no: un componente
que el encuadre corto no recibe nivel, porque inventarlo seria afirmar una vertebra
que la imagen no muestra.
"""

import numpy as np

from pfi_ai_service.real_inference_runtime import (
    build_masks,
    build_measurements,
    build_segmentation,
    lumbar_disc_levels,
    name_posterior_elements,
    name_vertebral_bodies,
    split_vertebral_anatomical_instances,
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


def test_el_arco_hereda_el_nivel_del_cuerpo_donde_nace():
    bodies, posterior, _ = spine()
    body_names = ["L2", "L3", "L4", "L5"]
    assert name_posterior_elements(posterior, bodies, body_names) == body_names


def test_un_arco_sin_cuerpo_nombrado_enfrente_queda_sin_nivel():
    bodies, posterior, _ = spine()
    names = name_posterior_elements(posterior, bodies, ["L2", None, "L4", "L5"])
    assert names == ["L2", None, "L4", "L5"]


def test_un_posterior_elongado_caudo_cranealmente_usa_su_origen_y_no_el_maximo_solapamiento():
    """Regresion real: el extremo craneal es L3 aunque el componente solape mas L4."""
    l3 = block(107, 134, 40, 80)
    l4 = block(141, 165, 40, 80)
    posterior = block(127, 153, 110, 140)

    # La regla historica elegia L4: 12 filas de solapamiento contra 7 con L3.
    assert name_posterior_elements([posterior], [l3, l4], ["L3", "L4"]) == ["L3"]


def test_un_posterior_que_se_extiende_sobre_dos_cuerpos_conserva_el_superior_donde_nace():
    superior = block(20, 50, 40, 80)
    inferior = block(60, 90, 40, 80)
    posterior = block(45, 80, 110, 140)

    assert name_posterior_elements([posterior], [superior, inferior], ["L2", "L3"]) == ["L2"]


def test_un_origen_ambiguo_entre_dos_cuerpos_no_inventa_nivel():
    body_a = block(20, 55, 40, 80)
    body_b = block(45, 80, 40, 80)
    posterior = block(50, 70, 110, 140)

    assert name_posterior_elements([posterior], [body_a, body_b], ["L2", "L3"]) == [None]


def test_nombrar_posteriores_no_modifica_las_mascaras():
    bodies, posterior, _ = spine()
    originals = [mask.copy() for mask in bodies + posterior]

    name_posterior_elements(posterior, bodies, ["L2", "L3", "L4", "L5"])

    for actual, original in zip(bodies + posterior, originals):
        assert np.array_equal(actual, original)


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


def named_spine(disc_levels: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = [(10 + index * 30, 35 + index * 30) for index in range(len(disc_levels) + 1)]
    bodies = [block(top, bottom, 40, 80, size=280) for top, bottom in rows]
    discs = [
        block(rows[index][1], rows[index + 1][0], 40, 80, size=280)
        for index in range(len(disc_levels))
    ]
    return bodies, discs


def test_t12_l1_mas_lumbares_nombra_t12_hasta_s1():
    levels = ["T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    bodies, discs = named_spine(levels)

    assert lumbar_disc_levels(len(discs)) == levels
    assert name_vertebral_bodies(bodies, discs, levels) == ["T12", "L1", "L2", "L3", "L4", "L5", "S1"]


def test_t11_t12_mas_t12_l1_y_lumbares_nombra_t11_hasta_s1():
    levels = ["T11-T12", "T12-L1", "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]
    bodies, discs = named_spine(levels)

    assert lumbar_disc_levels(len(discs)) == levels
    assert name_vertebral_bodies(bodies, discs, levels) == ["T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1"]


def test_limite_local_recupera_cuerpos_y_posteriores_t11_t12_sin_cambiar_mediciones():
    prediction = np.zeros((220, 220), dtype=np.uint8)
    discs = [
        (11, 20, 127, 152),
        (38, 48, 119, 145),
        (68, 78, 111, 138),
        (98, 109, 102, 131),
        (132, 141, 96, 126),
        (164, 170, 92, 126),
        (187, 210, 98, 129),
    ]
    bodies = [
        (0, 15, 128, 156),
        (17, 43, 121, 151),
        (44, 73, 113, 144),
        (74, 103, 105, 137),
        (107, 134, 98, 130),
        (141, 165, 93, 125),
        (167, 197, 94, 126),
    ]
    posterior = [
        (8, 26, 169, 189),
        (39, 60, 163, 185),
        (67, 92, 155, 181),
        (97, 123, 147, 176),
        (127, 151, 140, 172),
        (152, 166, 138, 156),
        (176, 185, 142, 152),
    ]
    for top, bottom, left, right in discs:
        prediction[top:bottom, left:right] = 3
    for top, bottom, left, right in bodies + posterior:
        prediction[top:bottom, left:right] = 1

    segmentation = build_segmentation("sagittal_spider", "sagittal", prediction)
    body_levels = [item["level"] for item in segmentation["instances"] if item["label"] == "vertebra"]
    posterior_levels = [item["level"] for item in segmentation["instances"] if item["label"] == "posterior_element"]

    assert body_levels == ["T11", "T12", "L1", "L2", "L3", "L4", "L5"]
    assert posterior_levels == ["T11", "T12", "L1", "L2", "L3", "L4", "L5"]

    confidence = np.ones_like(prediction, dtype=np.float32)
    masks = build_masks(
        "sagittal_spider",
        "sagittal",
        prediction,
        confidence,
        "series-sagittal",
        7,
    )
    mask_body_levels = [item["level"] for item in masks if item["label"] == "vertebra"]
    mask_posterior_levels = [item["level"] for item in masks if item["label"] == "posterior_element"]
    assert mask_body_levels == body_levels
    assert mask_posterior_levels == posterior_levels
    assert {item["classId"] for item in masks if item["label"] in {"vertebra", "posterior_element"}} == {1}

    measurements = build_measurements(
        "sagittal_spider",
        "sagittal",
        prediction,
        confidence,
        (0.7, 0.7),
        7,
    )
    vertebral_levels = {
        item["level"]
        for item in measurements
        if item["labelKey"].startswith("vertebral_") and item["level"]
    }
    assert "T11" not in vertebral_levels
    assert "T12" not in vertebral_levels


def test_encuadre_superior_sin_ancla_suficiente_deja_niveles_en_null():
    bodies, posterior, _ = spine()
    partial_discs = [block(50, 60, 40, 80), block(90, 100, 40, 80)]
    levels = lumbar_disc_levels(len(partial_discs))
    split_bodies, split_posterior = split_vertebral_anatomical_instances(bodies + posterior, partial_discs)
    body_names = name_vertebral_bodies(split_bodies, partial_discs, levels)

    assert levels == [None, None]
    assert set(body_names) == {None}
    assert set(name_posterior_elements(split_posterior, split_bodies, body_names)) == {None}


def test_build_masks_no_compara_arrays_para_nombrar_cuerpos_sin_nivel():
    prediction = np.zeros((80, 80), dtype=np.uint8)
    prediction[10:25, 20:35] = 1
    prediction[40:55, 20:35] = 1
    confidence = np.ones_like(prediction, dtype=np.float32)

    masks = build_masks("sagittal_spider", "sagittal", prediction, confidence, "series-sagittal", 0)

    vertebra_ids = [item["id"] for item in masks if item["label"] == "vertebra"]
    assert vertebra_ids == ["mask-sagittal-vertebra-b1", "mask-sagittal-vertebra-b2"]
