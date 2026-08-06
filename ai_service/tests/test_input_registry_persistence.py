"""El registro de inputs sobrevive a un reinicio del proceso.

Motivo concreto, verificado el 2026-08-06 sobre una corrida real: minutos despues de
completarse un estudio, sus dos series axiales devolvian 404. El registro vivia solo en
memoria del proceso, asi que cualquier reinicio -o un segundo worker- invalidaba todo
inputId ya entregado al cliente, aunque el archivo siguiera en disco.

Rompe P10.6 de forma directa: la clasificacion subarticular necesita resolver el inputId
de la serie axial sobre la que el medico marco el receso.
"""
import pytest

import pfi_ai_service.input_registry as registry


@pytest.fixture()
def clean_registry(tmp_path, monkeypatch):
    """Registro vacio sobre un directorio propio, restaurado al terminar.

    No se recarga el modulo con importlib: otros modulos ya importaron sus nombres, y
    reemplazarlo les deja referencias a un registro distinto del que usa el servicio.
    Se guarda y restaura el estado global, que es lo que el test realmente necesita.
    """
    monkeypatch.setenv("PFI_UPLOAD_DIR", str(tmp_path / "inputs"))
    previous = dict(registry._INPUT_REGISTRY)
    previous_loaded = registry._REGISTRY_LOADED
    registry._INPUT_REGISTRY.clear()
    registry._REGISTRY_LOADED = False
    yield registry
    registry._INPUT_REGISTRY.clear()
    registry._INPUT_REGISTRY.update(previous)
    registry._REGISTRY_LOADED = previous_loaded


def restart_process(module):
    """Lo que le pasa al registro cuando el contenedor se recrea: memoria en cero."""
    module._INPUT_REGISTRY.clear()
    module._REGISTRY_LOADED = False


def _record(input_id, path, plane="axial", analyzable=True):
    return registry.InputRecord(
        input_id=input_id,
        case_id="CASE-1",
        plane=plane,
        path=path,
        format="dcm",
        size=123,
        source_key="upload",
        analyzable=analyzable,
    )


def test_un_input_registrado_se_resuelve_tras_reiniciar_el_proceso(clean_registry, tmp_path):
    serie = tmp_path / "serie"
    serie.mkdir()
    (serie / "1.dcm").write_bytes(b"x")
    clean_registry.remember_input(_record("inp_abc", serie))

    restart_process(clean_registry)
    assert clean_registry._INPUT_REGISTRY == {}

    recuperado = clean_registry.resolve_registered_input("inp_abc")
    assert recuperado.input_id == "inp_abc"
    assert recuperado.plane == "axial"
    assert recuperado.path == serie


def test_se_conservan_los_campos_que_deciden_si_una_serie_puede_inferir(clean_registry, tmp_path):
    serie = tmp_path / "solo_vista"
    serie.mkdir()
    clean_registry.remember_input(_record("inp_view", serie, plane="sagittal", analyzable=False))

    restart_process(clean_registry)
    recuperado = clean_registry.resolve_registered_input("inp_view")

    # Si `analyzable` no sobreviviera, una serie de solo-vista volveria a ser elegible
    # como entrada de inferencia despues de cada reinicio.
    assert recuperado.analyzable is False
    assert recuperado.plane == "sagittal"


def test_un_input_cuyo_archivo_desaparecio_no_se_rehidrata(clean_registry, tmp_path):
    serie = tmp_path / "efimera"
    serie.mkdir()
    clean_registry.remember_input(_record("inp_gone", serie))
    serie.rmdir()

    restart_process(clean_registry)

    with pytest.raises(clean_registry.InputRegistryError) as exc:
        clean_registry.resolve_registered_input("inp_gone")
    assert exc.value.status_code == 404


def test_un_indice_corrupto_no_tumba_el_servicio(clean_registry, tmp_path):
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "registry.json").write_text("{no es json", encoding="utf-8")

    restart_process(clean_registry)

    # Se comporta como un registro vacio, no como una excepcion al arrancar.
    with pytest.raises(clean_registry.InputRegistryError):
        clean_registry.resolve_registered_input("inp_cualquiera")


def test_varios_inputs_conviven_en_el_indice(clean_registry, tmp_path):
    for name in ("a", "b", "c"):
        path = tmp_path / name
        path.mkdir()
        clean_registry.remember_input(_record(f"inp_{name}", path))

    restart_process(clean_registry)

    for name in ("a", "b", "c"):
        assert clean_registry.resolve_registered_input(f"inp_{name}").path == tmp_path / name
