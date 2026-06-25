"""Demo inicial Streamlit del MVP.

Esta app es un placeholder de producto. La inferencia real se conectará cuando
exista un checkpoint entrenado y un pipeline estable.
"""

import streamlit as st


st.set_page_config(page_title="PFI RM Lumbar MVP", layout="wide")

st.title("MVP - Análisis asistido de RM lumbar sagital")
st.warning(
    "Prototipo académico. No emite diagnósticos, no recomienda tratamientos "
    "y no reemplaza el criterio profesional."
)

st.markdown(
    """
    Esta interfaz será utilizada para:

    1. cargar o seleccionar una RM lumbar sagital de ejemplo;
    2. ejecutar segmentación automática;
    3. visualizar máscaras superpuestas;
    4. revisar mediciones geométricas derivadas de máscaras;
    5. exportar una salida estructurada editable.
    """
)

st.info("Próximo paso: conectar carga de imagen, modelo entrenado y visualización de overlays.")
