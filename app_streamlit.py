import streamlit as st
import cv2
import numpy as np
import json
import os

CONFIG_PATH = 'config_faces.json'
COLOR_OK    = (34, 197, 94)
COLOR_UNK   = (239, 68, 68)

st.set_page_config(
    page_title="Reconocimiento Facial ITSE",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource(show_spinner="Cargando modelo ArcFace (~170 MB)...")
def cargar_recursos():
    from insightface.app import FaceAnalysis

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    embs_ref = {
        clase: np.array(emb, dtype=np.float32)
        for clase, emb in config['embeddings_ref'].items()
    }

    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    return app, embs_ref, config


def similitud_coseno(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def reconocer(app, embs_ref, img_bgr, umbral):
    """
    Detecta rostros y los reconoce comparando embeddings ArcFace.
    Retorna imagen anotada y lista de resultados.
    """
    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces      = app.get(img_bgr)
    resultados = []

    for face in faces:
        emb_query = face.normed_embedding
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Comparar contra todos los perfiles
        mejor_nombre = 'Desconocido'
        mejor_sim    = -1.0
        for nombre, emb_ref in embs_ref.items():
            sim = similitud_coseno(emb_query, emb_ref)
            if sim > mejor_sim:
                mejor_sim    = sim
                mejor_nombre = nombre

        if mejor_sim < umbral:
            mejor_nombre = 'Desconocido'

        es_conocido = mejor_nombre != 'Desconocido'
        color       = COLOR_OK if es_conocido else COLOR_UNK

        # Dibujar bounding box
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

        # Etiqueta
        etiqueta = f"{mejor_nombre}  {mejor_sim*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(
            etiqueta, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1
        )
        cv2.rectangle(img_rgb,
                      (x1, y1 - th - 10), (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(img_rgb, etiqueta, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        resultados.append({
            'nombre':    mejor_nombre,
            'similitud': mejor_sim,
            'conocido':  es_conocido
        })

    return img_rgb, resultados


def main():
    st.title("🎓 Reconocimiento Facial — ITSE")
    st.markdown(
        "**InsightFace ArcFace** (ResNet-100, ONNX) — "
        "red neuronal preentrenada con 5M caras."
    )

    try:
        app, embs_ref, config = cargar_recursos()
    except FileNotFoundError:
        st.error(f"No se encontró `{CONFIG_PATH}`.")
        st.info("Ejecuta `registrar_personas.py` en Colab y sube el archivo a GitHub.")
        return
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return

    clases = config.get('clases', [])

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")
        umbral = st.slider(
            "Umbral de similitud",
            min_value=0.10, max_value=0.90,
            value=float(config.get('umbral', 0.40)),
            step=0.05,
            help="Sube para ser más estricto. Baja si no reconoce a nadie."
        )
        st.markdown("---")
        st.markdown("### Personas registradas")
        for c in clases:
            st.markdown(f"✅ {c}")
        st.markdown("---")
        st.info(
            "🟢 Verde = reconocido\n\n"
            "🔴 Rojo = desconocido\n\n"
            "% = similitud con el perfil"
        )

    st.success(
        f"✅ Modelo: **ArcFace ResNet-100** (ONNX) | "
        f"{len(clases)} persona(s) registradas"
    )

    tab1, tab2 = st.tabs(["📷 Cámara en Vivo", "🖼️ Subir Imagen"])

    # ── Tab 1: Cámara ─────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            foto = st.camera_input("📸 Apunta la cámara y toma una foto")
        with col2:
            st.markdown("### 📊 Detecciones")
            info_ph = st.empty()

        if foto is not None:
            data  = np.frombuffer(foto.getvalue(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            with st.spinner("Analizando..."):
                img_proc, resultados = reconocer(app, embs_ref, frame, umbral)

            col1.image(img_proc, channels='RGB', use_container_width=True)

            if resultados:
                txt = "".join(
                    f"{'✅' if r['conocido'] else '❓'} **{r['nombre']}** "
                    f"— {r['similitud']*100:.0f}%\n\n"
                    for r in resultados
                )
                info_ph.markdown(txt)
            else:
                info_ph.markdown("_Sin rostros detectados_")

    # ── Tab 2: Subir Imagen ───────────────────────────────────
    with tab2:
        archivo = st.file_uploader(
            "Sube una foto para reconocer",
            type=['jpg', 'jpeg', 'png']
        )
        if archivo:
            data  = np.frombuffer(archivo.read(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            with st.spinner("Analizando..."):
                img_proc, resultados = reconocer(app, embs_ref, frame, umbral)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(img_proc, channels='RGB', use_container_width=True)
            with c2:
                st.markdown("### Resultados")
                if resultados:
                    for r in resultados:
                        if r['conocido']:
                            st.success(
                                f"✅ **{r['nombre']}**\n\n"
                                f"Similitud: {r['similitud']*100:.0f}%"
                            )
                        else:
                            st.warning(
                                f"❓ Desconocido\n\n"
                                f"Similitud máx: {r['similitud']*100:.0f}%"
                            )
                else:
                    st.info("No se detectaron rostros en la imagen.")


if __name__ == '__main__':
    main()
