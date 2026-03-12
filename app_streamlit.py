"""
APP STREAMLIT - Reconocimiento Facial con DeepFace (ArcFace)
=============================================================
Reconocimiento en tiempo real comparando embeddings ArcFace.

Archivos necesarios:
  - config_deepface.json   (generado por registrar_personas.py)

Requisitos:
  pip install streamlit deepface tf-keras opencv-python numpy

Ejecutar:
  streamlit run app_streamlit.py
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
import os
import urllib.request
from deepface import DeepFace
from deepface.modules import verification

CONFIG_PATH = 'config_deepface.json'
PROTO_PATH  = 'deploy.prototxt'
DNN_PATH    = 'res10_300x300_ssd.caffemodel'
COLOR_OK    = (34, 197, 94)
COLOR_UNK   = (239, 68, 68)

st.set_page_config(
    page_title="Reconocimiento Facial ITSE",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def cargar_recursos():
    # Cargar config con embeddings de referencia
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Convertir embeddings a numpy
    embs_ref = {
        clase: np.array(emb, dtype=np.float32)
        for clase, emb in config['embeddings_ref'].items()
    }

    # Detector de rostros (OpenCV DNN)
    proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    if not os.path.exists(PROTO_PATH):
        urllib.request.urlretrieve(proto_url, PROTO_PATH)
    if not os.path.exists(DNN_PATH):
        urllib.request.urlretrieve(model_url, DNN_PATH)
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, DNN_PATH)

    # Pre-calentar ArcFace con una imagen dummy
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    try:
        DeepFace.represent(dummy, model_name=config['modelo'],
                           enforce_detection=False)
    except Exception:
        pass

    return embs_ref, config, net


def detectar_rostros(net, img_bgr, umbral_det=0.5):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()
    rostros = []
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf > umbral_det:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            rostros.append((x1, y1, x2, y2, conf))
    return rostros


def recortar_rostro(img_bgr, x1, y1, x2, y2):
    bw, bh = x2 - x1, y2 - y1
    mg  = int(min(bw, bh) * 0.15)
    x1  = max(0, x1 - mg);  y1 = max(0, y1 - mg)
    x2  = min(img_bgr.shape[1], x2 + mg)
    y2  = min(img_bgr.shape[0], y2 + mg)
    return img_bgr[y1:y2, x1:x2]


def similitud_coseno(a, b):
    """Similitud coseno entre dos vectores (1 = idéntico, 0 = opuesto)."""
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def reconocer_rostro(rostro_bgr, embs_ref, config):
    """
    Extrae embedding ArcFace del rostro y lo compara
    contra todos los perfiles de referencia.
    Retorna (nombre, similitud).
    """
    modelo  = config.get('modelo', 'ArcFace')
    umbral  = config.get('umbral', 0.40)

    try:
        resultado = DeepFace.represent(
            img_path          = rostro_bgr,
            model_name        = modelo,
            detector_backend  = 'skip',   # ya recortamos el rostro
            enforce_detection = False
        )
        emb_query = np.array(resultado[0]['embedding'], dtype=np.float32)
    except Exception:
        return 'Desconocido', 0.0

    mejor_nombre = 'Desconocido'
    mejor_sim    = -1.0

    for nombre, emb_ref in embs_ref.items():
        sim = similitud_coseno(emb_query, emb_ref)
        if sim > mejor_sim:
            mejor_sim    = sim
            mejor_nombre = nombre

    if mejor_sim < umbral:
        return 'Desconocido', mejor_sim
    return mejor_nombre, mejor_sim


def procesar_frame(frame, embs_ref, config, net, umbral_override):
    img_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rostros_bb = detectar_rostros(net, frame)
    resultados = []

    for (x1, y1, x2, y2, conf_det) in rostros_bb:
        recorte = recortar_rostro(frame, x1, y1, x2, y2)
        if recorte.size == 0:
            continue

        # Sobrescribir umbral con el del slider
        config_tmp = dict(config)
        config_tmp['umbral'] = umbral_override

        nombre, similitud = reconocer_rostro(recorte, embs_ref, config_tmp)
        es_conocido = nombre != 'Desconocido'
        color       = COLOR_OK if es_conocido else COLOR_UNK

        # Dibujar
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{nombre}  {similitud*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.rectangle(img_rgb, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_rgb, etiqueta, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        resultados.append({
            'nombre': nombre, 'similitud': similitud,
            'conocido': es_conocido
        })

    return img_rgb, resultados


def main():
    st.title("🎓 Reconocimiento Facial — ITSE")
    st.markdown(
        "**DeepFace ArcFace** — red neuronal preentrenada con millones de caras. "
        "Reconocimiento por similitud de embeddings."
    )

    try:
        embs_ref, config, net = cargar_recursos()
    except FileNotFoundError:
        st.error(f"No se encontró '{CONFIG_PATH}'.")
        st.info("Ejecuta primero `registrar_personas.py` en Colab y sube el archivo.")
        return
    except Exception as e:
        st.error(f"Error al cargar: {e}")
        return

    clases = config.get('clases', [])

    with st.sidebar:
        st.header("⚙️ Configuración")
        umbral = st.slider(
            "Umbral de similitud (coseno)",
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
        f"✅ Modelo: **{config.get('modelo', 'ArcFace')}** | "
        f"{len(clases)} persona(s) registradas"
    )

    tab1, tab2 = st.tabs(["📷 Cámara en Vivo", "🖼️ Subir Imagen"])

    # ── Cámara ────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            btn_start = st.button("▶ Iniciar cámara", type="primary")
            btn_stop  = st.button("⏹ Detener")
            frame_ph  = st.empty()
        with col2:
            st.markdown("### 📊 Detecciones")
            info_ph = st.empty()

        if btn_start:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("No se pudo abrir la cámara.")
                return
            st.session_state['running'] = True
            t_prev = time.time()

            while st.session_state.get('running', False):
                ret, frame = cap.read()
                if not ret:
                    break

                img_proc, resultados = procesar_frame(
                    frame, embs_ref, config, net, umbral
                )
                t_now  = time.time()
                fps    = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now
                cv2.putText(img_proc, f"FPS: {fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                frame_ph.image(img_proc, channels='RGB', use_container_width=True)

                if resultados:
                    txt = "".join(
                        f"{'✅' if r['conocido'] else '❓'} **{r['nombre']}** "
                        f"— {r['similitud']*100:.0f}%\n\n"
                        for r in resultados
                    )
                    info_ph.markdown(txt)
                else:
                    info_ph.markdown("_Sin rostros detectados_")

                if btn_stop:
                    st.session_state['running'] = False
                    break
            cap.release()

    # ── Imagen ────────────────────────────────────────────────
    with tab2:
        archivo = st.file_uploader("Sube una foto", type=['jpg', 'jpeg', 'png'])
        if archivo:
            data  = np.frombuffer(archivo.read(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_proc, resultados = procesar_frame(
                frame, embs_ref, config, net, umbral
            )
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
                    st.info("No se detectaron rostros.")


if __name__ == '__main__':
    main()
