import streamlit as st
import cv2
import numpy as np
import json
import os

CONFIG_PATH = 'config_faces.json'

st.set_page_config(
    page_title="Face Recognition — ITSE",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a;
    color: #e8e8e8;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #222;
}

[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

h1, h2, h3 {
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: -0.02em;
}

.block-container { padding: 2rem 2.5rem; max-width: 1200px; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #222;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #666 !important;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 0.75rem 1.5rem;
    border: none;
    text-transform: uppercase;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #e8e8e8 !important;
    border-bottom: 1px solid #e8e8e8 !important;
}

.stButton > button {
    background: #e8e8e8;
    color: #0a0a0a;
    border: none;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    text-transform: uppercase;
    transition: background 0.15s;
    width: 100%;
}

.stButton > button:hover { background: #ffffff; }

.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }

[data-testid="stFileUploader"] {
    background: #111;
    border: 1px dashed #333;
    border-radius: 4px;
    padding: 1rem;
}

[data-testid="stCameraInput"] label { color: #666 !important; font-size: 0.8rem; }

.result-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}

.result-card.known { border-left: 3px solid #4ade80; }
.result-card.unknown { border-left: 3px solid #f87171; }

.result-name {
    font-family: 'DM Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #e8e8e8;
}

.result-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.25rem;
}

.person-tag {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    margin: 0.15rem;
}

.stat-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    text-align: center;
    margin-bottom: 1rem;
}

.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    color: #e8e8e8;
}

.stat-label {
    font-size: 0.72rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.1rem;
}

hr { border-color: #1e1e1e; margin: 1.25rem 0; }

.stSpinner > div { color: #666 !important; }

[data-testid="stImage"] img { border-radius: 4px; }

p, label, .stMarkdown { color: #888 !important; }
.stMarkdown strong { color: #e8e8e8 !important; }

[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #e8e8e8 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_recursos():
    from insightface.app import FaceAnalysis

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)

    embs_ref = {
        clase: np.array(emb, dtype=np.float32)
        for clase, emb in config['embeddings_ref'].items()
    }

    modelo = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    modelo.prepare(ctx_id=0, det_size=(640, 640))

    return modelo, embs_ref, config


def similitud_coseno(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def procesar(modelo, embs_ref, img_bgr, umbral):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces   = modelo.get(img_bgr)
    resultados = []

    for face in faces:
        emb   = face.normed_embedding
        x1, y1, x2, y2 = face.bbox.astype(int)

        mejor_nombre = 'Desconocido'
        mejor_sim    = -1.0
        for nombre, ref in embs_ref.items():
            sim = similitud_coseno(emb, ref)
            if sim > mejor_sim:
                mejor_sim    = sim
                mejor_nombre = nombre

        if mejor_sim < umbral:
            mejor_nombre = 'Desconocido'

        conocido = mejor_nombre != 'Desconocido'
        color    = (74, 222, 128) if conocido else (248, 113, 113)

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        etiqueta = f"{mejor_nombre}  {mejor_sim*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(img_rgb, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img_rgb, etiqueta, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (10, 10, 10), 1)

        resultados.append({
            'nombre':    mejor_nombre,
            'similitud': mejor_sim,
            'conocido':  conocido
        })

    return img_rgb, resultados


def tarjeta_resultado(r):
    cls   = "known" if r['conocido'] else "unknown"
    nombre = r['nombre']
    score  = f"{r['similitud']*100:.1f}% similitud"
    st.markdown(f"""
    <div class="result-card {cls}">
        <div class="result-name">{nombre}</div>
        <div class="result-score">{score}</div>
    </div>
    """, unsafe_allow_html=True)


# ── App ────────────────────────────────────────────────────────

try:
    modelo, embs_ref, config = cargar_recursos()
    cargado = True
except FileNotFoundError:
    cargado = False
except Exception as e:
    st.error(str(e))
    cargado = False

clases = config.get('clases', []) if cargado else []

# Sidebar
with st.sidebar:
    st.markdown("### Face Recognition")
    st.markdown("---")

    if cargado:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{len(clases)}</div><div class="stat-label">personas</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="stat-value">512</div><div class="stat-label">dim emb.</div></div>', unsafe_allow_html=True)

    st.markdown("**Umbral de similitud**")
    umbral = st.slider(
        "", min_value=0.10, max_value=0.90,
        value=float(config.get('umbral', 0.40)) if cargado else 0.40,
        step=0.05, label_visibility="collapsed"
    )
    st.caption("Sube el valor para ser mas estricto.")

    st.markdown("---")
    st.markdown("**Personas registradas**")
    if cargado:
        tags = "".join(f'<span class="person-tag">{c}</span>' for c in clases)
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.caption("Modelo no cargado.")

    st.markdown("---")
    st.caption("ArcFace ResNet-100 · ONNX · InsightFace")


# Main
st.markdown("# Face Recognition")
st.markdown("ArcFace ResNet-100 — reconocimiento por similitud de embeddings.")
st.markdown("---")

if not cargado:
    st.warning("No se encontro `config_faces.json`. Ejecuta `registrar_personas.py` en Colab y sube el archivo al repositorio.")
    st.stop()

tab1, tab2 = st.tabs(["Camara", "Subir imagen"])

with tab1:
    col_img, col_res = st.columns([3, 1])
    with col_img:
        foto = st.camera_input("", label_visibility="collapsed")
    with col_res:
        st.markdown("**Resultado**")
        resultado_ph = st.empty()

    if foto is not None:
        data  = np.frombuffer(foto.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        with st.spinner("Procesando..."):
            img_proc, resultados = procesar(modelo, embs_ref, frame, umbral)
        col_img.image(img_proc, channels='RGB', use_container_width=True)
        with resultado_ph.container():
            if resultados:
                for r in resultados:
                    tarjeta_resultado(r)
            else:
                st.caption("Sin rostros detectados.")

with tab2:
    col_img2, col_res2 = st.columns([3, 1])
    with col_img2:
        archivo = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    with col_res2:
        st.markdown("**Resultado**")
        resultado_ph2 = st.empty()

    if archivo is not None:
        data  = np.frombuffer(archivo.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        with st.spinner("Procesando..."):
            img_proc, resultados = procesar(modelo, embs_ref, frame, umbral)
        col_img2.image(img_proc, channels='RGB', use_container_width=True)
        with resultado_ph2.container():
            if resultados:
                for r in resultados:
                    tarjeta_resultado(r)
            else:
                st.caption("Sin rostros detectados.")
