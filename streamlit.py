"""Skin disease screening app (single-image inference).

Run:
    .\\.venv\\Scripts\\activate
    streamlit run streamlit.py
"""

from pathlib import Path
from typing import List, Tuple, Optional
import json

import numpy as np
import streamlit as st
import tensorflow as tf
import gdown
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Chickenpox",
    "Cowpox",
    "Dermatofibroma",
    "Healthy",
    "HFMD",
    "Measles",
    "Melanocytic nevi",
    "unknown"
]

CLASS_INFO = {
    "Actinic keratoses": "Rough, scaly patches from sun exposure; can be precancerous.",
    "Basal cell carcinoma": "Common skin cancer type; usually grows slowly.",
    "Benign keratosis-like lesions": "Usually non-cancerous skin growths.",
    "Chickenpox": "Viral illness causing itchy rash and blisters.",
    "Cowpox": "Rare poxvirus infection; lesions can appear on skin.",
    "Dermatofibroma": "Common benign skin nodule.",
    "Healthy": "No obvious lesion pattern detected by this model.",
    "HFMD": "Hand, foot, and mouth disease; viral rash condition.",
    "Measles": "Highly contagious viral illness with rash and fever.",
    "Melanocytic nevi": "Moles; usually benign but should be monitored.",
}

PREVENTIVE_MEASURES = {
    "Actinic keratoses": (
        "Use broad-spectrum sunscreen daily, avoid peak UV hours, wear protective clothing, "
        "and schedule regular skin checks."
    ),
    "Basal cell carcinoma": (
        "Limit sun exposure, use SPF 30+ sunscreen, avoid tanning beds, and seek early "
        "dermatology evaluation for new or non-healing lesions."
    ),
    "Benign keratosis-like lesions": (
        "Protect skin from chronic sun damage, avoid scratching lesions, and monitor for "
        "changes in color, size, or shape."
    ),
    "Chickenpox": (
        "Follow vaccination schedules, maintain hygiene, avoid close contact during active "
        "infection, and keep nails trimmed to reduce skin infection risk."
    ),
    "Cowpox": (
        "Use gloves when handling potentially infected animals, wash hands thoroughly, and "
        "avoid touching skin lesions."
    ),
    "Dermatofibroma": (
        "Reduce repeated skin trauma, avoid picking skin nodules, and monitor persistent "
        "or changing lesions."
    ),
    "Healthy": (
        "Continue skin protection habits: sunscreen, hydration, gentle skincare, and regular "
        "self-examination for new changes."
    ),
    "HFMD": (
        "Frequent handwashing, disinfect high-touch surfaces, avoid sharing utensils, and "
        "limit close contact during infection."
    ),
    "Measles": (
        "Ensure MMR vaccination, practice respiratory hygiene, isolate infected individuals, "
        "and seek prompt medical care for symptoms."
    ),
    "Melanocytic nevi": (
        "Use sun protection, avoid excessive UV exposure, and monitor moles using ABCDE "
        "warning signs with periodic dermatologist checks."
    ),
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
DEFAULT_MODEL_PATH = str(MODEL_DIR / "efficient_skin_1.keras")
MODEL_DRIVE_URL = "https://drive.google.com/file/d/18yszajZvwtwtj_BBCKJ0g7cCF5Qbd_nF/view?usp=drive_link"
TEMP_DRIVE_URL = "https://drive.google.com/file/d/1a823eaF-vkGdVE_Iq_CGl9ey1lz1qu1W/view?usp=drive_link"
IMG_SIZE = 224
CONF_THRESHOLD_DEFAULT = 0.75
CONF_THRESHOLD_NEVI = 0.75
CONF_THRESHOLD_BKL = 0.84
MARGIN_THRESHOLD = 0.28
DERM_FALLBACK_MIN_PROB = 0.15
DERM_FALLBACK_MAX_GAP = 0.20
NEVI_FALLBACK_MIN_PROB = 0.20
NEVI_FALLBACK_MAX_GAP = 0.20


@st.cache_resource
def load_model_cached(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data
def load_temperature_cached(temperature_path: str) -> float:
    path = Path(temperature_path)
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("temperature", 1.0))
    except Exception:
        return 1.0


def _extract_drive_id(url: str) -> Optional[str]:
    if "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    if "id=" in url:
        return url.split("id=")[1].split("&")[0]
    return None


def _looks_like_html(sample: bytes) -> bool:
    head = sample.lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def _validate_download(dest_path: Path) -> bool:
    if not dest_path.exists() or dest_path.stat().st_size < 1024:
        return False
    with dest_path.open("rb") as f:
        sample = f.read(2048)
    if _looks_like_html(sample):
        return False
    return True


def download_model_from_drive(url: str, dest_path: Path) -> bool:
    file_id = _extract_drive_id(url)
    if not file_id:
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(gdown_url, str(dest_path), quiet=False)
        return _validate_download(dest_path)
    except Exception:
        return False


def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return probs
    safe_probs = np.clip(probs, 1e-8, 1.0)
    scaled = np.power(safe_probs, 1.0 / temperature)
    denom = np.sum(scaled)
    if denom <= 0:
        return probs
    return scaled / denom


def preprocess_image(uploaded_file, img_size: int = IMG_SIZE) -> np.ndarray:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid or corrupted image file.") from exc

    image = image.resize((img_size, img_size))
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_top_k(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_names: List[str],
    temperature: float = 1.0,
    k: int = 3,
) -> List[Tuple[str, float]]:
    probs = model.predict(image_batch, verbose=0)[0]
    probs = apply_temperature_scaling(probs, temperature)
    top_idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in top_idx]


def get_prediction_quality(top3: List[Tuple[str, float]]) -> Tuple[bool, str]:
    best_label, best_prob = top3[0]
    second_label, second_prob = top3[1]

    required_conf_by_class = {
        "Melanocytic nevi": CONF_THRESHOLD_NEVI,
        "Benign keratosis-like lesions": CONF_THRESHOLD_BKL,
    }
    required_conf = required_conf_by_class.get(best_label, CONF_THRESHOLD_DEFAULT)
    margin = best_prob - second_prob

    conf_ok = best_prob >= required_conf
    margin_ok = margin >= MARGIN_THRESHOLD

    if conf_ok and margin_ok:
        return True, ""

    if not conf_ok and not margin_ok:
        return (
            False,
            (
                "Low confidence and close probabilities between top classes "
                f"({best_label} vs {second_label})."
            ),
        )
    if not conf_ok:
        return False, f"Prediction confidence is below {required_conf * 100:.0f}% threshold."
    return False, f"Top-1 and Top-2 are too close (margin {margin * 100:.2f}%)."


def apply_confusion_fallback(top3: List[Tuple[str, float]]) -> Tuple[str, float, str]:
    best_label, best_prob = top3[0]
    prob_map = {label: prob for label, prob in top3}

    # Dermatofibroma is often confused with AK/Nevi in this model.
    derm_prob = prob_map.get("Dermatofibroma")
    if (
        best_label in {"Actinic keratoses", "Melanocytic nevi"}
        and derm_prob is not None
        and derm_prob >= DERM_FALLBACK_MIN_PROB
        and (best_prob - derm_prob) <= DERM_FALLBACK_MAX_GAP
    ):
        return (
            "Dermatofibroma",
            float(derm_prob),
            "Adjusted using confusion rule (Dermatofibroma fallback).",
        )

    # Melanocytic nevi is often confused with BKL/BCC in this model.
    nevi_prob = prob_map.get("Melanocytic nevi")
    if (
        best_label in {"Benign keratosis-like lesions", "Basal cell carcinoma"}
        and nevi_prob is not None
        and nevi_prob >= NEVI_FALLBACK_MIN_PROB
        and (best_prob - nevi_prob) <= NEVI_FALLBACK_MAX_GAP
    ):
        return (
            "Melanocytic nevi",
            float(nevi_prob),
            "Adjusted using confusion rule (Melanocytic nevi fallback).",
        )

    return best_label, best_prob, ""


def render_homepage() -> None:
    st.markdown(
        """
        <div class="content-shell">
        <div class="section-card">
            <h2 class="section-title">Application Overview</h2>
            <p class="lead">
                This application analyzes uploaded skin images using a trained deep-learning model
                and returns the most likely class with confidence scores.
            </p>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### How it works")
        st.markdown("1. Upload a skin image in JPG, JPEG, PNG, or WEBP format.")
        st.markdown("2. The model preprocesses the image and runs inference.")
        st.markdown("3. Results are ranked and displayed with confidence scores.")
    with c2:
        st.markdown("### What you get")
        st.markdown("- Predicted condition name")
        st.markdown("- Confidence percentage")
        st.markdown("- Top-3 probability table")
        st.markdown("- Disease description and preventive guidance")


def render_about() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h2 class="section-title">About This Tool</h2>
            <p class="lead">
                Diseases used in this project with their short descriptions and preventive measures.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = []
    for disease in CLASS_NAMES:
        if disease == "unknown":
            continue
        cards.append(
            (
                disease,
                CLASS_INFO.get(disease, "No description available."),
                PREVENTIVE_MEASURES.get(
                    disease, "Consult a healthcare professional for guidance."
                ),
            )
        )

    card_html = "\n".join(
        f"""
        <div class="note-card">
            <div class="note-pill">Condition Profile</div>
            <h3 class="note-title">{disease}</h3>
            <div class="note-block">
                <div class="note-label">Description</div>
                <div class="note-text">{desc}</div>
            </div>
            <div class="note-block">
                <div class="note-label">Preventive Measures</div>
                <div class="note-text">{prev}</div>
            </div>
        </div>
        """
        for disease, desc, prev in cards
    )

    st.markdown(
        f"""
        <style>
        .note-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.2rem;
            margin-top: 0.6rem;
        }}
        .note-card {{
            background: #ffffff;
            border: 1px solid #d7e3ef;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 8px 18px rgba(15, 39, 66, 0.08);
        }}
        .note-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: #e7f7ef;
            color: #0f6b3d;
            font-weight: 600;
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
        }}
        .note-title {{
            margin: 0 0 0.7rem 0;
            color: #143657;
            font-size: 1.35rem;
        }}
        .note-block {{
            background: #f7fbff;
            border: 1px solid #d9e6f3;
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.6rem;
        }}
        .note-label {{
            font-weight: 700;
            color: #1c3f63;
            font-size: 0.85rem;
            margin-bottom: 0.2rem;
        }}
        .note-text {{
            color: #2b3f53;
            font-size: 0.9rem;
            line-height: 1.35;
        }}
        </style>
        <div class="note-grid">
            {card_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

   # st.warning(
      #  "Medical disclaimer: Predictions can be incorrect. For severe, worsening, or persistent"
      #  " symptoms, consult a dermatologist or healthcare professional."
    #)


def render_prediction() -> None:
    st.markdown(
        """
        <div class="section-card">
            <h2 class="section-title">Prediction</h2>
            <p class="lead">Upload an image to get the predicted class, confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config_col, upload_col = st.columns([1, 1.7], gap="large")
    model_path = DEFAULT_MODEL_PATH
    temp_path_default = str(Path(model_path).with_suffix(".temperature.json"))
    temperature_path = temp_path_default

    model_file = Path(model_path)
    if not model_file.exists():
        with st.spinner("Model file missing. Downloading..."):
            ok = download_model_from_drive(MODEL_DRIVE_URL, model_file)
        if not ok or not model_file.exists():
            st.error(
                "Model file not found and download failed. "
                "Please check the Drive link or add the model file locally."
            )
            return
    else:
        if not _validate_download(model_file):
            model_file.unlink(missing_ok=True)
            with st.spinner("Model file invalid. Re-downloading..."):
                ok = download_model_from_drive(MODEL_DRIVE_URL, model_file)
            if not ok or not model_file.exists():
                st.error(
                    "Model file is invalid and re-download failed. "
                    "Please check the Drive link or add the model file locally."
                )
                return

    temperature_file = Path(temperature_path)
    if not temperature_file.exists():
        with st.spinner("Temperature file missing. Downloading..."):
            download_model_from_drive(TEMP_DRIVE_URL, temperature_file)

    try:
        with st.spinner("Loading model..."):
            model = load_model_cached(str(model_file))
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return

    temperature = load_temperature_cached(temperature_path)
    if temperature != 1.0:
        st.caption(f"Using calibrated temperature: {temperature:.4f}")

    with upload_col:
        uploaded = st.file_uploader(
            "Upload a skin image",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=False,
        )

    if uploaded is None:
        st.info("Upload an image to start prediction.")
        return

    try:
        preview = Image.open(uploaded).convert("RGB")
        show_col, result_col = st.columns([1.1, 1], gap="large")
        with show_col:
            st.image(preview, caption="Uploaded image", use_container_width=True)
        uploaded.seek(0)
        image_batch = preprocess_image(uploaded)
    except ValueError as exc:
        st.warning(str(exc))
        return
    except Exception:
        st.warning("Could not read image. Please upload a valid image file.")
        return

    with st.spinner("Running prediction..."):
        top3 = predict_top_k(model, image_batch, CLASS_NAMES, temperature=temperature, k=3)

    best_label, best_prob = top3[0]
    final_label, final_prob, adjustment_note = apply_confusion_fallback(top3)
    top3_for_quality = top3.copy()
    if final_label != best_label:
        top3_for_quality = [(final_label, final_prob)] + [item for item in top3 if item[0] != final_label]

    confident, quality_message = get_prediction_quality(top3_for_quality)
    with result_col:
        st.metric("Predicted class", final_label)
        if final_label != "unknown":
            st.metric("Confidence", f"{final_prob * 100:.2f}%")
        if adjustment_note:
            st.caption(adjustment_note)
        if not confident:
            st.warning(
                f"Uncertain prediction: {quality_message} Please consult a doctor for proper diagnosis."
            )
        else:
            info_text = CLASS_INFO.get(final_label)
            if info_text:
                st.info(f"Disease description: {info_text}")
            preventive_text = PREVENTIVE_MEASURES.get(final_label)
            if preventive_text:
                st.success(f"Preventive measures: {preventive_text}")

    if final_label != "unknown":
        st.info(
            "If symptoms are severe, worsening, or persistent, consult a dermatologist or healthcare professional."
        )


def main() -> None:
    st.set_page_config(page_title="Skin Disease App", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background: #e9edf3;
        }
        .main .block-container {
            padding-top: 0.7rem;
            max-width: 1200px;
        }
        .hero-banner {
            background: linear-gradient(120deg, #0a4f8f, #1c88c7);
            padding: 1.5rem 1.5rem 1rem 1.5rem;
            border-radius: 0 0 30px 30px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 8px 18px rgba(0, 47, 93, 0.18);
        }
        .hero-banner h1 {
            margin: 0;
            font-size: 2.25rem;
            line-height: 1.15;
            letter-spacing: 0.3px;
        }
        .hero-banner p {
            margin-top: 0.6rem;
            font-size: 1.05rem;
            color: rgba(255,255,255,0.95);
        }
        .section-card {
            background: #e9edf3;
            border-radius: 14px;
            padding: 0.2rem 0 0.4rem 0;
        }
        .content-shell {
            max-width: 100%;
            margin: 0 0 0.6rem 0;
        }
        .section-title {
            font-weight: 700;
            color: #0f2742;
            margin-bottom: 0.4rem;
            font-size: 2rem;
        }
        .lead {
            color: #243b53;
            font-size: 0.95rem;
        }
        div[role="radiogroup"] {
            display: flex;
            justify-content: center;
            gap: 0.7rem;
            margin: 0.35rem 0 0.85rem 0;
        }
        div[role="radiogroup"] > label {
            background: #e8f0f7;
            border: 1px solid #b7cbe0;
            color: #214769;
            border-radius: 999px;
            padding: 0.1rem 0.9rem;
        }
        div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
            display: none;
        }
        div[role="radiogroup"] > label:has(input:checked) {
            background: #1f6fa8;
            color: #ffffff;
            border-color: #1f6fa8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-banner">
            <h1>Skin Disease Screening</h1>
            <p>AI-powered skin image analysis to estimate likely conditions with confidence scores.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Homepage", "About", "Prediction"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Homepage":
        render_homepage()
    elif page == "About":
        render_about()
    else:
        render_prediction()


if __name__ == "__main__":
    main()
