# app.py
import streamlit as st
from PIL import Image
import io
import pandas as pd
from src.models.classifier import WasteClassifier
import time

st.set_page_config(page_title="Waste Sorting Demo", layout="centered")

st.title("Waste Sorting — TrashNet Demo")
st.markdown("Upload an image or use the camera to classify waste into categories (plastic, paper, glass, metal, cardboard, trash).")

# Load model
MODEL_PATH = "models/resnet_trashnet.pth"
clf = WasteClassifier(MODEL_PATH)

if not clf.is_ready():
    st.warning("Model not found at `models/resnet_trashnet.pth`.\n\n"
               "Please train the model using `src/models/train_model.py` or place a trained checkpoint there.\n"
               "The UI will still allow you to upload images but it won't produce predictions until the model exists.")

# Session state: history table
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload image(s) (jpg/png)", accept_multiple_files=True, type=["jpg","jpeg","png"])

    if uploaded:
        for up in uploaded:
            bytes_data = up.read()
            pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            st.image(pil, caption=f"Uploaded: {up.name}", use_column_width=True)
            if clf.is_ready():
                try:
                    preds = clf.predict(pil, topk=6)
                    df = pd.DataFrame(preds, columns=["class","probability"])
                    st.dataframe(df.style.format({"probability":"{:.3f}"}))
                    st.bar_chart(df.set_index("class")["probability"])
                    # append history
                    st.session_state.history.insert(0, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": up.name, "top1": preds[0][0], "p": float(preds[0][1])})
                except Exception as e:
                    st.error("Prediction error: " + str(e))
            else:
                st.info("Model missing — no prediction.")

with col2:
    st.subheader("Webcam (snapshot)")
    cam = st.camera_input("Take a picture with your webcam")
    if cam:
        pil_cam = Image.open(cam).convert("RGB")
        st.image(pil_cam, caption="Camera snapshot", use_column_width=True)
        if clf.is_ready():
            try:
                preds = clf.predict(pil_cam, topk=6)
                df = pd.DataFrame(preds, columns=["class","probability"])
                st.dataframe(df.style.format({"probability":"{:.3f}"}))
                st.bar_chart(df.set_index("class")["probability"])
                st.session_state.history.insert(0, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "file": "webcam", "top1": preds[0][0], "p": float(preds[0][1])})
            except Exception as e:
                st.error("Prediction error: " + str(e))
        else:
            st.info("Model missing — no prediction.")

st.markdown("---")
st.subheader("Session prediction history (this session only)")
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)
    st.table(hist_df)
else:
    st.write("No predictions yet this session.")

st.markdown("---")
st.caption("Note: For best results, supply clean, centered photos of a single waste item. Lighting and background affect performance.")
