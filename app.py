import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DeepGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    try:

        model = tf.keras.models.load_model(
            "deepfake_final_model.keras"
        )

        return model

    except Exception as e:

        st.error("❌ Unable to load the model.")

        st.code(str(e))

        return None


model = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ DeepGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Deepfake Image Detection System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# ABOUT
# =========================================================

with st.expander("ℹ️ About DeepGuard AI"):

    st.write("""
    DeepGuard AI is an AI-based deepfake image detection
    system.

    The system uses a trained MobileNetV2 deep learning model
    to classify an uploaded image as REAL or FAKE.

    The uploaded image is resized to 224 × 224 pixels before
    being passed to the trained model.
    """)


# =========================================================
# MODEL STATUS
# =========================================================

if model is not None:

    st.success("🟢 Model Loaded Successfully")

else:

    st.error("🔴 Model Loading Failed")


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Upload JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # -----------------------------------------------------
    # Resize
    # -----------------------------------------------------

    image = image.resize(
        (224, 224)
    )


    # -----------------------------------------------------
    # Convert to NumPy
    # -----------------------------------------------------

    image_array = np.array(
        image
    ).astype(
        np.float32
    )


    # -----------------------------------------------------
    # Add batch dimension
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]


    return float(prediction)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # -------------------------------------------------
        # Display image
        # -------------------------------------------------

        st.subheader("🖼️ Uploaded Image")

        st.image(
            image,
            caption="Image selected for analysis",
            use_container_width=True
        )


        # -------------------------------------------------
        # Analyze Button
        # -------------------------------------------------

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            if model is None:

                st.error(
                    "Model is not available."
                )

            else:

                # -----------------------------------------
                # Prediction
                # -----------------------------------------

                with st.spinner(
                    "🤖 AI is analyzing the image..."
                ):

                    prediction = predict_image(
                        image
                    )


                # -----------------------------------------
                # Probability
                # -----------------------------------------

                real_probability = prediction * 100

                fake_probability = (
                    1 - prediction
                ) * 100


                # -----------------------------------------
                # Classification
                # -----------------------------------------

                if prediction >= 0.5:

                    result = "REAL"

                    confidence = real_probability

                else:

                    result = "FAKE"

                    confidence = fake_probability


                # -----------------------------------------
                # Result
                # -----------------------------------------

                st.divider()

                st.subheader(
                    "🎯 Detection Result"
                )


                if result == "REAL":

                    st.success(
                        "✅ REAL IMAGE"
                    )

                else:

                    st.error(
                        "⚠️ FAKE / DEEPFAKE IMAGE"
                    )


                # -----------------------------------------
                # Confidence
                # -----------------------------------------

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )


                st.progress(
                    int(
                        min(
                            max(
                                confidence,
                                0
                            ),
                            100
                        )
                    )
                )


                # -----------------------------------------
                # Probability Details
                # -----------------------------------------

                st.subheader(
                    "📊 Prediction Analysis"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "REAL Probability",
                        f"{real_probability:.2f}%"
                    )


                with col2:

                    st.metric(
                        "FAKE Probability",
                        f"{fake_probability:.2f}%"
                    )


                # -----------------------------------------
                # Confidence Interpretation
                # -----------------------------------------

                if confidence >= 80:

                    st.success(
                        "🟢 High confidence prediction."
                    )

                elif confidence >= 50:

                    st.warning(
                        "🟡 Moderate confidence prediction."
                    )

                else:

                    st.warning(
                        "🟠 Low confidence prediction."
                    )


                # -----------------------------------------
                # Technical Details
                # -----------------------------------------

                with st.expander(
                    "🔧 Technical Details"
                ):

                    st.write(
                        "**Model:** MobileNetV2"
                    )

                    st.write(
                        "**Input Size:** 224 × 224 × 3"
                    )

                    st.write(
                        "**Classification Threshold:** 0.5"
                    )

                    st.write(
                        f"**Raw Model Output:** "
                        f"{prediction:.6f}"
                    )


    except Exception as e:

        st.error(
            "⚠️ Unable to process this image."
        )

        st.code(
            str(e)
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DeepGuard AI • Deepfake Detection using MobileNetV2"
)