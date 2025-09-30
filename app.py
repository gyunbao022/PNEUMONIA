
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from PIL import Image

# ============== 설정 ==============
# 1. 파일 경로 및 모델 설정
MODEL_PATH = "models/hazard_resnet50_strio_0.keras"
IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "PNEUMONIA"]
LAST_CONV_LAYER_NAME = "conv5_block3_out"  # ResNet50의 마지막 활성화 레이어

# ============== 모델 로딩 ==============
# @st.cache_resource: 모델을 한번만 로드하여 앱 성능을 높입니다.
@st.cache_resource
def load_my_model():
    """
    Keras 내부 그래프 오류를 피하기 위해,
    모델 구조를 직접 생성하고 가중치만 불러옵니다.
    """
    try:
        # 모델 구조 정의
        def build_model(input_shape, num_classes):
            base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
            inputs = keras.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
            x = keras.layers.Dropout(0.3, name="dropout")(x)
            outputs = keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
            model = keras.Model(inputs, outputs)
            return model

        # 모델 뼈대 생성
        model = build_model(input_shape=IMG_SIZE + (3,), num_classes=len(CLASSES))
        # 가중치 로드
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
        return None

model = load_my_model()

# ============== Grad-CAM 함수 ==============
# 가장 안정적인 최종 버전 함수를 사용합니다.
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = keras.Model(
            [model.input], [model.get_layer('resnet50').get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        # 오류 발생 시 None을 반환하여 앱이 멈추지 않도록 합니다.
        print(f"Grad-CAM 생성 중 오류 발생: {e}")
        return None

def superimpose_gradcam(original_img, heatmap, alpha=0.6):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + original_img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# ============== Streamlit UI 구성 ==============
st.set_page_config(page_title="폐렴 진단 보조 시스템", layout="wide")
st.title("🫁 폐렴 X-ray 진단 보조 AI")
st.write("ResNet50 모델을 사용하여 폐렴 여부를 예측하고, Grad-CAM으로 판단 근거를 시각화합니다.")

if model is None:
    st.error("모델을 불러올 수 없습니다. `models` 폴더에 모델 파일이 있는지 확인해주세요.")
else:
    uploaded_file = st.file_uploader("X-ray 이미지를 업로드하세요.", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        # 이미지 전처리
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMG_SIZE)
        original_img = np.array(image) # 시각화용 원본

        img_array = np.expand_dims(original_img, axis=0)
        img_array_preprocessed = preprocess_input(img_array.copy()) # 모델 예측용

        st.image(image, caption="업로드된 이미지", width=300)

        if st.button("분석 실행"):
            with st.spinner('AI가 이미지를 분석 중입니다...'):
                # 1. 예측 실행 (안정적인 부분)
                prediction_probs = model.predict(img_array_preprocessed)[0]
                prediction_idx = np.argmax(prediction_probs)
                pred_label = CLASSES[prediction_idx]
                pred_confidence = prediction_probs[prediction_idx] * 100

                st.subheader("📊 분석 결과")
                if pred_label == "PNEUMONIA":
                    st.error(f"**'{pred_label}'**일 확률이 **{pred_confidence:.2f}%** 입니다.")
                else:
                    st.success(f"**'{pred_label}'**일 확률이 **{pred_confidence:.2f}%** 입니다.")

                # 2. Grad-CAM 시각화 (실패할 수 있는 부분)
                st.subheader("💡 AI의 판단 근거 (Grad-CAM)")
                heatmap = make_gradcam_heatmap(img_array_preprocessed, model, LAST_CONV_LAYER_NAME, pred_index=prediction_idx)

                if heatmap is not None:
                    grad_cam_img = superimpose_gradcam(original_img, heatmap)
                    st.image(grad_cam_img, caption="Grad-CAM 시각화 결과 (붉은 영역이 판단의 주요 근거)", use_column_width=True)
                else:
                    st.warning("Grad-CAM 시각화 생성에 실패했습니다. 이는 Keras 모델의 내부 그래프 문제일 수 있으나, 예측 결과는 신뢰할 수 있습니다.")
