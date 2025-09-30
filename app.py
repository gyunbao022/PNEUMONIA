import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# ============== 설정 ==============
MODEL_PATH = "models/hazard_resnet50_strio_0.keras"
IMG_SIZE = (224, 224)
CLASSES = ["NORMAL", "PNEUMONIA"]

# ============== 모델 로딩 ==============
@st.cache_resource
def load_my_model():
    try:
        def build_model(input_shape, num_classes):
            base_model = ResNet50(include_top=False, weights=None, input_shape=input_shape)
            inputs = keras.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
            x = keras.layers.Dropout(0.3, name="dropout")(x)
            outputs = keras.layers.Dense(num_classes, activation="softmax", name="dense")(x)
            model = keras.Model(inputs, outputs)
            return model
        
        model = build_model(input_shape=IMG_SIZE + (3,), num_classes=len(CLASSES))
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"모델 로딩 중 오류가 발생했습니다: {e}")
        return None

model = load_my_model()

# ============== LIME 시각화 함수 (새로 추가됨) ==============
@st.cache_data # LIME 계산 결과를 캐싱하여 반복 실행 시 속도 향상
def get_lime_explanation(image_array, model):
    """
    LIME을 사용하여 모델의 예측에 대한 시각적 설명을 생성합니다.
    """
    try:
        # LIME 이미지 설명기 생성
        explainer = lime_image.LimeImageExplainer()

        # 모델 예측 함수 정의: LIME은 (N, height, width, 3) 형태의 입력을 받아
        # (N, num_classes) 형태의 확률을 반환하는 함수를 요구합니다.
        def predict_fn(images):
            # LIME이 생성한 이미지는 0-255 범위이므로, 모델에 맞게 전처리합니다.
            images_preprocessed = preprocess_input(images)
            return model.predict(images_preprocessed)

        # 설명 생성 (시간이 다소 걸릴 수 있습니다)
        # top_labels=2: 상위 2개 클래스에 대한 설명을 모두 생성
        # num_samples=1000: 더 정확한 설명을 위해 1000개의 샘플 이미지를 생성 (값을 줄이면 속도가 빨라짐)
        explanation = explainer.explain_instance(
            image_array, 
            predict_fn, 
            top_labels=2, 
            hide_color=0, 
            num_samples=1000
        )
        return explanation
    except Exception as e:
        st.error(f"LIME 설명 생성 중 오류 발생: {e}")
        return None

# ============== Streamlit UI 구성 ==============
st.set_page_config(page_title="폐렴 진단 보조 시스템", layout="wide")
st.title("🫁 폐렴 X-ray 진단 보조 AI (LIME 적용)")
st.write("ResNet50 모델을 사용하여 폐렴 여부를 예측하고, LIME으로 판단 근거를 시각화합니다.")

if model is None:
    st.error("모델을 불러올 수 없습니다. `models` 폴더에 모델 파일이 있는지 확인해주세요.")
else:
    uploaded_file = st.file_uploader("X-ray 이미지를 업로드하세요.", type=["jpeg", "jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMG_SIZE)
        original_img = np.array(image)

        st.image(image, caption="업로드된 이미지", width=300)

        if st.button("분석 실행"):
            with st.spinner('AI가 이미지를 분석 중입니다... (LIME 분석은 시간이 다소 소요될 수 있습니다)'):
                # 1. 예측 실행
                img_array_for_pred = np.expand_dims(original_img, axis=0)
                img_array_preprocessed = preprocess_input(img_array_for_pred.copy())
                
                prediction_probs = model.predict(img_array_preprocessed)[0]
                prediction_idx = np.argmax(prediction_probs)
                pred_label = CLASSES[prediction_idx]
                pred_confidence = prediction_probs[prediction_idx] * 100

                st.subheader("📊 분석 결과")
                if pred_label == "PNEUMONIA":
                    st.error(f"**'{pred_label}'**일 확률이 **{pred_confidence:.2f}%** 입니다.")
                else:
                    st.success(f"**'{pred_label}'**일 확률이 **{pred_confidence:.2f}%** 입니다.")

                # 2. LIME 시각화 실행
                st.subheader("💡 AI의 판단 근거 (LIME)")
                explanation = get_lime_explanation(original_img, model)

                if explanation:
                    # LIME 결과를 이미지로 변환
                    # 예측된 클래스(prediction_idx)에 긍정적인 영향을 준 영역만(positive_only=True) 표시
                    image, mask = explanation.get_image_and_mask(
                        prediction_idx, 
                        positive_only=True, 
                        num_features=5, 
                        hide_rest=False
                    )
                    
                    # Matplotlib를 사용하여 이미지 출력
                    fig, ax = plt.subplots()
                    ax.imshow(mark_boundaries(image, mask))
                    ax.axis('off')
                    st.pyplot(fig)
                    st.info("초록색으로 표시된 영역이 모델이 현재와 같이 예측하는 데 긍정적인 영향을 미친 주요 근거입니다.")
                else:
                    st.warning("LIME 시각화 생성에 실패했습니다.")
