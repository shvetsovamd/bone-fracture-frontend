# -*- coding: utf-8 -*-

import streamlit as st
import requests
import cv2
import numpy as np

st.set_page_config(page_title="Bone Fracture Detection", layout="centered")

st.title("Детекция переломов на рентгеновских снимках")
st.write("Загрузите снимок и выберите архитектуру нейросети для анализа.")

BACKEND_URL = "http://localhost:8000/predict"

model_type = st.selectbox("Выберите модель ИИ:", ["YOLOv11", "RT-DETR"])
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок (JPG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Ошибка: Загруженный файл поврежден или не является допустимым изображением!")
    else:
        st.success("Изображение успешно загружено и валидировано.")
        st.image(uploaded_file, caption="Исходный рентгеновский снимок", use_container_width=True)

        if st.button("Запустить анализ переломов"):
            with st.spinner("Модель ИИ анализирует снимок..."):
                try:
                    files = {"image": (uploaded_file.name, file_bytes, uploaded_file.type)}
                    data = {"model_type": model_type}

                    response = requests.post(BACKEND_URL, files=files, data=data)

                    if response.status_code == 200:
                        results = response.json()["detections"]

                        if len(results) == 0:
                            st.balloons()
                            st.success("Переломов не обнаружено!")
                        else:
                            st.warning(f"Обнаружено потенциальных переломов: {len(results)}")

                            output_image = image.copy()
                            for det in results:
                                xmin, ymin, xmax, ymax = det["box"]
                                label = f"{det['label']} ({det['confidence']})"

                                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                                cv2.putText(output_image, label, (xmin, ymin - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                            st.image(output_image_rgb, caption="Результат детекции", use_container_width=True)
                    else:
                        error_msg = response.json().get('detail', 'Неизвестная ошибка')
                        st.error(f"Ошибка сервера: {error_msg}")

                except requests.exceptions.ConnectionError:
                    st.error("Не удалось связаться с Backend-сервером. Убедитесь, что app_backend.py запущен.")
