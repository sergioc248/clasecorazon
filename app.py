import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Cargar el modelo y el escalador guardado
model = joblib.load('modelo_knn.bin')
scaler = joblib.load('esclador.bin')

# Título y autor
st.title('Asistente Cardíaco')
st.subheader('Autor: Sergio Cuadros')

# Instrucciones de uso
st.write("""
    Ingrese su edad y niveles de colesterol para determinar si tiene problemas cardíacos.
    Los valores de edad deben estar entre 18 y 80 años, y los niveles de colesterol entre 100 y 600.
    Si el resultado es 1, podría tener problemas cardíacos, si es 0, no tiene problemas.
""")

# Crear un formulario para ingresar los datos
tab1, tab2 = st.tabs(["Ingreso de Datos", "Resultado"])

with tab1:
    # Slider para la edad
    edad = st.slider('Seleccione su edad', min_value=18, max_value=80, step=1)

    # Slider para el colesterol
    colesterol = st.slider('Seleccione su nivel de colesterol', min_value=100, max_value=600, step=1)

    # Convertir los datos a un DataFrame
    input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

    # Escalar los datos con el MinMaxScaler cargado
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción con el modelo
    prediccion = model.predict(input_data_scaled)

with tab2:
    # Mostrar el resultado de la predicción
    if prediccion == 0:
        st.write("No tiene problemas de corazón.")
        st.image("https://colombianadetrasplantes.com/web/wp-content/uploads/2023/05/01-PORTADA.-01-scaled.jpg")
    else:
        st.write("Tiene problemas de corazón.")
        st.image("https://cloudfront-us-east-1.images.arcpublishing.com/infobae/WRI4UH2CFFG3PFSLDLXBXW4YV4.jpg")
