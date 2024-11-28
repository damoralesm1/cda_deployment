import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# Cargar el modelo
model = joblib.load("decision_tree_pipeline.pkl")

# Definir las columnas relevantes
relevant_columns = [
    'Inversion Planeada', 'Meta Leads', 'Meta Conversaciones',
    'Meta Clicks', 'Meta Seguidores', 'Meta interacciones', 'Meta Alcance', 'Campaign type'
]

# Opciones para Campaign type
campaign_types = ['BRANDING', 'PERFORMANCE', 'TACTICOS', 'MARKETING CLOUD']

# Función para cargar imágenes como base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Cargar las imágenes locales en base64
honda_logo = get_base64_image("honda.png")
vml_logo = get_base64_image("vml.png")

# Crear la cabecera con HTML y CSS
header_html = f"""
<div style="
    background-color: #c8102e;
    padding: 20px;
    border-radius: 0px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;  /* Ocupa todo el ancho */
    box-sizing: border-box;
    margin: 0;
">
    <div style="color: white; font-size: 24px; font-weight: bold;">
        Predicción de Éxito en Campañas Publicitarias
    </div>
    <div style="display: flex; gap: 10px;">
        <img src="data:image/png;base64,{honda_logo}" alt="Honda Logo" style="height: 28px;">
        <img src="data:image/png;base64,{vml_logo}" alt="VML Logo" style="height: 28px;">
    </div>
</div>
"""

# Mostrar la cabecera en la app
st.markdown(header_html, unsafe_allow_html=True)
# Interfaz de usuario
# st.title("Predicción de Éxito en Campañas Publicitarias")

# Crear inputs para las variables
st.subheader("Por favor, ingresa los valores de las siguientes variables:")

inputs = {}

# Campo numérico para las variables numéricas
for col in relevant_columns[:-1]:  # Excluir 'Campaign type'
    inputs[col] = st.number_input(
        f"{col}:",
        step=0.01,
        key=col
    )

# Dropdown para 'Campaign type'
selected_campaign_type = st.selectbox("Selecciona el tipo de campaña:", campaign_types)
inputs['Campaign type'] = selected_campaign_type

# Convertir inputs a un DataFrame con las columnas esperadas
input_df = pd.DataFrame([inputs], columns=relevant_columns)

# Botón para predecir
if st.button("Predecir"):
    try:
        # Realizar predicción y obtener probabilidades
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Mostrar resultados con mensajes intuitivos
        if prediction == 1:
            st.success(f"¡La campaña será exitosa! ({prediction_proba[1] * 100:.2f}% de confianza)")
        else:
            st.error(f"La campaña no será exitosa ({prediction_proba[0] * 100:.2f}% de confianza)")
    
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
