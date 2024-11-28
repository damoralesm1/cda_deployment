import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Asegúrate de importar pandas

# Cargar el modelo
model = joblib.load("decision_tree_pipeline.pkl")

# Definir las columnas relevantes
relevant_columns = [
    'Inversion Planeada', 'Meta Leads', 'Meta Conversaciones',
    'Meta Clicks', 'Meta Seguidores', 'Meta interacciones', 'Meta Alcance', 'Campaign type'
]

# Opciones para Campaign type
campaign_types = ['BRANDING', 'PERFORMANCE', 'TACTICOS', 'MARKETING CLOUD']

# Interfaz de usuario
st.title("Predicción de Campañas Publicitarias")

# Crear inputs para las variables
st.subheader("Por favor ingresa los valores de las siguientes variables:")

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
        # Realizar predicción
        prediction = model.predict(input_df)
        st.write(f"Predicción del modelo: {prediction[0]}")
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")
