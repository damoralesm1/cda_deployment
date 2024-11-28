import streamlit as st
import joblib
import numpy as np
import pandas as pd

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
st.title("Predicción de Éxito en Campañas Publicitarias")

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
