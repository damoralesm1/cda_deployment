import streamlit as st
import joblib
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

# Helpers para los campos
helpers = {
    'Inversion Planeada': "Indica la inversión planeada en pesos colombianos (valor decimal).",
    'Meta Leads': "Número objetivo de leads que esperas alcanzar con la campaña.",
    'Meta Conversaciones': "Número de conversaciones objetivo generadas por la campaña.",
    'Meta Clicks': "Número total de clics que esperas lograr.",
    'Meta Seguidores': "Número objetivo de seguidores nuevos obtenidos.",
    'Meta interacciones': "Número esperado de interacciones en redes sociales.",
    'Meta Alcance': "Cantidad de usuarios que esperas que la campaña alcance.",
    'Campaign type': "Selecciona el tipo de campaña que estás planificando."
}

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

# Crear inputs para las variables
st.subheader("Por favor, ingresa los valores de las siguientes variables:")

inputs = {}

# Crear columnas
col1, col2 = st.columns(2)

# Campo numérico para "Inversion Planeada" en la primera columna
with col1:
    inputs['Inversion Planeada'] = st.number_input(
        f"Inversión Planeada (en pesos colombianos):",
        min_value=0.0,
        step=1000.0,
        key='Inversion Planeada'
    )
    st.markdown(f"<small style='color: gray;'>{helpers['Inversion Planeada']}</small>", unsafe_allow_html=True)

# Campos numéricos para las demás variables en las dos columnas
for idx, col in enumerate(relevant_columns[1:-1]):  # Excluir 'Campaign type'
    with col1 if idx % 2 == 0 else col2:
        inputs[col] = st.number_input(
            f"{col}:",
            min_value=0,
            step=1,
            key=col
        )
        st.markdown(f"<small style='color: gray;'>{helpers[col]}</small>", unsafe_allow_html=True)

# Dropdown para 'Campaign type' en la segunda columna
with col2:
    selected_campaign_type = st.selectbox(
        "Selecciona el tipo de campaña:",
        campaign_types
    )
    st.markdown(f"<small style='color: gray;'>{helpers['Campaign type']}</small>", unsafe_allow_html=True)
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
