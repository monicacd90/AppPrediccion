import streamlit as st
import numpy as np
from joblib import load

# Cargar el modelo y el scaler
modelo_completo = load('modelo_svm_escalado.joblib')
scaler = modelo_completo['scaler']
modelo_svm = modelo_completo['modelo']

st.set_page_config(page_title="Predicción Prequirúrgica", page_icon="🧠")
st.title("🧠 Predicción de Éxito Quirúrgico con SVM")
st.write("Completa los siguientes campos para obtener una predicción basada en tu modelo.")

# Inputs personalizados
k_pre = st.number_input("Potasio prequirúrgico (K precirugía)", value=4.0)
folico = st.number_input("Ácido Fólico prequirúrgico", value=10.0)
fa = st.number_input("Fosfatasa Alcalina (FA) prequirúrgica", value=90.0)
talla = st.number_input("Talla (en metros)", value=1.70)
transferrina = st.number_input("Transferrina prequirúrgica", value=250.0)
peso = st.number_input("Peso prequirúrgico (kg)", value=80.0)
imc = st.number_input("IMC prequirúrgico", value=27.5)
tg = st.number_input("Triglicéridos prequirúrgicos", value=150.0)
beck = st.number_input("Puntaje de Depresión (Beck)", value=10.0)
insulina = st.number_input("Insulina prequirúrgica", value=15.0)

# Botón para hacer la predicción
if st.button("📊 Predecir resultado"):
    datos_usuario = np.array([[k_pre, folico, fa, talla, transferrina,
                               peso, imc, tg, beck, insulina]])
    datos_escalados = scaler.transform(datos_usuario)
    probabilidad = modelo_svm.predict_proba(datos_escalados)[0][1]
    prediccion = int(probabilidad >= 0.56)

    st.subheader("Resultado de la Predicción")
    st.write(f"Probabilidad estimada de éxito: **{probabilidad:.2%}**")
    if prediccion == 1:
        st.success("✅ Se predice ÉXITO quirúrgico.")
    else:
        st.error("❌ Se predice NO ÉXITO quirúrgico.")
