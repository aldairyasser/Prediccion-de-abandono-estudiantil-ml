import streamlit as st
import functions as ft

#basic setup and layout
ft.config_page()

#Iniciamos la página en 0
if "datos" not in st.session_state:
    st.session_state["datos"] = 0

# Opciones del menú
menu = st.sidebar.selectbox("PÁGINAS", ("1. INTRODUCCIÓN 📜", "2. DATOS DE NEGOCIO 📁", "3. INSIGHTS DEL EDA 🔎", "4. PREDICCIÓN UNITARIA 🎓", "5. PREDICCIÓN CSV 🗂️"))

if menu == "1. INTRODUCCIÓN 📜":
    ft.home()

elif menu == "2. DATOS DE NEGOCIO 📁":
    ft.carga_datos()

elif menu == "3. INSIGHTS DEL EDA 🔎":
    ft.coclu_eda()

elif menu == "4. PREDICCIÓN UNITARIA 🎓":
    ft.predi_uni()

elif menu == "5. PREDICCIÓN CSV 🗂️":
    ft.predi_csv()