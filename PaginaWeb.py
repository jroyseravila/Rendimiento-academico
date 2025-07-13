import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# === Cargar modelo DEME ===
modelo_dict = joblib.load("DEME_Model.pkl")
modelos = modelo_dict["modelos"]
pesos = modelo_dict["pesos"]
scaler = modelo_dict["scaler"]
variables = modelo_dict["variables"]

# === Funciones para predicción ===
def obtener_predicciones_proba(X_input):
    predicciones = {}
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_input)[:, 1]
        predicciones[nombre] = proba
    return predicciones

def ensamblar_dinamico(predicciones, pesos):
    n = len(next(iter(predicciones.values())))
    resultado = np.zeros(n)
    for nombre, proba in predicciones.items():
        resultado += pesos[nombre] * proba
    return (resultado >= 0.5).astype(int)

# === Diccionarios ===
app_mode_dict = {
    1: "1st phase - general contingent", 2: "Ordinance No. 612/93", 5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses", 10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)", 17: "2nd phase - general contingent",
    18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)", 39: "Over 23 years old", 42: "Transfer",
    43: "Change of course", 44: "Technological specialization diploma holders", 51: "Change of institution/course",
    53: "Short cycle diploma holders", 57: "Change of institution/course (International)"
}

course_dict = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture", 9147: "Management",
    9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management", 9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening attendance)"
}

occupation_dict = {
    0: "Student", 1: "Legislative/Executive Rep/Managers", 2: "Intellectual/Scientific Activities",
    3: "Intermediate Technicians", 4: "Administrative staff", 5: "Personal Services and Sellers",
    6: "Farmers/Fishermen", 7: "Industry/Construction Workers", 8: "Machine Operators",
    9: "Unskilled Workers", 10: "Armed Forces", 90: "Other Situation", 99: "(blank)",
    122: "Health professionals", 123: "Teachers", 125: "ICT Specialists", 131: "Science/Engineering Technicians",
    132: "Health Technicians", 134: "Legal/Social/Sports Technicians", 141: "Office workers",
    143: "Accounting/Finance Operators", 144: "Other Admin Support", 151: "Personal service workers",
    152: "Sellers", 153: "Personal care workers", 171: "Skilled construction workers",
    173: "Craftsmen/Artisans", 175: "Food/Clothing Workers", 191: "Cleaning workers",
    192: "Unskilled Agricultural Workers", 193: "Unskilled Industry Workers", 194: "Meal prep assistants",
    101: "Armed Forces Officers", 102: "Armed Forces Sergeants", 103: "Other Armed Forces personnel",
    112: "Admin/Commercial Directors", 114: "Service Directors", 121: "Science/Math/Eng Specialists",
    124: "Finance/Admin Specialists", 135: "ICT Technicians", 154: "Security Personnel",
    161: "Skilled Agricultural Workers", 163: "Subsistence Farmers", 172: "Metalworkers",
    174: "Electricians/Electronics", 181: "Plant Operators", 182: "Assembly Workers",
    183: "Vehicle Drivers", 195: "Street Vendors"
}

# === Explicación del ensamble ===
def mostrar_explicacion_modelos(predicciones):
    st.subheader("Explicación del Ensamble de Modelos")
    promedio_por_modelo = {nombre: round(np.mean(pred) * 100, 2) for nombre, pred in predicciones.items()}
    df_modelos = pd.DataFrame(list(promedio_por_modelo.items()), columns=["Modelo", "Porcentaje de confianza"])
    mejor_modelo = df_modelos.loc[df_modelos["Porcentaje de confianza"].idxmax()]
    st.dataframe(df_modelos.sort_values(by="Porcentaje de confianza", ascending=False), use_container_width=True)
    st.success(f"Modelo más seguro: {mejor_modelo['Modelo']} con {mejor_modelo['Porcentaje de confianza']}%")


# === Configuración de Streamlit ===
st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide"
)

st.sidebar.title("Menú")
usuario = st.sidebar.selectbox("Tipo de usuario", ["Estudiante con Archivo", "Formulario Estudiante", "Administrador"])

if usuario == "Estudiante con Archivo":
    st.title("Predicción Académica para Estudiantes")
    archivo = st.file_uploader("Sube tu archivo CSV con tus datos", type=["csv"])

    if archivo:
        df = pd.read_csv(archivo)
        X = df[variables]
        X_scaled = scaler.transform(X)

        predicciones = obtener_predicciones_proba(X_scaled)
        resultado = ensamblar_dinamico(predicciones, pesos)

        df["Resultado"] = np.where(resultado == 1, "Graduado ✅", "Deserción ❌")
        st.write("Resultado de tu evaluación:")
        st.dataframe(df[["Resultado"] + variables])

elif usuario == "Formulario Estudiante":
    st.title("Formulario Anónimo para Estudiantes")
    with st.form("formulario_anonimo"):
        app_mode = st.selectbox("¿Cómo accediste a la universidad?", options=list(app_mode_dict.keys()), format_func=lambda x: app_mode_dict[x])
        course = st.selectbox("¿Qué carrera estudias?", options=list(course_dict.keys()), format_func=lambda x: course_dict[x])
        mother_occ = st.selectbox("¿Cuál es la ocupación de tu madre?", options=list(occupation_dict.keys()), format_func=lambda x: occupation_dict[x])
        father_occ = st.selectbox("¿Cuál es la ocupación de tu padre?", options=list(occupation_dict.keys()), format_func=lambda x: occupation_dict[x])
        adm_grade = st.slider("¿Cuál fue tu nota de admisión?", 0.0, 200.0, 150.0)
        scholarship = st.radio("¿Tienes beca?", [1, 0], format_func=lambda x: "Sí" if x==1 else "No")
        age_enroll = st.slider("¿Qué edad tenías al matricularte?", 15, 40, 18)
        prev_grade = st.slider("Nota de estudios anteriores", 0.0, 20.0, 12.0)
        grade_1sem = st.slider("Promedio del primer semestre", 0.0, 20.0, 14.0)
        grade_2sem = st.slider("Promedio del segundo semestre", 0.0, 20.0, 14.0)
        approved_1sem = st.slider("Asignaturas aprobadas en primer semestre", 0, 20, 5)
        approved_2sem = st.slider("Asignaturas aprobadas en segundo semestre", 0, 20, 5)
        eval_1sem = st.slider("Evaluaciones en el primer semestre", 0, 20, 6)
        eval_2sem = st.slider("Evaluaciones en el segundo semestre", 0, 20, 6)

        enviar = st.form_submit_button("Enviar y Evaluar")

    if enviar:
        tuition_up = 0
        entrada = pd.DataFrame([[app_mode, course, mother_occ, father_occ, adm_grade,
                                 tuition_up, scholarship, age_enroll, prev_grade,
                                 grade_1sem, grade_2sem, approved_1sem, approved_2sem,
                                 eval_1sem, eval_2sem]], columns=variables)
        entrada_scaled = scaler.transform(entrada)
        predicciones = obtener_predicciones_proba(entrada_scaled)
        resultado = ensamblar_dinamico(predicciones, pesos)

        estado = "Graduado" if resultado[0] == 1 else "Deserción"
        color = "green" if resultado[0] == 1 else "red"
        confianza = np.mean(list(predicciones.values()), axis=0)[0] * 100

        st.markdown(f"### Estado Predicho: <span style='color:{color}'>{estado}</span>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confianza,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confianza (%)"},
            gauge = {'axis': {'range': [0, 100]},
                     'bar': {'color': color}}
        ))
        st.plotly_chart(fig, use_container_width=True)

else:
    opcion = st.sidebar.radio("Selecciona una opción", ["Evaluación Individual", "Evaluación por Archivo"])

    if opcion == "Evaluación Individual":
        st.title("Evaluación Individual del Estudiante")
        with st.form("formulario_admin"):
            col1, col2 = st.columns(2)
            with col1:
                app_mode = st.selectbox("Application mode", options=list(app_mode_dict.keys()), format_func=lambda x: app_mode_dict[x])
                course = st.selectbox("Course", options=list(course_dict.keys()), format_func=lambda x: course_dict[x])
                mother_occ = st.selectbox("Mother's occupation", options=list(occupation_dict.keys()), format_func=lambda x: occupation_dict[x])
                father_occ = st.selectbox("Father's occupation", options=list(occupation_dict.keys()), format_func=lambda x: occupation_dict[x])
                adm_grade = st.slider("Admission grade", 0.0, 200.0, 150.0)
                scholarship = st.radio("Scholarship holder", [1, 0], format_func=lambda x: "Sí" if x==1 else "No")

            with col2:
                age_enroll = st.number_input("Age at enrollment", min_value=15, max_value=80, value=18)
                prev_grade = st.slider("Previous qualification (grade)", 0.0, 200.0, 120.0)
                grade_1sem = st.slider("Curricular units 1st sem (grade)", 0.0, 20.0, 14.0)
                grade_2sem = st.slider("Curricular units 2nd sem (grade)", 0.0, 20.0, 14.0)
                approved_1sem = st.number_input("1st sem approved units", 0, 20, 5)
                approved_2sem = st.number_input("2nd sem approved units", 0, 20, 5)
                eval_1sem = st.number_input("1st sem evaluations", 0, 20, 6)
                eval_2sem = st.number_input("2nd sem evaluations", 0, 20, 6)

            submitted = st.form_submit_button("Evaluar Estudiante")

        if submitted:
            tuition_up = 0
            entrada = pd.DataFrame([[app_mode, course, mother_occ, father_occ, adm_grade,
                                     tuition_up, scholarship, age_enroll, prev_grade,
                                     grade_1sem, grade_2sem, approved_1sem, approved_2sem,
                                     eval_1sem, eval_2sem]], columns=variables)
            entrada_scaled = scaler.transform(entrada)
            predicciones = obtener_predicciones_proba(entrada_scaled)
            resultado = ensamblar_dinamico(predicciones, pesos)

            estado = "Graduado" if resultado[0] == 1 else "Deserción"
            color = "green" if resultado[0] == 1 else "red"
            confianza = np.mean(list(predicciones.values()), axis=0)[0] * 100

            st.markdown(f"### Estado Predicho: <span style='color:{color}'>{estado}</span>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confianza,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confianza (%)"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': color}}
            ))
            st.plotly_chart(fig, use_container_width=True)
             #  Meta-explicación 
            mostrar_explicacion_modelos(predicciones)

    elif opcion == "Evaluación por Archivo":
        st.title("Evaluación por Dataset")
        archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

        if archivo:
            df = pd.read_csv(archivo)
            st.write("Vista previa de los datos:", df.head())

            X = df[variables]
            X_scaled = scaler.transform(X)

            predicciones = obtener_predicciones_proba(X_scaled)
            resultado = ensamblar_dinamico(predicciones, pesos)

            df["Estado Predicho"] = np.where(resultado == 1, "Graduado", "Deserción")
            df["Confianza (%)"] = np.mean(list(predicciones.values()), axis=0) * 100

            st.write("Resultados de la predicción:")
            st.dataframe(df[["Estado Predicho", "Confianza (%)"] + variables])
