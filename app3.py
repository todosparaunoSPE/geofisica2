# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:47:10 2024

@author: jperezr
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parámetros del modelo
nx = 50  # Número de celdas en el eje x
ny = 50  # Número de celdas en el eje y
nz = 10  # Número de celdas en el eje z (para visualización 3D)
dx = 1.0  # Tamaño de la celda en el eje x
dy = 1.0  # Tamaño de la celda en el eje y

# Crear la cuadrícula de presiones y estado de fractura (2D)
pressure = np.zeros((nx, ny))  # Matriz de presiones inicialmente cero
fracture = np.zeros((nx, ny), dtype=bool)  # Matriz de estado de fractura (False = no fracturado)

# Crear la cuadrícula de estado de fractura en 3D
fracture_3d = np.zeros((nx, ny, nz), dtype=bool)  # Matriz de estado de fractura en 3D

# Añadir sliders en la barra lateral para ajustar los parámetros de simulación
st.sidebar.header('Parámetros de Simulación')
fracture_pressure_mean = st.sidebar.slider('Presión de fracturamiento media', 0.0, 100.0, 50.0)
fracture_pressure_std = st.sidebar.slider('Desviación estándar de la presión de fracturamiento', 0.0, 20.0, 5.0)
rock_strength_mean = st.sidebar.slider('Resistencia de la roca media', 0.0, 100.0, 40.0)
rock_strength_std = st.sidebar.slider('Desviación estándar de la resistencia de la roca', 0.0, 20.0, 5.0)

# Parámetros para anisotropía de la roca
stress_orientation = np.random.uniform(0, 2*np.pi, size=(nx, ny))  # Orientación del estrés

# Dataframes para mostrar propiedades de las celdas antes y después de la simulación
df_data_before = pd.DataFrame(index=range(nx*ny), columns=['X', 'Y', 'Fracture Pressure', 'Rock Strength'])
df_data_after = pd.DataFrame(index=range(nx*ny), columns=['X', 'Y', 'Fracture Pressure', 'Rock Strength'])

# Llenar los dataframes con datos aleatorios
for i in range(nx):
    for j in range(ny):
        df_data_before.loc[i*ny + j] = [i, j, np.random.normal(fracture_pressure_mean, fracture_pressure_std),
                                         np.random.normal(rock_strength_mean, rock_strength_std)]

# Función para aplicar la presión de fracturamiento en una ubicación (2D)
def apply_fracture(x, y):
    if not fracture[x, y]:
        fracture[x, y] = True
        pressure[x, y] = np.random.normal(fracture_pressure_mean, fracture_pressure_std)

# Función para aplicar la presión de fracturamiento en una ubicación (3D)
def apply_fracture_3d(x, y, z):
    if not fracture_3d[x, y, z]:
        fracture_3d[x, y, z] = True
        pressure[x, y] = np.random.normal(fracture_pressure_mean, fracture_pressure_std)

# Función para simular la propagación de fracturas con anisotropía y variabilidad espacial
def simulate_fracture_propagation(num_steps, save_interval):
    fig = go.Figure(data=go.Heatmap(z=pressure, colorscale='Viridis'))
    fig.update_layout(title='Pressure Field', xaxis_title='X', yaxis_title='Y')

    # Mostrar el gráfico inicial
    heatmap_placeholder = st.empty()
    heatmap_placeholder.plotly_chart(fig)

    for step in range(num_steps):
        # Seleccionar ubicación aleatoria para evaluar propagación de fracturas
        x = np.random.randint(0, nx)
        y = np.random.randint(0, ny)
        z = np.random.randint(0, nz)  # Para la visualización 3D

        # Verificar si la celda no está fracturada y evaluar propagación
        if not fracture_3d[x, y, z]:
            # Calcular la diferencia de presión con la presión de fracturamiento
            delta_pressure = np.random.normal(fracture_pressure_mean, fracture_pressure_std) - pressure[x, y]

            # Calcular la dirección de propagación de la fractura basada en el estrés
            stress_angle = stress_orientation[x, y]
            propagation_direction = np.array([np.cos(stress_angle), np.sin(stress_angle)])

            # Calcular la tasa de crecimiento de la fractura basada en criterios de tensión
            stress_criteria = delta_pressure / np.random.normal(rock_strength_mean, rock_strength_std)
            growth_criteria = np.random.uniform(0, 1)

            if growth_criteria < stress_criteria:
                # Aplicar fractura en la ubicación seleccionada
                apply_fracture_3d(x, y, z)

                # Actualizar datos después de aplicar la fractura
                df_data_after.loc[x*ny + y] = [x, y, pressure[x, y], np.random.normal(rock_strength_mean, rock_strength_std)]

                # Actualizar el mapa de calor cada intervalo de guardado
                if step % save_interval == 0 or step == num_steps - 1:
                    fig.update_traces(z=pressure)
                    heatmap_placeholder.plotly_chart(fig)

    # Filtrar registros NaN en df_data_after
    df_data_after_clean = df_data_after.dropna()

    # Mostrar el DataFrame después de la simulación en Streamlit (solo registros con datos)
    st.subheader('Datos de Propiedades de las Celdas (Después de la Simulación)')
    st.dataframe(df_data_after_clean)

    return df_data_after_clean

# Mostrar el DataFrame antes de la simulación en Streamlit
st.subheader('Datos de Propiedades de las Celdas (Antes de la Simulación)')
st.dataframe(df_data_before)

# Añadir controles en Streamlit para aplicar la presión de fracturamiento
num_steps = st.slider('Número de pasos de propagación de fractura', 1, 1000, 100)
save_interval = st.slider('Intervalo de guardado', 1, 100, 10)

if st.button('Simular Propagación de Fracturas'):
    df_data_after_clean = simulate_fracture_propagation(num_steps, save_interval)

    # Visualización de fracturas en 3D
    x_3d, y_3d, z_3d = np.where(fracture_3d)
    fracture_fig_3d = go.Figure(data=[go.Scatter3d(x=x_3d, y=y_3d, z=z_3d, mode='markers', marker=dict(color='red', size=3))])
    fracture_fig_3d.update_layout(title='Fracture State (3D)', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), 
                                  width=800, height=600)  # Ajustar el tamaño del gráfico 3D
    st.plotly_chart(fracture_fig_3d)

    # Análisis Estadístico
    st.header('Análisis Estadístico')

    # Distribución de Presión de Fracturamiento y Resistencia de la Roca (Antes)
    st.subheader('Distribución de Propiedades (Antes de la Simulación)')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df_data_before['Fracture Pressure'].dropna(), bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Distribución de Presión de Fracturamiento')
    ax[0].set_xlabel('Presión de Fracturamiento')
    ax[0].set_ylabel('Frecuencia')

    sns.histplot(df_data_before['Rock Strength'].dropna(), bins=30, kde=True, ax=ax[1])
    ax[1].set_title('Distribución de Resistencia de la Roca')
    ax[1].set_xlabel('Resistencia de la Roca')
    ax[1].set_ylabel('Frecuencia')

    st.pyplot(fig)

    # Distribución de Presión de Fracturamiento y Resistencia de la Roca (Después)
    st.subheader('Distribución de Propiedades (Después de la Simulación)')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df_data_after_clean['Fracture Pressure'].dropna(), bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Distribución de Presión de Fracturamiento')
    ax[0].set_xlabel('Presión de Fracturamiento')
    ax[0].set_ylabel('Frecuencia')

    sns.histplot(df_data_after_clean['Rock Strength'].dropna(), bins=30, kde=True, ax=ax[1])
    ax[1].set_title('Distribución de Resistencia de la Roca')
    ax[1].set_xlabel('Resistencia de la Roca')
    ax[1].set_ylabel('Frecuencia')

    st.pyplot(fig)

# Sección de Ayuda
with st.sidebar:
    st.header('Ayuda')
    st.markdown("""
    - **Presión de fracturamiento media:** Valor promedio de la presión necesaria para fracturar la roca.
    - **Desviación estándar de la presión de fracturamiento:** Variabilidad en la presión de fracturamiento.
    - **Resistencia de la roca media:** Valor promedio de la resistencia de la roca.
    - **Desviación estándar de la resistencia de la roca:** Variabilidad en la resistencia de la roca.
    - **Número de pasos de propagación de fractura:** Número total de iteraciones para la simulación.
    - **Intervalo de guardado:** Frecuencia con la que se actualiza el mapa de calor durante la simulación.
    """)
    
# Copyright
st.markdown("""
---
© 2024 jahoperi. Todos los derechos reservados.
""")

            
