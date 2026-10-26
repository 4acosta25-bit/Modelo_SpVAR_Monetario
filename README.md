# Modelo de Vector Autorregresivo Espacial (SpVAR) Monetario - Colombia

Aplicación econométrica e interactiva desarrollada en Python y Streamlit para el análisis y simulación de la transmisión espacial de la oferta monetaria ($M_2$) y su efecto sobre el Índice de Precios al Consumidor (IPC) a nivel departamental en Colombia.

## 🚀 Características
- **Carga Topológica:** Geometría de Colombia libre de observaciones aisladas.
- **Selección de Matriz Espacial ($W$):** Evaluación automática del Índice de Moran sobre matrices de contigüidad y $k$-vecinos más cercanos ($KNN$).
- **Estimación Econométrica:** Modelo $SpVAR$ OLS con matriz de covarianza robusta frente a heterocedasticidad $HC1$.
- **Impactos y Multiplicadores:** Descomposición de efectos Directos, Indirectos (*Spillovers*) y Totales mediante la Inversa de Leontief Espacial $(I - ho W)^{-1}$.
- **Visualizaciones:** Mapas coropléticos de Folium y scatterplot 3D en Plotly.

## 🛠️ Instalación Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Despliegue
Diseñado para su ejecución directa en **Streamlit Community Cloud**.
