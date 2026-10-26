# ==============================================================================
# APLICACIÓN DE MODELACIÓN ECONOMÉTRICA ESPACIAL (SpVAR Monetario Colombia)
# Despliegue: Streamlit Community Cloud (streamlit.app)
# ==============================================================================

import warnings
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import scipy.stats as stats
import statsmodels.api as sm

import libpysal
from libpysal.weights import Queen, Rook, KNN
from esda.moran import Moran, Moran_Local

import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import streamlit as st

warnings.filterwarnings('ignore')
np.random.seed(42)

# Configuración de página
st.set_page_config(
    page_title="SpVAR Monetario Colombia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para contraste profesional
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #0D6EFD; font-weight: 700; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #6C757D; margin-bottom: 20px; }
    .card-box { background-color: #FFFFFF; padding: 18px; border-radius: 8px; border: 1px solid #DEE2E6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Modelo Vector Autorregresivo Espacial (SpVAR) Monetario</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transmisión Espacial de la Oferta Monetaria e Inflación Departamental en Colombia</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS GEOGRÁFICOS Y GENERACIÓN DE PANEL SINTÉTICO ROBUSTO
# ------------------------------------------------------------------------------
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/mdeandal/colombia-json/master/colombia.json",
    "https://raw.githubusercontent.com/resilience-building/colombia-geojson/master/colombia-departments.json",
    "https://gist.githubusercontent.com/john-guerra/43c7656821b2f972bf0f/raw/colombia.geo.json"
]

@st.cache_data
def get_colombia_geodata() -> gpd.GeoDataFrame:
    gdf = None
    for url in GEOJSON_URLS:
        try:
            temp_gdf = gpd.read_file(url)
            if temp_gdf is not None and len(temp_gdf) > 0:
                gdf = temp_gdf
                break
        except Exception:
            continue

    if gdf is None:
        colombia_dptos = [
            ("Amazonas", -70.5, -1.5), ("Antioquia", -75.5, 6.5), ("Arauca", -70.7, 6.8),
            ("Atlántico", -74.9, 10.8), ("Bolívar", -74.3, 8.5), ("Boyacá", -73.0, 5.5),
            ("Caldas", -75.5, 5.2), ("Caquetá", -74.0, 1.0), ("Casanare", -71.5, 5.3),
            ("Cauca", -76.8, 2.5), ("Cesar", -73.5, 9.3), ("Chocó", -76.8, 5.2),
            ("Córdoba", -75.8, 8.3), ("Cundinamarca", -74.1, 4.8), ("Guainía", -68.5, 2.5),
            ("Guaviare", -72.0, 2.2), ("Huila", -75.5, 2.5), ("La Guajira", -72.5, 11.5),
            ("Magdalena", -74.2, 10.2), ("Meta", -73.0, 3.5), ("Nariño", -77.3, 1.5),
            ("Norte de Santander", -72.8, 8.0), ("Putumayo", -76.0, 0.5), ("Quindío", -75.7, 4.5),
            ("Risaralda", -75.9, 4.9), ("San Andrés", -81.7, 12.5), ("Santander", -73.2, 6.8),
            ("Sucre", -75.2, 9.2), ("Tolima", -75.2, 3.8), ("Valle del Cauca", -76.3, 3.8),
            ("Vaupés", -70.0, 0.8), ("Vichada", -69.5, 4.5)
        ]
        names = [d[0] for d in colombia_dptos]
        polygons = [box(d[1] - 0.5, d[2] - 0.5, d[1] + 0.5, d[2] + 0.5) for d in colombia_dptos]
        gdf = gpd.GeoDataFrame({
            'code': [str(i+1).zfill(2) for i in range(len(names))],
            'dpto_name': names,
            'geometry': polygons
        }, crs="EPSG:4326")

    if 'DPTO' in gdf.columns:
        gdf['code'] = gdf['DPTO'].astype(str).str.zfill(2)
    elif 'id' in gdf.columns:
        gdf['code'] = gdf['id'].astype(str).str.zfill(2)
    elif 'code' not in gdf.columns:
        gdf['code'] = [str(i+1).zfill(2) for i in range(len(gdf))]

    if 'NOMBRE_DPT' in gdf.columns:
        gdf['dpto_name'] = gdf['NOMBRE_DPT']
    elif 'NAME_1' in gdf.columns:
        gdf['dpto_name'] = gdf['NAME_1']
    elif 'dpto_name' not in gdf.columns:
        gdf['dpto_name'] = [f"Departamento_{i+1}" for i in range(len(gdf))]

    gdf = gdf[['code', 'dpto_name', 'geometry']].copy()
    gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid].reset_index(drop=True)

    gdf_projected = gdf.to_crs(epsg=3857)
    centroids_proj = gdf_projected.geometry.centroid.to_crs(epsg=4326)
    gdf['centroid_lon'] = centroids_proj.x
    gdf['centroid_lat'] = centroids_proj.y

    n = len(gdf)
    gdf['VIIRS_Lights'] = np.round(np.random.lognormal(mean=2.5, sigma=0.8, size=n), 2)

    coords = np.column_stack([gdf['centroid_lon'], gdf['centroid_lat']])
    w_init = KNN.from_array(coords, k=min(4, n-1))
    w_init.transform = 'R'
    W_mat = w_init.full()[0]

    u_m2 = np.random.normal(loc=7.0, scale=1.5, size=n)
    m2_val = np.linalg.solve(np.eye(n) - 0.40 * W_mat, u_m2 + 0.15 * gdf['VIIRS_Lights'])
    gdf['M2_Growth'] = np.round(np.clip(m2_val, 1.0, 18.0), 2)

    u_inf = np.random.normal(loc=3.5, scale=0.8, size=n)
    inf_val = np.linalg.solve(np.eye(n) - 0.35 * W_mat, 0.3 * gdf['M2_Growth'] + 0.2 * (W_mat @ gdf['M2_Growth']) + u_inf)
    gdf['Inflation_IPC'] = np.round(np.clip(inf_val, 1.5, 15.0), 2)

    return gdf

gdf_data = get_colombia_geodata()

# ------------------------------------------------------------------------------
# 2. AUDITORÍA DE PESOS ESPACIALES Y ESTIMACIÓN MODELO SpVAR
# ------------------------------------------------------------------------------
@st.cache_data
def run_spvar_model(_gdf: gpd.GeoDataFrame):
    y = _gdf['Inflation_IPC'].values
    coords = np.column_stack([_gdf['centroid_lon'], _gdf['centroid_lat']])
    n = len(_gdf)

    candidates = {}
    for k in [3, 4, 5]:
        if k < n:
            w_knn = KNN.from_array(coords, k=k)
            w_knn.transform = 'R'
            mi_knn = Moran(y, w_knn)
            candidates[f'KNN_{k}'] = (w_knn, mi_knn.I, mi_knn.p_norm)

    best_name = max(candidates.keys(), key=lambda k: (candidates[k][2] < 0.05, candidates[k][1]))
    best_w, best_I, best_p = candidates[best_name]
    W_mat = best_w.full()[0]

    y1 = _gdf['Inflation_IPC'].values
    y2 = _gdf['M2_Growth'].values
    x_proxy = _gdf['VIIRS_Lights'].values

    Wy1 = W_mat @ y1
    Wy2 = W_mat @ y2

    X = np.column_stack([np.ones(n), y2, Wy1, Wy2, x_proxy])
    model = sm.OLS(y1, X)
    results = model.fit(cov_type='HC1')

    coeffs = results.params
    p_values = results.pvalues

    I_mat = np.eye(n)
    leontief_matrix = np.linalg.inv(I_mat - coeffs[2] * W_mat)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(coeffs[2] * W_mat))))

    S_m2 = leontief_matrix @ (coeffs[1] * I_mat + coeffs[3] * W_mat)
    direct_impact = float(np.mean(np.diag(S_m2)))
    total_impact = float(np.mean(np.sum(S_m2, axis=1)))
    indirect_impact = float(total_impact - direct_impact)

    lisa = Moran_Local(y1, best_w)
    _gdf['LISA_Q'] = lisa.q
    _gdf['LISA_p'] = lisa.p_sim

    summary_dict = {
        'summary_text': str(results.summary()),
        'coefficients': {
            'Intercept (Alpha)': coeffs[0],
            'Local M2 (Beta)': coeffs[1],
            'Spatial Lag IPC (Rho)': coeffs[2],
            'Spatial Cross M2 (Theta)': coeffs[3],
            'Proxy VIIRS (Gamma)': coeffs[4]
        },
        'p_values': {
            'Local M2 (Beta)': p_values[1],
            'Spatial Lag IPC (Rho)': p_values[2],
            'Spatial Cross M2 (Theta)': p_values[3]
        },
        'impacts': {
            'Direct_Impact': direct_impact,
            'Indirect_Impact (Spillover)': indirect_impact,
            'Total_Impact': total_impact
        },
        'stability': {
            'Spectral_Radius': spectral_radius,
            'Is_Stable': bool(spectral_radius < 1.0)
        },
        'gdf_processed': _gdf,
        'W_name': best_name,
        'Moran_I': float(best_I),
        'Moran_p': float(best_p)
    }
    return summary_dict

model_results = run_spvar_model(gdf_data)

# ------------------------------------------------------------------------------
# 3. CONSTRUCCIÓN DE LA INTERFAZ
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Mapas Espaciales", 
    "📊 Estructura Espacial 3D", 
    "💡 Política Monetaria", 
    "📑 Auditoría Econométrica"
])

with tab1:
    col_var, col_info = st.columns([1, 3])
    with col_var:
        selected_var = st.selectbox(
            "Seleccionar Variable Macro para Mapa Coroplético",
            options=["Inflation_IPC", "M2_Growth", "VIIRS_Lights"],
            format_func=lambda x: "Inflación IPC (%)" if x == "Inflation_IPC" else ("Crecimiento M2 (%)" if x == "M2_Growth" else "Radiancia VIIRS")
        )
        st.info("El mapa interactivo renderiza la distribución departamental con tooltips detallados de las variables macroeconómicas.")

    center_lat = float(gdf_data['centroid_lat'].mean())
    center_lon = float(gdf_data['centroid_lon'].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Light Gray Canvas"
    )

    folium.Choropleth(
        geo_data=gdf_data.__geo_interface__,
        name='choropleth',
        data=gdf_data,
        columns=['code', selected_var],
        key_on='feature.properties.code',
        fill_color='YlOrRd' if selected_var == "Inflation_IPC" else 'Viridis',
        fill_opacity=0.85,
        line_opacity=0.3,
        legend_name=f"{selected_var}"
    ).add_to(m)

    folium.features.GeoJson(
        gdf_data,
        style_function=lambda x: {'fillColor': '#ffffff00', 'color': '#000000', 'weight': 0.5},
        highlight_function=lambda x: {'fillColor': '#00000000', 'color': '#000000', 'weight': 2},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['dpto_name', 'Inflation_IPC', 'M2_Growth', 'VIIRS_Lights'],
            aliases=['Departamento:', 'Inflación (%):', 'M2 Crecimiento (%):', 'Luces VIIRS:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    ).add_to(m)

    st_folium(m, width=1000, height=520, returned_objects=[])

with tab2:
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=gdf_data['centroid_lon'],
        y=gdf_data['centroid_lat'],
        z=gdf_data['Inflation_IPC'],
        mode='markers+text',
        text=gdf_data['dpto_name'],
        textposition="top center",
        marker=dict(
            size=np.maximum(gdf_data['M2_Growth'] * 2.2, 4.0),
            color=gdf_data['VIIRS_Lights'],
            colorscale='Plasma',
            colorbar=dict(title="VIIRS Lights", x=0.9),
            opacity=0.9,
            line=dict(width=1, color='#212529')
        )
    )])
    fig_3d.update_layout(
        title="Estructura Espacial 3D: Inflación (Elevación Z) vs Oferta Monetaria (Tamaño)",
        scene=dict(
            xaxis_title='Longitud',
            yaxis_title='Latitud',
            zaxis_title='Inflación IPC (%)',
            xaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#E9ECEF"),
            yaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#E9ECEF"),
            zaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#E9ECEF"),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8F9FA",
        font=dict(color="#212529", family="sans-serif"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    c = model_results['coefficients']
    p = model_results['p_values']
    imp = model_results['impacts']
    st_diag = model_results['stability']

    st.subheader("Informe de Política Monetaria y Transmisión Espacial")
    st.markdown(f"**Matriz Seleccionada:** `{model_results['W_name']}` | **Moran's I:** `{model_results['Moran_I']:.4f}` (p-val: `{model_results['Moran_p']:.4f}`)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Impacto Directo", f"{imp['Direct_Impact']:.4f}")
    c2.metric("Impacto Indirecto (Spillover)", f"{imp['Indirect_Impact (Spillover)']:.4f}")
    c3.metric("Impacto Total Espacial", f"{imp['Total_Impact']:.4f}")

    st.markdown("#### Mecanismos de Transmisión Evaluados:")
    st.write(f"- **Cruce Local ($\\beta$):** Un incremento del 1% en la oferta monetaria regional (M2) genera un cambio directo de **{c['Local M2 (Beta)']:.3f}%** en la inflación regional (p-value: {p['Local M2 (Beta)']:.4f}).")
    st.write(f"- **Contagio Inflacionario Directo ($\\rho$):** El desbordamiento de inflación de departamentos vecinos sobre la inflación local es de **{c['Spatial Lag IPC (Rho)']:.3f}** (p-value: {p['Spatial Lag IPC (Rho)']:.4f}).")
    st.write(f"- **Desbordamiento Cruzado de Dinero ($\\theta$):** El crecimiento del M2 en regiones vecinas impacta la inflación local en un **{c['Spatial Cross M2 (Theta)']:.3f}%** (p-value: {p['Spatial Cross M2 (Theta)']:.4f}).")

    st.markdown("#### Diagnóstico de Estabilidad del Sistema Dinámico:")
    st.write(f"Radio Espectral: `{st_diag['Spectral_Radius']:.4f}`")
    if st_diag['Is_Stable']:
        st.success("SISTEMA ESTABLE (Radio Espectral < 1)")
    else:
        st.error("SISTEMA INESTABLE (Radio Espectral ≥ 1)")

with tab4:
    st.subheader("Salida del Estimador OLS con Errores Estándar Robustos (HC1)")
    st.text(model_results['summary_text'])
