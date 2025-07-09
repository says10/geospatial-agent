import streamlit as st
from src.geospatial_agent.interface.streamlit_app.components.map_renderer import render_map, render_dual_map, render_temporal_map
from src.geospatial_agent.interface.streamlit_app.components.graph_renderer import render_workflow_graph
from src.geospatial_agent.interface.streamlit_app.components.audio_renderer import render_audio_input_if_visible
import geopandas as gpd
from shapely.geometry import Polygon
import streamlit.components.v1 as components
import json
import time
import datetime

st.set_page_config(page_title="Geospatial Agent", layout="wide")

# --- Default Workflow Plan and Edges for Graph ---
workflow_plan = [
    {"function_name": "data_sources", "parameters": {"sources": ["Satellite", "GIS", "User Upload"]}},
    {"function_name": "satellite_images", "parameters": {"api": "STAC"}},
    {"function_name": "gis_layers", "parameters": {"db": "PostGIS"}},
    {"function_name": "byo_data", "parameters": {"formats": ["GeoTIFF", "Shapefile"]}},
    {"function_name": "input_parser", "parameters": {"input": "User Query"}},
    {"function_name": "reasoning_pipeline", "parameters": {"steps": "CoT Reasoning"}},
    {"function_name": "llm_tool_planner", "parameters": {"tools": ["GeoPandas", "Rasterio"]}},
    {"function_name": "rag_lookup", "parameters": {"docs": "Knowledgebase"}},
    {"function_name": "knowledgebase", "parameters": {"type": "GDAL/QGIS Docs"}},
    {"function_name": "critic", "parameters": {"mode": "Auto-Repair"}},
    {"function_name": "execution_engine", "parameters": {"engine": "GeoPandas"}},
    {"function_name": "interactive_viz", "parameters": {"output": "Dual Map"}},
    {"function_name": "map_output", "parameters": {"format": "Folium/Leaflet"}},
    {"function_name": "cot_timeline", "parameters": {"panel": "CoT Steps"}},
    {"function_name": "exec_summary", "parameters": {"format": "PDF/JSON"}}
]
workflow_edges = [
    (0, 1, "satellite"),
    (0, 2, "gis"),
    (0, 3, "byo"),
    (1, 4, "parse"),
    (2, 4, "parse"),
    (3, 4, "parse"),
    (4, 5, "reason"),
    (5, 6, "plan"),
    (6, 7, "lookup"),
    (7, 8, "kb"),
    (6, 9, "critic"),
    (6, 10, "execute"),
    (10, 11, "visualize"),
    (11, 12, "output"),
    (5, 13, "timeline"),
    (12, 14, "summary")
]

# --- Sidebar: Chain-of-Thought Reasoning ---

st.sidebar.title(" Chain-of-Thought")

# Determine which map mode and dataset(s) are being used for CoT
map_mode = st.session_state.get("map_view_select", "Single Map")
single_map_choice = st.session_state.get("single_map_choice", "Observed")

if map_mode == "Single Map":
    cot_steps = [
        "Parse user query",
        f"Select '{single_map_choice}' flood data",
        "Filter by selected date",
        "Visualize flood zones on map"
    ]
else:
    cot_steps = [
        "Parse user query",
        "Select 'Observed' and 'Predicted' flood data",
        "Filter both by selected date",
        "Visualize side-by-side comparison"
    ]

# Initialize session state for CoT animation and results
if "cot_animating" not in st.session_state:
    st.session_state.cot_animating = False
if "cot_step" not in st.session_state:
    st.session_state.cot_step = 0
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# Sidebar rendering
for i, step in enumerate(cot_steps):
    if st.session_state.cot_animating and i == st.session_state.cot_step:
        st.sidebar.markdown(
            f"<div style='background-color:#1976d2;color:#fff;padding:6px;border-radius:4px;font-weight:bold;'>➡️ Step {i+1}: {step}</div>",
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(f"Step {i+1}: {step}")
st.sidebar.markdown("---")
if st.session_state.cot_animating:
    st.sidebar.info(f"Processing step {st.session_state.cot_step+1}...")
elif st.session_state.show_results:
    st.sidebar.success("Workflow complete!")
else:
    st.sidebar.info("Status: Ready for your query!")

# --- Main App Layout ---
st.title("🌍 Geospatial Reasoning Agent")
st.markdown("""
Describe your geospatial challenge or question below. The agent will use intelligent reasoning to generate, explain, and execute a multi-step geospatial workflow—automatically integrating your data and visualizing the results.
""")

# --- Main UI Columns ---
col_left, col_right = st.columns([7, 1], gap="large")

with col_left:
    # Remove map view toggle, always use Dual Map
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)  # Add top spacing for visual comfort
    # Audio input appears above the text input if activated
    audio_file = render_audio_input_if_visible()
    user_query = st.text_input(
        "Describe your geospatial task:",
        "Map flood risk for my city",
        help="Describe what you want to analyze."
    )
    st.info('Example: "Map flood-prone zones near Bengaluru"', icon="💡")
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # Spacing before submit
    col_input, col_submit = st.columns([5, 2], gap="small")
    with col_input:
        pass  # Used for alignment only
    with col_submit:
        submit = st.button(
            "Submit",
            use_container_width=True,
            help="Submit your query for processing."
        )
        st.markdown(
            """
            <style>
            div[data-testid="baseButton-secondary"] button {
                background: linear-gradient(90deg, #1976d2 0%, #1565c0 100%);
                color: #fff;
                font-weight: 600;
                border-radius: 6px;
                border: none;
                box-shadow: 0 2px 8px rgba(25,118,210,0.08);
                transition: background 0.2s;
            }
            div[data-testid="baseButton-secondary"] button:hover {
                background: linear-gradient(90deg, #2196f3 0%, #1976d2 100%);
                color: #fff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)  # Spacing after submit

    # --- CoT Animation Trigger ---
    if (submit or audio_file) and not st.session_state.cot_animating and not st.session_state.show_results:
        st.session_state.cot_animating = True
        st.session_state.cot_step = 0
        st.session_state.show_results = False
        st.rerun()

    # --- CoT Animation Logic ---
    if st.session_state.cot_animating:
        if st.session_state.cot_step < len(cot_steps) - 1:
            time.sleep(0.5)
            st.session_state.cot_step += 1
            st.rerun()
        else:
            time.sleep(1)
            st.session_state.cot_animating = False
            st.session_state.show_results = True
            st.rerun()

    # --- Map Output Area ---
    if st.session_state.show_results:
        st.success("Your query has been received! Please scroll down to view the results.")
        st.markdown("### 🗺️ Dual Map: Context vs. Flood Prediction Output")
        st.info(
            "**Left Map:** Context layers for Bengaluru (land cover, study area, observed flood extent).\n"
            "**Right Map:** Predicted flood extent for Bengaluru, plus study area.\n"
            "Use the map controls to zoom, pan, and compare both sides."
        )
        # --- Temporal Playback Controls (move to top) ---
        # Load all available times from both observed and predicted flood data
        all_times = set()
        for gdf_path in ["mock_data/mock_flood_observed.geojson", "mock_data/mock_flood_predicted.geojson"]:
            gdf = gpd.read_file(gdf_path)
            # Convert all time values to datetime.date objects
            for t in gdf["time"].unique():
                if isinstance(t, str):
                    all_times.add(datetime.datetime.strptime(t, "%Y-%m-%d").date())
                elif hasattr(t, 'date'):
                    all_times.add(t.date())
        all_times = sorted(list(all_times))
        # Use a slider for date selection (user-friendly for a small set of dates)
        selected_time = st.slider(
            "Select date for flood visualization:",
            min_value=all_times[0],
            max_value=all_times[-1],
            value=all_times[0]
        )
        selected_time_str = selected_time.strftime("%Y-%m-%d")
        st.markdown(f"**Selected date:** {selected_time_str}")
        import geopandas as gpd
        import folium
        from streamlit_folium import folium_static
        from folium.plugins import DualMap
        landcover = gpd.read_file("mock_data/mock_landcover.geojson")
        study_area = gpd.read_file("mock_data/mock_study_area.geojson")
        observed_flood = gpd.read_file("mock_data/mock_flood_observed.geojson")
        observed_flood["time"] = observed_flood["time"].astype(str)
        predicted_flood = gpd.read_file("mock_data/mock_flood_predicted.geojson")
        predicted_flood["time"] = predicted_flood["time"].astype(str)
        # Filter observed and predicted flood layers by selected time
        observed_flood = observed_flood[observed_flood["time"] == selected_time_str]
        predicted_flood = predicted_flood[predicted_flood["time"] == selected_time_str]
        # Ensure 'time' is string for Folium/GeoJSON serialization
        observed_flood["time"] = observed_flood["time"].astype(str)
        predicted_flood["time"] = predicted_flood["time"].astype(str)
        center = [12.95, 77.62]
        m = DualMap(location=center, zoom_start=12)
        # Left: Land cover, study area, observed flood
        folium.GeoJson(
            landcover,
            name="Land Cover",
            style_function=lambda f: {"fillColor": {"Urban": "gray", "Forest": "green", "Barren": "brown", "Agriculture": "yellow"}.get(f["properties"]["class"], "blue"),
                                  "color": "black", "weight": 1, "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(fields=["class"])
        ).add_to(m.m1)
        folium.GeoJson(
            study_area,
            name="Study Area",
            style_function=lambda f: {"color": "red", "weight": 2, "fillOpacity": 0},
            tooltip="Study Area"
        ).add_to(m.m1)
        folium.GeoJson(
            observed_flood,
            name="Observed Flood",
            style_function=lambda f: {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.4},
            tooltip="Observed Flood"
        ).add_to(m.m1)
        # Right: Predicted flood, study area
        folium.GeoJson(
            predicted_flood,
            name="Predicted Flood",
            style_function=lambda f: {"fillColor": "purple", "color": "purple", "weight": 1, "fillOpacity": 0.5},
            tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], labels=False, sticky=True, localize=True, style=("background-color: #e1bee7; color: #4a148c; font-weight: bold; padding: 4px; border-radius: 4px;"),
                                          prefix="Predicted Flood")
        ).add_to(m.m2)
        folium.GeoJson(
            study_area,
            name="Study Area",
            style_function=lambda f: {"color": "red", "weight": 2, "fillOpacity": 0},
            tooltip="Study Area"
        ).add_to(m.m2)
        # Place legends as expanders side by side above the map
        legend_left, legend_right = st.columns(2)
        with legend_left:
            with st.expander("Legend", expanded=False):
                st.markdown("""
                **Land Cover Colors:**  
                - Gray: Urban  
                - Green: Forest  
                - Yellow: Agriculture  
                - Brown: Barren  
                **Blue:** Observed Flood  
                **Red outline:** Study Area
                """)
        with legend_right:
            with st.expander("Legend", expanded=False):
                st.markdown("""
                **Purple:** Predicted Flood  
                **Red outline:** Study Area
                """)
        folium_static(m, width=1200, height=600)
        # Add vertical space between map and legends
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("### Workflow Graph")
        graph_html = render_workflow_graph(workflow_plan, workflow_edges)
        components.html(graph_html, height=900, width=1800)

    else:
        st.info("Submit a query to start the geospatial workflow!")

with col_right:
    st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)  # Align mic button with input
    if 'show_mic' not in st.session_state:
        st.session_state['show_mic'] = False
    mic_pressed = st.button("🎤 Speak", key="mic_btn")
    if mic_pressed:
        st.session_state['show_mic'] = True 