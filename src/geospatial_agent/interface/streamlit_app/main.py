import streamlit as st
from src.geospatial_agent.interface.streamlit_app.components.map_renderer import render_map, render_dual_map, render_temporal_map
from src.geospatial_agent.interface.streamlit_app.components.graph_renderer import render_workflow_graph
from src.geospatial_agent.interface.streamlit_app.components.audio_renderer import render_audio_input_if_visible
import geopandas as gpd
from shapely.geometry import Polygon
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Geospatial Agent", layout="wide")

# --- Sidebar: Chain-of-Thought Reasoning ---

st.sidebar.title(" Chain-of-Thought")
cot_steps = [
    "1. Parse user query",
    "2. Retrieve relevant tools/data",
    "3. Plan workflow steps",
    "4. Execute geoprocessing",
    "5. Visualize results"
]
for i, step in enumerate(cot_steps, 1):
    st.sidebar.markdown(f"**Step {i}:** {step}")

st.sidebar.markdown("---")
st.sidebar.info("Status: Ready for your query!")

# --- Main App Layout ---
st.title("🌍 Geospatial Reasoning Agent")
st.markdown("""
Enter a geospatial analysis task (e.g., *Map flood risk for this area*). The agent will plan and visualize the workflow step-by-step.
""")

# --- Main UI Columns ---
col_left, col_right = st.columns([7, 1], gap="large")

with col_left:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)  # Add top spacing for visual comfort
    map_view = st.selectbox(
        "Map View",
        ["Single Map", "Dual Map (Before/After)"],
        index=0,
        key="map_view_select",
        help="Single Map: Visualize a single dataset with temporal playback. Dual Map: Compare two datasets side by side with synchronized temporal playback."
    )
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)  # Spacing between controls
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
        # Style the submit button for a professional look
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

    # --- Map Output Area ---
    if map_view == "Dual Map (Before/After)":
        if submit or audio_file:
            if submit:
                st.success("Your query has been received! Please scroll down to view the results.")
            st.markdown("### 🗺️ Dual Map Temporal Playback: Prediction vs. Observation")
            st.info("Compare the evolution of Prediction and Observation datasets over time.")
            import json
            import geopandas as gpd
            # Load mock datasets for Prediction and Observation
            with open("data/static/mock_pred.geojson", "r") as f:
                pred_geojson = json.load(f)
            with open("data/static/mock_obs.geojson", "r") as f:
                obs_geojson = json.load(f)
            # Extract all unique timestamps from both datasets
            def extract_times(geojson):
                times = set()
                for feat in geojson.get("features", []):
                    tlist = feat["properties"].get("times")
                    if tlist:
                        times.update(tlist)
                return times
            all_times = sorted(list(extract_times(pred_geojson) | extract_times(obs_geojson)))
            if all_times:
                # Synchronized time slider for both maps
                time_idx = st.slider("Select time index", 0, len(all_times)-1, 0, format="%d")
                selected_time = all_times[time_idx]
                st.write(f"Showing features for: {selected_time}")
                # Filter each dataset for the selected time
                def filter_by_time(geojson, t):
                    feats = [f for f in geojson.get("features", []) if t in f["properties"].get("times", [])]
                    return {"type": "FeatureCollection", "features": feats}
                pred_filtered = filter_by_time(pred_geojson, selected_time)
                obs_filtered = filter_by_time(obs_geojson, selected_time)
                # Convert filtered features to GeoDataFrames with CRS set
                if pred_filtered["features"]:
                    gdf_pred = gpd.GeoDataFrame.from_features(pred_filtered["features"])
                    gdf_pred.set_crs("EPSG:4326", inplace=True)
                else:
                    gdf_pred = gpd.GeoDataFrame(columns=["geometry"])
                if obs_filtered["features"]:
                    gdf_obs = gpd.GeoDataFrame.from_features(obs_filtered["features"])
                    gdf_obs.set_crs("EPSG:4326", inplace=True)
                else:
                    gdf_obs = gpd.GeoDataFrame(columns=["geometry"])
                # Render the dual map with professional labels
                render_dual_map(gdf_pred, gdf_obs, left_title="Prediction", right_title="Observation")
            else:
                st.warning("No time-aware features found in either dataset.")
        else:
            st.info("Submit a query to start the geospatial workflow!")
    elif map_view == "Single Map":
        if submit or audio_file:
            if submit:
                st.success("Your query has been received! Please scroll down to view the results.")
            st.markdown("### 🗺️ Single Map Temporal Playback")
            st.info("Explore how a single dataset changes over time using the slider below.")
            import json
            # Load mock dataset for single map
            mock_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[77.60, 12.90], [77.605, 12.905], [77.61, 12.91], [77.608, 12.915], [77.60, 12.912], [77.60, 12.90]]]
                        },
                        "properties": {
                            "zone_name": "Zone 1",
                            "risk_level": "Low",
                            "times": ["2023-07-01T00:00:00Z"]
                        }
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[77.62, 12.92], [77.625, 12.925], [77.63, 12.93], [77.628, 12.935], [77.62, 12.932], [77.62, 12.92]]]
                        },
                        "properties": {
                            "zone_name": "Zone 2",
                            "risk_level": "Medium",
                            "times": ["2023-07-02T00:00:00Z"]
                        }
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[77.64, 12.94], [77.645, 12.945], [77.65, 12.95], [77.648, 12.955], [77.64, 12.952], [77.64, 12.94]]]
                        },
                        "properties": {
                            "zone_name": "Zone 3",
                            "risk_level": "High",
                            "times": ["2023-07-03T00:00:00Z"]
                        }
                    }
                ]
            }
            # Extract all unique timestamps from the mock dataset
            def extract_times(geojson):
                times = set()
                for feat in geojson.get("features", []):
                    tlist = feat["properties"].get("times")
                    if tlist:
                        times.update(tlist)
                return times
            all_times = sorted(list(extract_times(mock_geojson)))
            if all_times:
                time_idx = st.slider("Select time index", 0, len(all_times)-1, 0, format="%d")
                selected_time = all_times[time_idx]
                st.write(f"Showing features for: {selected_time}")
                # Filter the dataset for the selected time
                def filter_by_time(geojson, t):
                    feats = [f for f in geojson.get("features", []) if t in f["properties"].get("times", [])]
                    return {"type": "FeatureCollection", "features": feats}
                filtered = filter_by_time(mock_geojson, selected_time)
                import geopandas as gpd
                if filtered["features"]:
                    gdf = gpd.GeoDataFrame.from_features(filtered["features"])
                    gdf.set_crs("EPSG:4326", inplace=True)
                else:
                    gdf = gpd.GeoDataFrame(columns=["geometry"])
                # Render the single map with professional styling
                render_map(gdf)
            else:
                st.warning("No time-aware features found in the dataset.")
        else:
            st.info("Submit a query to start the geospatial workflow!")
    else:
        if submit or audio_file:
            with st.spinner("Processing your query, please wait..."):
                run_workflow(user_query=user_query if submit else None, audio_file=audio_file if audio_file else None, map_view=map_view)
        else:
            st.info("Submit a query to start the geospatial workflow!")

with col_right:
    st.markdown("<div style='height: 38px;'></div>", unsafe_allow_html=True)  # Align mic button with input
    if 'show_mic' not in st.session_state:
        st.session_state['show_mic'] = False
    mic_pressed = st.button("🎤 Speak", key="mic_btn")
    if mic_pressed:
        st.session_state['show_mic'] = True 