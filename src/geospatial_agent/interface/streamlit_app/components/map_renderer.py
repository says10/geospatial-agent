import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon
from folium.plugins import DualMap
from folium.plugins import TimestampedGeoJson

def render_map(gdf, map_height=400):
    """
    Renders a Folium map from a GeoPandas GeoDataFrame in Streamlit.
    Args:
        gdf (GeoDataFrame): The geospatial data to plot.
        map_height (int): Height of the map in pixels.
    """
    if gdf.empty:
        st.warning("No geospatial data to display.")
        return
    centroid = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, height=map_height)

    # Define color mapping for risk levels
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }

    # Style function for polygons
    def style_function(feature):
        risk = feature['properties'].get('risk_level', 'Low')
        return {
            'fillColor': risk_colors.get(risk, 'gray'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        }

    # Add polygons with popups/tooltips
    folium.GeoJson(
        gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['zone_name', 'risk_level']),
        popup=folium.GeoJsonPopup(fields=['zone_name', 'risk_level'])
    ).add_to(m)

    folium_static(m, width=700, height=map_height)

def render_dual_map(left_gdf, right_gdf, left_title="Before", right_title="After", map_height=500):
    """
    Renders a dual map using Folium's DualMap plugin for side-by-side comparison.
    
    Args:
        left_gdf (GeoDataFrame): Data for the left map
        right_gdf (GeoDataFrame): Data for the right map  
        left_title (str): Title for the left map
        right_title (str): Title for the right map
        map_height (int): Height of the map in pixels
    """
    # Calculate centroid from both datasets
    if not left_gdf.empty and not right_gdf.empty:
        combined_geom = left_gdf.geometry.unary_union.union(right_gdf.geometry.unary_union)
        centroid = combined_geom.centroid
    elif not left_gdf.empty:
        centroid = left_gdf.geometry.unary_union.centroid
    elif not right_gdf.empty:
        centroid = right_gdf.geometry.unary_union.centroid
    else:
        st.warning("No geospatial data to display.")
        return
    
    # Create dual map
    m = DualMap(location=[centroid.y, centroid.x], zoom_start=10)
    
    # Define color mapping for risk levels
    risk_colors = {
        'Low': 'green',
        'Medium': 'orange', 
        'High': 'red'
    }
    
    # Style function for polygons
    def style_function(feature):
        risk = feature['properties'].get('risk_level', 'Low')
        return {
            'fillColor': risk_colors.get(risk, 'gray'),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        }
    
    # Add data to left map
    if not left_gdf.empty:
        folium.GeoJson(
            left_gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['zone_name', 'risk_level']),
            popup=folium.GeoJsonPopup(fields=['zone_name', 'risk_level'])
        ).add_to(m.m1)
        
        # Add title to left map
        folium.map.Marker(
            [centroid.y + 0.01, centroid.x - 0.01],
            icon=folium.DivIcon(
                html=f'<div style="background-color: white; padding: 5px; border: 2px solid black; font-weight: bold;">{left_title}</div>',
                icon_size=(100, 20),
                icon_anchor=(0, 0)
            )
        ).add_to(m.m1)
    
    # Add data to right map
    if not right_gdf.empty:
        folium.GeoJson(
            right_gdf,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=['zone_name', 'risk_level']),
            popup=folium.GeoJsonPopup(fields=['zone_name', 'risk_level'])
        ).add_to(m.m2)
        
        # Add title to right map
        folium.map.Marker(
            [centroid.y + 0.01, centroid.x - 0.01],
            icon=folium.DivIcon(
                html=f'<div style="background-color: white; padding: 5px; border: 2px solid black; font-weight: bold;">{right_title}</div>',
                icon_size=(100, 20),
                icon_anchor=(0, 0)
            )
        ).add_to(m.m2)
    
    # Display the dual map
    folium_static(m, width=1200, height=map_height)

def render_temporal_map(geojson_data, map_height=500):
    """
    Renders a Folium map with a TimestampedGeoJson plugin for temporal playback in Streamlit.
    Args:
        geojson_data (dict): GeoJSON data with time properties (e.g., 'times' or 'time' in properties).
        map_height (int): Height of the map in pixels.
    """
    import json
    if not geojson_data:
        st.warning("No temporal GeoJSON data to display.")
        return
    # Center map on first feature or default
    try:
        features = geojson_data.get('features', [])
        if features:
            first_geom = features[0]['geometry']
            coords = first_geom.get('coordinates', [[0, 0]])
            # Support for Polygon/Point
            if first_geom['type'] == 'Point':
                center = coords[::-1]
            else:
                center = coords[0][0][::-1] if isinstance(coords[0][0], (list, tuple)) else coords[0][::-1]
        else:
            center = [0, 0]
    except Exception:
        center = [0, 0]
    m = folium.Map(location=center, zoom_start=10, height=map_height)
    TimestampedGeoJson(
        data=geojson_data,
        transition_time=500,
        period="PT1H",  # 1 hour, adjust as needed
        add_last_point=True,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD HH:mm:ss',
        time_slider_drag_update=True
    ).add_to(m)
    folium_static(m, width=900, height=map_height)
