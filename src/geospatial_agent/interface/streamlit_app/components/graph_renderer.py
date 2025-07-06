from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

# Mapping from function_name to user-friendly labels
FRIENDLY_LABELS = {
    "data_sources": "Data Sources",
    "satellite_images": "Satellite Images (STAC APIs)",
    "gis_layers": "GIS Layers (Local/PostGIS)",
    "byo_data": "Bring-Your-Own Data (GeoTIFF, Shapefile)",
    "input_parser": "Input Parser & Layer Loader",
    "reasoning_pipeline": "Reasoning & Execution Pipeline",
    "llm_tool_planner": "LLM Tool Planner (CoT Reasoning)",
    "rag_lookup": "RAG Lookup",
    "knowledgebase": "Knowledgebase / RAG (GDAL/QGIS Docs + Past Workflows)",
    "critic": "Critic & Auto-Repair (Error Handling)",
    "execution_engine": "Execution Engine (GeoPandas / Rasterio / PyQGIS)",
    "interactive_viz": "Interactive Visualization",
    "map_output": "Map Output (Mapbox / Leaflet)",
    "cot_timeline": "CoT Timeline Panel",
    "exec_summary": "Exec Summary Download (PDF/JSON)",
    "user_refine_aoi": "User Refine AOI"
}
# Color mapping for step types
COLOR_MAP = {
    # 1. Deep Blue
    "data_sources": "#1976d2",
    # 2,3,4. Light Blue
    "satellite_images": "#90caf9",
    "gis_layers": "#90caf9",
    "byo_data": "#90caf9",
    # 6. Bright Green
    "reasoning_pipeline": "#00e676",
    # 5,7,11,10. Green
    "input_parser": "#43a047",
    "llm_tool_planner": "#43a047",
    "execution_engine": "#43a047",
    "critic": "#43a047",
    # 9. Orange
    "knowledgebase": "#ffb300",
    # 12. Bright Pink
    "interactive_viz": "#e040fb",
    # 13,14,15. Light Pink
    "map_output": "#f8bbd0",
    "cot_timeline": "#f8bbd0",
    "exec_summary": "#f8bbd0",
    # User Refine AOI (optional, dashed feedback)
    "user_refine_aoi": "#bdbdbd"
}

def render_workflow_graph(plan, edges=None, height=600, width="100%"):
    net = Network(height=f"{height}px", width=width, directed=True, notebook=False)
    net.barnes_hut()
    for idx, step in enumerate(plan):
        fn = step['function_name']
        friendly = FRIENDLY_LABELS.get(fn, fn.replace('_', ' ').title())
        border_color = COLOR_MAP.get(fn, "#1976d2")
        label = f"{friendly}"
        title = f"<b>{friendly}</b><br><b>Parameters:</b> {step['parameters']}"
        net.add_node(
            idx,
            label=label,
            title=title,
            shape='box',
            color={
                'background': '#fff',
                'border': border_color,
                'highlight': {
                    'background': '#e3f2fd',
                    'border': border_color
                }
            },
            borderWidth=3,
            shadow=True,
            shapeProperties={'borderRadius': 16},
            font={
                'size': 22,
                'face': 'Segoe UI, Arial, sans-serif',
                'color': '#222',
                'bold': True
            },
            margin=20
        )
    # Add edges
    if edges:
        for from_idx, to_idx, label in edges:
            if label == "dashed":
                net.add_edge(from_idx, to_idx, width=3, color="#757575", arrows="to", dashes=True)
            else:
                net.add_edge(from_idx, to_idx, width=3, color="#757575", arrows="to", label=label)
    else:
        for idx in range(len(plan) - 1):
            net.add_edge(idx, idx + 1, width=3, color="#757575", arrows="to")
    # Use hierarchical layout for left-to-right flow
    net.set_options("""
    var options = {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed",
          "levelSeparation": 350,
          "nodeSpacing": 300,
          "treeSpacing": 400
        }
      },
      "nodes": {
        "borderWidth": 3,
        "shadow": true,
        "shape": "box",
        "shapeProperties": {
          "borderRadius": 16
        },
        "font": {
          "size": 22,
          "face": "Segoe UI, Arial, sans-serif",
          "color": "#222",
          "bold": true
        },
        "margin": 20
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.2
          }
        },
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal",
          "roundness": 0.4
        },
        "color": {
          "color": "#757575",
          "highlight": "#1976d2",
          "inherit": false
        },
        "font": {
          "size": 20,
          "align": "top",
          "background": "#fff",
          "strokeWidth": 3,
          "strokeColor": "#fff",
          "bold": true
        }
      },
      "physics": {
        "enabled": false
      }
    }
    """)
    return net.generate_html()
