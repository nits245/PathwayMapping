import sys
import tempfile
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import math
from datetime import datetime
from data_processing import DataProcessing
from traffic_flow_predictor import TrafficFlowPredictor
from graph_time_integration import enrich_graph_with_travel_times


def process_location_data(filepath):
    location_data = (
        pd.read_excel(filepath, sheet_name='Data', skiprows=1)
        .dropna()
    )
    location_data.columns = location_data.columns.str.strip()
    location_data = location_data[['Location', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()
    return location_data

def generate_map_from_graph(G, paths=None, lat_offset=0.0015, lon_offset=0.0013):
    lats = []
    lons = []
    labels = []
    
    for node_id, attrs in G.nodes(data=True):
        # apply offset to shift nodes up and right in the visual map
        lats.append(attrs['x'] + lat_offset)
        lons.append(attrs['y'] + lon_offset)
        #labels.append(f"Node {node_id}")
        labels.append(f"SCATS {attrs['scats_number']}")  # Display SCATS number

    fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers+text',
        text=labels,
        textfont=dict(family="Arial Black", size=12, color="black"),
        marker=dict(size=14, color='blue'),
        textposition="top right"
    ))
    
    fig.update_traces(text=labels, mode='markers+text')

    if paths:
        other_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, path in enumerate(paths):
            offset_lat = i * 0.00003  # slight vertical nudge
            offset_lon = i * 0.00003  # slight horizontal nudge

            path_lats = [G.nodes[n]['x'] + lat_offset + offset_lat for n in path]
            path_lons = [G.nodes[n]['y'] + lon_offset + offset_lon for n in path]
            
            color = 'red' if i == 0 else other_colors[(i - 1) % len(other_colors)]

            fig.add_trace(go.Scattermapbox(
                lat=path_lats,
                lon=path_lons,
                mode='lines+markers',
                line=dict(width=4, color=color),
                marker=dict(size=6, color=color),
                name=f'Path {i+1}'
            ))


        
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=np.mean(lats), lon=np.mean(lons)),
            zoom=14
        ),
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}
    )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    # embed javascript to update marker and label size on zoom in and out
    post_script = '''
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    gd.on('plotly_relayout', function(eventdata){
        if(eventdata['mapbox.zoom']){
            var z = eventdata['mapbox.zoom'];
            var newSize = Math.max(5, 20 - z * 1.5);
            Plotly.restyle(gd, {'marker.size': newSize});
        }
    });
    '''
    fig.write_html(temp_file.name, include_plotlyjs='cdn', post_script=post_script)
    return temp_file.name



def show_map_gui(html_path):
    app = QApplication(sys.argv)
    web = QWebEngineView()
    web.load(QUrl.fromLocalFile(html_path))
    window = QMainWindow()
    window.setWindowTitle("SCATS Location Map")
    window.setCentralWidget(web)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())

