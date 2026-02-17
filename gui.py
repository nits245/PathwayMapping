import sys
import tempfile
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, \
    QPushButton, QComboBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
import math
from datetime import datetime
from data_processing import DataProcessing
from traffic_flow_predictor import TrafficFlowPredictor
from graph_time_integration import enrich_graph_with_travel_times
from graph_pathfinding import yen_k_shortest_paths
from networkx_search_algorithms import ASTAR, Dijkstra, DFS, BFS, GBFS, IDASTAR
import itertools

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
        lats.append(attrs['x'] + lat_offset)
        lons.append(attrs['y'] + lon_offset)
        labels.append(f"SCATS {attrs['scats_number']}")

    fig = go.Figure(go.Scattermap(
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
                name=f'Path {i + 1}'
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


class MapWindow(QMainWindow):
    def __init__(self, scats_filepath, model_path='models/gru_model.keras'):
        super().__init__()
        self.setWindowTitle("SCATS Map with Pathfinding")
        self.init_ui()
        self.model_paths = {
            "LSTM": 'models/lstm_model.keras',
            "GRU": 'models/gru_model.keras',
            "CNN": 'models/conv1d.keras'
        }
        # Initialize data processing
        dp = DataProcessing(scats_filepath)
        dp.process_scats_data()
        self.base_graph = dp.create_DiGraph()
        
        # Add forced edges from graph_pathfinding
        self.add_forced_edges(dp)
        self.scats_filepath = scats_filepath

        
        # Setup UI

        self.update_predictor()
        # Show initial map
        self.show_initial_map()

    def add_forced_edges(self, dp):
        """Add forced edges between disconnected nodes"""
        # Victoria Street to Barker's Road connection
        if 27 in self.base_graph.nodes and 138 in self.base_graph.nodes:
            self.add_bidirectional_edge(27, 138, dp)

        # Node 119 to 134 connection
        if 119 in self.base_graph.nodes and 134 in self.base_graph.nodes:
            self.add_bidirectional_edge(119, 134, dp)

        # SCATS 3001 connections
        nodes_3001 = [nid for nid, data in self.base_graph.nodes(data=True)
                      if data.get('scats_number') == 3001]
        for u, v in itertools.combinations(nodes_3001, 2):
            self.add_bidirectional_edge(u, v, dp, scats=3001)

    def add_bidirectional_edge(self, u, v, dp, scats=None):
        """Add bidirectional edge between nodes"""
        lat_u, lon_u = self.base_graph.nodes[u]['x'], self.base_graph.nodes[u]['y']
        lat_v, lon_v = self.base_graph.nodes[v]['x'], self.base_graph.nodes[v]['y']
        dist = dp.haversine_distance(lat_u, lon_u, lat_v, lon_v)
        scats_num = scats or self.base_graph.nodes[u]['scats_number']
        self.base_graph.add_edge(u, v, distance_km=dist, scat_point=scats_num)
        self.base_graph.add_edge(v, u, distance_km=dist, scat_point=scats_num)

    def init_ui(self):
        """Initialize the user interface"""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Map view
        self.web = QWebEngineView()
        main_layout.addWidget(self.web, 4)
        
        # Control panel
        control = self.create_control_panel()
        main_layout.addWidget(control, 1)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_control_panel(self):
        """Create the control panel"""
        control = QWidget()
        layout = QVBoxLayout()
        
        # Time input
        layout.addWidget(QLabel("Time (HHMM):"))
        self.time_input = QLineEdit("0900")
        layout.addWidget(self.time_input)
        
        # SCATS inputs
        layout.addWidget(QLabel("Origin SCATS:"))
        self.origin_input = QLineEdit()
        layout.addWidget(self.origin_input)
        
        layout.addWidget(QLabel("Destination SCATS:"))
        self.dest_input = QLineEdit()
        layout.addWidget(self.dest_input)
        
        # Algorithm selection
        layout.addWidget(QLabel("Algorithm:"))
        self.algo_box = QComboBox()
        self.algo_box.addItems(["ASTAR", "DIJKSTRA", "IDASTAR", "GBFS", "BFS", "DFS"])
        layout.addWidget(self.algo_box)
        
        # Model selection
        layout.addWidget(QLabel("Model:"))
        self.model_box = QComboBox()
        self.model_box.addItems(["LSTM", "CNN", "GRU"])
        layout.addWidget(self.model_box)
        
        # Number of paths
        layout.addWidget(QLabel("Number of paths (k):"))
        self.k_input = QLineEdit("5")
        layout.addWidget(self.k_input)
        
        # Search button
        self.find_button = QPushButton("Find Paths")
        self.find_button.clicked.connect(self.on_find_path)
        layout.addWidget(self.find_button)
        
        # Info label
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)
        
        control.setLayout(layout)
        return control

    def show_initial_map(self):
        """Show initial map without paths"""
        html = generate_map_from_graph(self.base_graph)
        self.web.load(QUrl.fromLocalFile(html))
    def update_predictor(self):


        current_model = self.model_box.currentText()
        model_path = self.model_paths[current_model]
        print(model_path)
        self.predictor = TrafficFlowPredictor(self.scats_filepath, model_path)
    def on_find_path(self):
        """Handle search button click"""
        self.find_button.setEnabled(False)
        self.info_label.setText("Searching...")
        self.update_predictor()
        scats_filepath = "datasets/Scats Data October 2006.xls"
        try:
            # Get inputs
            time_str = self.time_input.text().strip()
            origin = int(self.origin_input.text().strip())
            dest = int(self.dest_input.text().strip())
            k = int(self.k_input.text().strip())
            
            # Get algorithm and model
            algo = self.algo_box.currentText().upper()
            model = self.model_box.currentText()
            # Process the search
            paths, times = self.find_paths(time_str, origin, dest, algo, k)
            
            if paths:
                total_time = sum(times)
                path_info = "\n".join(
                    f"Path {i+1}: {tt:.1f}s"
                    for i, tt in enumerate(times)
                )
                self.info_label.setText(
                    f"Found {len(paths)} paths.\nTotal time: {total_time:.1f}s\n{path_info}"
                )
                
                # Update map with paths
                tm = datetime.strptime(time_str, '%H%M').strftime('%H:%M')
                G = enrich_graph_with_travel_times(self.base_graph.copy(), self.predictor, tm)
                html = generate_map_from_graph(G, paths=paths)
                self.web.load(QUrl.fromLocalFile(html))
            else:
                self.info_label.setText("No paths found")
                
        except ValueError as e:
            self.info_label.setText(f"Error: {str(e)}")
        except Exception as e:
            self.info_label.setText(f"Error: {str(e)}")
        finally:
            self.find_button.setEnabled(True)

    def find_paths(self, time_str, origin, dest, algo, k):
        """Find k-shortest paths between origin and destination"""
        # Map SCATS numbers to nodes
        scats_to_nodes = {}
        for u, v, data in self.base_graph.edges(data=True):
            sensor = data.get('scat_point')
            if sensor is not None:
                scats_to_nodes.setdefault(sensor, set()).add(u)
        
        # Find closest nodes for origin and destination
        orig_nodes = scats_to_nodes.get(origin)
        dest_nodes = scats_to_nodes.get(dest)
        if not orig_nodes or not dest_nodes:
            raise ValueError(f"Invalid SCATS numbers")
        
        best_u, best_v = min(
            ((u, v, math.hypot(
                self.base_graph.nodes[u]['x'] - self.base_graph.nodes[v]['x'],
                self.base_graph.nodes[u]['y'] - self.base_graph.nodes[v]['y']
            )) for u in orig_nodes for v in dest_nodes),
            key=lambda x: x[2]
        )[:2]
        
        # Enrich graph with travel times
        tm = datetime.strptime(time_str, '%H%M').strftime('%H:%M')
        G = enrich_graph_with_travel_times(self.base_graph.copy(), self.predictor, tm)
        
        # Get algorithm class
        alg_map = {
            'ASTAR': ASTAR,
            'DIJKSTRA': Dijkstra,
            'IDASTAR': IDASTAR,
            'GBFS': GBFS,
            'BFS': BFS,
            'DFS': DFS
        }
        Alg = alg_map[algo]
        
        # Find paths using Yen's algorithm
        return yen_k_shortest_paths(G, best_u, best_v, Alg, k)

if __name__ == "__main__":
    scats_fp = "datasets/Scats Data October 2006.xls"
    # Default model (use GRU)
    #model_path = 'models/gru_model.keras'
    app = QApplication(sys.argv)
    window = MapWindow(scats_fp)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())
