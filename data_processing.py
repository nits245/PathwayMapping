import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import re
import networkx as nx
import math

class DataProcessing:
    def __init__(self, filepath):
        self.filepath = filepath
        self.window_size = 4
        self.traffic_long = None
        self.scaler = None
        # _loc_scaler will be set after processing data
        self._loc_scaler = None

    def process_scats_data(self):
        # Load and clean
        traffic_data = (
            pd.read_excel(self.filepath, sheet_name='Data', skiprows=1)
            .dropna()
        )
        traffic_data.columns = traffic_data.columns.str.strip()

        # Keep relevant columns
        relevant_columns = [
            'SCATS_Number', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date'
        ] + [f'V{str(i).zfill(2)}' for i in range(96)]
        traffic_data = traffic_data[relevant_columns]

        # Melt and create timestamps
        traffic_long = traffic_data.melt(
            id_vars=['SCATS_Number', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date'],
            var_name='Interval',
            value_name='Traffic_Volume'
        )
        traffic_long['Timestamp'] = (
            pd.to_datetime(traffic_long['Date']).dt.normalize() +
            traffic_long['Interval'].str.extract(r'V(\d+)').astype(int)[0]
                .apply(lambda x: timedelta(minutes=15 * x))
        )
        traffic_long = traffic_long[[
            'SCATS_Number', 'Traffic_Volume',
            'NB_LATITUDE', 'NB_LONGITUDE', 'Timestamp'
        ]].sort_values(by=[
            'SCATS_Number', 'NB_LATITUDE', 'NB_LONGITUDE', 'Timestamp'
        ]).reset_index(drop=True)

        # Scale traffic volume
        self.scaler = MinMaxScaler()
        traffic_long['Traffic_Volume'] = self.scaler.fit_transform(
            traffic_long[['Traffic_Volume']]
        )
        self.traffic_long = traffic_long

        # Fit location scaler on unique coordinates
        coords = (
            self.traffic_long[['NB_LATITUDE', 'NB_LONGITUDE']]
            .drop_duplicates()
            .values
        )
        self._loc_scaler = MinMaxScaler().fit(coords)

        # Prepare sequences if needed
        values = self.traffic_long['Traffic_Volume'].values
        X, y = self.create_sequences(values)
        return X, y, self.scaler

    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)

    def get_sequence_by_time(self, scat_number: int, time_of_day: str):
        target_time = pd.to_datetime(time_of_day).time()
        full = self.traffic_long.copy()
        full['Timestamp'] = pd.to_datetime(full['Timestamp'])
        full = full[full['SCATS_Number'] == scat_number]
        mask = full['Timestamp'].dt.time == target_time
        full_time = full[mask].sort_values('Timestamp')
        if len(full_time) < self.window_size:
            raise ValueError(
                f"Need ≥{self.window_size} records for SCATS {scat_number} at {time_of_day}, got {len(full_time)}"
            )
        vals = full_time['Traffic_Volume'].values[-self.window_size:]
        return vals.reshape(1, self.window_size, 1)

    def get_location_vector(self, scat_number: int) -> np.ndarray:
        """
        Returns a 5-dimensional location feature vector for the given SCATS sensor.
        Here: normalized latitude & longitude plus three zero-padding dimensions.
        """
        if self.traffic_long is None or self._loc_scaler is None:
            raise RuntimeError("Call process_scats_data() before using get_location_vector()")
        # Filter for this sensor
        df = self.traffic_long[self.traffic_long['SCATS_Number'] == scat_number]
        if df.empty:
            raise KeyError(f"No metadata for SCATS {scat_number}")
        lat = float(df.iloc[0]['NB_LATITUDE'])
        lon = float(df.iloc[0]['NB_LONGITUDE'])
        # Scale lat/lon
        scaled_lat, scaled_lon = self._loc_scaler.transform([[lat, lon]])[0]
        # Pack into 5-D vector
        return np.array([scaled_lat, scaled_lon, 0.0, 0.0, 0.0], dtype=float)

    def extract_street_names(self, desc):
        # unchanged
        if pd.isna(desc):
            return []
        s = str(desc).strip()
        m = re.match(r"(.+?)\s+(?:N|S|E|W|SW|NE|NW|SE)\s+of\s+(.+)", s, re.IGNORECASE)
        if m:
            a, b = m.group(1), m.group(2)
            clean = lambda x: re.sub(
                r"\s*(ROAD|RD|STREET|ST|AVENUE|AVE|BOULEVARD|BLVD)\s*$", "", x, flags=re.IGNORECASE
            ).strip()
            return [clean(a), clean(b)]
        part = re.split(r",|\s+(?:N|S|E|W|SW|NE|NW|SE)\s+", s, flags=re.IGNORECASE)[0]
        return [re.sub(
            r"\s*(ROAD|RD|STREET|ST|AVENUE|AVE|BOULEVARD|BLVD)\s*$", "", part, flags=re.IGNORECASE
        ).strip()]

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Earth radius in kilometers
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    def create_graph(self, origin: int, destination: int):
        # unchanged
        full = pd.read_excel(self.filepath, sheet_name='Data', skiprows=1)
        full.columns = full.columns.str.strip()
        full.rename(columns={full.columns[1]: 'STREET_DESCRIPTION'}, inplace=True)
        cleaned = full.dropna(subset=['SCATS_Number', 'STREET_DESCRIPTION', 'NB_LATITUDE', 'NB_LONGITUDE'])
        cleaned = cleaned[~((cleaned.NB_LATITUDE == 0.0) & (cleaned.NB_LONGITUDE == 0.0))]

        nodes, coords_to_id = {}, {}
        next_id = 1
        for _, row in cleaned.iterrows():
            lat, lon = float(row.NB_LATITUDE), float(row.NB_LONGITUDE)
            key = (lat, lon)
            if key not in coords_to_id:
                coords_to_id[key] = next_id
                nodes[next_id] = (lat, lon)
                next_id += 1

        street_pts = {}
        for _, row in cleaned.iterrows():
            lat, lon = float(row.NB_LATITUDE), float(row.NB_LONGITUDE)
            nid = coords_to_id[(lat, lon)]
            for street in self.extract_street_names(row.STREET_DESCRIPTION):
                if street:
                    street_pts.setdefault(street, []).append((nid, lat, lon))

        edges = set()
        for pts in street_pts.values():
            uniq = {nid: (lat, lon) for nid, lat, lon in pts}
            seq = [(nid, *coord) for nid, coord in uniq.items()]
            if len(seq) < 2:
                continue
            lats = [lat for _nid, lat, _ in seq]
            lons = [lon for _nid, _, lon in seq]
            idx = 2 if np.var(lons) > np.var(lats) else 1
            ordered = sorted(seq, key=lambda x: x[idx])
            for (n1, *_), (n2, *_) in zip(ordered, ordered[1:]):
                edges.add((n1, n2))
                edges.add((n2, n1))

        graph_path = 'graph/output.txt'
        with open(graph_path, 'w') as f:
            f.write("Nodes:\n")
            for nid, (lat, lon) in sorted(nodes.items()):
                f.write(f"{nid}: ({lat},{lon})\n")
            f.write("Edges with distance (km):\n")
            for n1, n2 in sorted(edges):
                lat1, lon1 = nodes[n1]
                lat2, lon2 = nodes[n2]
                dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                f.write(f"({n1},{n2}): {dist:.3f}\n")
        return graph_path
    
    def create_DiGraph(self):
        # unchanged
        full = pd.read_excel(self.filepath, sheet_name='Data', skiprows=1)
        full.columns = full.columns.str.strip()
        full.rename(columns={full.columns[1]: 'STREET_DESCRIPTION'}, inplace=True)
        cleaned = full.dropna(subset=['SCATS_Number', 'STREET_DESCRIPTION', 'NB_LATITUDE', 'NB_LONGITUDE'])
        cleaned = cleaned[~((cleaned.NB_LATITUDE == 0.0) & (cleaned.NB_LONGITUDE == 0.0))]

        coords_to_id = {}
        nodes = {}
        next_id = 1
        for _, row in full.iterrows():
            lat, lon = float(row.NB_LATITUDE), float(row.NB_LONGITUDE)
            coord = (lat, lon)
            if coord not in coords_to_id:
                coords_to_id[coord] = next_id
                nodes[next_id] = coord
                next_id += 1

        sensor_at_node = {
            coords_to_id[(r.NB_LATITUDE, r.NB_LONGITUDE)]: int(r.SCATS_Number)
            for _, r in full.iterrows()
        }

        street_groups = {}
        for _, row in full.iterrows():
            nid = coords_to_id[(row.NB_LATITUDE, row.NB_LONGITUDE)]
            for street in self.extract_street_names(row.STREET_DESCRIPTION):
                street_groups.setdefault(street, set()).add(nid)

        # Build the graph with street-based edges
        G = nx.DiGraph()
        for nid, (lat, lon) in nodes.items():
            G.add_node(
                nid,
                x=lat,
                y=lon,
                scats_number=sensor_at_node[nid]   # <-- add this
            )
        for street, nset in street_groups.items():
            seq = list(nset)
            lats = [nodes[n][0] for n in seq]
            lons = [nodes[n][1] for n in seq]
            key = (max(lons)-min(lons) > max(lats)-min(lats))
            seq.sort(key=lambda n: nodes[n][1 if key else 0])
            for a, b in zip(seq, seq[1:]):
                lat1, lon1 = nodes[a]
                lat2, lon2 = nodes[b]
                dist = self.haversine_distance(lat1, lon1, lat2, lon2)
                G.add_edge(a, b,
                           distance_km=dist,
                           scat_point=sensor_at_node[a])
                G.add_edge(b, a,
                           distance_km=dist,
                           scat_point=sensor_at_node[b])
        return G

    def process_location_data(self):
        # unchanged
        location_data = (
            pd.read_excel(self.filepath, sheet_name='Data', skiprows=1)
            .dropna()
        )
        location_data.columns = location_data.columns.str.strip()
        return location_data[['Location', 'NB_LATITUDE', 'NB_LONGITUDE']].drop_duplicates()

if __name__ == '__main__':
    dp = DataProcessing('datasets/Scats Data October 2006.xls')
    G = dp.create_DiGraph()
    
    """ print("Created nodes with coordinates:")
    for node_id, attrs in G.nodes(data=True):
        print(f"Node {node_id}: Latitude = {attrs['x']}, Longitude = {attrs['y']}, SCATS = {attrs['scats_number']}") """
