from h3 import h3
import random
import pandas as pd


def bucket_data_to_h3(resolution, df_path="fake_data.csv"):
    df = pd.read_csv(df_path, index_col=0)
    h3_ids = []
    for row in df.iterrows():
        h3_ids.append(h3.geo_to_h3(row["lat"], row["lon"], resolution))

    df["h3_id"] = h3_ids
    return df


def generate_k_ring(data):
    """
    data
        - resolution
        - coord
            - lat
            - lon
        - k

    returns: hexagons: []
    """

    resolution = data["resolution"]
    lat, lon = data["coord"]["lat"], data["coord"]["lon"]
    k = data["k"]

    h3_address = h3.geo_to_h3(lat, lon, resolution)
    ring = h3.k_ring_distances(h3_address, k)

    hexagons = []
    for hex_set in ring:
        for hex_address in hex_set:
            hexagons.append(
                {
                    "boundary": h3.h3_to_geo_boundary(hex_address),
                    "p": random.random(),
                    "h3_id": hex_address,
                }
            )
    return hexagons
