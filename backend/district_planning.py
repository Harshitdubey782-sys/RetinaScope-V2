
from pathlib import Path

import pandas as pd
import geopandas as gpd


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

PLANNING_PATH = BASE_DIR / "planning_data.csv"
SHAPEFILE_PATH = BASE_DIR / "district_shapes" / "output.shp"


def load_planning_data():
    return pd.read_csv(PLANNING_PATH)


def load_district_shapes():
    gdf = gpd.read_file(SHAPEFILE_PATH)

    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)

    return gdf


def normalize_name(series):
    return (
        series.astype(str)
        .str.upper()
        .str.replace("&", "AND", regex=False)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_map_data(selected_state="All States"):
    planning = load_planning_data().copy()
    shapes = load_district_shapes().copy()

    # Normalize state and district names
    shapes["_state_key"] = normalize_name(shapes["statename"])
    shapes["_district_key"] = normalize_name(shapes["distname"])

    planning["_state_key"] = normalize_name(planning["state"])
    planning["_district_key"] = normalize_name(planning["district"])

    # Filter before merging
    if selected_state != "All States":
        state_key = normalize_name(
            pd.Series([selected_state])
        ).iloc[0]

        shapes = shapes[
            shapes["_state_key"] == state_key
        ].copy()

        planning = planning[
            planning["_state_key"] == state_key
        ].copy()

    merged = shapes.merge(
        planning,
        on=["_state_key", "_district_key"],
        how="inner",
        suffixes=("_shape", "_plan")
    )

    merged = merged.dropna(subset=["priority"]).copy()

    if merged.empty:
        return merged

    # Simplify geometry for faster map rendering
    merged["geometry"] = merged.geometry.simplify(
        tolerance=0.005,
        preserve_topology=True
    )

    merged["map_id"] = merged.index.astype(str)

    return merged
