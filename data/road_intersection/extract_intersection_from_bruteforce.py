
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


# --- Utility: Interpolate chainage for a point on a road ---
def interpolate_chainage_fast(road_lat, road_lon, road_chainage, lat, lon):
    dists = np.sqrt((road_lat - lat) ** 2 + (road_lon - lon) ** 2)
    if len(dists) < 2:
        return None
    idx_sorted = np.argsort(dists)[:2]
    i1, i2 = idx_sorted[0], idx_sorted[1]
    ch1, ch2 = road_chainage[i1], road_chainage[i2]
    d1, d2 = dists[i1], dists[i2]
    w = 0.5 if (d1 + d2) == 0 else d2 / (d1 + d2)
    chainage = ch1 * w + ch2 * (1 - w)
    insert_after = min(i1, i2)
    return chainage, insert_after, max(d1, d2), i1



# --- Main: Find close roads and prepare for intersection detection ---
def get_intersection_df(roads_preprocessed, roads_shp):
    """
    Given a preprocessed roads DataFrame and a roads shapefile GeoDataFrame,
    returns a DataFrame of crossing rows (intersections) to be concatenated to the original roads DataFrame.
    """
    # Prepare road geometries
    roads_csv = roads_preprocessed.copy()
    roads_csv = roads_csv.reset_index(drop=True)
    roads_csv["idx"] = roads_csv.index
    roads_csv["crossing"] = None
     # --- Precompute CSV road LineStrings and lat/lon arrays (for later use) ---
    road_lines, road_latlon, road_chainage, road_firstrow = {}, {}, {}, {}
    for road, group in roads_csv.groupby("road"):
        arr_lat = group["lat"].values
        arr_lon = group["lon"].values
        arr_chainage = group["chainage"].values
        points = [
            Point(lon, lat)
            for lon, lat in zip(arr_lon, arr_lat)
            if not np.isnan(lat) and not np.isnan(lon)
        ]
        if len(points) > 1:
            road_lines[road] = LineString(points)
            road_latlon[road] = (arr_lat, arr_lon)
            road_chainage[road] = arr_chainage
            road_firstrow[road] = group.iloc[0]
    print(f"Prepared {len(road_lines)} road geometries from CSV for intersection processing.")


    # --- Find roads within 2km of each road using minimum LineString distance ---
    DEGREE_THRESHOLD = 2 / 111.0  # 1 degree ~ 111km
    # Build a reduced, non-redundant close_roads_dict using symmetry
    road_names_sorted = sorted(road_lines.keys())
    close_roads_dict = {road: [] for road in road_names_sorted}
    for i, road1 in enumerate(road_names_sorted):
        line1 = road_lines[road1]
        for j in range(i + 1, len(road_names_sorted)):
            road2 = road_names_sorted[j]
            line2 = road_lines[road2]
            if line1.distance(line2) <= DEGREE_THRESHOLD:
                close_roads_dict[road1].append(road2)
    print("Computed reduced, non-redundant roads within 2km of each road:")
    #for road, close_list in close_roads_dict.items():
    #    print(f"{road}: {close_list}")


    # --- Detect intersections between close road pairs ---
    intersection_threshold_deg = 0.0045  # ~500m (0.0045 deg at equator)
    dedup_threshold_deg = 0.018  # ~2km (same as proximity threshold)
    intersections = []  # (road1, idx1, road2, idx2, lat, lon)
    intersection_registry = {}  # (road1, road2) -> list of (lat, lon)

    for road1, close_roads in close_roads_dict.items():
        print(f"Processing road {road1} with {len(close_roads)} close roads...")
        arr_lat1, arr_lon1 = road_latlon[road1]
        for road2 in close_roads:
            arr_lat2, arr_lon2 = road_latlon[road2]
            for i1, (lat1, lon1) in enumerate(zip(arr_lat1, arr_lon1)):
                for i2, (lat2, lon2) in enumerate(zip(arr_lat2, arr_lon2)):
                    d = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
                    if d <= intersection_threshold_deg:
                        # Deduplication: check if an intersection for this pair exists within 2km
                        pair_key = tuple(sorted([road1, road2]))
                        already = False
                        if pair_key in intersection_registry:
                            for prev_lat, prev_lon in intersection_registry[pair_key]:
                                if np.sqrt((lat1 - prev_lat) ** 2 + (lon1 - prev_lon) ** 2) < dedup_threshold_deg:
                                    already = True
                                    break
                        if already:
                            continue
                        # Register this intersection
                        intersection_registry.setdefault(pair_key, []).append((lat1, lon1))
                        intersections.append((road1, i1, road2, i2, lat1, lon1))

    print(f"Detected {len(intersections)} unique intersections between close road pairs.")
    #for road1, i1, road2, i2, lat, lon in intersections:
    #    print(f"Intersection: {road1} (idx {i1}) <-> {road2} (idx {i2}) at ({lat}, {lon})")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    for road, line in road_lines.items():
        x, y = line.xy
        plt.plot(x, y, color='gray', linewidth=0.5)
    for road1, i1, road2, i2, lat, lon in intersections:
        plt.plot(lon, lat, 'ro')
    plt.title("Detected Intersections (Red Dots) Between Close Roads")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid()
    plt.show()

    # --- Prepare intersection rows for output DataFrame ---
    new_rows = []
    new_idx_counter = roads_csv["idx"].max() + 1 if not roads_csv.empty else 0
    intersection_id = 0
    for road1, i1, road2, i2, lat, lon in intersections:
        # For each intersection, create two rows (one for each road)
        for road, idx, other_road, other_idx in [(road1, i1, road2, i2), (road2, i2, road1, i1)]:
            base_row = road_firstrow[road].copy()
            base_row["chainage"] = road_chainage[road][idx]
            base_row["lat"] = lat
            base_row["lon"] = lon
            base_row["lrp"] = f"LRP_CROSS_{new_idx_counter}"
            base_row["idx"] = new_idx_counter
            base_row["type"] = "Crossing"
            base_row["gap"] = ""
            base_row["bridgedual"] = ""
            base_row["condition"] = ""
            base_row["crossing"] = None  # Will be filled after both rows are created
            new_rows.append((road, other_road, new_idx_counter, base_row))
            new_idx_counter += 1

    # Now fill the 'crossing' column to point to the paired row
    # Build a mapping from (road, other_road, lat, lon) to idx
    row_lookup = {}
    for road, other_road, idx, row in new_rows:
        row_lookup[(road, other_road, row["lat"], row["lon"])] = idx
    # Assign crossing idx
    final_rows = []
    for road, other_road, idx, row in new_rows:
        crossing_idx = row_lookup.get((other_road, road, row["lat"], row["lon"]), None)
        row["crossing"] = crossing_idx
        final_rows.append(row)

    print(f"Generated {len(final_rows)} new crossing rows from intersections.")
    if final_rows:
        df_out = pd.DataFrame(final_rows)
    else:
        df_out = pd.DataFrame(columns=roads_csv.columns)

    return df_out
