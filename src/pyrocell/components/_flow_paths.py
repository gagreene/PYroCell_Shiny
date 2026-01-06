#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rasterio as rio
import fiona
import geopandas as gpd

from collections import deque, defaultdict
from math import sqrt
from shapely.geometry import LineString
from typing import Optional


# Declare global variables
suppress_msgs = False

# D8 direction code -> (dr, dc) using ArcGIS convention
_D8_CODE_TO_OFFSET: dict[int, tuple[int, int]] = {
    1:   (0,  1),    # E
    2:   (1,  1),    # SE
    4:   (1,  0),    # S
    8:   (1, -1),    # SW
    16:  (0, -1),    # W
    32:  (-1, -1),   # NW
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
}


def _log(msg: str):
    if not suppress_msgs:
        print(msg)


def assign_strahler_order(flowpaths_path: str,
                          out_path: str,
                          stahler_cutoff: int = 3,
                          coord_decimals: int = 3) -> None:
    """
    Assign Strahler stream order to each flow path segment.

    Assumes:
        - Each feature is a LineString representing a stream segment.
        - Coordinates are in downstream order (start = upstream, end = downstream).
        - Segments meet at exactly matching endpoints (within rounding tolerance).

    :param flowpaths_path: Input stream segments shapefile.
    :param out_path: Output shapefile path with 'Strahler' field.
    :param stahler_cutoff: Minimum Strahler order to retain (not used here).
    :param coord_decimals: Decimal precision for snapping endpoints.
    :return: None
    """
    gdf = gpd.read_file(flowpaths_path)

    if any(gdf.geometry.type == 'MultiLineString'):
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    if not all(gdf.geometry.type == 'LineString'):
        _log('\t\tAll geometries must be LineStrings after explode.')
        return

    n_segments = len(gdf)
    if n_segments == 0:
        _log('\t\tNo segments found in input.')
        return

    def _round_coord(x: float, y: float) -> tuple[float, float]:
        return (round(float(x), coord_decimals),
                round(float(y), coord_decimals))

    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []

    for geom in gdf.geometry:
        coords = list(geom.coords)
        if len(coords) < 2:
            _log('\t\tSegment with fewer than 2 vertices encountered.')
            return
        x0, y0 = coords[0]
        x1, y1 = coords[-1]
        starts.append(_round_coord(x0, y0))
        ends.append(_round_coord(x1, y1))

    start_map: dict[tuple[float, float], list[int]] = defaultdict(list)
    end_map: dict[tuple[float, float], list[int]] = defaultdict(list)

    for i in range(n_segments):
        start_map[starts[i]].append(i)
        end_map[ends[i]].append(i)

    parents: list[list[int]] = [[] for _ in range(n_segments)]
    children: list[list[int]] = [[] for _ in range(n_segments)]

    for i in range(n_segments):
        parents[i] = end_map.get(starts[i], [])

    for i in range(n_segments):
        children[i] = start_map.get(ends[i], [])

    strahler: list[int] = [-1] * n_segments
    remaining_parents: list[int] = [len(parents[i]) for i in range(n_segments)]

    q: deque[int] = deque()

    # Seed headwater segments
    for i in range(n_segments):
        if remaining_parents[i] == 0:
            strahler[i] = 1
            q.append(i)

    while q:
        i = q.popleft()

        for child in children[i]:
            if remaining_parents[child] <= 0:
                continue
            remaining_parents[child] -= 1

            if remaining_parents[child] == 0:
                upstream_orders = [strahler[p] for p in parents[child] if p >= 0 and strahler[p] > 0]
                if not upstream_orders:
                    strahler[child] = 1
                else:
                    max_order = max(upstream_orders)
                    count_max = sum(1 for o in upstream_orders if o == max_order)
                    if count_max >= 2:
                        strahler[child] = max_order + 1
                    else:
                        strahler[child] = max_order
                q.append(child)

    # Fallback for any unassigned (cycles / oddities)
    for i in range(n_segments):
        if strahler[i] < 0:
            strahler[i] = 1

    gdf['strahler'] = strahler
    gdf = gdf[gdf['strahler'] >= stahler_cutoff]
    gdf.to_file(out_path, driver='ESRI Shapefile')

    return


def compute_d8_flow_direction(elev_path: str,
                              out_fdir_path: str,
                              suppress_messages: bool,
                              nodata: Optional[float] = None) -> None:
    """
    Compute D8 flow direction (ArcGIS-style coded) from a DEM.

    Direction codes (ArcGIS convention):
        1   = E   (0, +1)
        2   = SE  (+1, +1)
        4   = S   (+1, 0)
        8   = SW  (+1, -1)
        16  = W   (0, -1)
        32  = NW  (-1, -1)
        64  = N   (-1, 0)
        128 = NE  (-1, +1)

    Cells with no lower neighbor (pits/flats) get value 0.
    DEM nodata cells remain nodata.

    :param elev_path: Path to input DEM raster.
    :param out_fdir_path: Path to output flow direction raster.
    :param suppress_messages: Whether to suppress log messages.
    :param nodata: Optional DEM nodata override.
    :return: None
    """
    global suppress_msgs
    suppress_msgs = suppress_messages

    with rio.open(elev_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile.copy()

        if nodata is None:
            nodata_val = src.nodata
            if nodata_val is None:
                _log('\t\tDEM nodata is not set; pass nodata explicitly.')
                return
        else:
            nodata_val = nodata

        rows, cols = src.height, src.width
        cell_width = src.transform.a
        cell_height = -src.transform.e  # affine.e is usually negative

    valid = np.isfinite(dem) & (dem != nodata_val)

    diag_dist = sqrt(cell_width**2 + cell_height**2)

    # (dr, dc, distance, code)
    neighbors = [
        (0,  1, cell_width,       1),    # E
        (1,  1, diag_dist,        2),    # SE
        (1,  0, cell_height,      4),    # S
        (1, -1, diag_dist,        8),    # SW
        (0, -1, cell_width,      16),    # W
        (-1, -1, diag_dist,      32),    # NW
        (-1,  0, cell_height,    64),    # N
        (-1,  1, diag_dist,      128),   # NE
    ]

    fdir = np.zeros((rows, cols), dtype=np.int32)  # 0 = pit/flat

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue

            z = dem[r, c]
            best_slope = 0.0
            best_code = 0

            for dr, dc, dist, code in neighbors:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                if not valid[rr, cc]:
                    continue

                dz = z - dem[rr, cc]
                if dz > 0:
                    slope = dz / dist
                    if slope > best_slope:
                        best_slope = slope
                        best_code = code

            fdir[r, c] = best_code

    # Apply nodata
    out_nodata = nodata_val if nodata_val is not None else -9999
    fdir[~valid] = out_nodata

    profile.update(
        dtype='int32',
        count=1,
        nodata=out_nodata
    )

    with rio.open(out_fdir_path, 'w', **profile) as dst:
        dst.write(fdir.astype(np.int32), 1)

    return


def extract_stream_segments_from_acc(elev_path: str,
                                     fdir_path: str,
                                     acc_path: str,
                                     out_shp: str,
                                     acc_threshold: Optional[float] = None,
                                     nodata: Optional[float] = None) -> None:
    """
    Extract stream segments as polylines from accumulation + D8 flow direction.

    Steps:
        1. Load DEM (for transform & CRS), flow direction (D8), and accumulation.
        2. Define stream cells where accumulation >= acc_threshold.
        3. Build a directed graph of stream cells:
               each cell has at most one downstream neighbor (D8),
               count upstream neighbors for each cell.
        4. Identify segment 'start' cells:
               upstream_count == 0  (headwaters), or
               upstream_count >= 2 (junctions).
        5. For each start cell, follow D8 flow until:
               stream ends, or
               next junction/outlet,
           to create a polyline segment.
        6. Write segments to a shapefile.

    NOTE (FIX): We no longer use a global 'visited' set here, so junction
    cells can be both the end of an upstream segment and the start of
    downstream segments. Each segment only tracks its own visited cells.

    :param elev_path: Path to DEM (for CRS & transform).
    :param fdir_path: D8 flow direction raster (ArcGIS codes).
    :param acc_path: Flow accumulation raster.
    :param acc_threshold: Threshold for defining stream cells.
    :param out_shp: Output shapefile path for stream segments.
    :param nodata: Optional nodata override (for fdir/acc).
    :return: None
    """
    # Load DEM for grid & transform
    with rio.open(elev_path) as dem_src:
        transform = dem_src.transform
        crs = dem_src.crs
        rows, cols = dem_src.height, dem_src.width

    # Load flow direction
    with rio.open(fdir_path) as fdir_src:
        fdir = fdir_src.read(1)
        fdir_nodata = fdir_src.nodata if nodata is None else nodata

    # Load accumulation
    with rio.open(acc_path) as acc_src:
        acc = acc_src.read(1).astype(np.float64)
        acc_nodata = acc_src.nodata if acc_src.nodata is not None else -9999.0

    if acc_threshold is None:
        # Default: 0.5% of max accumulation value
        valid_acc = np.isfinite(acc) & (acc != acc_nodata)
        max_acc = np.max(acc[valid_acc]) if np.any(valid_acc) else 0.0
        acc_threshold = 0.025 * max_acc

    # Stream mask: accumulation above threshold & valid direction
    valid_fdir = np.isfinite(fdir) & (fdir != fdir_nodata) & (fdir != 0)
    valid_acc = np.isfinite(acc) & (acc != acc_nodata)
    is_stream = (acc >= acc_threshold) & valid_fdir & valid_acc

    size = rows * cols
    flat_is_stream = is_stream.ravel()
    flat_fdir = fdir.ravel()

    # For each cell: receiver within stream (or -1)
    receiver = np.full(size, -1, dtype=np.int32)
    upstream_count = np.zeros(size, dtype=np.int32)

    for idx in range(size):
        if not flat_is_stream[idx]:
            continue

        code = int(flat_fdir[idx])
        offset = _D8_CODE_TO_OFFSET.get(code)
        if offset is None:
            continue

        r = idx // cols
        c = idx % cols
        dr, dc = offset
        rr = r + dr
        cc = c + dc

        if 0 <= rr < rows and 0 <= cc < cols:
            nbr_idx = rr * cols + cc
            if flat_is_stream[nbr_idx]:
                receiver[idx] = nbr_idx
                upstream_count[nbr_idx] += 1

    # Identify start cells (headwaters or junctions)
    start_indices: list[int] = []
    for idx in range(size):
        if not flat_is_stream[idx]:
            continue
        if upstream_count[idx] == 0 or upstream_count[idx] >= 2:
            start_indices.append(idx)

    # Prepare shapefile
    schema = {
        'geometry': 'LineString',
        'properties': {'id': 'int'}
    }
    crs_mapping = crs.to_dict() if hasattr(crs, 'to_dict') else crs

    with fiona.open(out_shp, 'w',
                    driver='ESRI Shapefile',
                    crs=crs_mapping,
                    schema=schema) as shp:

        line_id = 1

        for start_idx in start_indices:
            # For each start, we allow revisiting junction cells that may also
            # be the endpoints of other segments. Only guard against cycles
            # within this single segment.
            path_idxs: list[int] = []
            visited_seg: set[int] = set()
            current = start_idx

            while True:
                if current in visited_seg:
                    break  # protect against loops
                visited_seg.add(current)
                path_idxs.append(current)

                rec = receiver[current]
                if rec < 0:
                    break  # end of stream (no downstream stream cell)

                # stop segment when next cell is a junction or outlet
                if upstream_count[rec] != 1:
                    path_idxs.append(rec)
                    break

                current = rec

            if len(path_idxs) < 2:
                continue

            # Convert raster indices to coordinates
            coords: list[tuple[float, float]] = []
            for idx in path_idxs:
                r = idx // cols
                c = idx % cols
                x, y = rio.transform.xy(transform, r, c)
                coords.append((x, y))

            shp.write({
                'geometry': LineString(coords).__geo_interface__,
                'properties': {'id': line_id}
            })
            line_id += 1

    return


def flow_accumulation_from_direction(fdir_path: str,
                                     out_acc_path: str,
                                     weight_path: Optional[str] = None,
                                     nodata: Optional[float] = None) -> None:
    """
    Compute flow accumulation from a D8 flow direction raster.

    Each valid cell contributes:
        - weight value (if weight_path is provided), or
        - 1.0 (unweighted).

    The flow graph is built using D8 direction codes:
        1, 2, 4, 8, 16, 32, 64, 128 (ArcGIS convention).

    :param fdir_path: Path to D8 flow direction raster (int codes).
    :param out_acc_path: Path to output accumulation raster.
    :param weight_path: Optional weight raster (same shape).
    :param nodata: Optional nodata override for fdir.
    :return: None
    """
    with rio.open(fdir_path) as src:
        fdir = src.read(1)
        profile = src.profile.copy()

        if nodata is None:
            nodata_val = src.nodata
            if nodata_val is None:
                nodata_val = 0  # treat 0 as no-flow / nodata in many D8 grids
        else:
            nodata_val = nodata

        rows, cols = src.height, src.width

    # Valid cells: not nodata and not 0
    valid = np.isfinite(fdir) & (fdir != nodata_val) & (fdir != 0)
    size = rows * cols

    if weight_path is not None:
        with rio.open(weight_path) as w_src:
            weights = w_src.read(1).astype(np.float64)
            if w_src.shape != (rows, cols):
                _log('\t\tWeight raster must match flow direction raster dimensions.')
                return
    else:
        weights = np.ones_like(fdir, dtype=np.float64)

    weights[~valid] = 0.0

    flat_fdir = fdir.ravel()
    flat_valid = valid.ravel()
    flat_weights = weights.ravel()

    receiver = np.full(size, -1, dtype=np.int32)
    inflow_count = np.zeros(size, dtype=np.int32)

    # Build graph: each cell -> its receiver (if any)
    for idx in range(size):
        if not flat_valid[idx]:
            continue

        code = int(flat_fdir[idx])
        offset = _D8_CODE_TO_OFFSET.get(code)
        if offset is None:
            continue

        r = idx // cols
        c = idx % cols
        dr, dc = offset
        rr = r + dr
        cc = c + dc

        if 0 <= rr < rows and 0 <= cc < cols:
            nbr_idx = rr * cols + cc
            if flat_valid[nbr_idx]:
                receiver[idx] = nbr_idx
                inflow_count[nbr_idx] += 1

    # Topological accumulation
    acc = flat_weights.copy()

    q: deque[int] = deque(
        i for i in range(size)
        if flat_valid[i] and inflow_count[i] == 0
    )

    while q:
        i = q.popleft()
        rec = receiver[i]
        if rec >= 0:
            acc[rec] += acc[i]
            inflow_count[rec] -= 1
            if inflow_count[rec] == 0:
                q.append(rec)

    acc_grid = acc.reshape(rows, cols)
    acc_grid[~valid] = nodata_val

    profile.update(
        dtype='float32',
        count=1,
        nodata=float(nodata_val)
    )

    with rio.open(out_acc_path, 'w', **profile) as dst:
        dst.write(acc_grid.astype(np.float32), 1)

    return
