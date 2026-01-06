import os
import numpy as np
from osgeo import gdal, ogr, osr
from shapely.geometry import shape, Point, LineString, MultiLineString, Polygon, MultiPolygon, box
from shapely.ops import unary_union, nearest_points, polygonize
from geopandas import GeoDataFrame, GeoSeries
from rasterio.features import shapes as rio_shapes
from scipy.ndimage import binary_dilation
import copy

# Enable GDAL exceptions
gdal.UseExceptions()

# Declare global variables
suppress_msgs = False


def _log(msg: str):
    if not suppress_msgs:
        print(msg)


def _buffer_edges(travel_times_arr: np.ndarray, run_time: float):
    # Replace non-finite values with run_time for contouring
    cleaned = np.where(np.isfinite(travel_times_arr), travel_times_arr, run_time)

    # Detect if cleaned travel times is on the edge/boundary of self.max_grid_shape
    # If touching, buffer the touched edges to avoid contour generation issues
    edges_to_buffer = []
    if np.any(np.isfinite(travel_times_arr[0, :])):
        edges_to_buffer.append('top')
    if np.any(np.isfinite(travel_times_arr[-1, :])):
        edges_to_buffer.append('bottom')
    if np.any(np.isfinite(travel_times_arr[:, 0])):
        edges_to_buffer.append('left')
    if np.any(np.isfinite(travel_times_arr[:, -1])):
        edges_to_buffer.append('right')

    # Initialize pad tracking variables for each side
    pad_top = pad_bottom = pad_left = pad_right = 0

    # Apply buffering/padding if edges are touched
    if edges_to_buffer:
        buffer_width = 1  # Number of cells to buffer
        if 'top' in edges_to_buffer:
            cleaned = np.pad(cleaned, ((buffer_width, 0), (0, 0)), constant_values=run_time)
            pad_top = buffer_width
        if 'bottom' in edges_to_buffer:
            cleaned = np.pad(cleaned, ((0, buffer_width), (0, 0)), constant_values=run_time)
            pad_bottom = buffer_width
        if 'left' in edges_to_buffer:
            cleaned = np.pad(cleaned, ((0, 0), (buffer_width, 0)), constant_values=run_time)
            pad_left = buffer_width
        if 'right' in edges_to_buffer:
            cleaned = np.pad(cleaned, ((0, 0), (0, buffer_width)), constant_values=run_time)
            pad_right = buffer_width
    return cleaned, pad_top, pad_bottom, pad_left, pad_right


def _adjust_transform_for_padding(process_transform, pad_top: int, pad_left: int, cell_w: float, cell_h: float):
    transform = copy.deepcopy(process_transform)
    if any([pad_top, pad_left]):
        origin_x, origin_y = transform.c, transform.f
        origin_x -= pad_left * cell_w
        origin_y += pad_top * cell_h
        transform = type(transform)(transform.a, transform.b, origin_x, transform.d, transform.e, origin_y)
    return transform


def _create_raster(cleaned: np.ndarray):
    rows, cols = cleaned.shape
    driver = gdal.GetDriverByName('MEM')
    raster = driver.Create('', cols, rows, 1, gdal.GDT_Float32)
    return raster


def _write_raster(cleaned, raster, transform, crs, nodata_value: float):
    raster.SetGeoTransform(transform.to_gdal())
    raster.SetProjection(crs.to_wkt())
    band = raster.GetRasterBand(1)
    band.WriteArray(cleaned)
    band.SetNoDataValue(nodata_value)
    return band


def _generate_contour_layer(band, contour_levels: np.ndarray, nodata_value: float, crs):
    # Create an OGR layer for contours
    driver = ogr.GetDriverByName('Memory')
    datasource = driver.CreateDataSource('out')
    layer = datasource.CreateLayer('contours', srs=osr.SpatialReference(wkt=crs.to_wkt()))

    # Add a field for the contour values
    field_defn = ogr.FieldDefn('arr_time', ogr.OFTReal)
    layer.CreateField(field_defn)

    # Generate the contours using GDALContourGenerateEx
    options = [
        f'FIXED_LEVELS={" ,".join(map(str, contour_levels))}',
        f'ELEV_FIELD=arr_time',
        f'NODATA={nodata_value}',
    ]
    try:
        gdal.ContourGenerateEx(srcBand=band, dstLayer=layer, options=options)
    except RuntimeError as e:
        raise RuntimeError(f'GDALContourGenerateEx failed: {e}')

    gdf = _ogr_layer_to_gdf(layer, crs)

    return gdf


def _ogr_layer_to_gdf(layer, crs):
    contours = []
    layer.ResetReading()
    # Extract geometries and values from the OGR layer
    for feature in layer:
        geometry = feature.GetGeometryRef()
        value = feature.GetField('arr_time')
        if geometry and geometry.IsValid():  # Ensure the geometry is valid
            wkt = geometry.ExportToWkt()
            if 'nan' not in wkt.lower():  # Exclude geometries with NaN coordinates
                contours.append((wkt, value))
    if not contours:
        _log('\t\tNo valid contours were generated.')
        return None
    # Create a GeoDataFrame for the contours
    gdf = GeoDataFrame(contours, columns=['wkt', 'arr_time'])
    gdf = gdf.set_geometry(GeoSeries.from_wkt(gdf['wkt'])).drop(columns='wkt')
    gdf.set_crs(crs, inplace=True)
    gdf = gdf.dissolve(by='arr_time').reset_index()
    gdf = gdf[gdf['arr_time'] > 0]
    return gdf


def _extract_nonfuel_polygons(mask: np.ndarray, transform) -> list:
    return [
        shape(geom)
        for geom, value in rio_shapes(mask.astype('uint8'), mask=mask, transform=transform)
        if value == 1
    ]


def _save_contour(contour_out, contour_ext: str, gdf_polygons, srs, fire_id_val: str, out_folder_val: str):
    data_dict = {
        'geometry': contour_out,
        'arr_time': gdf_polygons['arr_time'],
    }
    if contour_ext == 'poly':
        data_dict['area_ha'] = contour_out.to_crs(epsg=6933).area / 10000.0
    data_dict['perim_m'] = contour_out.to_crs(epsg=6933).length
    contour_gdf = GeoDataFrame(data=data_dict, crs=srs)
    out_name = f'{fire_id_val}_arrival_contours_{contour_ext}.shp'
    contour_file = os.path.join(out_folder_val, out_name)
    contour_gdf.to_file(contour_file, driver='ESRI Shapefile')
    _log(f'\t\tContours saved to: {contour_file}')


def _save_perimeter(gdf_polygons, run_time: float, srs, fire_id_val: str, out_folder_val: str):
    perim_gdf = gdf_polygons[gdf_polygons['arr_time'] == run_time]
    if not perim_gdf.empty:
        unified_polygons = unary_union(perim_gdf.geometry)
        if isinstance(unified_polygons, Polygon):
            unified_polygons = [unified_polygons]
        elif isinstance(unified_polygons, MultiPolygon):
            unified_polygons = list(unified_polygons.geoms)
        perimeter_gdf = GeoDataFrame(geometry=unified_polygons, crs=srs)
        perimeter_gdf['area_ha'] = perimeter_gdf['geometry'].to_crs(epsg=6933).area / 10000.0
        perimeter_gdf['perim_m'] = perimeter_gdf['geometry'].to_crs(epsg=6933).length
        out_name = f'{fire_id_val}_burn_perimeter.shp'
        perimeter_file = os.path.join(out_folder_val, out_name)
        perimeter_gdf.to_file(perimeter_file, driver='ESRI Shapefile')
        _log(f'\t\tBurn perimeter saved to: {perimeter_file}')
    else:
        _log('\t\tNo valid perimeter polygons were generated.')


def gen_arrival_contours(
    interval_mins: int,
    travel_times: np.ndarray,
    total_run_time: float,
    process_transform,
    crs,
    cell_width: float,
    cell_height: float,
    process_grid_shape: tuple,
    land_cover: np.ndarray,
    ignition_type: str,
    remove_poly_ign: bool,
    ignition_geom,
    out_request: list,
    out_folder: str,
    fire_id: str,
    suppress_messages: bool,
) -> None:
    global suppress_msgs
    suppress_msgs = suppress_messages

    if travel_times is None:
        _log('\t\tTravel times have not been calculated. Run the PYroCell model first.')
        return
    if total_run_time is None:
        _log('\t\tMaximum simulation time (run_time) has not been set. Ensure PYroCell was run correctly.')
        return
    if interval_mins is None or interval_mins <= 0:
        _log('\t\t"interval_mins" must be a positive integer.')
        return

    contour_levels = np.arange(0, total_run_time + interval_mins, interval_mins)
    nodata_value = -9999.0

    # Clean and buffer travel times
    cleaned_travel_times, pad_top, pad_bottom, pad_left, pad_right = _buffer_edges(travel_times, total_run_time)
    # Adjust transform if padding was applied
    transform = _adjust_transform_for_padding(process_transform, pad_top, pad_left, cell_width, cell_height)
    # Create in-memory raster for cleaned travel times
    raster = _create_raster(cleaned_travel_times)

    # Generate a binary mask of valid data (non-nodata values)
    valid_mask = cleaned_travel_times != nodata_value
    # Identify edges of valid data using dilation
    edge_mask = binary_dilation(valid_mask) & ~valid_mask
    # Add edges to the cleaned_travel_times using the maximum values in the travel time dataset
    cleaned_travel_times[edge_mask] = np.nanmax(cleaned_travel_times)

    # Write the cleaned travel times to the raster and set nodata value
    band = _write_raster(cleaned_travel_times, raster, transform, crs, nodata_value)

    # Generate contour layer from the raster band
    gdf = _generate_contour_layer(band, contour_levels, nodata_value, crs)
    if gdf is None:
        return

    # Determine valid data extent
    ivalid_rows, ivalid_cols = np.where(np.isfinite(travel_times))
    if ivalid_rows.size == 0 or ivalid_cols.size == 0:
        _log('\t\tNo valid travel time data found within the specified run time.')
        return
    min_row, max_row = ivalid_rows.min(), ivalid_rows.max()
    min_col, max_col = ivalid_cols.min(), ivalid_cols.max()

    # Get non-fuel polygons within valid travel time bounds
    land_cover_subset = land_cover[min_row:max_row + 1, min_col:max_col + 1]
    nonfuel_mask = ~np.isin(land_cover_subset, [1, 2])
    cell_nonfuel_polygons = _extract_nonfuel_polygons(nonfuel_mask, process_transform)
    cell_nonfuel_union = unary_union(cell_nonfuel_polygons)

    # Identify non-fuel cells adjacent to burned cells (edge)
    burned_mask = (cleaned_travel_times < total_run_time)
    padded_land_cover = np.pad(
        land_cover,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=1
    ) if any([pad_top, pad_bottom, pad_left, pad_right]) else land_cover
    edge_nonfuel_mask = ~np.isin(padded_land_cover, [1, 2])
    dilated_burned = binary_dilation(burned_mask, structure=np.ones((3, 3)))
    adjacent_edge_nonfuel_mask = dilated_burned & edge_nonfuel_mask
    edge_nonfuel_polygons = _extract_nonfuel_polygons(adjacent_edge_nonfuel_mask, process_transform)
    edge_nonfuel_union = unary_union(edge_nonfuel_polygons)

    # Combine all non-fuel areas
    nonfuel_union = unary_union([cell_nonfuel_union, edge_nonfuel_union])

    # Shift contour points to nearest edge of non-fuel areas
    def _shift_all_points_to_edge(geom):
        if geom.is_empty:
            return geom
        if geom.geom_type == 'LineString':
            new_coords = [
                nearest_points(Point(pt), edge_nonfuel_union)[1].coords[0]
                if edge_nonfuel_union.distance(Point(pt)) <= (cell_width + cell_height) / 4
                else pt
                for pt in geom.coords
            ]
            return LineString(new_coords)
        elif geom.geom_type == 'MultiLineString':
            return MultiLineString([_shift_all_points_to_edge(part) for part in geom.geoms])
        return geom
    gdf['geometry'] = gdf['geometry'].apply(_shift_all_points_to_edge)

    # Dissolve all features by the 'arr_time' field to create multipart geometries
    arrival_lines = gdf.dissolve(by='arr_time').reset_index()

    # Convert lines to polygons and include both exteriors and holes
    polygons = []
    for geom, arr_time in zip(arrival_lines.geometry, arrival_lines['arr_time']):
        if geom.geom_type in ['LineString', 'MultiLineString']:
            # Combine the geometry into a single unified linework
            merged_line = unary_union(geom)
            # Generate all polygons from the linework
            raw_polygons = list(polygonize(merged_line))

            # Group exteriors and interiors
            exterior_polygons = []
            interior_rings = []
            for raw_polygon in raw_polygons:
                if raw_polygon.is_valid and raw_polygon.area > 0:
                    # Check if it's an exterior or interior ring
                    if any(raw_polygon.intersects(other) and raw_polygon.area < other.area
                           for other in raw_polygons if raw_polygon != other):
                        # It's an interior (hole)
                        interior_rings.append(raw_polygon)
                    else:
                        # It's an exterior polygon
                        exterior_polygons.append(raw_polygon)

            # Construct final polygons with associated holes
            # Build individual polygons with holes
            poly_list = []
            for _exterior in exterior_polygons:
                holes = [_interior.exterior for _interior in interior_rings
                         if _interior.intersects(_exterior) and (_interior.area < _exterior.area)]
                poly_list.append(Polygon(_exterior.exterior, holes))
            if poly_list:
                polygons.append((MultiPolygon(poly_list) if len(poly_list) > 1 else poly_list[0], arr_time))
            else:
                _log(f'\t\tNo perimeters identified for arrival time: {arr_time}')
        else:
            _log(f'\t\tUnsupported geometry type: {geom.geom_type}')

    def _clip_polygons_to_extent(poly_tuple):
        geom, arr_time = poly_tuple
        if geom.is_empty:
            return (geom, arr_time)
        travel_times_bounds = box(
            process_transform.c,
            process_transform.f + process_transform.e * process_grid_shape[0],
            process_transform.c + process_transform.a * process_grid_shape[1],
            process_transform.f
        )
        return (geom.intersection(travel_times_bounds), arr_time)
    polygons = [_clip_polygons_to_extent(p) for p in polygons]
    polygons = [p for p in polygons if not p[0].is_empty]

    # Erase non-fuel cells from the polygons
    final_polygons = []
    for poly, arr_time in polygons:
        if poly.is_valid and not poly.is_empty:
            # Use buffer(0) to fix potential geometry issues before difference
            try:
                erased = poly.buffer(0).difference(nonfuel_union.buffer(0))
            except Exception:
                erased = poly.difference(nonfuel_union)
            if erased.is_empty or not erased.is_valid:
                continue
            final_polygons.append((erased, arr_time))

    # Create GeoDataFrame from final polygons
    gdf_polygons = GeoDataFrame(
        {'geometry': [p for p, t in final_polygons], 'arr_time': [t for p, t in final_polygons]},
        crs=crs
    )

    # Dissolve polygons by 'arr_time' to combine multipart geometries
    gdf_polygons = gdf_polygons.dissolve(by='arr_time').reset_index()

    if (ignition_type == 'polygon') and remove_poly_ign:
        def _erase_ignition_area(_geom):
            if _geom.is_empty:
                return _geom
            try:
                erased_geom = _geom.difference(ignition_geom.buffer(0))
            except Exception:
                erased_geom = _geom
            return erased_geom
        gdf_polygons['geometry'] = gdf_polygons['geometry'].apply(_erase_ignition_area)
        # Remove any empty geometries resulting from the difference operation
        gdf_polygons = gdf_polygons[~gdf_polygons['geometry'].is_empty].reset_index(drop=True)

    # Cycle through each polygon, starting at last, and intersect each prior polygon with the next later one
    # Keep only the intersecting portions to remove overlaps from earlier arrival times
    if not gdf_polygons.empty:
        gdf_polygons = gdf_polygons.sort_values(by='arr_time', ascending=False).reset_index(drop=True)
        for later_idx in range(0, len(gdf_polygons) - 1):
            # Use the current polygon as the "later" reference
            later_geom = gdf_polygons.iloc[later_idx]['geometry']
            if later_idx == 0:
                # Add or replace the latest polygon for its arrival time
                gdf_polygons.at[later_idx, 'geometry'] = later_geom

            # For each prior arrival time, intersect it with the next later polygon
            for earlier_idx in range(later_idx + 1, len(gdf_polygons)):
                current_row = gdf_polygons.iloc[earlier_idx]
                current_geom = current_row['geometry']
                try:
                    intersected = current_geom.intersection(later_geom.buffer(0))
                except Exception:
                    intersected = current_geom.intersection(later_geom)
                if not intersected.is_empty and intersected.is_valid:
                    # Add or replace with the more recent version for this arrival time
                    gdf_polygons.at[earlier_idx, 'geometry'] = intersected

        # Resort gdf_polygons by ascending arrival time
        gdf_polygons = gdf_polygons.sort_values(by='arr_time').reset_index(drop=True)

    # Save contours if requested
    if ('cont_poly' in out_request) or ('cont_line' in out_request):
        if not gdf_polygons.empty:
            if 'cont_poly' in out_request:
                _save_contour(gdf_polygons.geometry, 'poly', gdf_polygons, crs, fire_id, out_folder)
            if 'cont_line' in out_request:
                _save_contour(gdf_polygons.boundary, 'line', gdf_polygons, crs, fire_id, out_folder)
        else:
            _log('\t\tNo valid polygons available to generate contours.')

    # Export the burn perimeter if requested
    if 'perim' in out_request:
        _save_perimeter(gdf_polygons, total_run_time, crs, fire_id, out_folder)