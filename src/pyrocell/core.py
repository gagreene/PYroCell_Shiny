import os
import shutil
from datetime import datetime as dt
import math
import numpy as np
from osgeo import gdal
from rasterio.features import geometry_mask, shapes as rio_shapes
import rasterio as rio
from rasterio import windows, features
from geopandas import read_file
from shapely.geometry import Polygon, MultiPolygon, box, shape
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
from shapely import segmentize
from pyproj import Transformer
from heapq import heappop, heappush
from numba import float64
from numba import jit
from scipy.optimize import root_scalar
from typing import Optional, Union
import copy
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pyrocell.components import process_outputs as pout
from dependencies.flame_components import flame_components as fc

# Enable GDAL exceptions
gdal.UseExceptions()


class PYRO:
    def __init__(self,
                 fb_plugin: str,
                 ignition_path: str,
                 fire_date: Union[int, str],
                 elevation_path: str,
                 slope_path: str,
                 aspect_path: str,
                 out_request: Optional[list],
                 out_folder: str,
                 fire_id: Optional[str] = None,
                 ellipse_breadth: float = 3,
                 process_buffer: int = 1000,
                 use_accel: bool = True,
                 float_dtype: Union[np.dtype, type] = np.float32,
                 suppress_messages: bool = False,
                 use_breaching: bool = False,
                 breach_type: str = 'default',
                 remove_poly_ign: bool = False,
                 ign_segment_length: float = 10.0,
                 **fb_kwargs) -> None:
        """
        Initialize the PYroCell model with various environmental and configuration
        parameters essential for simulating fire spread.

        :param fb_plugin: The fire behaviour plugin to use (default is 'cffbps').
        :param ignition_path: Path to the ignition shapefile.
        :param fire_date: The date to use for FBP modelling (format: yyyymmdd).
        :param elevation_path: Path to the elevation dataset.
        :param slope_path: Path to the slope dataset (in percent).
        :param aspect_path: Path to the aspect dataset (in degrees).
        :param out_request: List of requested datasets to output.
        :param out_folder: Directory path for output results to be stored.
        :param fire_id: ID to apply to the output files as a prefix (e.g., f'{fire_id}_...').
        :param ellipse_breadth: Number of cells to use for the ellipse breadth.
        :param process_buffer: The default buffer value (meters) to use for processing extents.
        :param use_accel: Use acceleration for ignition point.
        :param float_dtype: Maximum array float datatype to use for processing - affects speed & precision.
            Decreases processing time, but produces slightly different fire spread results from the
            standard Numpy calculations.
        :param suppress_messages: If True, suppresses print statements from PyroCell.
        :param bros_scaler: Scaling factor for BROS (Back Rate of Spread) calculations.
        :param use_breaching: If True, uses breaching calculations for fire spread.
        :param breach_type: Type of breaching calculation to use ('default', 'prometheus', or 'simple').
        :param remove_poly_ign: If True, removes polygon ignition areas from output datasets.
        :param ign_segment_length: Segment length (in meters) to use when densifying line/polygon ignition geometries.
        """
        # Assign input values
        self.ignition_path = ignition_path
        self.fb_plugin = fb_plugin
        self.fire_date = fire_date
        self.elevation_path = elevation_path
        self.slope_path = slope_path
        self.aspect_path = aspect_path
        self.out_request = out_request
        self.out_folder = out_folder
        self.fire_id = fire_id if fire_id else 'Fire' + dt.now().strftime('%Y%m%d%H%M')
        self.ellipse_breadth = ellipse_breadth
        self.process_buffer = process_buffer  # Buffer distance in meters
        self.use_accel = use_accel
        self.float_dtype = float_dtype
        self.suppress_messages = suppress_messages
        self.use_breaching = use_breaching
        self.breach_type = breach_type
        self.remove_poly_ign = remove_poly_ign
        self.ignition_segment_length = ign_segment_length
        self.fb_kwargs = fb_kwargs

        # Initialize flags
        self.init_ignition = True
        self.spread_fire = True

        # Initialize runtime variables
        self.num_time_steps = None
        self.time_step = None
        self.current_dset = None
        self.total_run_time = None
        self.interval_mins = None

        # Initialize fire behaviour engine
        self.fb_engine = None

        # Initialize ignition parameters
        self.ignition_geom = None
        self.ignition_type = None
        self.ignition_point_cell = None
        self.ignition_point_geom = None

        # Initialize input raster specs
        self.max_grid_shape = None
        self.max_transform = None
        self.crs = None
        self.no_data = None
        self.cell_width = None
        self.cell_height = None

        # Initialize processing arrays
        self.travel_times = None
        self.lat = None
        self.long = None
        self.elevation = None
        self.elevation_dtype = None
        self.slope = None
        self.slope_dtype = None
        self.slope_rad = None
        self.aspect = None
        self.aspect_dtype = None
        self.aspect_array = None
        self.wsv = None
        self.raz = None
        self.hfi = None
        self.hros = None
        self.bros = None
        self.ros = None
        self.accel = None
        self.land_cover = None

        # Initialize processing variables
        self.buffer_default = self.process_buffer
        self.lb_ratio = None
        self.process_grid = None
        self.process_transform = None
        self.process_grid_shape = None
        self.process_grid_prior = None
        self.process_transform_prior = None
        self.process_grid_shape_prior = None
        self.process_row_offset = None
        self.process_col_offset = None
        self.valid_cells_mask = None

        # Initialize PYroCell-specific output parameters
        self.valid_outputs = ['perim', 'cont_line', 'cont_poly', 'flow_paths', 'major_flow_paths', 'tt', 'ros', 'fi']

        # Create temp folder
        self.temp_folder = os.path.join(self.out_folder, f'temp_{self.fire_id}')
        os.makedirs(self.temp_folder, exist_ok=True)

        # Define adjacent cell offsets for breaching
        self.adjacent_offsets = {
            'quad1': [(-1, 0), (0, 1)],
            'quad2': [(0, 1), (1, 0)],
            'quad3': [(1, 0), (0, -1)],
            'quad4': [(0, -1), (-1, 0)],
        }

        # Verify input data
        self._verify_inputs()

        # Get raster specifications
        self._get_raster_specs()

        # Export grid row and column indices for debugging
        self.counter = 0
        # self._export_grid_rows_cols(grid_source = 'max')

    def _log(self, msg: str):
        if not self.suppress_messages:
            print(msg)

    def _export_grid_rows_cols(self, grid_source: str = 'process'):
        """
        Export grid row and column indices as separate rasters for debugging.

        :return: None
        """
        # Assuming self.process_grid_shape = (rows, cols)
        if grid_source == 'max':
            rows, cols = self.max_grid_shape
            self.process_transform = self.max_transform
        else:
            rows, cols = self.process_grid_shape
        row_indices = np.arange(rows).reshape(-1, 1) * np.ones((1, cols), dtype=int)
        col_indices = np.arange(cols).reshape(1, -1) * np.ones((rows, 1), dtype=int)
        self._save_output(out_array=row_indices,
                          out_path=os.path.join(self.out_folder,
                                                f'row_indices_{self.time_step}_{self.counter}.tif'))
        self._save_output(out_array=col_indices,
                          out_path=os.path.join(self.out_folder,
                                                f'col_indices_{self.time_step}_{self.counter}.tif'))
        self.counter += 1
        return

    @staticmethod
    def _calc_raster_stats(src: rio.DatasetReader) -> rio.DatasetReader:
        """
        Function to recalculate statistics for each band of a rasterio dataset reader object

        :param src: input rasterio dataset reader object in 'r+' mode
        :return: rasterio dataset reader object in 'r+' mode
        """
        try:
            # Calculate statistics for all bands
            stats = src.stats()

            # Update dataset tags with the new statistics
            for i, band in enumerate(src.indexes):
                # Convert the Statistics object to a dictionary
                stats_dict = {
                    'min': stats[i].min,
                    'max': stats[i].max,
                    'mean': stats[i].mean,
                    'std': stats[i].std
                }
                src.update_tags(band, **stats_dict)
            return src
        except Exception:
            for bidx in src.indexes:
                try:
                    src.statistics(bidx, clear_cache=True)
                except rio.errors.StatisticsError as e:
                    print(f'Rasterio Calculate Statistics Error: {e}')
                    continue
        return src

    def _verify_inputs(self) -> None:
        """
        Function to verify the Class inputs.

        :return: None
        """
        if not isinstance(self.ignition_path, str):
            raise TypeError('The ignition_path parameter must be a string data type')

        if not isinstance(self.fire_date, (str, int)):
            raise TypeError('The fire_date parameter must be a string or int data type')
        elif isinstance(self.fire_date, str):
            self.fire_date = int(self.fire_date)

        if not isinstance(self.elevation_path, str):
            raise TypeError('The elevation_path parameter must be a string data type')
        else:
            self.elevation = rio.open(self.elevation_path)
            self.elevation_dtype = self.elevation.dtypes[0]
            if np.dtype(self.elevation_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.elevation_dtype = self.float_dtype

        if not isinstance(self.slope_path, str):
            raise TypeError('The slope_path parameter must be a string data type')
        else:
            self.slope = rio.open(self.slope_path)
            self.slope_dtype = self.slope.dtypes[0]
            if np.dtype(self.slope_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.slope_dtype = self.float_dtype

        if not isinstance(self.aspect_path, str):
            raise TypeError('The aspect_path parameter must be a string data type')
        else:
            self.aspect = rio.open(self.aspect_path)
            self.aspect_dtype = self.aspect.dtypes[0]
            if np.dtype(self.aspect_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.aspect_dtype = self.float_dtype

        if not isinstance(self.out_folder, str):
            raise TypeError('The out_folder parameter must be a string data type')

        if not isinstance(self.use_breaching, bool):
            raise TypeError('The use_breaching parameter must be a boolean data type')

        if not isinstance(self.breach_type, str):
            raise TypeError('The breach_type parameter must be a string data type')
        elif self.breach_type not in ['default', 'prometheus', 'simple']:
            raise ValueError('The breach_type parameter must be either "default", "prometheus", or "simple"')

        return

    def _verify_output_requests(self):
        """
        Verify the requested output datasets.
        :return: None
        """
        if self.out_request is not None:
            invalid_output_requests = []
            for out_item in self.out_request:
                if out_item not in self.valid_outputs + self.fb_engine.valid_outputs:
                    invalid_output_requests.append(out_item)
            if invalid_output_requests:
                self._save_text(
                    out_path=os.path.join(self.out_folder, f'{self.fire_id}_InvalidOutputRequests.txt'),
                    out_msg=f'Invalid output requests: {invalid_output_requests}.\n'
                            f'Valid options are: {self.valid_outputs + self.fb_engine.valid_outputs}'
                )

            self.out_request = [out_item for out_item in self.out_request
                                if out_item not in invalid_output_requests] if self.out_request else []

            if not self.out_request:
                self._log('No valid output requests specified. Exiting model run.')
                self.spread_fire = False

        return

    def _get_raster_specs(self) -> None:
        """
        Calculate the input raster specifications, including:
        the array shape, the transform, the projection, the no data value,
        and the cell width and height.

        :return: Tuple containing the grid shape, the raster transform,
            the raster projection (crs), the raster no data value, the
            raster cell width, and the raster cell height.
        """
        # Load the elevation raster to define grid shape and transform
        with rio.open(self.elevation_path) as src:
            self.max_grid_shape = src.shape
            self.max_transform = src.transform
            self.crs = src.crs  # Set the CRS from the reference raster
            self.no_data = -999  # Was src.nodata

        # Get input raster cell width and height
        self.cell_width, self.cell_height = abs(self.max_transform.a), abs(self.max_transform.e)

        return

    def _init_datasets(self) -> None:
        """
        Initialize processing datasets.

        :return: None
        """
        if self.travel_times is None:
            self._log('Initializing processing datasets...')

        # Initialize slope radians array
        self._log('\tGenerating slope radians array')
        # Extract slope array using the processing grid extent
        slope_array = self.slope.read(
            1,
            window=self.process_window,
            out_shape=self.process_grid_shape
        ).astype(self.float_dtype)
        self.slope_rad = np.arctan(slope_array / 100.0)
        del slope_array

        # Initialize aspect array
        self._log('\tExtracting aspect array')
        self.aspect_array = self.aspect.read(
            1,
            window=self.process_window,
            out_shape=self.process_grid_shape
        ).astype(self.float_dtype)

        # Generate lat and long rasters
        self._log('\tGenerating lat and long rasters')
        lat_file = os.path.join(self.temp_folder, f'{self.fire_id}_latitude.tif')
        lon_file = os.path.join(self.temp_folder, f'{self.fire_id}_longitude.tif')
        self.long, self.lat = self._get_grid_coords(
            src=self.elevation,
            out_file_x=lon_file,
            out_file_y=lat_file,
            window=self.process_window,
            dtype=self.float_dtype
        )

        # Initialize processing and output datasets
        process_data = ['travel_times', 'wsv', 'raz', 'hfi', 'ros', 'hros', 'bros', 'accel']

        # Initialize processing arrays
        if self.travel_times is None:
            self._log('\tInitializing processing arrays')
            # Initialize the fire behaviour engine
            self._init_fb_engine()

            # Initialize processing arrays using the processing grid
            process_dict = {
                var: np.full(self.process_grid_shape, np.inf).astype(self.float_dtype) for var in process_data
            }

            # Set the processing arrays as class attributes
            self._set_params(process_dict)
            del process_dict
        else:
            self._log('\tExpanding processing arrays')
            # Update the fire behaviour engine with the new processing window
            self.fb_engine.set_processing_window(
                self.process_window, self.process_grid_shape, self.process_row_offset, self.process_col_offset
            )

            # Expand existing processing arrays to match the current processing extent,
            # while retaining existing values and adding new values as np.inf
            for var in process_data:
                existing_array = getattr(self, var)
                new_array = np.full(self.process_grid_shape, np.inf).astype(self.float_dtype)
                # Determine where to place the existing array in the new array
                start_row = self.process_row_offset
                end_row = start_row + existing_array.shape[0]
                start_col = self.process_col_offset
                end_col = start_col + existing_array.shape[1]
                # Copy the entire existing array into the new array at the correct offset
                new_array[start_row:end_row, start_col:end_col] = existing_array
                setattr(self, var, new_array)

        # Initialize land cover dataset
        self._log(f'\tGenerating land cover dataset')
        self.land_cover = self.fb_engine.get_land_cover_array()

        return

    def _init_fb_engine(self) -> None:
        """
        Initialize the fire behaviour engine based on the selected plugin.
        """
        # Initialize fire behaviour plugin
        self._log(f'\tInitializing {self.fb_plugin} fire behaviour plugin')
        if self.fb_plugin == 'cffbps':
            from src.pyrocell.plugins import fbp_plugin as fbp

            pyro_kwargs = {
                'process_window': self.process_window,
                'process_grid_shape': self.process_grid_shape,
                'out_folder': self.out_folder,
                'fire_date': self.fire_date,
                'elevation': self.elevation,
                'slope': self.slope,
                'aspect': self.aspect,
                'lat_path': self.lat,
                'long_path': self.long,
                'float_dtype': self.float_dtype,
            }
            fb_init_keys = ['fueltype_path', 'pct_cnfr', 'pct_deadfir',
                            'grass_fuelload', 'grass_curingfactor', 'use_gpu',
                            'percentile_growth']
            fb_subset = {k: self.fb_kwargs[k] for k in fb_init_keys if k in self.fb_kwargs}

            # Initialize the FBP plugin
            self.fb_engine = fbp.FBP_PLUGIN(**{**pyro_kwargs, **fb_subset})

        # Verify output requests
        self._log(f'\t\tVerifying output requests')
        self._verify_output_requests()

        return

    def _init_ign_geoms(self) -> None:
        """
        Initialize ignition geometry. Supports Point, LineString, Polygon (with holes).

        - Point: Single ignition cell
        - LineString: Ignition along the line
        - Polygon: Interior burned, boundary spreads
        - Polygon with holes: Holes also ignite if surrounded by burnable fuel types
        """
        # Read ignition geometry
        gdf = read_file(self.ignition_path)
        if gdf.empty:
            self._log('No features found in the ignition dataset.')
            self._save_text(out_path=os.path.join(self.out_folder, f'{self.fire_id}_NoFeaturesInIgnitionFile.txt'),
                            out_msg='No features found in the ignition dataset.')
            self.spread_fire = False
            return
        if gdf.crs != self.crs:
            gdf = gdf.to_crs(self.crs)
        geometry = unary_union(gdf.geometry)

        # Remove geometry that is not within the bounds of self.process_window
        bounds_box = box(*rio.windows.bounds(self.process_window, self.max_transform))
        geometry = geometry.intersection(bounds_box)

        # Check if geometry is empty after intersection, and exit simulation if so
        if geometry.is_empty:
            self._log('Ignition geometry is outside the raster bounds.')
            self._save_text(out_path=os.path.join(self.out_folder, f'{self.fire_id}_IgnitionOutsideRasterBounds.txt'),
                            out_msg='Ignition geometry is outside the raster bounds.')
            self.spread_fire = False
            return

        # Assign ignition geometry
        self.ignition_geom = geometry

        # Get geometry type and convert to lower case
        geom_type = geometry.geom_type.lower()

        if 'point' in geom_type:
            self.ignition_type = 'point'
        elif 'line' in geom_type:
            self.ignition_type = 'line'
        elif 'polygon' in geom_type:
            self.ignition_type = 'polygon'
        else:
            self._log(f'Unsupported geometry type: {geometry.geom_type}')
            self._save_text(out_path=os.path.join(self.out_folder, f'{self.fire_id}_UnsupportedIgnitionGeomType.txt'),
                            out_msg=f'Unsupported ignition geometry type: {geometry.geom_type}')
            self.spread_fire = False
            return

        self.ignition_cells = []
        burned_interior_mask = np.zeros(self.process_grid_shape, dtype=bool)

        def _coords_to_cell(x, y) -> tuple[int, int, float, float]:
            gx, gy = self._geographic_to_grid(x, y, transform_type='process')
            row, col = int(gy), int(gx)
            sub_row, sub_col = gy - row, gx - col
            return row, col, sub_row, sub_col

        # Handle Point
        if self.ignition_type == 'point':
            row, col, sub_row, sub_col = _coords_to_cell(geometry.x, geometry.y)
            if self.land_cover[row, col] in [1, 2]:  # Only add if burnable fuel type
                self.ignition_cells.append((row, col, sub_row, sub_col))

        # Handle Line / MultiLine
        if self.ignition_type == 'line':
            # Densify line geometry
            gdf['geometry'] = gdf['geometry'].apply(lambda geom: segmentize(geom, self.ignition_segment_length))
            line_geometry = unary_union(gdf.geometry)
            # Clip to processing bounds
            line_geometry = line_geometry.intersection(bounds_box)
            gdf['geometry'] = [line_geometry]  # Update gdf with clipped geometry

            # Add ignition points along the line to ignition_cells
            for line in gdf.geometry:
                if line.is_empty:
                    continue
                if line.geom_type == 'LineString':
                    coords = list(line.coords)
                elif line.geom_type == 'MultiLineString':
                    coords = [c for ls in line.geoms for c in ls.coords]
                else:
                    continue
                for x, y in coords:
                    row, col, sub_row, sub_col = _coords_to_cell(x, y)
                    if 0 <= row < self.process_grid_shape[0] and 0 <= col < self.process_grid_shape[1]:
                        if self.land_cover[row, col] in [1, 2]:  # Only add if burnable fuel type
                            self.ignition_cells.append((row, col, sub_row, sub_col))

        # Handle Polygon / MultiPolygon
        if self.ignition_type == 'polygon':
            if geometry.geom_type.startswith('Multi'):
                polygons = list(geometry.geoms)
            else:
                polygons = [geometry]

            for poly in polygons:
                # Burn polygon interior (excluding holes)
                interior_mask = geometry_mask(
                    [poly],
                    out_shape=self.process_grid_shape,
                    transform=self.process_transform,
                    invert=True
                )
                self.travel_times[interior_mask] = 0
                burned_interior_mask[interior_mask] = True

                # Rasterize polygon to identify burned cells
                mask = PYRO._rasterize_geometry(geom=self.ignition_geom,
                                                out_shape=self.travel_times.shape,
                                                transform=self.process_transform)

                # Convert burned cell mask to polygon(s) for boundary extraction
                shapes_gen = rio_shapes(mask.astype(np.uint8), mask=mask, transform=self.process_transform)
                burned_polys = [shape(geom) for geom, val in shapes_gen if val == 1]

                # Buffer the burned polygons by 0.1 meters
                burned_polys = [bp.buffer(0.1) for bp in burned_polys]

                # Union the burned polygons with the original polygon to adjust for rasterization artifacts
                burned_union = unary_union(burned_polys)
                poly = poly.union(burned_union)

                # Densify exterior boundary using shapely `segmentize`
                densified_poly = segmentize(poly, self.ignition_segment_length)
                # Clip densified polygon to processing bounds
                densified_poly = densified_poly.intersection(bounds_box)
                # Get exterior coordinates
                if densified_poly.geom_type == 'Polygon':
                    exterior_coords = list(densified_poly.exterior.coords)
                    interiors = densified_poly.interiors
                elif densified_poly.geom_type == 'MultiPolygon':
                    exterior_coords = [coord for poly in densified_poly.geoms for coord in poly.exterior.coords]
                    interiors = [ring for poly in densified_poly.geoms for ring in poly.interiors]
                else:
                    exterior_coords = []
                    interiors = []

                # Add ignition along exterior boundary
                for x, y in exterior_coords:
                    row, col, sub_row, sub_col = _coords_to_cell(x, y)
                    if 0 <= row < self.process_grid_shape[0] and 0 <= col < self.process_grid_shape[1]:
                        if (not burned_interior_mask[row, col]) and (self.land_cover[row, col] in [1, 2]):
                            self.ignition_cells.append((row, col, sub_row, sub_col))

                # Handle holes (interiors) with densification
                for ring in interiors:
                    hole_poly = Polygon(ring)
                    hole_mask = geometry_mask(
                        [hole_poly],
                        out_shape=self.process_grid_shape,
                        transform=self.process_transform,
                        invert=True
                    )

                    # Check if the hole contains any burnable cells
                    hole_rows, hole_cols = np.where(hole_mask)
                    has_burnable = False
                    for r, c in zip(hole_rows, hole_cols):
                        if self.land_cover[r, c] in [1, 2]:
                            has_burnable = True
                            break

                    if not has_burnable:
                        continue  # skip this hole entirely

                    # Add ignition along hole boundary
                    hole_coords = list(ring.coords)
                    for x, y in hole_coords:
                        row, col, sub_row, sub_col = _coords_to_cell(x, y)
                        if not burned_interior_mask[row, col]:
                            self.ignition_cells.append((row, col, sub_row, sub_col))

        if len(self.ignition_cells) == 0:
            # Ignition is located in a non-burnable fuel type
            # Save FireInNonFuel text file and return empty queues
            self._log('\tIgnition(s) located in a non-burnable fuel type. No fire spread will be simulated.')
            self._save_text(out_path=os.path.join(self.out_folder, f'{self.fire_id}_IgnitionInNonFuel.txt'),
                            out_msg='Ignition(s) located in non-burnable fuel type.')
            self.spread_fire = False

        return

    def _set_processing_extent(self, queue_list: list) -> list:
        """
        Calculate and set the current geographic processing extent, buffered by a specified distance.
        The extent is determined by the ignition cells (if initializing ignition) or by all valid cells
        in the travel times array, as well as any cells in the priority and carry-over queues.
        The resulting extent is clipped to the bounds of the input raster.

        :param queue_list: List of queues to process
        :return: None
        """
        def _unique_cells_from_queue(_queue) -> set:
            """Extract unique (row, col) tuples from a nested queue structure."""
            unique = set()
            if not _queue:
                return unique
            for sublist in _queue:
                if isinstance(sublist[0], float):
                    unique.add((sublist[1], sublist[2]))
                    continue
                for cell_data in sublist:
                    if isinstance(cell_data, (list, tuple)) and len(cell_data) > 0:
                        cell = cell_data[0]
                        if isinstance(cell, tuple) and len(cell) == 2:
                            unique.add(cell)
                        else:
                            self._log(f'Unexpected format in queue cell: {cell}')
            return unique

        # Assign current processing extent variables to the prior variables
        self.process_grid_prior = self.process_grid
        self.process_transform_prior = self.process_transform
        self.process_grid_shape_prior = self.process_grid_shape

        if self.init_ignition:
            # Use buffer around ignition geometry to define the processing extent
            gdf = read_file(self.ignition_path)
            if gdf.crs != self.crs:
                gdf = gdf.to_crs(self.crs)
            geometry = unary_union(gdf.geometry)
            del gdf
            if geometry.is_empty:
                raise ValueError('Ignition geometry is empty after reprojecting to raster CRS.')
            min_x, min_y, max_x, max_y = geometry.bounds

            # Convert geographic extent to grid indices and back to ensure alignment with cell centers
            cmin, rmax = self._geographic_to_grid(min_x, min_y, transform_type='max')
            cmax, rmin = self._geographic_to_grid(max_x, max_y, transform_type='max')
            min_x, min_y = self._grid_to_geographic(
                math.floor(cmin), math.floor(rmax), get_cell_center=True, transform_type='max'
            )
            max_x, max_y = self._grid_to_geographic(
                math.floor(cmax), math.floor(rmin), get_cell_center=True, transform_type='max'
            )
        else:
            # Use all valid travel time and queue cells to define the extent
            self.valid_cells_mask = np.isfinite(self.travel_times)
            cells = set(zip(*np.where(self.valid_cells_mask)))
            for queue in queue_list:
                cells.update(_unique_cells_from_queue(queue))
            if cells:
                rows, cols = zip(*cells)
                rmin, rmax = min(rows), max(rows)
                cmin, cmax = min(cols), max(cols)
                min_x, min_y = self._grid_to_geographic(
                    col=cmin, row=rmax, get_cell_center=True, transform_type='process'
                )
                max_x, max_y = self._grid_to_geographic(
                    col=cmax, row=rmin, get_cell_center=True, transform_type='process'
                )
            else:
                # No valid cells; use prior processing extent
                return queue_list

        # Apply processing buffer
        buffer_x_conversion = math.ceil(self.process_buffer / self.cell_width) * self.cell_width
        buffer_y_conversion = math.ceil(self.process_buffer / self.cell_height) * self.cell_height
        min_x, min_y = min_x - buffer_x_conversion, min_y - buffer_y_conversion
        max_x, max_y = max_x + buffer_x_conversion, max_y + buffer_y_conversion

        # Clip the extent to the maximum raster bounds
        raster_min_x, raster_max_y = self._grid_to_geographic(
            col=0, row=0, get_cell_center=True, transform_type='max'
        )
        raster_max_x, raster_min_y = self._grid_to_geographic(
            col=self.max_grid_shape[1], row=self.max_grid_shape[0], get_cell_center=True, transform_type='max'
        )

        # Clip the extent to the maximum raster bounds
        min_x = max(min_x, raster_min_x)
        max_x = min(max_x, raster_max_x)
        min_y = max(min_y, raster_min_y)
        max_y = min(max_y, raster_max_y)

        # Convert geographic extent to grid indices using cell width and height
        left_col, top_row = self._geographic_to_grid(min_x, max_y, transform_type='max')
        right_col, bottom_row = self._geographic_to_grid(max_x, min_y, transform_type='max')
        top_row = int(top_row)
        left_col = int(left_col)
        bottom_row = int(bottom_row)
        right_col = int(right_col)

        # Set the transform for the processing window using floating point column and row values
        min_x = self.max_transform.c + left_col * self.cell_width
        max_x = self.max_transform.c + right_col * self.cell_width
        min_y = self.max_transform.f - bottom_row * self.cell_height
        max_y = self.max_transform.f - top_row * self.cell_height
        self.process_transform = rio.transform.from_bounds(
            min_x, min_y, max_x, max_y,
            right_col - left_col,
            bottom_row - top_row
        )

        # Define the processing window
        self.process_window = rio.windows.from_bounds(
            min_x, min_y, max_x, max_y,
            transform=self.max_transform
        ).round_offsets().round_lengths()

        # Round to nearest integer for safe indexing
        self.process_grid = (left_col, top_row, right_col, bottom_row)
        self.process_grid_shape = (self.process_window.height, self.process_window.width)

        if not self.init_ignition:
            # Determine row and column offsets
            self.process_row_offset = self.process_grid_prior[1] - self.process_grid[1]
            self.process_col_offset = self.process_grid_prior[0] - self.process_grid[0]

        # Update cell indices in the input queues to reflect changes in the processing grid extent
        updated_queues = self._update_queue_grid_rows_cols(queue_list=copy.deepcopy(queue_list))
        del queue_list

        # Initialize datasets
        self._init_datasets()

        # For debugging
        # self._export_grid_rows_cols()

        return updated_queues

    @staticmethod
    def _update_queue_grid_cells(queue: list, row_offset: int, col_offset: int) -> list:
        """
        Update cell indices in the queue by specified offsets.

        :param queue: Priority queue (list of tuples) or carry over queue (list of lists of tuples)
        :param row_offset: Integer row offset to apply
        :param col_offset: Integer column offset to apply
        :return: Updated queue with adjusted cell indices
        """
        # Priority queue: list of tuples (current_time, row, col)
        if isinstance(queue[0], tuple):
            updated = []
            for item in queue:
                if isinstance(item, tuple) and len(item) >= 3:
                    new_item = (item[0], item[1] + row_offset, item[2] + col_offset) + item[3:]
                    updated.append(new_item)
                else:
                    print(f'Unexpected format in priority queue item: {item}')
            return updated
        # Carry over queue: list of lists of tuples
        elif isinstance(queue[0], list):
            for sublist in queue:
                for i, cell_data in enumerate(sublist):
                    if isinstance(cell_data, (list, tuple)) and len(cell_data) > 0:
                        cell = cell_data[0]
                        if isinstance(cell, tuple) and len(cell) == 2:
                            new_cell = (cell[0] + row_offset, cell[1] + col_offset)
                            new_cell_data = [new_cell,] + list(cell_data[1:])
                            sublist[i] = new_cell_data
                        else:
                            print(f'Unexpected format in carry over queue cell: {cell}')
            return queue
        else:
            print(f'Unknown queue structure: {queue}')
            return queue

    def _update_queue_grid_rows_cols(self, queue_list: list) -> list:
        """
        Update the cell indices in the input queues to reflect changes in the processing grid extent.
        Handles both priority queues (list of tuples) and carry over queues (list of lists of tuples).

        :param queue_list: List of queues to update
        :return: Updated list of queues with adjusted cell indices
        """
        if self.process_grid_prior is not None:
            if (self.process_row_offset == 0) and (self.process_col_offset == 0):
                return queue_list
            updated_queues = []
            for queue in queue_list:
                if not queue:
                    updated_queues.append([])
                    continue
                updated_queue = self._update_queue_grid_cells(queue, self.process_row_offset, self.process_col_offset)
                updated_queues.append(updated_queue)
            return updated_queues
        else:
            return queue_list

    def _burn_ignition(self,
                       dset_run_time: float,
                       accel_runtime_carry_over_queue: list) -> tuple[list, list, list]:
        """
        Burn the ignition geometry, set initial travel times, and build priority queues.

        :param dset_run_time: The dataset's maximum run time.
        :param accel_runtime_carry_over_queue: The acceleration run time carry over queue.
        :return: priority_queue, carry_over_queue, accel_runtime_carry_over_queue
        """
        # Initialize queues
        accel_priority_queue = []
        accel_carry_over_queue = []
        priority_queue = []
        carry_over_queue = []

        # Ensure only point feature types use acceleration
        calc_accel = self.use_accel and (self.ignition_type == 'point')

        for grid_row, grid_col, sub_row, sub_col in self.ignition_cells:

            target_offsets = self._gen_ellipse_offsets(spread_direction=self.raz[grid_row, grid_col],
                                                       lb_ratio=self.lb_ratio[grid_row, grid_col],
                                                       breadth=self.ellipse_breadth,
                                                       hros=self.hros[grid_row, grid_col],
                                                       bros=self.bros[grid_row, grid_col],
                                                       add_source_cell=True)
            # target_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for d_row, d_col in target_offsets:
                target_row, target_col = grid_row + d_row, grid_col + d_col
                # Ensure target cell is within the processing grid/extent
                if 0 <= target_row < self.process_grid_shape[0] and 0 <= target_col < self.process_grid_shape[1]:
                    if self.land_cover[grid_row, grid_col] in [1, 2] and \
                            self.land_cover[target_row, target_col] in [1, 2]:

                        intersected_cells = self._calc_travel_times(
                            grid_row=grid_row + sub_row,
                            grid_col=grid_col + sub_col,
                            target_d_row=d_row,
                            target_d_col=d_col,
                            use_exact=True,
                            calc_accel=calc_accel,
                            current_time=0
                        )

                        if intersected_cells:
                            if calc_accel:
                                (accel_priority_queue, accel_carry_over_queue,
                                 accel_runtime_carry_over_queue) = self._process_travel_times_accel(
                                    process_cells=intersected_cells,
                                    current_time=0,
                                    dset_run_time=dset_run_time,
                                    accel_priority_queue=accel_priority_queue,
                                    accel_carry_over_queue=accel_carry_over_queue,
                                    accel_runtime_carry_over_queue=accel_runtime_carry_over_queue
                                )
                            else:
                                priority_queue, carry_over_queue = self._process_travel_times(
                                    process_cells=intersected_cells,
                                    current_time=0,
                                    dset_run_time=dset_run_time,
                                    priority_queue=priority_queue,
                                    carry_over_queue=carry_over_queue
                                )

        # Process accel priority queue
        while accel_priority_queue:
            current_time, grid_row, grid_col = heappop(accel_priority_queue)
            if (current_time > self.travel_times[grid_row, grid_col]) or (current_time > dset_run_time):
                continue

            target_offsets = self._gen_ellipse_offsets(spread_direction=self.raz[grid_row, grid_col],
                                                       lb_ratio=self.lb_ratio[grid_row, grid_col],
                                                       breadth=self.ellipse_breadth,
                                                       hros=self.hros[grid_row, grid_col],
                                                       bros=self.bros[grid_row, grid_col])

            for d_row, d_col in target_offsets:
                target_row, target_col = grid_row + d_row, grid_col + d_col
                # Ensure target cell is within the processing grid/extent
                if 0 <= target_row < self.process_grid_shape[0] and 0 <= target_col < self.process_grid_shape[1]:
                    if self.land_cover[grid_row, grid_col] in [1, 2] and \
                            self.land_cover[target_row, target_col] in [1, 2]:

                        intersected_cells = self._calc_travel_times(
                            grid_row=grid_row,
                            grid_col=grid_col,
                            target_d_row=d_row,
                            target_d_col=d_col,
                            calc_accel=True,
                            current_time=current_time
                        )

                        if intersected_cells:
                            (accel_priority_queue, accel_carry_over_queue,
                             accel_runtime_carry_over_queue) = self._process_travel_times_accel(
                                process_cells=intersected_cells,
                                current_time=current_time,
                                dset_run_time=dset_run_time,
                                accel_priority_queue=accel_priority_queue,
                                accel_carry_over_queue=accel_carry_over_queue,
                                accel_runtime_carry_over_queue=accel_runtime_carry_over_queue
                            )

            # Promote accel carry over to normal carry over
            priority_queue, carry_over_queue = self._process_carryover(
                priority_queue=priority_queue,
                carry_over_queue=accel_carry_over_queue,
                dset_run_time=dset_run_time
            )

        return priority_queue, carry_over_queue, accel_runtime_carry_over_queue

    @staticmethod
    @jit(float64(float64, float64, float64, float64, float64), cache=True, nopython=True)
    def _calc_ros(spread_direction: float,
                  hros: float,
                  bros: float,
                  raz: float,
                  lb_ratio: float) -> float:
        """
        Calculate the rate of spread (ROS) dynamically for head, backing, and flanking directions,
        using Numba JIT precompiled code to speed up processing.

        :param spread_direction: The direction of fire spread (compass degrees).
        :param hros: Head fire rate of spread.
        :param bros: Backing fire rate of spread.
        :param raz: Relative azimuth (wind direction).
        :param lb_ratio: Precomputed length-to-breadth ratio.
        :return: Calculated directional rate of spread (ROS).
        """
        # Relative angle to wind (theta)
        theta = abs(spread_direction - raz) % 360
        if theta < 180:
            theta = 360 - theta  # Normalize to [180, 360]

        # Set theta to 270.001 if it equals 270 to avoid undefined value error
        if theta == 270.0:
            theta = 270.001

        # Calculate flanking rate of spread (fros)
        fros = (hros + bros) / (lb_ratio * 2)

        # Calculate ROS
        theta_rad = math.radians(theta)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)

        term1 = (hros - bros) / (2 * cos_theta)
        term2 = (hros + bros) / (2 * cos_theta)

        term3_numerator = (
                fros * cos_theta * math.sqrt((fros**2) * (cos_theta**2) + (hros * bros) * (sin_theta**2)) -
                ((hros**2 - bros**2) * (sin_theta**2)) / 4
        )

        term3_denominator = ((fros**2) * (cos_theta**2) + (((hros + bros) / 2)**2) * (sin_theta**2))

        # Handle zero denominator
        if term3_denominator == 0:
            # Default to infinity
            return math.inf

        # Return the final ROS value
        return term1 + term2 * term3_numerator / term3_denominator

    @staticmethod
    @jit(float64(float64, float64, float64, float64), cache=True, nopython=True)
    def _calc_slope_dist(travel_dist: float,
                         travel_angle_deg: float,
                         slope_rad: float,
                         aspect_deg: float) -> float:
        """
        Adjust the travel distance based on slope and direction of travel,
        using vertical rise and projected cross-slope distance.

        :param travel_dist: Base travel distance in meters.
        :param travel_angle_deg: Direction of travel in degrees.
        :param slope_rad: Slope in radians for the current cell.
        :param aspect_deg: Aspect (slope direction) of the current cell
            in degrees (0 = North, 90 = East, etc.).
        :return: Slope-adjusted distance.
        """
        if slope_rad == 0:
            return travel_dist
        else:
            # Calculate the absolute difference between the travel angle and the slope aspect
            # This gives the relative alignment of the travel direction with the slope aspect
            cross_slope_angle_deg = abs(travel_angle_deg - aspect_deg)

            # Adjust the cross-slope angle to align with compass directions
            # Ensure that the angle lies between -180 and 180 degrees
            cross_slope_angle_ras = math.radians(
                180 - cross_slope_angle_deg if cross_slope_angle_deg > 180 else cross_slope_angle_deg - 180
            )

            # Calculate the distance component along the cross-slope direction
            # This is the projection of the travel distance along the slope
            cross_slope_dist_y = travel_dist * math.cos(cross_slope_angle_ras)

            # Calculate the vertical rise due to the slope
            # Vertical rise = cross-slope distance * tangent of the slope angle
            slope_rise = cross_slope_dist_y * math.tan(slope_rad)

            # Use the Pythagorean theorem to calculate the slope-adjusted distance
            # Slope-adjusted distance = sqrt(horizontal^2 + vertical^2)
            slope_dist = math.sqrt(travel_dist**2 + slope_rise**2)

            return slope_dist

    @staticmethod
    def _calc_breaching_dist(breach_type: str,
                             fire_type: int,
                             fire_intensity: float,
                             stand_ht: float) -> float:
        """
        Calculate the breaching distance threshold for a fire.
        The breaching distance threshold is calculated as 1.5 times the flame length.
        This "rule of thumb" is per Alexander et al. (2004) - Incorporating breaching and spotting
        considerations into PROMETHEUS - the Canadian wildland fire growth model.

        :param breach_type: Type of breaching model ('default' or 'prometheus').
        :param fire_type: Fire type (1 = surface, 2 = passive crown/intermittent crown, 3 = active crown/crown).
        :param fire_intensity: Fire intensity (kW/m).
        :param stand_ht: Stand height (m).
        :return: Maximum distance (threshold) to breach fuel break (m).
        """
        # Calculate flame length based on fire intensity and type
        if fire_type == 3:
            if breach_type == 'prometheus':
                return stand_ht * 2.5
            else:
                model = 'Butler_HEAD'
        else:
            model = 'Byram_HEAD'
            stand_ht = 0.0
        flame_length = fc.getFlameLength(model=model, fire_intensity=fire_intensity)

        return 1.5 * (flame_length + stand_ht)

    @staticmethod
    @jit(float64(float64, float64, float64), cache=True, nopython=True)
    def _calc_accel_dist_at_time(ros_eq: float, accel_param: float, time: float) -> float:
        """
        Estimate the head fire spread distance at the provided time under acceleration.

            D = ROS_eq * (t + (e^(-a * t) / a) - (1 / a))

        :param ros_eq: Equilibrium rate of spread for the current cell.
        :param accel_param: Acceleration parameter for the current cell.
        :param time: Fire burning time for the current cell.
        :return: Estimated acceleration rate of spread.
        """
        return ros_eq * (time + (math.exp(-accel_param * time) / accel_param) - (1 / accel_param))

    @staticmethod
    @jit(float64(float64, float64, float64), cache=True, nopython=True)
    def _calc_accel_ros_at_time(ros_eq: float, accel_param: float, time: float) -> float:
        """
        Estimate the acceleration (a) rate of spread at the provided time using the equilibrium rate of spread (ros_eq).

            ROS_t = ROS_eq * (1 - e^(-a * t))

        :param ros_eq: Equilibrium rate of spread for the current cell.
        :param accel_param: Acceleration parameter for the current cell.
        :param time: Fire burning time for the current cell.
        :return: Estimated acceleration rate of spread.
        """
        return ros_eq * (1 - math.exp(-accel_param * time))

    @staticmethod
    def _calc_accel_time_at_dist(accel_param: float, ros_eq: float, dist: float) -> float:
        """
        Estimate the time (in minutes) required to travel a given distance under acceleration conditions,
        using Newton-Raphson method to solve:

            dist = ROS_eq * [t + (exp(-a * t) - 1)/a]

        where:
            - a = acceleration parameter
            - ROS_eq = equilibrium rate of spread
            - t = time (minutes)
            - dist = travel distance (meters)

        :param accel_param: Acceleration parameter (a)
        :param ros_eq: Equilibrium rate of spread (ROS_eq) in m/min
        :param dist: Travel distance in meters
        :return: Time in minutes required to traverse the distance, or np.inf if invalid
        """

        # ### Solve for t_d in Equation 71: D = ROS_eq * (t_d + e^(-a*t_d)/a - 1/a)
        # Equation to solve for t_d
        def _equation(t_d):
            return ros_eq * (t_d + math.exp(-accel_param * t_d) / accel_param - 1 / accel_param) - dist

        # Default solution to None
        solution = None

        # Use a numerical solver to find t_d
        for upper_lim in range(1000, 30000, 1000):
            try:
                solution = root_scalar(_equation, bracket=[0, upper_lim], method='brentq')
                break
            except ValueError:
                pass

        if solution is not None:
            if solution.converged:
                return solution.root
            else:
                raise ValueError('Solution did not converge - unable to determine travel time during acceleration')
        else:
            raise ValueError(f'Unable to calculate acceleration time at distance {dist}')

    @staticmethod
    @jit(float64(float64), cache=True, nopython=True)
    def _calc_accel_time_99pct(a_param: float) -> float:
        """
        Calculate the time (in minutes) at which ROS_t == 0.99 * ROS_eq.

        :param a_param: Acceleration parameter.
        :return: The minute at which ROS_t >= 0.99 * ROS_eq.
        """
        return -math.log(0.01) / a_param if a_param != 0 else math.inf

    def _calc_travel_times(self,
                           grid_row: float,
                           grid_col: float,
                           target_d_row: int,
                           target_d_col: int,
                           use_exact: bool = False,
                           calc_accel: bool = False,
                           current_time: Optional[float] = None) -> list[tuple]:
        """
        Calculates:
            (a) the cells intersected by a line from the center cell (or ignition point) to the target cell,
            (b) the segment lengths within each intersected cell,
            (c) fire behaviour for each line segment
            (d) the rate of spread (theta_ros) for each line segment,
            (e) the travel times for each line segment.

        :param grid_row: Grid cell row of the current location.
        :param grid_col: Grid cell column of the current location.
        :param target_d_row: Row index of the target cell.
        :param target_d_col: Column index of the target cell.
        :param use_exact: If True, use exact grid cell coordinates instead of the cell center.
        :param calc_accel: If True, calculate acceleration travel times.
        :return: None if calc_accel is False, otherwise a list of tuples containing:
                    - Cell coordinates
                    - Travel angle (degrees)
                    - Slope-adjusted distance
                    - ROS
                    - Travel time
        """
        # Start and end points for the line
        if use_exact:
            # Use exact ignition point coordinates
            center_row = grid_row - int(grid_row)
            center_col = grid_col - int(grid_col)
        else:
            # Use the center of the source (0, 0) cell
            center_row, center_col = 0.5, 0.5

        start_x = center_col * self.cell_width
        start_y = center_row * self.cell_height
        end_x = (target_d_col + 0.5) * self.cell_width
        end_y = (target_d_row + 0.5) * self.cell_height

        dx = end_x - start_x
        dy = end_y - start_y
        polar_angle = math.atan2(-dy, dx)  # Angle in radians from the polar coordinate system
        travel_angle_deg = float((90 - math.degrees(polar_angle)) % 360)  # Convert to compass degrees

        x, y = start_x, start_y
        d_row, d_col = 0, 0
        intersected_cells = []
        prior_x, prior_y = x, y

        # Initialize breaching variables
        cell_counter = 0
        dist_breached = 0.0
        prior_grid_row = None
        prior_grid_col = None
        prior_theta_ros = None

        if calc_accel:
            # Initialize acceleration variables
            prior_travel_time = current_time
            accel = None
            max_accel_time = None
            accum_slope_dist = PYRO._calc_accel_dist_at_time(ros_eq=self.hros[int(grid_row), int(grid_col)],
                                                             accel_param=self.accel[int(grid_row), int(grid_col)],
                                                             time=current_time)

        while (d_row, d_col) != (target_d_row, target_d_col):
            # Current grid coordinates
            current_grid_row = int(grid_row) + d_row
            current_grid_col = int(grid_col) + d_col

            # Determine the next intersection points with cell boundaries
            next_vertical_x = (d_col + 1) * self.cell_width if dx > 0 else d_col * self.cell_width
            next_horizontal_y = (d_row + 1) * self.cell_height if dy > 0 else d_row * self.cell_height

            t_to_vertical = (next_vertical_x - x) / dx if dx != 0 else float('inf')
            t_to_horizontal = (next_horizontal_y - y) / dy if dy != 0 else float('inf')

            if t_to_vertical < t_to_horizontal:
                x = next_vertical_x
                y += t_to_vertical * dy
                d_col = d_col + (1 if dx > 0 else -1)
            elif t_to_vertical == t_to_horizontal:
                x = next_vertical_x
                y = next_horizontal_y
                d_col = d_col + (1 if dx > 0 else -1)
                d_row = d_row + (1 if dy > 0 else -1)
            else:
                x += t_to_horizontal * dx
                y = next_horizontal_y
                d_row = d_row + (1 if dy > 0 else -1)

            # Check if the next cell is non-fuel
            lc_type = self.land_cover[current_grid_row, current_grid_col]
            lc_type_next = self.land_cover[int(grid_row) + d_row, int(grid_col) + d_col]
            non_fuel_encountered = (lc_type not in [1, 2])
            non_fuel_encountered_next = (lc_type_next not in [1, 2])

            # End calculations if not breaching, and either current or next cells are non-fuel
            if not self.use_breaching and (non_fuel_encountered or non_fuel_encountered_next):
                return []

            # Check for non-fuel cells in adjacent cells
            at_cell_corner = (abs((x / self.cell_width) % 1) < 0.1) and (abs((y / self.cell_height) % 1) < 0.1)
            if at_cell_corner and not (non_fuel_encountered or non_fuel_encountered_next):
                # Map travel angles to adjacent cell offsets
                if 0 < travel_angle_deg <= 90:
                    offsets = self.adjacent_offsets['quad1']
                elif 90 < travel_angle_deg <= 180:
                    offsets = self.adjacent_offsets['quad2']
                elif 180 < travel_angle_deg <= 270:
                    offsets = self.adjacent_offsets['quad3']
                else:  # if 270 < travel_angle_deg <= 360 or travel_angle_deg == 0:
                    offsets = self.adjacent_offsets['quad4']
                if offsets:
                    adj1_row = current_grid_row + offsets[0][0]
                    adj1_col = current_grid_col + offsets[0][1]
                    adj2_row = current_grid_row + offsets[1][0]
                    adj2_col = current_grid_col + offsets[1][1]

                    adj1_lc = self.land_cover[adj1_row, adj1_col] if (
                        0 <= adj1_row < self.process_grid_shape[0] and 0 <= adj1_col < self.process_grid_shape[1]
                    ) else None
                    adj2_lc = self.land_cover[adj2_row, adj2_col] if (
                        0 <= adj2_row < self.process_grid_shape[0] and 0 <= adj2_col < self.process_grid_shape[1]
                    ) else None
                    if any(lc not in [1, 2] for lc in [adj1_lc, adj2_lc]):
                        if self.use_breaching:
                            if self.breach_type == 'simple':
                                prior_x, prior_y = x, y
                                continue
                            else:
                                pass
                        else:
                            return []

            # Calculate the segment length from prior position to current position
            segment_length = float(math.sqrt((x - prior_x)**2 + (y - prior_y)**2))

            # Handle zero-length segments
            if segment_length <= 0:
                if non_fuel_encountered:
                    if self.use_breaching:
                        prior_x, prior_y = x, y
                        continue
                    else:
                        return []
                else:
                    prior_x, prior_y = x, y
                    continue

            # Calculate slope-adjusted distance
            slope_dist = self._calc_slope_dist(
                travel_dist=segment_length,
                travel_angle_deg=travel_angle_deg,
                slope_rad=self.slope_rad[current_grid_row, current_grid_col],
                aspect_deg=self.aspect_array[current_grid_row, current_grid_col]
            )

            # PROCESS BREACHING
            if non_fuel_encountered:
                # If breaching is True, and the fire is not in the acceleration phase,
                # check if the cell can be breached
                if self.use_breaching and (self.breach_type in ['default', 'prometheus']) and not calc_accel:
                    # Add the segment length to dist_breached
                    dist_breached += slope_dist

                    # Rescale prior_grid_row and prior_grid_col to match the process_extent
                    # rescaled_row = prior_grid_row - self.process_grid[1]  # Subtract top-left row
                    # rescaled_col = prior_grid_col - self.process_grid[0]  # Subtract top-left column

                    # Calculate the current fire type using rescaled row/col values
                    # (to match the processing extent)
                    fire_type = self.fb_engine.get_fire_type(
                        ros=prior_theta_ros,
                        row=prior_grid_row,
                        col=prior_grid_col
                    )

                    # Calculate the current fire type using the full row/col values
                    stand_ht = self.fb_engine.get_stand_ht(
                        row=prior_grid_row,
                        col=prior_grid_col
                    )

                    # Calculate the fire intensity for the prior fuel cell by scaling HFI
                    # according to the ratio of theta ROS to the head fire ROS
                    # FI = Hwr, where H and w are constant for the cell (w is not dependent on a
                    # fire behaviour metric). Therefore, FI is proportional to ROS.
                    scaled_fire_intensity = (self.hfi[prior_grid_row, prior_grid_col] *
                                             (prior_theta_ros / self.hros[prior_grid_row, prior_grid_col]))

                    # Calculate the breaching distance threshold
                    breach_dist = self._calc_breaching_dist(
                        breach_type=self.breach_type,
                        fire_type=fire_type,
                        fire_intensity=scaled_fire_intensity,
                        stand_ht=stand_ht
                    )

                    # If the non-fuel segment length is greater than the breach distance, terminate calculation
                    if (slope_dist > breach_dist) or (dist_breached > breach_dist):
                        return []
                    else:
                        prior_x, prior_y = x, y
                        continue
                else:
                    return []
            else:
                # Reset dist_breached if fuel cell encountered
                dist_breached = 0.0

                # Get the lb_ratio
                lb_ratio = self.lb_ratio[current_grid_row, current_grid_col]

                # If in acceleration phase, adjust lb_ratio
                if calc_accel:
                    accum_slope_dist += slope_dist
                    accel = self.accel[current_grid_row, current_grid_col]
                    max_accel_time = PYRO._calc_accel_time_99pct(accel)
                    hros_accel_travel_time = PYRO._calc_accel_time_at_dist(
                        accel_param=accel,
                        ros_eq=self.hros[current_grid_row, current_grid_col],
                        dist=accum_slope_dist - (slope_dist/2))
                    lb_ratio = (lb_ratio - 1) * (1 - math.exp(-accel * hros_accel_travel_time)) + 1

                # Calculate rate of spread (ROS) for the current position and spread direction
                theta_ros_eq = self._calc_ros(
                    spread_direction=travel_angle_deg,
                    hros=self.hros[current_grid_row, current_grid_col],
                    bros=self.bros[current_grid_row, current_grid_col],
                    raz=self.raz[current_grid_row, current_grid_col],
                    lb_ratio=lb_ratio
                )

                # Assumed that processing is faster to assign these three variables than it is
                # to calculate fire_type, stand_ht, scaled_fire_intensity and breach_dist for each fuel cell
                prior_grid_row = current_grid_row
                prior_grid_col = current_grid_col
                prior_theta_ros = theta_ros_eq

            # Calculate the cell travel time
            if theta_ros_eq > 0:
                if calc_accel:
                    # if cell_counter > 0:
                    travel_time = PYRO._calc_accel_time_at_dist(accel_param=accel,
                                                                ros_eq=theta_ros_eq,
                                                                dist=accum_slope_dist)
                    # else:
                    #     travel_time = PYRO._calc_accel_time_at_dist(accel_param=accel,
                    #                                                 ros_eq=theta_ros_eq,
                    #                                                 dist=slope_dist)
                    # accum_accel_time += travel_time
                    segment_travel_time = travel_time - prior_travel_time
                    prior_travel_time = travel_time

                    if cell_counter > 0:
                        # Adjust theta_ros_eq to account for acceleration and assign to theta_ros
                        theta_ros = PYRO._calc_accel_ros_at_time(ros_eq=theta_ros_eq,
                                                                 accel_param=accel,
                                                                 time=travel_time)
                    else:
                        # Adjust theta_ros_eq to account for acceleration and assign to theta_ros
                        theta_ros = PYRO._calc_accel_ros_at_time(ros_eq=theta_ros_eq,
                                                                 accel_param=accel,
                                                                 time=current_time)
                else:
                    travel_time = slope_dist / theta_ros_eq

                    # Assign theta_ros_eq to theta_ros
                    theta_ros = theta_ros_eq
            else:
                travel_time = float('inf')
                theta_ros = None

            # Add the data to the intersected cells list
            if calc_accel:
                if max_accel_time is not None:
                    intersected_cells.append(((current_grid_row, current_grid_col),
                                              travel_angle_deg,  accum_slope_dist, slope_dist,
                                              theta_ros, segment_travel_time, max_accel_time))
                else:
                    raise ValueError('Error: Unable to calculate max_accel_time.')
            else:
                intersected_cells.append(((current_grid_row, current_grid_col),
                                          travel_angle_deg, slope_dist,
                                          theta_ros, travel_time))
            prior_x, prior_y = x, y
            cell_counter += 1

        # Handle the final segment reaching the target cell
        target_grid_row = int(grid_row) + target_d_row
        target_grid_col = int(grid_col) + target_d_col

        # Calculate the final segment length
        final_length = math.sqrt((end_x - prior_x)**2 + (end_y - prior_y)**2)

        # Calculated slope-adjusted distance
        slope_dist = self._calc_slope_dist(
            travel_dist=final_length,
            travel_angle_deg=travel_angle_deg,
            slope_rad=self.slope_rad[target_grid_row, target_grid_col],
            aspect_deg=self.aspect_array[target_grid_row, target_grid_col]
        )

        # Get the lb_ratio
        lb_ratio = self.lb_ratio[target_grid_row, target_grid_col]

        # If in acceleration phase, adjust lb_ratio
        accel = None
        if calc_accel:
            accum_slope_dist += slope_dist
            accel = self.accel[target_grid_row, target_grid_col]
            max_accel_time = self._calc_accel_time_99pct(accel)
            hros_accel_travel_time = self._calc_accel_time_at_dist(
                accel_param=accel,
                ros_eq=self.hros[target_grid_row, target_grid_col],
                dist=accum_slope_dist)
            lb_ratio = (lb_ratio - 1) * (1 - math.exp(-accel * hros_accel_travel_time)) + 1

        # Calculate the rate of spread (ROS) for the current position
        theta_ros_eq = self._calc_ros(
            spread_direction=travel_angle_deg,
            hros=self.hros[target_grid_row, target_grid_col],
            bros=self.bros[target_grid_row, target_grid_col],
            raz=self.raz[target_grid_row, target_grid_col],
            lb_ratio=lb_ratio
        )

        if calc_accel:
            travel_time = self._calc_accel_time_at_dist(accel, theta_ros_eq, accum_slope_dist)
            segment_travel_time = travel_time - prior_travel_time
            if max_accel_time is not None:
                # accum_accel_time += travel_time
                # if cell_counter > 0:
                # Adjust theta_ros_eq to account for acceleration and assign to theta_ros
                theta_ros = self._calc_accel_ros_at_time(ros_eq=theta_ros_eq,
                                                         accel_param=accel,
                                                         time=travel_time)
                # else:
                #     # Adjust theta_ros_eq to account for acceleration and assign to theta_ros
                #     theta_ros = self._calc_accel_ros_at_time(ros_eq=theta_ros_eq,
                #                                              accel_param=accel,
                #                                              time=current_time)

                # Add the data to the intersected cells list
                intersected_cells.append(((target_grid_row, target_grid_col),
                                          travel_angle_deg, accum_slope_dist, slope_dist,
                                          theta_ros, segment_travel_time, max_accel_time))
            else:
                raise ValueError('Error: Unable to calculate max_accel_time.')
        else:
            if theta_ros_eq > 0:
                travel_time = slope_dist / theta_ros_eq

                # Assign theta_ros_eq to theta_ros
                theta_ros = theta_ros_eq
            else:
                travel_time = math.inf
                theta_ros = math.inf

            # Add the data to the intersected cells list
            intersected_cells.append(((target_grid_row, target_grid_col),
                                      travel_angle_deg, slope_dist,
                                      theta_ros, travel_time))

        return intersected_cells

    def _close_datasets(self):
        """
        Function to close opened rasterio DatasetReader objects.

        :return: None
        """
        datasets = [
            self.elevation, self.slope, self.aspect, self.lat, self.long
        ]
        for dataset in datasets:
            if dataset and isinstance(dataset, rio.DatasetReader) and not dataset.closed:
                name = None
                if any(name in dataset.name for name in ['_latitude.tif', '_longitude.tif']):
                    name = dataset.name
                dataset.close()
                del dataset
                if name is not None:
                    os.remove(name)

        # Close fire behaviour plugin datasets
        self.fb_engine.close_datasets()

        # Delete the temp directory and contents, if it exists
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)

        return

    def _fire_on_edge(self) -> bool:
        """
        Check if fire has reached an edge of the processing window,
        excluding edges that also align with the full dataset bounds.

        :return: True if fire is on a non-dataset edge, False otherwise
        """
        # Get window indices for the processing window
        window = self.process_window
        row_off, col_off = int(window.row_off), int(window.col_off)
        height, width = int(window.height), int(window.width)

        # Dataset edge limits
        max_row, max_col = self.max_grid_shape

        fire_on_top = np.any(self.travel_times[0, 0:-1] < np.inf) and row_off > 0
        fire_on_bottom = np.any(self.travel_times[-1, 0:-1] < np.inf) and (row_off + height) < max_row
        fire_on_left = np.any(self.travel_times[0:-1, 0] < np.inf) and col_off > 0
        fire_on_right = np.any(self.travel_times[0:-1, -1] < np.inf) and (col_off + width) < max_col

        return fire_on_top or fire_on_bottom or fire_on_left or fire_on_right

    @staticmethod
    def _gen_ellipse_offsets(spread_direction: float,
                             lb_ratio: float,
                             breadth: float,
                             hros: float,
                             bros: float,
                             add_source_cell: bool = False) -> list[tuple[int, int]]:
        """
        Generate (d_row, d_col) offsets intercepted by a rotated rectangle
        based on the lb_ratio and the provided breadth.

        This version ensures all *touched* cells are included, not just
        those whose centers fall within the rectangle.

        :param spread_direction: Direction of the rectangle in degrees (clockwise from north).
        :param lb_ratio: Length-to-breadth ratio of the rectangle.
        :param breadth: Total breadth (in cells) of the rectangle (shorter side).
        :param hros: Head fire rate of spread.
        :param bros: Backfire rate of spread.
        :param add_source_cell: Boolean flag indicating if the source cell (0, 0) should be included.
        :return: List of (d_row, d_col) relative offsets from the source cell.
        """
        # Solve for rectangle dimensions
        length = lb_ratio * breadth
        a = length / 2  # semi-major axis
        b = breadth / 2  # semi-minor axis
        # c = math.sqrt(a**2 - b**2)  # distance from center to focus

        if (hros + bros) == 0:
            return [(0, 0)]
        else:
            # Get the back:head ROS ratio to adjust rectangle offset (dy)
            bh_ros_ratio = bros / (hros + bros)
            # Calculate vertical (y-axis) offset
            dy = a - length * bh_ros_ratio

        # Generate the base (unrotated) rectangle centered at (0, 0)
        base_rect = box(-b, -a, b, a)
        # base_rect = box(-b, -a - dy, b, a + 2 * dy)

        # Create the axis-aligned (unrotated) rectangle and center it at (0, dy)
        base_rect = translate(base_rect, xoff=0, yoff=dy)

        # Carve out column 0 cells where rows < -1 or rows > 1
        # Column 0 in grid coords spans x ∈ [-0.5, 0.5]
        # Keep only y ∈ [-1.5, 1.5] in that column; subtract the rest
        minx, miny, maxx, maxy = base_rect.bounds
        col0_xmin, col0_xmax = -0.5001, 0.5001
        keep_ymin, keep_ymax = -1.5, 1.5

        # Carve out row 0 cells where cols < -1 or cols > 1
        # Row 0 in grid coords spans y ∈ [-0.5, 0.5]
        # Keep only x ∈ [-1.5, 1.5] in that row; subtract the rest
        row0_ymin, row0_ymax = -0.5001, 0.5001
        keep_xmin, keep_xmax = -1.5, 1.5

        # Parts to remove from column 0 and row 0
        center_cut = box(-0.5001, -0.5001, 0.5001, 0.5001)  # center cell
        cut_top = box(col0_xmin, keep_ymax, col0_xmax, maxy)  # rows < -1 (y > 1.5)
        cut_btm = box(col0_xmin, miny, col0_xmax, keep_ymin)  # rows > 1 (y < -1.5)
        cut_left = box(minx, row0_ymin, keep_xmin, row0_ymax)   # cols < -1 (x < -1.5)
        cut_right = box(keep_xmax, row0_ymin, maxx, row0_ymax)  # cols > 1 (x > 1.5)

        # If add_source_cell is True, include the (0,0) cell
        notch_mask = cut_top.union(cut_btm).union(cut_left).union(cut_right).union(center_cut)
        # Only subtract what overlaps the rectangle
        notched_rect = base_rect.difference(notch_mask)

        # Rotate notched rectangle
        rectangle = rotate(notched_rect, -spread_direction, origin=(0, 0), use_radians=False)

        # Determine bounding box in grid coordinates
        minx, miny, maxx, maxy = rectangle.bounds
        row_min, row_max = int(np.floor(-maxy)), int(np.ceil(-miny))
        col_min, col_max = int(np.floor(minx)), int(np.ceil(maxx))

        # Identify all grid cells touched by the rotated rectangle
        intersecting_cells = []
        for d_row in range(row_min, row_max + 1):
            for d_col in range(col_min, col_max + 1):
                # Each cell is a square from (x - 0.5, y - 0.5) to (x + 0.5, y + 0.5)
                cell_geom = box(d_col - 0.4999, -d_row - 0.4999, d_col + 0.4999, -d_row + 0.4999)
                if rectangle.intersects(cell_geom):
                    intersecting_cells.append((d_row, d_col))

        if add_source_cell:
            return [(0, 0)] + intersecting_cells
        else:
            return intersecting_cells

    def _geographic_to_grid(self,
                            x_coord: float,
                            y_coord: float,
                            transform_type: str = 'process') -> tuple[float, float]:
        """
        Converts geographic coordinates to grid cell coordinates.

        :param x_coord: X coordinate in geographic EPSG units.
        :param y_coord: Y coordinate in geographic EPSG units.
        :param transform_type: 'max' for full dataset transform, 'process' for current processing extent transform
        :return: Grid cell coordinates at the geographic (x, y) location.
        """
        if transform_type == 'process':
            return ~self.process_transform * (x_coord, y_coord)

        return ~self.max_transform * (x_coord, y_coord)

    @staticmethod
    def _get_grid_coords(src: rio.DatasetReader,
                         out_file_x: str,
                         out_file_y: str,
                         window: rio.windows.Window,
                         out_crs: str = 'EPSG:4326',
                         dtype: np.dtype = np.float32) -> tuple[str, str]:
        """
        Function returns two X and Y rasters with cell values matching the grid cell coordinates (one for Xs, one for Ys)

        :param src: a rasterio dataset reader object
        :param out_file_x: path to output raster for X coordinates
        :param out_file_y: path to output raster for Y coordinates
        :param window: rasterio window object defining the subset to process
        :param out_crs: string defining new projection (e.g., 'EPSG:4326')
        :param dtype: numpy data type for output rasters (default is np.float32)
        :return: a tuple (X, Y) of rasterio dataset reader objects in 'r' mode
        """
        # Get the coordinate reference system (CRS) of the input raster
        src_crs = src.crs

        # Read the subset using the provided window
        subset = src.read(1, window=window)
        subset_transform = src.window_transform(window)
        subset_rows, subset_cols = subset.shape

        # Get the geolocation of the top-left corner and pixel size for the subset
        x_start, y_start = subset_transform * (0, 0)
        x_end, y_end = subset_transform * (subset_cols, subset_rows)
        pixel_size_x = (x_end - x_start) / subset_cols
        pixel_size_y = (y_end - y_start) / subset_rows

        # Create a new affine transformation matrix for EPSG:4326
        transformer = Transformer.from_crs(src_crs, out_crs, always_xy=True)

        # Calculate the x & y coordinates for each cell in the subset
        x_coords = np.linspace(x_start + pixel_size_x / 2, x_end - pixel_size_x / 2, subset_cols)
        y_coords = np.linspace(y_start + pixel_size_y / 2, y_end - pixel_size_y / 2, subset_rows)
        lon, lat = np.meshgrid(x_coords, y_coords)
        lon, lat = transformer.transform(lon.flatten(), lat.flatten())

        # Reshape the lon and lat arrays to match the shape of the subset raster
        lon = lon.reshape(subset_rows, subset_cols).astype(dtype)
        lat = lat.reshape(subset_rows, subset_cols).astype(dtype)

        # Create output profiles for x and y coordinate rasters, setting dtype to the specified parameter
        profile = src.profile.copy()
        profile.update(dtype=dtype, height=subset_rows, width=subset_cols, transform=subset_transform)

        # Write X coordinate data to out_path_x
        with rio.open(out_file_x, 'w', **profile) as dst:
            dst.write(lon, 1)
            PYRO._calc_raster_stats(dst)

        # Write Y coordinate data to out_path_y
        with rio.open(out_file_y, 'w', **profile) as dst:
            dst.write(lat, 1)
            PYRO._calc_raster_stats(dst)

        return out_file_x, out_file_y

    def _grid_to_geographic(self,
                            col: Union[int, float],
                            row: Union[int, float],
                            transform_type: str = 'max',
                            get_cell_center: bool = False) -> tuple[float, float]:
        """
        Converts grid cell coordinates to geographic coordinates.

        :param col: Column index of the cell
        :param row: Row index of the cell
        :param transform_type: 'max' for full dataset transform, 'process' for current processing extent transform
        :param get_cell_center: If True, returns the geographic coordinate of the cell center.
            If False, returns the geographic coordinate of the actual cell position (top left corner).
        :return: Geographic (x, y) coordinates of the location in a cell
        """
        if transform_type == 'max':
            x, y = self.max_transform * (col, row)  # Top-left corner by default
        else:
            x, y = self.process_transform * (col, row)  # Top-left corner by default

        if get_cell_center:
            """
            Grid origin = top-left corner of the cell
            Geographic origin = bottom-left corner of the cell
            Y must be shifted down by half the cell height to mimic moving to a higher grid row
            X must be shifted right by half the cell width to mimic moving to a higher grid column
            """
            return x + self.cell_width / 2, y - self.cell_height / 2
        else:
            return x, y

    def _lb_ratio(self) -> np.ndarray:
        """
        Calculate the length-to-breadth ratio for each cell based on land cover and wind speed.

        :return: An array matching the grid shape with LB ratios.
        """
        # Create default array with 0 (safe fallback)
        lb_ratio = np.zeros_like(self.wsv, dtype=self.float_dtype)

        # Forest cells (land_cover == 1)
        forest_mask = self.land_cover == 1
        wsv_forest = self.wsv[forest_mask]
        lb_ratio[forest_mask] = 1 + 8.729 * (1 - np.exp(-0.03 * wsv_forest))**2.155

        # Grass cells (land_cover == 2)
        grass_mask = self.land_cover == 2
        wsv_grass = self.wsv[grass_mask]
        lb_ratio[grass_mask] = np.where(wsv_grass >= 1, 1.1 * wsv_grass**0.464, 1)

        return lb_ratio

    def _process_carryover_accel(self,
                                 accel_priority_queue: list,
                                 accel_carry_over_queue: list,
                                 dset_run_time: float) -> tuple[list, list, list]:
        """
        Process the acceleration carryover queue, ensuring that both the max acceleration run time
        and the dataset run time are not exceeded.

        :param accel_priority_queue: List of priority cells to process.
        :param accel_carry_over_queue: List of acceleration carry over cells to process.
        :param dset_run_time: The dataset's maximum run time.
        :return: The updated accel_priority, accel_carry_over, and accel_runtime_carry_over queues.
        """
        new_accel_carry_over_queue = []
        new_accel_runtime_carry_over_queue = []

        for carry_over_list in accel_carry_over_queue:
            target_grid_row, target_grid_col = carry_over_list[-1][0]
            current_time = carry_over_list[0][1]
            accel_time_remaining = self._calc_accel_time_99pct(
                self.accel[target_grid_row, target_grid_col]) - current_time
            dset_time_remaining = dset_run_time - current_time
            travel_time = 0
            theta_ros_accel = 0

            for i, [(current_grid_row, current_grid_col), _, spread_direction, slope_dist] in enumerate(
                    carry_over_list):
                # Get the adjusted lb_ratio for acceleration phase
                lb_ratio = self.lb_ratio[current_grid_row, current_grid_col]
                accel = self.accel[current_grid_row, current_grid_col]
                lb_ratio = (lb_ratio - 1) * (1 - math.exp(-accel * current_time)) + 1

                # Rate of spread
                theta_ros_eq = self._calc_ros(
                    spread_direction=spread_direction,
                    hros=self.hros[current_grid_row, current_grid_col],
                    bros=self.bros[current_grid_row, current_grid_col],
                    raz=self.raz[current_grid_row, current_grid_col],
                    lb_ratio=lb_ratio
                )

                # Adjust theta_ros_eq to account for acceleration and assign to theta_ros_accel
                theta_ros_accel = PYRO._calc_accel_ros_at_time(ros_eq=theta_ros_eq,
                                                               accel_param=accel,
                                                               time=current_time)

                if theta_ros_eq <= 0 or slope_dist <= 0:
                    continue

                # Travel time with acceleration
                travel_seg_time = self._calc_accel_time_at_dist(
                    accel_param=self.accel[current_grid_row, current_grid_col],
                    ros_eq=theta_ros_eq,
                    dist=slope_dist
                )

                # Check if the segment can be fully traversed
                if travel_seg_time <= accel_time_remaining and travel_seg_time <= dset_time_remaining:
                    travel_time += travel_seg_time
                    accel_time_remaining -= travel_seg_time
                    dset_time_remaining -= travel_seg_time
                else:
                    # Calculate partial progress
                    if dset_time_remaining < accel_time_remaining:
                        time_ratio = dset_time_remaining / travel_seg_time
                        new_time = current_time + dset_time_remaining
                        remaining_dist = slope_dist * (1 - time_ratio)
                        new_accel_runtime_carry_over_queue.append(
                            [[(current_grid_row, current_grid_col), new_time, spread_direction, remaining_dist]] +
                            carry_over_list[i + 1:]
                        )
                    else:
                        time_ratio = accel_time_remaining / travel_seg_time
                        new_time = current_time + accel_time_remaining
                        remaining_dist = slope_dist * (1 - time_ratio)
                        new_accel_carry_over_queue.append(
                            [[(current_grid_row, current_grid_col), new_time, spread_direction, remaining_dist]] +
                            carry_over_list[i + 1:]
                        )
                    return accel_priority_queue, new_accel_carry_over_queue, new_accel_runtime_carry_over_queue

            # Update travel times, output datasets, and priority queue
            new_time = current_time + travel_time
            if np.float32(new_time) < np.float32(self.travel_times[target_grid_row, target_grid_col]):
                self.travel_times[target_grid_row, target_grid_col] = new_time
                self.ros[target_grid_row, target_grid_col] = theta_ros_accel
                # self.fb_engine.process_timestep_vars(row=target_grid_row,col=target_grid_col)

                if new_time <= dset_run_time:
                    heappush(accel_priority_queue, (new_time, target_grid_row, target_grid_col))

        return accel_priority_queue, new_accel_carry_over_queue, new_accel_runtime_carry_over_queue

    def _process_carryover(self,
                           priority_queue: list,
                           carry_over_queue: list,
                           dset_run_time: float) -> Union[tuple[list, list], None]:
        """
        Process the carryover queue.

        :param priority_queue: List of priority cells to process.
        :param carry_over_queue: List of carry over cells to process.
        :param dset_run_time: The dataset's maximum run time.
        :return: The updated priority and carry over queues.
        """
        new_carry_over_queue = []
        for remaining_direction in carry_over_queue:
            target_grid_row, target_grid_col = remaining_direction[-1][0]
            current_time = remaining_direction[0][1]
            time_remaining = dset_run_time - current_time
            travel_time = 0
            travel_dist = 0
            theta_ros_eq = 0
            skip_target = False

            for i, [(current_grid_row, current_grid_col), _, spread_direction, slope_dist] \
                    in enumerate(remaining_direction):
                # Calculate the rate of spread (ROS) for the current position
                theta_ros_eq = self._calc_ros(
                    spread_direction=spread_direction,
                    hros=self.hros[current_grid_row, current_grid_col],
                    bros=self.bros[current_grid_row, current_grid_col],
                    raz=self.raz[current_grid_row, current_grid_col],
                    lb_ratio=self.lb_ratio[current_grid_row, current_grid_col]
                )

                # Calculate the segment travel time
                if theta_ros_eq == 0:
                    continue
                else:
                    segment_time = slope_dist / theta_ros_eq

                # Skip segments with invalid travel times or distances
                if segment_time <= 0 or slope_dist <= 0:
                    continue

                if (segment_time <= time_remaining) or (dset_run_time == self.total_run_time):
                    # If the segment can be fully traversed
                    travel_dist += slope_dist
                    time_remaining -= segment_time
                    travel_time += segment_time
                else:
                    # If only part of the segment can be traversed
                    partial_distance = slope_dist * (time_remaining / segment_time)
                    travel_dist += partial_distance
                    travel_time += time_remaining / segment_time

                    # Calculate the remaining distance for carry-over
                    remaining_distance = slope_dist - partial_distance

                    # Update the carry_over_queue directly
                    carry_over_list = [[(current_grid_row, current_grid_col),
                                        dset_run_time, spread_direction, remaining_distance]]

                    if i + 1 < len(remaining_direction):
                        carry_over_list.extend(
                            [[grid_coords, dset_run_time, angle, dist] for grid_coords, _, angle, dist in
                             remaining_direction[i + 1:]]
                        )

                    new_carry_over_queue.append(carry_over_list)
                    skip_target = True
                    break

            if not skip_target:
                # Update travel times and priority queue
                new_time = current_time + travel_time
                if np.float32(new_time) < np.float32(self.travel_times[target_grid_row, target_grid_col]):
                    self.travel_times[target_grid_row, target_grid_col] = new_time
                    self.ros[target_grid_row, target_grid_col] = theta_ros_eq
                    # self.fb_engine.process_timestep_vars(row=target_grid_row,col=target_grid_col)

                    if new_time <= dset_run_time:
                        # No remaining distances
                        heappush(priority_queue, (new_time, target_grid_row, target_grid_col))

        return priority_queue, new_carry_over_queue

    def _process_output_requests(self) -> None:
        """
        Function to calculate and export requested fire behaviour related outputs.

        :return: None
        """
        self._log('\nProcessing output requests...')
        # Handle the case when no travel times exist, or
        # when the only valid value within the fire run time is 0 (excluding np.inf)
        filtered = self.travel_times[(np.isfinite(self.travel_times)) & (self.travel_times <= self.total_run_time)]
        if np.all(self.travel_times == np.inf) or (filtered.size == 0) or (np.nanmax(filtered) == 0):
            self._log('\t\tNo travel times > 0 were calculated; skipping output processing.')
            self._save_text(out_path=os.path.join(self.out_folder, f'{self.fire_id}_NonSpreadingFire.txt'),
                            out_msg='Fire did not spread - no travel times > 0 were calculated.')
            return

        # Calculate the arrival contours and burn perimeter
        if ('perim' in self.out_request) or ('cont_poly' in self.out_request) or ('cont_line' in self.out_request):
            self._log('\tProcessing arrival time contours and/or burn perimeter...')
            pout.process_arrival_contours(
                out_request=self.out_request,
                interval_mins=self.interval_mins,
                total_run_time=self.total_run_time,
                crs=self.crs,
                process_transform=self.process_transform,
                process_grid_shape=self.process_grid_shape,
                cell_width=self.cell_width,
                cell_height=self.cell_height,
                ignition_type=self.ignition_type,
                ignition_geom=self.ignition_geom,
                land_cover=self.land_cover,
                travel_times=self.travel_times,
                fire_id=self.fire_id,
                out_folder=self.out_folder,
                remove_poly_ign=self.remove_poly_ign,
                suppress_messages=self.suppress_messages
            )

        # Remove values overlapping polygon ignition areas if requested
        if (self.ignition_type == 'polygon') & self.remove_poly_ign:
            mask = PYRO._rasterize_geometry(geom=self.ignition_geom,
                                            out_shape=self.travel_times.shape,
                                            transform=self.process_transform)
            self.travel_times[mask] = np.inf

        # Remove times that exceed run_time
        self._log('\tRemoving excessive travel times')
        self.travel_times[self.travel_times > self.total_run_time] = np.inf

        # Filter out non-raster and travel time requests from the output request
        fbp_request = [
            var for var in self.out_request
            if var not in ['perim', 'cont_poly', 'cont_line', 'flow_paths', 'major_flow_paths',  'tt']
        ]

        # Continue if there are remaining requested outputs
        if len(fbp_request) > 0:
            self.ros[np.isinf(self.travel_times)] = np.inf
            if 'ros' in fbp_request:
                self._log('\tProcessing the ros raster')
                out_path = os.path.join(self.out_folder, 'ros.tif')
                self._save_output(out_array=self.ros,
                                  out_path=out_path)

            # Filter out ros from the output request
            new_request = [var for var in fbp_request if var not in ['ros', 'fi']]

            # Ensure 'hfi' is requested if 'fi' is requested
            if 'fi' in fbp_request:
                new_request.append('hfi')

            # If there are any other requested outputs, run the fire behaviour model
            if len(new_request) > 0:
                # Run the fire behaviour model
                result = self._run_fire_behaviour(
                    out_request=new_request,
                    process_results=True
                )

                # Save requested outputs
                for var, array in list(zip(new_request, result)):
                    # Rename 'hfi' to 'fi' for output
                    if var == 'hfi':
                        var = 'fi'

                    self._log(f'\tProcessing the {var} raster')

                    # Skip the array if all values are NaN
                    if np.isnan(array).all():
                        self._log(f'\tSkipping {var} raster - all values are NaN.')
                        self._save_text(out_path=os.path.join(self.out_folder, f'{self.fire_id}_{var}.txt'),
                                        out_msg=f'No valid data to save in the {var} dataset.')
                        continue

                    # Determine the minimum valid no-data value based on dtype
                    dtype_info = (np.iinfo(array.dtype)
                                  if np.issubdtype(array.dtype, np.integer)
                                  else np.finfo(array.dtype))

                    # Use the minimum value of the dtype if self.no_data is out of bounds
                    self.no_data = (self.no_data
                                    if dtype_info.min <= self.no_data <= dtype_info.max
                                    else dtype_info.min)

                    # Adjust no-data values
                    array[np.isinf(self.travel_times)] = self.no_data

                    # Adjust zero values for fire behaviour outputs
                    array[self.travel_times == 0] = 0

                    # Save the full array with the updated transform
                    out_temp = os.path.join(self.out_folder, f'{var}.tif')
                    self._save_output(out_array=array,
                                      out_path=out_temp)

        if any(key in self.out_request for key in ('tt', 'flow_paths', 'major_flow_paths')):
            # Process the travel times output
            self._log('\tProcessing the travel time raster')
            out_path = os.path.join(self.out_folder, f'travel_times.tif')
            self._save_output(out_array=self.travel_times,
                              out_path=out_path)

            # Get the output travel times array path
            prefixed_name = f'{self.fire_id}_{os.path.basename(out_path)}'
            tt_out_path = os.path.join(self.out_folder, prefixed_name)

            # Process flow paths if requested
            fpath_list = [var for var in ['flow_paths', 'major_flow_paths'] if var in self.out_request]
            if fpath_list:
                self._log(f'\tProcessing the flow path dataset(s)')
                flow_paths_out = os.path.join(self.out_folder, f'{self.fire_id}_flow_paths.shp')
                major_flow_paths_out = os.path.join(self.out_folder, f'{self.fire_id}_major_flow_paths.shp')
                fpath_success, mfpath_success = pout.process_flow_paths(
                    flow_path_request=fpath_list,
                    travel_times_path=tt_out_path,
                    flow_paths_out=flow_paths_out,
                    major_flow_paths_out=major_flow_paths_out,
                    temp_folder=self.temp_folder,
                    suppress_messages=self.suppress_messages
                )
                if not fpath_success:
                    self._log('\t\tFlow path processing failed; skipping flow path outputs.')
                    for var in fpath_list:
                        self._save_text(
                            out_path=os.path.join(self.out_folder, f'{self.fire_id}_{var}.txt'),
                            out_msg='Flow path processing failed; no outputs were generated.'
                        )
                else:
                    if 'flow_paths' in fpath_list and fpath_success:
                        self._log(f'\t\tFlow paths saved to: {flow_paths_out}')
                    if 'major_flow_paths' in fpath_list:
                        if mfpath_success:
                            self._log(f'\t\tMajor flow paths saved to: {major_flow_paths_out}')
                        else:
                            self._log('\t\tMajor flow path processing failed; skipping flow path outputs.')
                            self._save_text(
                                out_path=os.path.join(self.out_folder, f'{self.fire_id}_major_flow_paths.txt'),
                                out_msg='Major flow path processing failed; no outputs were generated.'
                            )

            if 'tt' not in self.out_request:
                os.remove(tt_out_path)

        return

    def _process_travel_times_accel(self,
                                    process_cells: list,
                                    current_time: Optional[float],
                                    dset_run_time: float,
                                    accel_priority_queue: list,
                                    accel_carry_over_queue: list,
                                    accel_runtime_carry_over_queue: list) -> tuple[list, list, list]:
        """
        Update acceleration travel times and priority/carry-over queues.
        Sort segments based on proximity to the original grid cell center (start of the line).
        This function assumes acceleration time does not exceed the dataset runtime.

        :param process_cells: List of cells to process.
        :param current_time: The current travel time of the source cell.
        :param dset_run_time: The dataset's maximum run time.
        :param accel_priority_queue: List of acceleration priority cells to process.
        :param accel_carry_over_queue: List of acceleration carry over cells to process.
        :param accel_runtime_carry_over_queue: List of acceleration dataset run time carry over cells to process.
        :return: The revised priority and carry over queues.
        """
        target_grid_row, target_grid_col = process_cells[-1][0]
        time_consumed = current_time
        travel_time = 0
        theta_ros = 0

        for i, ((current_grid_row, current_grid_col),
                spread_angle, accum_slope_dist, slope_dist,
                theta_ros, segment_time, max_accel_time) in enumerate(process_cells):

            # Check if the time consumed exceeds the maximum acceleration time
            if time_consumed > max_accel_time:
                # This can only happen if a new fuel type is encountered that shifts the
                # acceleration time lower than what has already transpired. If this occurs,
                # the current and remaining cells should be pushed into the carry-over queue.

                # Push current and remaining segments into accel_carry_over_queue
                carry_over_list = [[(current_grid_row, current_grid_col),
                                    time_consumed, spread_angle, slope_dist]]
                # Add remaining segments, if any exist
                if len(process_cells[i + 1:]) > 0:
                    for grid_coords, angle, _, dist, _, _, _ in process_cells[i + 1:]:
                        carry_over_list.append([grid_coords, _, angle, dist])

                # Append the carry over list to the accel_carry_over_queue
                accel_carry_over_queue.append(carry_over_list)

                return accel_priority_queue, accel_carry_over_queue, accel_runtime_carry_over_queue

            # Get the time remaining for acceleration to reach equilibrium spread rate
            accel_time_remaining = max_accel_time - time_consumed

            # Get the time remaining for the dataset runtime
            dset_time_remaining = dset_run_time - time_consumed

            # Skip segments with invalid travel times or distances
            if segment_time <= 0 or slope_dist <= 0:
                continue

            if (segment_time <= accel_time_remaining) and (segment_time <= dset_time_remaining):
                # If the segment can be fully traversed without exceeding the acceleration time remaining
                # or the dataset runtime, or if this is the last dataset to process
                time_consumed += segment_time
                travel_time += segment_time
            else:
                cell_time_start = current_time + travel_time
                # If only part of the segment can be traversed...
                # Check which of the two time constraints is the limiting factor
                if dset_time_remaining < accel_time_remaining:
                    # If dset_time_remaining is smaller than accel_time_remaining, assign it as the limiting factor
                    time_consumed = dset_run_time
                else:
                    # Otherwise, max_accel_time is the limiting factor
                    time_consumed = max_accel_time
                # Calculate the distance that can be travelled within the remaining time
                accel_end_distance = PYRO._calc_accel_dist_at_time(
                    ros_eq=self.hros[current_grid_row, current_grid_col],
                    accel_param=self.accel[current_grid_row, current_grid_col],
                    time=time_consumed
                )
                accel_start_distance = PYRO._calc_accel_dist_at_time(
                    ros_eq=self.hros[current_grid_row, current_grid_col],
                    accel_param=self.accel[current_grid_row, current_grid_col],
                    time=cell_time_start
                )
                # Calculate the remaining distance for carry-over
                remaining_distance = slope_dist - (accel_end_distance - accel_start_distance)
                # Assign the remaining distance to the current segment
                carry_over_list = [[(current_grid_row, current_grid_col),
                                    time_consumed, spread_angle, remaining_distance]]
                # If there are remaining cell segments to process along the transect,
                # add them to the carry-over list as well
                if len(process_cells[i + 1:]) > 0:
                    for grid_coords, angle, _, dist, _, _, _ in process_cells[i + 1:]:
                        carry_over_list.append([grid_coords, time_consumed, angle, dist])

                if dset_time_remaining < accel_time_remaining:
                    # If run time was the limiting factor, add the carry over queue data
                    # to the accel_runtime_carry_over_queue. This queue will be processed using the
                    # dataset from the next time step.
                    accel_runtime_carry_over_queue.append(carry_over_list)

                else:
                    # Otherwise, add the carry over queue data to the accel_carry_over_queue
                    accel_carry_over_queue.append(carry_over_list)

                return accel_priority_queue, accel_carry_over_queue, accel_runtime_carry_over_queue

        # Update travel times, output datasets, and priority queue
        new_time = current_time + travel_time
        if np.float32(new_time) < np.float32(self.travel_times[target_grid_row, target_grid_col]):
            self.travel_times[target_grid_row, target_grid_col] = new_time
            self.ros[target_grid_row, target_grid_col] = theta_ros
            # self.fb_engine.process_timestep_vars(row=target_grid_row,col=target_grid_col)

            if new_time > 0:
                heappush(accel_priority_queue, (new_time, target_grid_row, target_grid_col))

        return accel_priority_queue, accel_carry_over_queue, accel_runtime_carry_over_queue

    def _process_travel_times(self,
                              process_cells: list,
                              current_time: Optional[float],
                              dset_run_time: float,
                              priority_queue: list,
                              carry_over_queue: list) -> tuple[list, list]:
        """
        Update travel times and priority/carry-over queues.
        Sort segments based on proximity to the original grid cell center (start of the line).

        :param process_cells: List of cells to process.
        :param current_time: The current travel time of the source cell.
        :param dset_run_time: The dataset's maximum run time.
        :param priority_queue: List of priority cells to process.
        :param carry_over_queue: List of carry over cells to process.
        :return: The revised priority and carry over queues.
        """
        target_grid_row, target_grid_col = process_cells[-1][0]
        time_remaining = dset_run_time - current_time
        travel_dist = 0
        travel_time = 0
        theta_ros_eq = 0
        is_final_dset = dset_run_time == self.total_run_time

        # Pre-filter invalid segments (zero or negative dist/time)
        process_cells = [
            cell for cell in process_cells
            if cell[2] > 0 and cell[4] > 0  # slope_dist > 0 and segment_time > 0
        ]

        for i, ((current_grid_row, current_grid_col),
                spread_angle, slope_dist, theta_ros_eq, segment_time) in enumerate(process_cells):

            if (segment_time <= time_remaining) or is_final_dset:
                # If the segment can be fully traversed
                travel_dist += slope_dist
                time_remaining -= segment_time
                travel_time += segment_time
            else:
                # Partial segment travel
                partial_distance = slope_dist * (time_remaining / segment_time)
                travel_dist += partial_distance
                travel_time += time_remaining / segment_time

                # Remaining distance for carry-over
                remaining_distance = slope_dist - partial_distance

                carry_over_list = [[(current_grid_row, current_grid_col),
                                    dset_run_time, spread_angle, remaining_distance]]

                if i + 1 < len(process_cells):
                    carry_over_list.extend([
                        [grid_coords, dset_run_time, angle, dist]
                        for grid_coords, angle, dist, _, _ in process_cells[i + 1:]
                    ])

                carry_over_queue.append(carry_over_list)
                return priority_queue, carry_over_queue

        # Update travel times, output datasets, and priority queue
        new_time = current_time + travel_time
        if np.float32(new_time) < np.float32(self.travel_times[target_grid_row, target_grid_col]):
            self.travel_times[target_grid_row, target_grid_col] = new_time
            self.ros[target_grid_row, target_grid_col] = theta_ros_eq
            # self.fb_engine.process_timestep_vars(row=target_grid_row,col=target_grid_col)

            if new_time <= dset_run_time:
                heappush(priority_queue, (new_time, target_grid_row, target_grid_col))

        return priority_queue, carry_over_queue

    @staticmethod
    def _rasterize_geometry(geom: Union[Polygon, MultiPolygon],
                            out_shape: tuple,
                            transform: rio.Affine) -> np.ndarray:
        """
        Rasterize a geometry to create a mask.

        :param geom: The geometry to rasterize (Polygon or MultiPolygon).
        :param out_shape: The shape of the output raster (rows, cols).
        :param transform: The affine transform for the output raster.
        :return: A boolean mask array where the geometry is rasterized.
        """
        shapes = [(geom, 1)]
        burned = features.rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=False,
            dtype=np.uint8
        )
        return burned.astype(bool)

    def _run_fire_behaviour(self,
                           out_request: list = None,
                           process_results: bool = False) -> Union[tuple, None]:
        """
        Function to run the fire behaviour model.

        :param out_request: List of requested output variables.
        :param process_results: Boolean flag to indicate if final results should be processed.
        :return: Tuple containing either:
            (1) the processing window and requested fire behaviour results if process_results == True; otherwise,
            (2) the requested fire behaviour results.
        """
        # Prepare new ROS array if processing final results
        if process_results:
            new_ros = self.ros.copy()
            new_ros[new_ros == self.no_data] = np.nan
        else:
            new_ros = None

        # Run fire behaviour model
        result, invalid_data = self.fb_engine.run_model(
            out_request=out_request,
            process_results=process_results,
            new_ros=new_ros,
            **{**{'dset_index': self.current_dset}, **self.fb_kwargs}
        )

        if invalid_data is not None:
            name_list = [name for name, res in invalid_data]
            self._log(f'The following input datasets do not contain valid data: {name_list}')
            self._save_text(
                out_path=os.path.join(self.out_folder, f'{self.fire_id}_InvalidInputData.txt'),
                out_msg=f'The following input datasets do not contain valid data: {name_list}'
            )
            self.spread_fire = False
            return None

        if process_results:
            return result
        else:
            # Get the results
            wsv, raz, hfi, hros, bros, accel = result

            # Handle the results
            self.wsv[0:self.process_window.height, 0:self.process_window.width] = wsv
            self.raz[0:self.process_window.height, 0:self.process_window.width] = raz
            self.hfi[0:self.process_window.height, 0:self.process_window.width] = hfi
            self.hros[0:self.process_window.height, 0:self.process_window.width] = hros
            self.bros[0:self.process_window.height, 0:self.process_window.width] = bros
            self.accel[0:self.process_window.height, 0:self.process_window.width] = accel

            # Delete temporary data
            del wsv, raz, hfi, hros, bros, accel, result

            # Precompute LB ratio based on updated WSV and land cover
            self.lb_ratio = self._lb_ratio()

            return None

    def _set_params(self, set_dict: dict) -> None:
        """
        Function to set parameters to specific values.

        :param set_dict: Dictionary of FBP parameter names and the values to assign to the FBP class object
        :return: None
        """
        # Iterate through the set dictionary and assign values to class attributes
        for key, update_array in set_dict.items():
            if hasattr(self, key):  # Check if the PYRO class has the attribute
                setattr(self, key, update_array)  # Update the class attribute with the new value
            else:
                raise AttributeError(f'PYRO class does not have an attribute named "{key}"')

        return

    def _save_output(self,
                     out_array: np.ndarray,
                     out_path: str) -> None:
        """
        Save the output arrays as a raster using the transform and CRS from the fuel type raster.

        :param out_array: The array of data to output as a raster.
        :param out_path: The path to save the output dataset.
        :return: None
        """
        with rio.open(self.elevation_path) as ref:
            # Prefix filename with fire_id
            prefixed_name = f'{self.fire_id}_{os.path.basename(out_path)}'
            final_out_path = os.path.join(self.out_folder, prefixed_name)

            # Copy the reference profile
            profile = ref.profile.copy()

            # Set a nodata value if not already defined
            profile.update(nodata=self.no_data)

            # Set non-finite values to nodata
            out_array[~np.isfinite(out_array)] = self.no_data

            # Update dtype in profile to match array
            if out_array.dtype == 'int64':
                profile.update(dtype=np.int16)
                out_array = out_array.astype(np.int16)
            elif 'float' in str(out_array.dtype):
                profile.update(dtype=self.float_dtype)
                out_array = out_array.astype(self.float_dtype)
            else:
                profile.update(dtype=out_array.dtype)

            # Find the bounding box of non-nodata values
            rows = np.any(out_array != self.no_data, axis=1)
            cols = np.any(out_array != self.no_data, axis=0)

            if not np.any(rows) or not np.any(cols):
                dset_name = os.path.basename(out_path).rstrip('.tif')
                self._log(f'\t\tNo valid data to save in the {dset_name} dataset; skipping output.')
                self._save_text(out_path=final_out_path.replace('.tif', '.txt'),
                                out_msg=f'No valid data to save in the {dset_name} dataset.')
                return

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop the array to this bounding box
            array_cropped = out_array[rmin:rmax + 1, cmin:cmax + 1]

            # Update the transform for the cropped array
            transform = self.process_transform
            new_transform = rio.transform.Affine(
                transform.a, transform.b,
                transform.c + cmin * transform.a,
                transform.d, transform.e,
                transform.f + rmin * transform.e
            )

            # Update the profile with new dimensions, transform, and dtype
            profile.update({
                'height': array_cropped.shape[0],
                'width': array_cropped.shape[1],
                'transform': new_transform
            })

            # Save the cropped array as a GeoTIFF
            self._log(f'\t\tSaving dataset to: {final_out_path}')
            with rio.open(final_out_path, 'w', **profile) as dst:
                # Write data to the new raster
                dst.write(array_cropped, 1)

                # Calculate statistics for all bands
                self._calc_raster_stats(dst)

        return

    @staticmethod
    def _save_text(out_path: str,
                   out_msg: str) -> None:
        """
        Save a text message to a specified file.

        :param out_path: The path to save the text file.
        :param out_msg: The message to write to the text file.
        :return: None
        """
        # Create text file and write the message
        with open(out_path, 'w') as msg_file:
            msg_file.write(out_msg)
        return

    def run_pyro(self,
                 num_time_steps: int = 1,
                 time_step_mins: int = 60,
                 contour_interval_mins: int = 60) -> None:
        """
        Run the Pyro Cell fire spread model over multiple time steps.

        :param num_time_steps: Number of time steps to run the model (default: 1).
        :param time_step_mins:
            Optional parameter specifying the runtime duration for each dataset in minutes (default: 60).
            Total run time is the time_step multiplied by the number of rasters in the hros_rasters list.
            Each raster will be run for the provided time_step.
        :param contour_interval_mins: Time interval in minutes for arrival time contours (default: 60).
        :return: Array representing the travel times for each cell
        """

        # Set class variables
        self.num_time_steps = num_time_steps
        self.time_step = time_step_mins
        self.interval_mins = contour_interval_mins
        self.total_run_time = num_time_steps * time_step_mins  # Total runtime in mins

        # Initialize queues
        priority_queue = []
        carry_over_queue = []
        accel_priority_queue = []
        accel_carry_over_queue = []
        accel_runtime_carry_over_queue = []

        # Generate a buffer off the processed cells
        self._log('Initializing processing extent...')
        _ = self._set_processing_extent(
            queue_list=[]
        )

        for dset in range(self.num_time_steps):
            while True:
                self.current_dset = dset
                dset_run_time = self.time_step * (1 + dset)
                self._log(f'Processing time step {dset + 1}: {dset_run_time} min')

                # Backup queues and travel times in case of rerun
                priority_queue_backup = copy.deepcopy(priority_queue)
                carry_over_queue_backup = copy.deepcopy(carry_over_queue)
                accel_priority_queue_backup = copy.deepcopy(accel_priority_queue)
                accel_carry_over_queue_backup = copy.deepcopy(accel_carry_over_queue)
                accel_runtime_carry_over_queue_backup = copy.deepcopy(accel_runtime_carry_over_queue)

                # Run the fire behaviour model
                self._run_fire_behaviour()
                if not self.spread_fire:
                    break

                # Initialize the fire ignition
                if self.init_ignition:

                    # Initialize the ignition geometry
                    self._init_ign_geoms()
                    if not self.spread_fire:
                        break

                    # Burn the ignition and get the priority and carry over queues
                    (priority_queue, carry_over_queue,
                     accel_runtime_carry_over_queue) = self._burn_ignition(
                        dset_run_time=dset_run_time,
                        accel_runtime_carry_over_queue=accel_runtime_carry_over_queue
                    )

                    # Only disable ignition if fire didn't hit the edge
                    if not self._fire_on_edge():
                        self.init_ignition = False
                else:

                    if self.use_accel:
                        # Process the acceleration runtime carry over queue (acceleration logic)
                        if len(accel_runtime_carry_over_queue) > 0:
                            (accel_priority_queue, accel_carry_over_queue,
                             accel_runtime_carry_over_queue) = self._process_carryover_accel(
                                accel_priority_queue=[],
                                accel_carry_over_queue=accel_runtime_carry_over_queue,
                                dset_run_time=dset_run_time
                            )

                        # Process the acceleration carry over queue (switch to standard carryover logic)
                        if len(accel_carry_over_queue) > 0:
                            priority_queue, carry_over_queue = self._process_carryover(
                                priority_queue=priority_queue,
                                carry_over_queue=accel_carry_over_queue,
                                dset_run_time=dset_run_time
                            )
                            accel_carry_over_queue = []

                    # Process the standard carry over queue
                    if len(carry_over_queue) > 0:
                        priority_queue, carry_over_queue = self._process_carryover(
                            priority_queue=priority_queue,
                            carry_over_queue=carry_over_queue,
                            dset_run_time=dset_run_time
                        )

                # Process the acceleration priority queue
                while accel_priority_queue:
                    current_time, grid_row, grid_col = heappop(accel_priority_queue)
                    if (current_time > self.travel_times[grid_row, grid_col]) or (current_time > dset_run_time):
                        continue

                    target_offsets = self._gen_ellipse_offsets(spread_direction=self.raz[grid_row, grid_col],
                                                               lb_ratio=self.lb_ratio[grid_row, grid_col],
                                                               breadth=self.ellipse_breadth,
                                                               hros=self.hros[grid_row, grid_col],
                                                               bros=self.bros[grid_row, grid_col])

                    for d_row, d_col in target_offsets:
                        target_row, target_col = grid_row + d_row, grid_col + d_col
                        # Ensure target cell is within the processing grid/extent
                        if 0 <= target_row < self.process_grid_shape[0] and 0 <= target_col < self.process_grid_shape[1]:

                            # Process cells if the source and target fuel types are either forest or grass
                            if (self.land_cover[grid_row, grid_col] in [1, 2] and
                                    self.land_cover[target_row, target_col] in [1, 2]):

                                # Calculate travel times for each cell between the current and target cells
                                intersected_cells = self._calc_travel_times(
                                    grid_row=grid_row,
                                    grid_col=grid_col,
                                    target_d_row=d_row,
                                    target_d_col=d_col,
                                    calc_accel=True,
                                    current_time=current_time
                                )

                                if len(intersected_cells) > 0:
                                    (accel_priority_queue, accel_carry_over_queue,
                                     accel_runtime_carry_over_queue) = self._process_travel_times_accel(
                                        process_cells=intersected_cells,
                                        current_time=current_time,
                                        dset_run_time=dset_run_time,
                                        accel_priority_queue=accel_priority_queue,
                                        accel_carry_over_queue=accel_carry_over_queue,
                                        accel_runtime_carry_over_queue=accel_runtime_carry_over_queue
                                    )

                # Process the standard priority queue
                while priority_queue:
                    current_time, grid_row, grid_col = heappop(priority_queue)
                    if (current_time > self.travel_times[grid_row, grid_col]) or (current_time > dset_run_time):
                        continue

                    target_offsets = self._gen_ellipse_offsets(spread_direction=self.raz[grid_row, grid_col],
                                                               lb_ratio=self.lb_ratio[grid_row, grid_col],
                                                               breadth=self.ellipse_breadth,
                                                               hros=self.hros[grid_row, grid_col],
                                                               bros=self.bros[grid_row, grid_col])

                    for d_row, d_col in target_offsets:
                        target_row, target_col = grid_row + d_row, grid_col + d_col
                        # Ensure target cell is within the processing grid/extent
                        if 0 <= target_row < self.process_grid_shape[0] and 0 <= target_col < self.process_grid_shape[1]:

                            # Process cells if the source and target fuel types are either forest or grass
                            if (self.land_cover[grid_row, grid_col] in [1, 2] and
                                    self.land_cover[target_row, target_col] in [1, 2]):

                                # Calculate travel times for each cell between the current and target cells
                                intersected_cells = self._calc_travel_times(
                                    grid_row=grid_row,
                                    grid_col=grid_col,
                                    target_d_row=d_row,
                                    target_d_col=d_col
                                )

                                if len(intersected_cells) > 0:
                                    priority_queue, carry_over_queue = self._process_travel_times(
                                        process_cells=intersected_cells,
                                        current_time=current_time,
                                        dset_run_time=dset_run_time,
                                        priority_queue=priority_queue,
                                        carry_over_queue=carry_over_queue
                                    )

                # Check for edge condition
                if self._fire_on_edge():
                    self.process_buffer += 1000
                    self._log(f'Fire reached processing edge at time step {dset + 1}')

                    if self.current_dset == 0:
                        # Reset ignition initialization
                        self.init_ignition = True

                    # Restore backups
                    priority_queue = copy.deepcopy(priority_queue_backup)
                    carry_over_queue = copy.deepcopy(carry_over_queue_backup)
                    accel_priority_queue = copy.deepcopy(accel_priority_queue_backup)
                    accel_carry_over_queue = copy.deepcopy(accel_carry_over_queue_backup)
                    accel_runtime_carry_over_queue = copy.deepcopy(accel_runtime_carry_over_queue_backup)

                    # Generate a buffer off the cells in travel_time, plus cells in priority & carry over queues
                    self._log(f'\tExpanding buffer by {self.process_buffer}m and retrying...')
                    (priority_queue, carry_over_queue,
                     accel_carry_over_queue, accel_runtime_carry_over_queue) = self._set_processing_extent(
                        queue_list=[priority_queue, carry_over_queue,
                                    accel_carry_over_queue, accel_runtime_carry_over_queue]
                    )

                    # Retry current timestep with new buffer
                    continue
                else:
                    self._log(f'\tTimestep complete')
                    # Process intermediate output datasets
                    self.fb_engine.update_tstep_data(
                        travel_times_array=self.travel_times,
                        time_step=self.time_step,
                        dset_run_time=dset_run_time,
                    )

                    # Break out of while loop if last dataset
                    if dset == self.num_time_steps - 1:
                        break

                    # Reset process_buffer after successful time step
                    self.process_buffer = self.buffer_default

                    # Generate a buffer off the cells in travel_time, plus cells in the carryover queues
                    self._log(f'Adjusting processing extent with a buffer of {self.process_buffer}m...')
                    (carry_over_queue, accel_carry_over_queue,
                     accel_runtime_carry_over_queue) = self._set_processing_extent(
                        queue_list=[carry_over_queue, accel_carry_over_queue, accel_runtime_carry_over_queue]
                    )

                    # Exit while loop and continue to next dset
                    break

            if not self.spread_fire:
                break

        if self.spread_fire:
            self._log('Fire spread modelling complete')
            # Process output data
            self._process_output_requests()

        # Close opened datasets
        self._close_datasets()

        return