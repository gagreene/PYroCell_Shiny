import os
from tqdm import tqdm
from shapely.geometry import Point, LineString, Polygon, MultiLineString, MultiPolygon
import geopandas as gpd
import concurrent.futures
import sys
sys.path.append(os.path.dirname(__file__))
from core import PYRO


def run_single_ignition(ignition_feature, shared_args: dict, fb_kwargs: dict, fire_id: str):
    """
    Internal function to run a single PYRO simulation for an ignition feature.

    :param ignition_feature: A GeoSeries row representing the ignition point/polygon.
    :param shared_args: Dictionary of arguments shared across all runs.
    :param fb_kwargs: Dictionary of arguments specific to the fire behavior model.
    :param fire_id: Unique identifier string for the fire (used in output naming).
    """
    # Get output folder and ensure it exists
    out_folder = shared_args['out_folder']
    os.makedirs(out_folder, exist_ok=True)

    # Assign fire_id
    fire_id = f'FID{fire_id}'

    # Save ignition feature to file with prefixed name
    ignition_path = os.path.join(out_folder, f'{fire_id}_ignition.shp')
    geometry = ignition_feature.geometry
    gdf = gpd.GeoDataFrame({'geometry': [geometry]}, crs=shared_args['crs'])
    gdf.to_file(ignition_path)

    # Verify geometry type
    if not isinstance(geometry, (Point, LineString, Polygon, MultiPolygon, MultiLineString)):
        print(f'Warning: Unexpected geometry type in fire_id {fire_id}: {geometry.geom_type}')

    # Remove out_folder from model_args (already passed)
    model_args = shared_args['model_args'].copy()
    model_args.pop('out_folder', None)

    # Instantiate and run the model
    pyro = PYRO(
        ignition_path=ignition_path,
        out_folder=out_folder,
        fire_id=fire_id,
        suppress_messages=shared_args.get('suppress_messages', True),
        **model_args,
        **fb_kwargs
    )

    # Run with custom time/interval if provided
    num_time_steps = shared_args.get('num_time_steps', 1)
    time_step_mins = shared_args.get('time_step_mins', 60)
    contour_interval_mins = shared_args.get('contour_interval_mins', 60)
    pyro.run_pyro(num_time_steps=num_time_steps,
                  time_step_mins=time_step_mins,
                  contour_interval_mins=contour_interval_mins)


def run_multiple_ignitions(ignition_path: str,
                           shared_model_args: dict,
                           fb_kwargs: dict,
                           max_workers: int = None,
                           num_time_steps: int = 1,
                           time_step_mins: int = 60,
                           contour_interval_mins: int = 60,
                           fire_id_field: str = None,
                           suppress_messages: bool = False):
    """
    Runs the PYRO model for multiple ignition points or polygons in parallel.

    This function reads ignition features from a file and processes each feature
    using the PYRO model. It supports parallel execution to improve performance
    when handling multiple ignition points or polygons.

    :param ignition_path: Path to the file containing ignition features (e.g., shapefile).
    :param shared_model_args: Dictionary of arguments shared across all model runs.
    :param fb_kwargs: Dictionary of arguments specific to the fire behavior model.
    :param max_workers: Maximum number of parallel workers to use. Defaults to None,
        which uses the number of available CPU cores.
    :param num_time_steps: Number of time steps to simulate. Defaults to 1.
    :param time_step_mins: Duration of each time step in minutes. Defaults to 60.
    :param contour_interval_mins: Interval in minutes for generating fire contours. Defaults to 60.
    :param fire_id_field: Name of the field in the ignition file to use as a unique fire ID.
        If None, a sequential ID is generated. Defaults to None.
    :param suppress_messages: Whether to suppress output messages during execution. Defaults to False.

    :returns: None
    """
    ign_gdf = gpd.read_file(ignition_path)
    crs = ign_gdf.crs
    out_folder = shared_model_args['out_folder']
    os.makedirs(out_folder, exist_ok=True)

    shared_args = {
        'crs': crs,
        'out_folder': out_folder,
        'model_args': shared_model_args.copy(),
        'num_time_steps': num_time_steps,
        'time_step_mins': time_step_mins,
        'contour_interval_mins': contour_interval_mins,
        'suppress_messages': suppress_messages
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Determine zero-padding width
        if fire_id_field and fire_id_field in ign_gdf.columns:
            fire_ids_raw = ign_gdf[fire_id_field].astype(str).str.strip()
            pad_width = len(str(fire_ids_raw.map(lambda x: ''.join(filter(str.isdigit, x))).astype(int).max()))
        else:
            pad_width = len(str(len(ign_gdf)))

        for i, row in ign_gdf.iterrows():
            if fire_id_field and fire_id_field in row:
                raw_val = str(row[fire_id_field]).strip()
                digits_only = ''.join(filter(str.isdigit, raw_val))
                fire_id = digits_only.zfill(pad_width) if digits_only else raw_val
            else:
                fire_id = str(i).zfill(pad_width)

            future = executor.submit(run_single_ignition, row, shared_args, fb_kwargs, fire_id)
            future.fire_id = fire_id  # attach custom attribute
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing Ignitions'):
            try:
                future.result()
            except Exception as e:
                fire_id = getattr(future, 'fire_id', 'unknown')
                print(f'Error in fire_id {fire_id}: {e}')