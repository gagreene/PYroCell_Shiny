import os
import glob
import time
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
from dependencies.cffdrs import cffbps as fbp
from src.pyrocell import core as pc
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

import geopandas as gpd
import folium
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

app_ui = ui.page_fluid(
    ui.h1('PYroCell Fire Growth Model'),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric('ellipse_breadth', 'Ellipse Breadth', 3),
            ui.input_numeric('process_buffer', 'Process Buffer', 1000),
            ui.input_checkbox('use_accel', 'Use Acceleration', False),
            ui.input_checkbox('use_breaching', 'Use Breaching', True),
            ui.input_select('breach_type', 'Breach Type', ['default', 'prometheus', 'simple']),
            ui.input_checkbox('remove_poly_ign', 'Remove Poly Ignition', True),
            ui.input_numeric('percentile_growth', 'Percentile Growth', 50),
            ui.input_numeric('num_time_steps', 'Number of Time Steps', 12),
            ui.input_numeric('time_step', 'Time Step Duration (mins)', 60),
            ui.input_numeric('interval_mins', 'Contour Interval (mins)', 60),
            ui.h4('Select Model Outputs'),
            ui.input_checkbox_group(
                id='out_request',
                label='',
                choices=['tt', 'flow_paths', 'major_flow_paths', 'ros', 'fire_type', 'fi', 'cfb'],
                selected=['tt', 'major_flow_paths']
            ),
            ui.input_action_button('run_model', 'Run Model'),
            # width='300px'  # Optional: set sidebar width
            ui.output_text_verbatim('result'),
        ),
        ui.panel_well(
            ui.h1('PYroCell Fire Growth Model'),
            ui.output_ui('results_map'),
        )
    )
)


def create_results_map(results_dir):
    shp_files = glob.glob(os.path.join(results_dir, '*.shp'))
    tif_files = glob.glob(os.path.join(results_dir, '*.tif'))

    all_bounds = []

    # Collect bounds from shapefiles
    for shp_file in shp_files:
        gdf = gpd.read_file(shp_file)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        if not gdf.empty:
            bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
            all_bounds.append(bounds)

    # Collect bounds from rasters
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            dst_crs = 'EPSG:4326'
            if src.crs != dst_crs:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                bounds = rasterio.transform.array_bounds(height, width, transform)
                minx, miny, maxx, maxy = bounds
            else:
                minx, miny, maxx, maxy = src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
            all_bounds.append([minx, miny, maxx, maxy])

    # Compute overall bounds
    if all_bounds:
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)
        center = [(miny + maxy) / 2, (minx + maxx) / 2]
        bounds = [[miny, minx], [maxy, maxx]]
        zoom_start = 6  # Initial zoom, will fit bounds below
    else:
        center = [54, -125]
        bounds = None
        zoom_start = 6

    m = folium.Map(location=center, zoom_start=zoom_start, tiles='Esri.WorldImagery')

    # Add shapefiles
    for shp_file in shp_files:
        gdf = gpd.read_file(shp_file)
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        if not gdf.empty:
            layer_name = os.path.splitext(os.path.basename(shp_file))[0]
            shp_group = folium.FeatureGroup(name=layer_name)
            folium.GeoJson(gdf).add_to(shp_group)
            shp_group.add_to(m)

    # Add rasters
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            dst_crs = 'EPSG:4326'
            if src.crs != dst_crs:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds)
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                dst = np.empty((src.count, height, width), dtype=src.dtypes[0])
                for i in range(src.count):
                    reproject(
                        source=rasterio.band(src, i + 1),
                        destination=dst[i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
                img = dst[0]
                bounds_img = rasterio.transform.array_bounds(height, width, transform)
                min_lat, min_lon, max_lat, max_lon = bounds_img[1], bounds_img[0], bounds_img[3], bounds_img[2]
            else:
                img = src.read(1)
                bounds_img = src.bounds
                min_lon, min_lat, max_lon, max_lat = bounds_img.left, bounds_img.bottom, bounds_img.right, bounds_img.top

            layer_name = os.path.splitext(os.path.basename(tif_file))[0]
            raster_group = folium.FeatureGroup(name=layer_name)
            folium.raster_layers.ImageOverlay(
                image=img,
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                opacity=0.5
            ).add_to(raster_group)
            raster_group.add_to(m)

    folium.LayerControl().add_to(m)

    # Fit map to bounds if available
    if bounds:
        m.fit_bounds(bounds)

    return m._repr_html_()



def server(input: Inputs, output: Outputs, session: Session):
    # Set up data directories
    data_dir = os.path.join(os.path.dirname(__file__), 'development', 'testing', 'test_data')
    wx_dir = os.path.join(data_dir, 'weather')
    results_dir = os.path.join(data_dir, '_results')
    os.makedirs(results_dir, exist_ok=True)

    @reactive.calc
    @reactive.event(input.run_model)
    def run_pyro_model():
        float_dtype = np.float32
        ellipse_breadth = input.ellipse_breadth()
        process_buffer = input.process_buffer()
        use_accel = input.use_accel()
        use_breaching = input.use_breaching()
        breach_type = input.breach_type()
        remove_poly_ign = input.remove_poly_ign()
        percentile_growth = input.percentile_growth()
        num_time_steps = input.num_time_steps()
        time_step = input.time_step()
        interval_mins = input.interval_mins()
        out_request = input.out_request()

        wx_date = 20230715
        bui = 82.382
        gcf = fbp.getSeasonGrassCuring(season='summer', province='BC')

        fire_id = 'TestFire'
        ign_path = os.path.join(data_dir, 'Ignitions.shp')
        ft_path = os.path.join(data_dir, 'FuelType.tif')
        lat_path = os.path.join(data_dir, 'LAT.tif')
        long_path = os.path.join(data_dir, 'LONG.tif')
        elv_path = os.path.join(data_dir, 'ELV.tif')
        slope_path = os.path.join(data_dir, 'GS.tif')
        aspect_path = os.path.join(data_dir, 'Aspect.tif')
        pc_path = os.path.join(data_dir, 'PC.tif')
        pdf_path = os.path.join(data_dir, 'PDF.tif')
        gfl_path = os.path.join(data_dir, 'GFL.tif')

        temp_list = sorted(glob.glob(os.path.join(wx_dir, 'temp*.tiff')))[0:num_time_steps]
        ws_list = sorted(glob.glob(os.path.join(wx_dir, 'ws*.tiff')))[0:num_time_steps]
        wd_list = sorted(glob.glob(os.path.join(wx_dir, 'wd*.tiff')))[0:num_time_steps]
        ffmc_list = sorted(glob.glob(os.path.join(wx_dir, 'hffmc*.tiff')))[0:num_time_steps]
        bui_list = [bui] * num_time_steps

        # Delete data in results directory if it exists
        if os.path.exists(results_dir):
            files = glob.glob(os.path.join(results_dir, '*'))
            for f in files:
                os.remove(f)

        fb_kwargs = {
            'fueltype_path': ft_path,
            'pct_cnfr': pc_path,
            'pct_deadfir': pdf_path,
            'grass_fuelload': gfl_path,
            'grass_curingfactor': gcf,
            'wind_speeds_list': ws_list,
            'wind_dirs_list': wd_list,
            'ffmc_list': ffmc_list,
            'bui_list': bui_list,
            'ws_domain_avg_value_current': None,
            'ws_domain_avg_value_new': None,
            'percentile_growth': percentile_growth,
            'use_gpu': False,
            'gpu_float_dtype': np.float32
        }

        start_time = time.time()
        pyro_model = pc.PYRO(
            fb_plugin='cffbps',
            ignition_path=ign_path,
            fire_date=wx_date,
            elevation_path=elv_path,
            slope_path=slope_path,
            aspect_path=aspect_path,
            out_folder=results_dir,
            out_request=out_request,
            fire_id=fire_id,
            ellipse_breadth=ellipse_breadth,
            process_buffer=process_buffer,
            use_accel=use_accel,
            float_dtype=float_dtype,
            suppress_messages=False,
            use_breaching=use_breaching,
            breach_type=breach_type,
            remove_poly_ign=remove_poly_ign,
            **fb_kwargs
        )
        pyro_model.run_pyro(
            num_time_steps=num_time_steps,
            time_step_mins=time_step,
            contour_interval_mins=interval_mins
        )
        elapsed_time = time.time() - start_time
        # output.result.set_value(f'PYroCell 🔥 modelling completed in {elapsed_time:.2f} seconds.')
        return f'Completed: {elapsed_time:.2f} s'

    @reactive.calc
    @reactive.event(input.run_model)
    def pyro_result():
        return run_pyro_model()

    @output
    @render.text
    def result():
        return pyro_result()

    @output
    @render.ui
    def results_map():
        elapsed = pyro_result()
        map_html = create_results_map(results_dir)
        return ui.TagList(
            ui.HTML(map_html),
            # ui.div(f"Processing time: {elapsed}", style="margin-top: 10px; font-weight: bold;")
        )

app = App(app_ui, server)