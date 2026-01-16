import os
import glob
import time
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))
from dependencies.cffdrs import cffbps as fbp
from src.pyrocell import core as pc
from shiny import App, ui, render, reactive, Inputs, Outputs, Session

app_ui = ui.page_fluid(
    ui.h1('PYroCell Fire Spread Model'),
    ui.h3('Configure Model Parameters'),
    ui.input_numeric("ellipse_breadth", "Ellipse Breadth", 3),
    ui.input_numeric("process_buffer", "Process Buffer", 1000),
    ui.input_checkbox("use_accel", "Use Acceleration", False),
    ui.input_checkbox("use_breaching", "Use Breaching", True),
    ui.input_select("breach_type", "Breach Type", ["default", "prometheus", "simple"]),
    ui.input_checkbox("remove_poly_ign", "Remove Poly Ignition", True),
    ui.input_numeric("percentile_growth", "Percentile Growth", 50),
    ui.input_numeric("num_time_steps", "Number of Time Steps", 12),
    ui.input_numeric("time_step", "Time Step (mins)", 60),
    ui.input_numeric("interval_mins", "Contour Interval (mins)", 60),
    ui.h3('Choose Model Outputs'),
    ui.input_action_button("run_model", "Run Model"),
    ui.output_text_verbatim("result")
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.event(input.run_model)
    def _():
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
        out_request = ['perim', 'cont_line', 'cont_poly', 'tt', 'flow_paths',
                       'major_flow_paths', 'ros', 'fire_type', 'fi', 'cfb']

        wx_date = 20230715
        bui = 82.382
        gcf = fbp.getSeasonGrassCuring(season='summer', province='BC')

        data_folder = os.path.join(os.path.dirname(__file__), 'test_data')
        fire_id = 'TestFire'
        ign_path = os.path.join(data_folder, 'Ignitions.shp')
        ft_path = os.path.join(data_folder, 'FuelType.tif')
        lat_path = os.path.join(data_folder, 'LAT.tif')
        long_path = os.path.join(data_folder, 'LONG.tif')
        elv_path = os.path.join(data_folder, 'ELV.tif')
        slope_path = os.path.join(data_folder, 'GS.tif')
        aspect_path = os.path.join(data_folder, 'Aspect.tif')
        pc_path = os.path.join(data_folder, 'PC.tif')
        pdf_path = os.path.join(data_folder, 'PDF.tif')
        gfl_path = os.path.join(data_folder, 'GFL.tif')

        wx_folder = os.path.join(data_folder, 'weather')
        temp_list = sorted(glob.glob(os.path.join(wx_folder, 'temp*.tiff')))[0:num_time_steps]
        ws_list = sorted(glob.glob(os.path.join(wx_folder, 'ws*.tiff')))[0:num_time_steps]
        wd_list = sorted(glob.glob(os.path.join(wx_folder, 'wd*.tiff')))[0:num_time_steps]
        ffmc_list = sorted(glob.glob(os.path.join(wx_folder, 'hffmc*.tiff')))[0:num_time_steps]
        bui_list = [bui] * num_time_steps

        results_folder = os.path.join(data_folder, '_Results')
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

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
        # pyro_model = pc.PYRO(
        #     fb_plugin='cffbps',
        #     ignition_path=ign_path,
        #     fire_date=wx_date,
        #     elevation_path=elv_path,
        #     slope_path=slope_path,
        #     aspect_path=aspect_path,
        #     out_folder=results_folder,
        #     out_request=out_request,
        #     fire_id=fire_id,
        #     ellipse_breadth=ellipse_breadth,
        #     process_buffer=process_buffer,
        #     use_accel=use_accel,
        #     float_dtype=float_dtype,
        #     suppress_messages=False,
        #     use_breaching=use_breaching,
        #     breach_type=breach_type,
        #     remove_poly_ign=remove_poly_ign,
        #     **fb_kwargs
        # )
        # pyro_model.run_pyro(
        #     num_time_steps=num_time_steps,
        #     time_step_mins=time_step,
        #     contour_interval_mins=interval_mins
        # )
        elapsed_time = time.time() - start_time
        # output.result.set_value(f'PYroCell 🔥 modelling completed in {elapsed_time:.2f} seconds.')
        yield f'PYroCell 🔥 modelling completed in {elapsed_time:.2f} seconds.'

app = App(app_ui, server)