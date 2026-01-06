import os
import numpy as np
import cffbps as fbp
from .. import pyro_cell as pc
from .utils.helpers import PyrocellOutputLocation, assert_equal_results


def test_pyrocell_point_ignition():
    # === Configuration ===
    # Set modelling parameters
    float_dtype = np.float64
    ellipse_breadth = 3
    process_buffer = 1000
    use_accel = False
    use_jit = True
    use_breaching = False
    percentile_growth = 50
    out_request = ["perim", "cont", "tt", "ros", "fire_type", "hfi", "cfb", "raz", "wsv", "ffc", "wfc", "sfc"]
    time_step = 240
    interval_mins = 60

    # Set weather and fuel parameters
    wx_date = 20230715
    ffmc = 87.3785
    bui = 82.382
    gcf = fbp.getSeasonGrassCuring(season="summer", province="BC")

    # Get paths to input folders and datasets
    data_folder = os.path.join(os.path.dirname(__file__), "test_data")
    ign_name = "test_ignitions/point/Test_Ignition_Point.shp"
    fueltype_name = "test_inputs/FuelType_cropped.tif"
    elv_name = "test_inputs/ELV_cropped.tif"
    slope_name = "test_inputs/GS_cropped.tif"
    aspect_name = "test_inputs/Aspect_cropped.tif"
    ws180_name = "test_inputs/WS_180_cropped.tif"
    wd180_name = "test_inputs/WD_180_cropped.tif"
    ws300_name = "test_inputs/WS_300_cropped.tif"
    wd300_name = "test_inputs/WD_300_cropped.tif"
    pc_name = "test_inputs/PC_cropped.tif"
    pdf_name = "test_inputs/PDF_cropped.tif"
    gfl_name = "test_inputs/GFL_cropped.tif"
    output_folder = "_Results"
    output_prefix = "PointTest"

    ign_path = os.path.join(data_folder, ign_name)
    ft_path = os.path.join(data_folder, fueltype_name)
    elv_path = os.path.join(data_folder, elv_name)
    slope_path = os.path.join(data_folder, slope_name)
    aspect_path = os.path.join(data_folder, aspect_name)
    ws180_path = os.path.join(data_folder, ws180_name)
    wd180_path = os.path.join(data_folder, wd180_name)
    ws300_path = os.path.join(data_folder, ws300_name)
    wd300_path = os.path.join(data_folder, wd300_name)
    pc_path = os.path.join(data_folder, pc_name)
    pdf_path = os.path.join(data_folder, pdf_name)
    gfl_path = os.path.join(data_folder, gfl_name)

    ws_list = [ws180_path, ws180_path, ws180_path, ws180_path, ws300_path, ws300_path, ws300_path, ws300_path][
        :1
    ]  # [0:5:4]
    wd_list = [wd180_path, wd180_path, wd180_path, wd180_path, wd300_path, wd300_path, wd300_path, wd300_path][
        :1
    ]  # [0:5:4]
    ffmc_list = [ffmc for _ in range(len(ws_list))]  # [0:5:4]
    bui_list = [bui for _ in range(len(ws_list))]  # [0:5:4]

    # Create results folder
    results_folder = os.path.join(data_folder, output_folder)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # ### RUN MTT MODELLING
    # The fire behavior model kwargs
    fb_kwargs = {
        "fueltype_path": ft_path,
        "pct_cnfr": pc_path,
        "pct_deadfir": pdf_path,
        "grass_fuelload": gfl_path,
        "grass_curingfactor": gcf,
        "wind_speeds_list": ws_list,
        "wind_dirs_list": wd_list,
        "ffmc_list": ffmc_list,
        "bui_list": bui_list,
        "use_gpu": False,
        # 'gpu_float_dtype': cp.float32,
        "percentile_growth": percentile_growth,
    }

    # Run PYro Cell
    pyro_model = pc.PYRO(
        fb_plugin="cffbps",
        ignition_path=ign_path,
        fire_date=wx_date,
        elevation_path=elv_path,
        slope_path=slope_path,
        aspect_path=aspect_path,
        out_folder=results_folder,
        out_request=out_request,
        fire_id=output_prefix,
        ellipse_breadth=ellipse_breadth,
        process_buffer=process_buffer,
        use_accel=use_accel,
        float_dtype=float_dtype,
        use_jit=use_jit,
        suppress_messages=False,
        use_breaching=use_breaching,
        **fb_kwargs,
    )
    pyro_model.run_pyro(num_time_steps=len(ws_list), time_step_mins=time_step, contour_interval_mins=interval_mins)

    # check that all the outputs are the same (or close enough)
    assert_equal_results(
        test_loc=PyrocellOutputLocation(data_folder, output_folder, output_prefix),
        golden_loc=PyrocellOutputLocation(data_folder, "test_golden_output/point", "TestFire"),
    )


def test_pyrocell_line_ignition():
    # === Configuration ===
    # Set modelling parameters
    float_dtype = np.float64
    ellipse_breadth = 3
    process_buffer = 1000
    use_accel = False
    use_jit = True
    use_breaching = False
    percentile_growth = 50
    out_request = ["perim", "cont", "tt", "ros", "fire_type", "hfi", "cfb", "raz", "wsv", "ffc", "wfc", "sfc"]
    time_step = 240
    interval_mins = 60

    # Set weather and fuel parameters
    wx_date = 20230715
    ffmc = 87.3785
    bui = 82.382
    gcf = fbp.getSeasonGrassCuring(season="summer", province="BC")

    # Get paths to input folders and datasets
    data_folder = os.path.join(os.path.dirname(__file__), "test_data")
    ign_name = "test_ignitions/line/Test_Ignition_Line.shp"
    fueltype_name = "test_inputs/FuelType_cropped.tif"
    elv_name = "test_inputs/ELV_cropped.tif"
    slope_name = "test_inputs/GS_cropped.tif"
    aspect_name = "test_inputs/Aspect_cropped.tif"
    ws180_name = "test_inputs/WS_180_cropped.tif"
    wd180_name = "test_inputs/WD_180_cropped.tif"
    ws300_name = "test_inputs/WS_300_cropped.tif"
    wd300_name = "test_inputs/WD_300_cropped.tif"
    pc_name = "test_inputs/PC_cropped.tif"
    pdf_name = "test_inputs/PDF_cropped.tif"
    gfl_name = "test_inputs/GFL_cropped.tif"
    output_folder = "_Results"
    output_prefix = "LineTest"

    ign_path = os.path.join(data_folder, ign_name)
    ft_path = os.path.join(data_folder, fueltype_name)
    elv_path = os.path.join(data_folder, elv_name)
    slope_path = os.path.join(data_folder, slope_name)
    aspect_path = os.path.join(data_folder, aspect_name)
    ws180_path = os.path.join(data_folder, ws180_name)
    wd180_path = os.path.join(data_folder, wd180_name)
    ws300_path = os.path.join(data_folder, ws300_name)
    wd300_path = os.path.join(data_folder, wd300_name)
    pc_path = os.path.join(data_folder, pc_name)
    pdf_path = os.path.join(data_folder, pdf_name)
    gfl_path = os.path.join(data_folder, gfl_name)

    ws_list = [ws180_path, ws180_path, ws180_path, ws180_path, ws300_path, ws300_path, ws300_path, ws300_path][
        :1
    ]  # [0:5:4]
    wd_list = [wd180_path, wd180_path, wd180_path, wd180_path, wd300_path, wd300_path, wd300_path, wd300_path][
        :1
    ]  # [0:5:4]
    ffmc_list = [ffmc for _ in range(len(ws_list))]  # [0:5:4]
    bui_list = [bui for _ in range(len(ws_list))]  # [0:5:4]

    # Create results folder
    results_folder = os.path.join(data_folder, output_folder)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # ### RUN MTT MODELLING
    # The fire behavior model kwargs
    fb_kwargs = {
        "fueltype_path": ft_path,
        "pct_cnfr": pc_path,
        "pct_deadfir": pdf_path,
        "grass_fuelload": gfl_path,
        "grass_curingfactor": gcf,
        "wind_speeds_list": ws_list,
        "wind_dirs_list": wd_list,
        "ffmc_list": ffmc_list,
        "bui_list": bui_list,
        "use_gpu": False,
        # 'gpu_float_dtype': cp.float32,
        "percentile_growth": percentile_growth,
    }

    # Run PYro Cell
    pyro_model = pc.PYRO(
        fb_plugin="cffbps",
        ignition_path=ign_path,
        fire_date=wx_date,
        elevation_path=elv_path,
        slope_path=slope_path,
        aspect_path=aspect_path,
        out_folder=results_folder,
        out_request=out_request,
        fire_id=output_prefix,
        ellipse_breadth=ellipse_breadth,
        process_buffer=process_buffer,
        use_accel=use_accel,
        float_dtype=float_dtype,
        use_jit=use_jit,
        suppress_messages=False,
        use_breaching=use_breaching,
        **fb_kwargs,
    )
    pyro_model.run_pyro(num_time_steps=len(ws_list), time_step_mins=time_step, contour_interval_mins=interval_mins)

    # check that all the outputs are the same (or close enough)
    assert_equal_results(
        test_loc=PyrocellOutputLocation(data_folder, output_folder, output_prefix),
        golden_loc=PyrocellOutputLocation(data_folder, "test_golden_output/line", "TestFire"),
    )


def test_pyrocell_polygon_ignition():
    # === Configuration ===
    # Set modelling parameters
    float_dtype = np.float64
    ellipse_breadth = 3
    process_buffer = 1000
    use_accel = False
    use_jit = True
    use_breaching = False
    percentile_growth = 50
    out_request = ["perim", "cont", "tt", "ros", "fire_type", "hfi", "cfb", "raz", "wsv", "ffc", "wfc", "sfc"]
    time_step = 240
    interval_mins = 60

    # Set weather and fuel parameters
    wx_date = 20230715
    ffmc = 87.3785
    bui = 82.382
    gcf = fbp.getSeasonGrassCuring(season="summer", province="BC")

    # Get paths to input folders and datasets
    data_folder = os.path.join(os.path.dirname(__file__), "test_data")
    ign_name = "test_ignitions/multiple_polygon/Test_Ignition_Multiple_Polygons.shp"
    fueltype_name = "test_inputs/FuelType_cropped.tif"
    elv_name = "test_inputs/ELV_cropped.tif"
    slope_name = "test_inputs/GS_cropped.tif"
    aspect_name = "test_inputs/Aspect_cropped.tif"
    ws180_name = "test_inputs/WS_180_cropped.tif"
    wd180_name = "test_inputs/WD_180_cropped.tif"
    ws300_name = "test_inputs/WS_300_cropped.tif"
    wd300_name = "test_inputs/WD_300_cropped.tif"
    pc_name = "test_inputs/PC_cropped.tif"
    pdf_name = "test_inputs/PDF_cropped.tif"
    gfl_name = "test_inputs/GFL_cropped.tif"
    output_folder = "_Results"
    output_prefix = "MultiplePolygonTest"

    ign_path = os.path.join(data_folder, ign_name)
    ft_path = os.path.join(data_folder, fueltype_name)
    elv_path = os.path.join(data_folder, elv_name)
    slope_path = os.path.join(data_folder, slope_name)
    aspect_path = os.path.join(data_folder, aspect_name)
    ws180_path = os.path.join(data_folder, ws180_name)
    wd180_path = os.path.join(data_folder, wd180_name)
    ws300_path = os.path.join(data_folder, ws300_name)
    wd300_path = os.path.join(data_folder, wd300_name)
    pc_path = os.path.join(data_folder, pc_name)
    pdf_path = os.path.join(data_folder, pdf_name)
    gfl_path = os.path.join(data_folder, gfl_name)

    ws_list = [ws180_path, ws180_path, ws180_path, ws180_path, ws300_path, ws300_path, ws300_path, ws300_path][
        :1
    ]  # [0:5:4]
    wd_list = [wd180_path, wd180_path, wd180_path, wd180_path, wd300_path, wd300_path, wd300_path, wd300_path][
        :1
    ]  # [0:5:4]
    ffmc_list = [ffmc for _ in range(len(ws_list))]  # [0:5:4]
    bui_list = [bui for _ in range(len(ws_list))]  # [0:5:4]

    # Create results folder
    results_folder = os.path.join(data_folder, output_folder)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    # ### RUN MTT MODELLING
    # The fire behavior model kwargs
    fb_kwargs = {
        "fueltype_path": ft_path,
        "pct_cnfr": pc_path,
        "pct_deadfir": pdf_path,
        "grass_fuelload": gfl_path,
        "grass_curingfactor": gcf,
        "wind_speeds_list": ws_list,
        "wind_dirs_list": wd_list,
        "ffmc_list": ffmc_list,
        "bui_list": bui_list,
        "use_gpu": False,
        # 'gpu_float_dtype': cp.float32,
        "percentile_growth": percentile_growth,
    }

    # Run PYro Cell
    pyro_model = pc.PYRO(
        fb_plugin="cffbps",
        ignition_path=ign_path,
        fire_date=wx_date,
        elevation_path=elv_path,
        slope_path=slope_path,
        aspect_path=aspect_path,
        out_folder=results_folder,
        out_request=out_request,
        fire_id=output_prefix,
        ellipse_breadth=ellipse_breadth,
        process_buffer=process_buffer,
        use_accel=use_accel,
        float_dtype=float_dtype,
        use_jit=use_jit,
        suppress_messages=False,
        use_breaching=use_breaching,
        **fb_kwargs,
    )
    pyro_model.run_pyro(num_time_steps=len(ws_list), time_step_mins=time_step, contour_interval_mins=interval_mins)

    # check that all the outputs are the same (or close enough)
    assert_equal_results(
        test_loc=PyrocellOutputLocation(data_folder, output_folder, output_prefix),
        golden_loc=PyrocellOutputLocation(data_folder, "test_golden_output/multiple_polygon", "TestFire"),
    )
