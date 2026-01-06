import cffbps
import cffbps_cupy
import math
import numpy as np
import cupy as cp
import rasterio as rio
from rasterio import windows
from typing import Union, Optional

class FBP_PLUGIN:
    def __init__(self,
                 process_window: windows,
                 process_grid_shape: tuple[int, int],
                 out_folder: str,
                 fire_date: Union[int, str],
                 elevation: rio.DatasetReader,
                 slope: rio.DatasetReader,
                 aspect: rio.DatasetReader,
                 fueltype_path: str,
                 lat_path: str,
                 long_path: str,
                 pct_cnfr: Union[str, float, int],
                 pct_deadfir: Union[str, float, int],
                 grass_fuelload: Union[str, float, int],
                 grass_curingfactor: Union[str, float, int],
                 d0: Optional[int] = None,
                 dj: Optional[int] = None,
                 ws_domain_avg_value_current: Optional[Union[float, int]] = None,
                 ws_domain_avg_value_new: Optional[Union[float, int]] = None,
                 percentile_growth: Optional[Union[float, int]] = None,
                 float_dtype: np.dtype = np.float32,
                 use_gpu: bool = False,
                 gpu_float_dtype: cp.dtype = cp.float32):
        """
        Initialize the FBP Plugin with parameters essential for simulating fire behavior.

        :param process_window: Rasterio window object defining the area to process.
        :param out_folder: Output folder path
        :param fire_date: The fire date (int or str) in YYYYMMDD format
        :param elevation: The elevation dataset reader object
        :param slope: The slope dataset reader object
        :param aspect: The aspect dataset reader object
        :param fueltype_path: Path to the fuel type dataset
        :param lat_path: Path to the latitude dataset
        :param long_path: Path to the longitude dataset
        :param pct_cnfr: Percentage of coniferous cover (%, ranging from 0-100)
        :param pct_deadfir: Percentage of dead fir cover (%, ranging from 0-100)
        :param grass_fuelload: Fuel load for grass (kg/m^2)
        :param grass_curingfactor: Grass curing factor (%, ranging from 0-100)
        :param d0: Julian date of minimum foliar moisture content (if None, it will be calculated based on latitude)
        :param dj: Julian date of modelled fire (if None, it will be calculated from wx_date)
        :param ws_domain_avg_value_current: The current wind speed domain average value applied the wind speed rasters.
            Used to unscale input wind speed grid values.
            ws_domain_avg_value_new must also be provided if this parameter is used.
        :param ws_domain_avg_value_new: The new domain average value to apply to the wind speed rasters.
            Used to rescale input wind speed grids after the current values are unscaled.
            ws_domain_avg_value_current must also be provided if this parameter is used.
        :param percentile_growth: Percentile growth value for rate of spread calculations
        :param float_dtype: Maximum data type for float processing (default is np.float32)
        :param use_gpu: Boolean indicating whether to use GPU for processing
        :param gpu_float_dtype: Maximum data type for GPU float processing (default is cp.float32)
        """
        # Initialize input datasets
        self.process_window = process_window
        self.process_grid_shape = process_grid_shape
        self.out_folder = out_folder
        self.fire_date = fire_date
        self.elevation = elevation
        self.slope = slope
        self.aspect = aspect
        self.lat_path = lat_path
        self.long_path = long_path
        self.fueltype_path = fueltype_path
        self.pct_cnfr = pct_cnfr
        self.pct_deadfir = pct_deadfir
        self.grass_fuelload = grass_fuelload
        self.grass_curingfactor = grass_curingfactor
        self.d0 = d0
        self.dj = dj
        self.ws_domain_avg_value_current = ws_domain_avg_value_current
        self.ws_domain_avg_value_new = ws_domain_avg_value_new
        self.float_dtype = float_dtype
        self.use_gpu = use_gpu
        self.gpu_float_dtype = gpu_float_dtype
        self.percentile_growth = percentile_growth

        # Initialize weather datasets
        self.wind_speeds_list = None
        self.wind_dirs_list = None
        self.ffmc_list = None
        self.bui_list = None

        # Initialize processing flags
        self.rescale_ws = False
        self.tstep_dsets_initialized = False

        # Initialize processing datasets
        self.process_row_offset = None
        self.process_col_offset = None
        self.fuel_type = None
        self.fuel_type_dtype = None
        self.fuel_type_array = None
        self.lat_array = None
        self.long_array = None
        self.ws = None
        self.ws_dtype = None
        self.wd = None
        self.wd_dtype = None
        self.ffmc = None
        self.ffmc_dtype = None
        self.bui = None
        self.bui_dtype = None
        self.pc = None
        self.pc_dtype = None
        self.pdf = None
        self.pdf_dtype = None
        self.gfl = None
        self.gfl_dtype = None
        self.gcf = None
        self.gcf_dtype = None

        # Initialize timestep-specific weather datasets
        self.tstep_window = None
        self.tstep_ws = None
        self.tstep_wd = None
        self.tstep_ffmc = None
        self.tstep_bui = None
        self.tstep_gcf = None
        self.tstep_fmc = None

        # Initialize output datasets
        self.valid_outputs = [
            'fire_type', 'fuel_type', 'ws', 'wd', 'm', 'fF', 'fW', 'ffmc', 'bui', 'isi', 'a', 'b', 'c',
            'rsz', 'sf', 'rsf', 'isf', 'rsi', 'wse1', 'wse2', 'wse', 'wsx', 'wsy', 'wsv', 'raz', 'q', 'bui0',
            'be', 'be_max', 'ffc', 'wfc', 'sfc', 'latn', 'dj', 'd0', 'nd', 'fmc', 'fme', 'csfi', 'rso',
            'bfw', 'bisi', 'bros', 'sros', 'cros', 'cbh', 'cfb', 'cfl', 'cfc', 'tfc', 'accel', 'fi_class'
        ]
        self.rso = None
        self.fmc = None

        # Verify input data
        self._verify_inputs()

        # Initialize time step-based arrays
        self._init_fuel_type_array()
        self._init_lat_long_arrays()
        self._init_tstep_data()

    def _verify_inputs(self) -> None:
        """
        Function to verify the Class inputs.

        :return: None
        """
        if not isinstance(self.fire_date, (str, int)):
            raise TypeError('The fire_date parameter must be a string or int data type')
        elif isinstance(self.fire_date, str):
            self.fire_date = int(self.fire_date)

        if not isinstance(self.fueltype_path, str):
            raise TypeError('The fueltype_path parameter must be a string data type')
        else:
            self.fuel_type = rio.open(self.fueltype_path)
            self.fuel_type_dtype = self.fuel_type.dtypes[0]
            if np.dtype(self.fuel_type_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.fuel_type_dtype = self.float_dtype

        if not isinstance(self.pct_cnfr, (str, float, int)):
            raise TypeError('The pct_cnfr parameter must be a string, float, or int data type')
        elif isinstance(self.pct_cnfr, str):
            self.pc = rio.open(self.pct_cnfr)
            self.pc_dtype = self.pc.dtypes[0]
            if np.dtype(self.pc_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.pc_dtype = self.float_dtype
        else:
            self.pc = self.pct_cnfr

        if not isinstance(self.pct_deadfir, (str, float, int)):
            raise TypeError('The pct_deadfir parameter must be a string, float, or int data type')
        elif isinstance(self.pct_deadfir, str):
            self.pdf = rio.open(self.pct_deadfir)
            self.pdf_dtype = self.pdf.dtypes[0]
            if np.dtype(self.pdf_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.pdf_dtype = self.float_dtype
        else:
            self.pdf = self.pct_deadfir

        if not isinstance(self.grass_fuelload, (str, float, int)):
            raise TypeError('The grass_fuelload parameter must be a string, float, or int data type')
        elif isinstance(self.grass_fuelload, str):
            self.gfl = rio.open(self.grass_fuelload)
            self.gfl_dtype = self.gfl.dtypes[0]
            if np.dtype(self.gfl_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.gfl_dtype = self.float_dtype
        else:
            self.gfl = self.grass_fuelload

        if not isinstance(self.grass_curingfactor, (str, float, int)):
            raise TypeError('The grass_curingfactor parameter must be a string, float, or int data type')
        elif isinstance(self.grass_curingfactor, str):
            self.gcf = rio.open(self.grass_curingfactor)
            self.gcf_dtype = self.gcf.dtypes[0]
            if np.dtype(self.gcf_dtype).itemsize > np.dtype(self.float_dtype).itemsize:
                self.gcf_dtype = self.float_dtype
        else:
            self.gcf = self.grass_curingfactor

        if self.d0 is not None:
            if not isinstance(self.d0, int):
                raise TypeError('The d0 parameter must be int data type')

        if self.dj is not None:
            if not isinstance(self.dj, int):
                raise TypeError('The dj parameter must be int data type')

        if self.ws_domain_avg_value_current is not None:
            if not isinstance(self.ws_domain_avg_value_current, (float, int)):
                raise TypeError('The ws_domain_avg_value parameter must be float or int data type')
            if self.ws_domain_avg_value_new is None:
                raise ValueError('The ws_domain_avg_value_new parameter must be provided '
                                 'if ws_scaling_value is provided')

        if self.ws_domain_avg_value_new is not None:
            if not isinstance(self.ws_domain_avg_value_new, (float, int)):
                raise TypeError('The ws_domain_avg_value_new parameter must be float or int data type')
            if self.ws_domain_avg_value_current is None:
                raise ValueError('The ws_domain_avg_value_current parameter must be provided '
                                 'if ws_domain_avg_value_new is provided')

        if (self.ws_domain_avg_value_current is not None) and (self.ws_domain_avg_value_new is not None):
            self.rescale_ws = True

        return

    def _init_fuel_type_array(self) -> None:
        """
        Function to initialize the fuel type array for the processing window.

        :return: None
        """
        window = self.process_window
        self.fuel_type_array = self.fuel_type.read(
            1,
            window=window
        ).astype(self.fuel_type_dtype)
        return

    def _init_lat_long_arrays(self) -> None:
        """
        Function to initialize the latitude and longitude arrays.

        :return: None
        """
        # Load latitude array
        with rio.open(self.lat_path) as lat:
            self.lat_array = lat.read(1).astype(self.float_dtype)

        # Load longitude array
        with rio.open(self.long_path) as long:
            self.long_array = long.read(1).astype(self.float_dtype)

        return

    def _init_weather_data(self,
                           dset_index: int,
                           wind_speeds_list: list[Union[str, float, int]],
                           wind_dirs_list: list[Union[str, float, int]],
                           ffmc_list: list[Union[str, float, int]],
                           bui_list: list[Union[str, float, int]]) -> None:
        """
        Function to initialize the weather data.

        :param dset_index: The index of the current dataset being processed
        :param wind_speeds_list: List of wind speed values or paths to gridded datasets
        :param wind_dirs_list: list of wind direction values or paths to gridded datasets
        :param ffmc_list: List of FFMC values or paths to gridded datasets
        :param bui_list: List of BUI values or paths to gridded datasets
        :return: None
        """
        self.wind_speeds_list = wind_speeds_list
        self.wind_dirs_list = wind_dirs_list
        self.ffmc_list = ffmc_list
        self.bui_list = bui_list

        if not isinstance(self.wind_speeds_list, list):
            raise TypeError('The wind_speeds_list parameter must be a list data type')

        if not isinstance(self.wind_dirs_list, list):
            raise TypeError('The wind_dirs_list parameter must be a list data type')

        # Helper to open raster or pass value
        def _open_or_value(item, float_dtype):
            if isinstance(item, str):
                ds = rio.open(item)
                dt = ds.dtypes[0]
                if np.dtype(dt).itemsize > np.dtype(float_dtype).itemsize:
                    dt = float_dtype
                return ds, dt
            else:
                return item, None

        # Get the wind speed data
        self.ws, self.ws_dtype = _open_or_value(self.wind_speeds_list[dset_index], self.float_dtype)

        # Get the wind direction data
        self.wd, self.wd_dtype = _open_or_value(self.wind_dirs_list[dset_index], self.float_dtype)

        # Get the ffmc data
        self.ffmc, self.ffmc_dtype = _open_or_value(self.ffmc_list[dset_index], self.float_dtype)

        # Get the bui data
        self.bui, self.bui_dtype = _open_or_value(self.bui_list[dset_index], self.float_dtype)

        return

    def _init_tstep_data(self):
        """
        Function to initialize the timestep-specific weather datasets.

        :return: None
        """
        process_data = ['tstep_ws', 'tstep_wd', 'tstep_ffmc', 'tstep_bui', 'tstep_gcf', 'tstep_fmc']

        # Initialize timestep-specific weather datasets
        if not self.tstep_dsets_initialized:
            # Initialize processing arrays using the processing grid
            process_dict = {var: np.full(self.process_grid_shape, np.inf).astype(self.float_dtype) for var in
                            process_data}

            # Set the processing arrays as class attributes
            self._set_params(process_dict)
            del process_dict
            self.tstep_dsets_initialized = True
        else:
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

        return

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

    def close_datasets(self):
        """
        Function to close opened rasterio DatasetReader objects.
        :return: None
        """
        datasets = [
            self.fuel_type, self.ws, self.wd, self.ffmc, self.bui, self.pc, self.pdf, self.gfl, self.gcf
        ]
        for dataset in datasets:
            if dataset and isinstance(dataset, rio.DatasetReader) and not dataset.closed:
                dataset.close()
        del datasets

        return

    def get_land_cover(self, row: int, col: int) -> int:
        """
        Function to determine land cover type (forest, grass, non-fuel).

        :param row: Row index of the cell
        :param col: Column index of the cell
        :return: Land cover type.
        """
        # Get the fuel type
        fuel_type = self.fuel_type_array[row, col]

        # Return the Land Cover value
        return 1 if fuel_type not in [14, 15, 19, 20] else 2 if fuel_type in [14, 15] else 0

    def get_land_cover_array(self) -> np.ndarray:
        """
        Function to generate and return a vectorized land cover mapping from the fuel type array:
          - 0 = non-fuel
          - 1 = forest
          - 2 = grass

        :return: Land cover type grid
        """
        # Get the land cover array using the process window to subset the fuel type array
        land_cover = np.full_like(self.fuel_type_array, 1)  # Default to forest
        land_cover[np.isin(self.fuel_type_array, [14, 15])] = 2  # Grass
        land_cover[np.isin(self.fuel_type_array, [19, 20])] = 0  # Non-fuel
        return land_cover

    def get_stand_ht(self, row: int, col: int) -> float:
        """
        Function to determine stand height based on the fuel type.

        :param row: Row index of the cell
        :param col: Column index of the cell
        :return: Stand height (m).
        """
        # Get the fuel type
        fuel_type = int(self.fuel_type_array[row, col])

        # Return the Stand Height value
        return cffbps.FBP().fbpCBH_CFL_HT_LUT[fuel_type][-1]

    def get_fire_type(self, ros: float, row: int, col: int) -> int:
        """
        Function to determine fire type (surface, crown, non-fuel).

        :param ros: Fire rate of spread (m/min)
        :param row: Row index of the cell
        :param col: Column index of the cell
        :return: Fire type (1 = surface, 2 = intermittent crown, 3 = crown).
        """
        # Get the critical surface fire rate of spread (rso) for the cell
        rso = self.rso[row, col]

        # Calculate the crown fraction burned for the cell
        cfb = max(0.0, 1 - math.exp(-0.23 * (ros - rso)))

        # Return the Fire Type value
        return 1 if (cfb <= 0.1) else 3 if (cfb >= 0.9) else 2

    def run_model(self,
                  out_request: list = None,
                  process_results: bool = False,
                  new_ros: np.ndarray = None,
                  **fb_kwargs) -> tuple[Union[None, list], Union[None, list]]:
        """
        Process a subset of datasets within self.process_extent and update fuel_type for non-infinite travel times.

        :param out_request: List of output parameters to request from the FBP model.
        :param process_results: Boolean indicating whether to process final results.
        :param new_ros: New rate of spread values to set in the FBP model.
        :return: None
        """
        # Set default output request if none provided
        if out_request is None:
            out_request = ['wsv', 'raz', 'hfi', 'hros', 'bros', 'accel', 'rso', 'fmc']

        # Extract weather-related kwargs
        wx_keys = ['dset_index', 'wind_speeds_list', 'wind_dirs_list', 'ffmc_list', 'bui_list']
        wx_kwargs = {k: fb_kwargs[k] for k in wx_keys if k in fb_kwargs}

        # Assign the processing window
        window = self.process_window

        # Initialize the weather data
        self._init_weather_data(**wx_kwargs)

        # Load subsets of the required rasters
        if self.use_gpu:
            fuel_type_subset = cp.asarray(self.fuel_type_array, cp.int8)  # Dataset already subset to window
            elevation_subset = cp.asarray(self.elevation.read(1, window=window), self.gpu_float_dtype)
            slope_subset = cp.asarray(self.slope.read(1, window=window), self.gpu_float_dtype)
            aspect_subset = cp.asarray(self.aspect.read(1, window=window), self.gpu_float_dtype)
            lat_subset = cp.asarray(self.lat_array, self.gpu_float_dtype)  # Dataset already subset to window
            long_subset = cp.asarray(self.long_array, self.gpu_float_dtype)  # Dataset already subset to window
            if process_results:
                if new_ros is None:
                    raise ValueError('new_ros must be provided if process_results is True')
                new_ros = cp.asarray(new_ros, dtype=new_ros.dtype)
        else:
            fuel_type_subset = self.fuel_type_array
            elevation_subset = self.elevation.read(1, window=window)
            slope_subset = self.slope.read(1, window=window)
            aspect_subset = self.aspect.read(1, window=window)
            lat_subset = self.lat_array  # Dataset already subset to window
            long_subset = self.long_array  # Dataset already subset to window

        def _read_and_convert(dataset, dtype, use_gpu):
            if isinstance(dataset, rio.DatasetReader):
                data = dataset.read(1, window=window)
                no_data = dataset.nodata
                if np.all(data == no_data):
                    return None
                if np.any(data == no_data):
                    data = np.where(data == no_data, np.nan, data).astype(self.float_dtype)
                    return cp.asarray(data, dtype=self.gpu_float_dtype) if use_gpu else data
                else:
                    return cp.asarray(data, dtype=self.gpu_float_dtype) if use_gpu else data
            else:
                return dataset

        pc_subset = _read_and_convert(self.pc, self.pc_dtype, self.use_gpu)
        pdf_subset = _read_and_convert(self.pdf, self.pdf_dtype, self.use_gpu)
        gfl_subset = _read_and_convert(self.gfl, self.gfl_dtype, self.use_gpu)
        if process_results:
            ws_subset = _read_and_convert(self.tstep_ws, self.ws_dtype, self.use_gpu)
            wd_subset = _read_and_convert(self.tstep_wd, self.wd_dtype, self.use_gpu)
            ffmc_subset = _read_and_convert(self.tstep_ffmc, self.ffmc_dtype, self.use_gpu)
            bui_subset = _read_and_convert(self.tstep_bui, self.bui_dtype, self.use_gpu)
            gcf_subset = _read_and_convert(self.gcf, self.gcf_dtype, self.use_gpu)
            fmc_subset = _read_and_convert(self.tstep_fmc, self.float_dtype, self.use_gpu)
        else:
            ws_subset = _read_and_convert(self.ws, self.ws_dtype, self.use_gpu)
            if self.rescale_ws:
                ws_subset = ws_subset * self.ws_domain_avg_value_new / self.ws_domain_avg_value_current
            wd_subset = _read_and_convert(self.wd, self.wd_dtype, self.use_gpu)
            ffmc_subset = _read_and_convert(self.ffmc, self.ffmc_dtype, self.use_gpu)
            bui_subset = _read_and_convert(self.bui, self.bui_dtype, self.use_gpu)
            gcf_subset = _read_and_convert(self.gcf, self.gcf_dtype, self.use_gpu)
            fmc_subset = None

        # Return the names of the None datasets
        none_dsets = []
        dset_name_list = ['ws', 'wd', 'ffmc', 'bui', 'pc', 'pdf', 'gfl', 'gcf']
        subset_list = [ws_subset, wd_subset, ffmc_subset, bui_subset, pc_subset, pdf_subset, gfl_subset, gcf_subset]
        for dset_name, dset in zip(dset_name_list, subset_list):
            if dset is None:
                none_dsets.append((dset_name, dset))

        if none_dsets:
            return (None, none_dsets)
        del subset_list

        # Pass the subsets to the FBP model
        if self.use_gpu:
            fbp = cffbps_cupy.FBP(cupy_float_type=self.gpu_float_dtype)
        else:
            fbp = cffbps.FBP()

        # Initialize the FBP model
        fbp.initialize(
            fuel_type=fuel_type_subset,
            wx_date=self.fire_date,
            lat=lat_subset,
            long=long_subset,
            elevation=elevation_subset,
            slope=slope_subset,
            aspect=aspect_subset,
            ws=ws_subset,
            wd=wd_subset,
            ffmc=ffmc_subset,
            bui=bui_subset,
            pc=pc_subset,
            pdf=pdf_subset,
            gfl=gfl_subset,
            gcf=gcf_subset,
            out_request=out_request,
            percentile_growth=self.percentile_growth
        )

        if process_results:
            # Run FBP
            fbp.invertWindAspect()
            fbp.calcSF()
            fbp.calcISZ()
            fbp.setParams({'fmc': fmc_subset})
            fbp.calcISI_RSI_BE()
            fbp.calcSFC()
            fbp.getCBH_CFL()
            fbp.calcROS()
            fbp.setParams({'hros': new_ros})
            fbp.calcCSFI()
            fbp.calcRSO()
            fbp.calcCFB()
            fbp.calcRosPercentileGrowth()
            fbp.calcFireType()
            fbp.calcCFC()
            fbp.calcC6hros()
            fbp.calcTFC()
            fbp.calcHFI()
            fbp.calcFireIntensityClass()

            # Get requested outputs
            out_params = fbp.getParams(out_request=out_request)

            # Clean up FBP object
            del fbp

            # Return requested outputs
            return out_params, None
        else:
            # Run FBP
            wsv, raz, hfi, hros, bros, accel, self.rso, self.fmc = fbp.runFBP()

            # Clean up FBP object
            del fbp

            # Return requested outputs
            return [wsv, raz, hfi, hros, bros, accel], None

    def update_tstep_data(self, travel_times_array: np.ndarray, time_step: int, dset_run_time: int):
        """
        Function to update the timestep-specific input data arrays based on the current time step
        and the travel time array.

        :param travel_times_array: Array of travel times for the current processing window
        :param time_step: The time step duration (minutes)
        :param dset_run_time: The cumulative run time of the datasets processed so far (minutes)
        :return: None
        """
        window = self.process_window
        # Update the timestep-specific weather data arrays where travel time is finite
        min_time = max(0, dset_run_time - time_step)
        finite_mask = np.isfinite(travel_times_array) & (travel_times_array > min_time)

        if isinstance(self.ws, rio.DatasetReader):
            self.tstep_ws[finite_mask] = self.ws.read(1, window=window)[finite_mask]
        else:
            self.tstep_ws[finite_mask] = self.ws
        if isinstance(self.wd, rio.DatasetReader):
            self.tstep_wd[finite_mask] = self.wd.read(1, window=window)[finite_mask]
        else:
            self.tstep_wd[finite_mask] = self.wd
        if isinstance(self.ffmc, rio.DatasetReader):
            self.tstep_ffmc[finite_mask] = self.ffmc.read(1, window=window)[finite_mask]
        else:
            self.tstep_ffmc[finite_mask] = self.ffmc
        if isinstance(self.bui, rio.DatasetReader):
            self.tstep_bui[finite_mask] = self.bui.read(1, window=window)[finite_mask]
        else:
            self.tstep_bui[finite_mask] = self.bui
        if isinstance(self.gcf, rio.DatasetReader):
            self.tstep_gcf[finite_mask] = self.gcf.read(1, window=window)[finite_mask]
        else:
            self.tstep_gcf[finite_mask] = self.gcf
        if isinstance(self.fmc, rio.DatasetReader):
            self.tstep_fmc[finite_mask] = self.fmc.read(1, window=window)[finite_mask]
        else:
            self.tstep_fmc[finite_mask] = self.fmc[finite_mask]
        return

    def set_processing_window(self,
                              new_window: windows,
                              new_shape: tuple[int, int],
                              new_row_offset: int,
                              new_col_offset: int) -> None:
        """
        Function to update the processing window.

        :param new_window: New rasterio window object defining the area to process.
        :param new_shape: New shape of the processing grid (rows, cols).
        :param new_row_offset: New row offset for the processing grid.
        :param new_col_offset: New column offset for the processing grid.
        :return: None
        """
        self.process_window = new_window
        self.process_grid_shape = new_shape
        self.process_row_offset = new_row_offset
        self.process_col_offset = new_col_offset
        self._init_fuel_type_array()
        self._init_lat_long_arrays()
        self._init_tstep_data()
        return
