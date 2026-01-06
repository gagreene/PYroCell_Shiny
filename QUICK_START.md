---
title: "Welcome to PYroCell"

format:
  pdf:
    pdf-engine: xelatex
    header-includes:
      - \usepackage{xcolor}
      - \definecolor{PyroTitle}{HTML}{53871C}
      - \definecolor{PyroSub}{HTML}{74BA2B}

      - \usepackage{titling}
      - \pretitle{\begin{center}\LARGE\bfseries\color{PyroTitle}}
      - \posttitle{\end{center}}

      - \usepackage{sectsty}
      - \sectionfont{\color{PyroTitle}}
      - \subsectionfont{\color{PyroSub}}

      - \usepackage{fvextra}
      - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
      - \fvset{fontsize=\small}

  gfm:
    execute: false
---
## Installation

### PYroCell

**PYroCell** is the core fire growth model.
https://github.com/gagreene/MinimumTravelTime

Create the Conda environment using the provided specification file:

```bash
conda create --name PyroCell --file conda-spec-file-windows.txt
```

---

### Project Structure

The following repositories must be cloned and placed within the same project directory so they can be imported by **PYroCell**:

```text
pyrocell_project/
├── PYroCell/
├── cffdrs/
├── flame_components/
└── ProcessGeospatial/
```

---

### Required Repositories

- **PYroCell** – Core fire growth model

- **CFFDRS** – Canadian Forest Fire Danger Rating System
  https://github.com/ForsiteConsultants/cffdrs

- **Flame Components** – Flame length and fire behavior component calculations
  https://github.com/ForsiteConsultants/flame_components

- **Process Geospatial** – Geospatial preprocessing utilities
  https://github.com/ForsiteConsultants/ProcessGeospatial




## Model Description

[TO DO]

## Model Input Variables

In **PYroCell**, fire spreads first from and Ignition point, line or polygon and then from cell to cell across a gridded landscape. Each pixel within this landscape is defined by its fuel and topographic characteristics:

- FBP Fuel Type and Attributes
- Elevation
- Slope
- Aspect

To calculate fire behaviour **PYroCell** also needs weather variables that can either vary from pixel to pixel or remain constant across the entire landscape.

- FFMC: Fine fuel moisture code
- BUI: Buildup index
- Wind speed
- Wind Direction

***

### Ignition

Ignitions must occur over **fuel-containing cells** for fires to initiate. If ignition **points** do not fall within fuel cells, those pixels will not ignite or spread fire. For **line** or **polygon** ignition features, fire spread will occur **only in the portions** of the geometry that intersect fuel-containing cells. Sections of lines or polygons that fall within **non-fuel cells** will not ignite or contribute to fire spread.


- ign_name: Path to a shapefile containing ignition geometry (**points**, **lines**, or **polygons**).
- fire_id_field: Name of the attribute field in the shapefile that contains a **unique ID** for each ignition feature. This ID will be appended to the fire simulation name.

``` python
 ign_name = 'NameOfShapeFile.shp'
 fire_id_field = 'Id'
```
### FBP fuel type and attributes

FBP fuels use **standardized numeric class codes** (integer values) where each number represents a specific **FBP fuel type** (e.g., `C1`, `D1`, `O1a`). In other words, the raster/array values are the **numeric codes**, which map to the **letter/number fuel classes** below.

- 1: 'C1',  # C-1
- 2: 'C2',  # C-2
- 3: 'C3',  # C-3
- 4: 'C4',  # C-4
- 5: 'C5',  # C-5
- 6: 'C6',  # C-6
- 7: 'C7',  # C-7
- 8: 'D1',  # D-1
- 9: 'D2',  # D-2
- 10: 'M1',  # M-1
- 11: 'M2',  # M-2
- 12: 'M3',  # M-3
- 13: 'M4',  # M-4
- 14: 'O1a',  # O-1a
- 15: 'O1b',  # O-1b
- 16: 'S1',  # S-1
- 17: 'S2',  # S-2
- 18: 'S3',  # S-3
- 19: 'NF',  # NF (non-fuel)
- 20: 'WA'  # WA (water)

FBP fuel **attributes** can be provided as either:
- **Spatial** (a raster/vector layer): provide a **file path**
- **Scalar** (single constant value): provide the **value directly**

**Attributes (spatial paths shown):**
```python
"fueltype_path": ft_path,      # spatial: path to fueltype codes layer (integer codes)
"pct_cnfr": pc_path,           # Percent conifer spatial or scalar
"pct_deadfir": pdf_path,       # Percent dead fir spatial or scalar
"grass_fuelload": gfl_path,    # Grass fuel load spatial or scalar
"grass_curingfactor": gcf,     # Grass curing factor spatial or scalar
```
### Topography: elevation, slope and aspect


Topographic inputs must be provided as **spatial layers** with the **same spatial extent and resolution** as the fuel layers.

Units must be:
- **Elevation**: metres above sea level (**masl**)
- **Slope**: percentage (**%**)
- **Aspect**: degrees (**0–360**, clockwise from north)

```python
elevation_path = elv_path,   # elevation in masl
slope_path = slope_path,     # slope in %
aspect_path = aspect_path,   # aspect in degrees
```

### Weather

Weather inputs may be provided as **scalar values** or as **spatial layers**.

**Units:**
- Wind speed: km/h
- Wind direction: degrees (0–360) (direction the wind is coming from)
- FFMC: dimensionless
- BUI: dimensionless

Scalar and spatial weather inputs may be mixed (e.g., spatial wind with scalar BUI),
provided temporal requirements are met. For **both scalar and spatial inputs**, weather data must:
- Have a **temporal frequency matching the model time-step duration**
  - Most commonly **hourly**
- Include **one scalar value or one spatial layer per model time step**

For example like this:

```python
wx_folder = os.path.join(data_folder, 'weather')
temp_list = sorted(glob.glob(os.path.join(wx_folder, 'temp*.tiff')))[0:num_time_steps]
ws_list = sorted(glob.glob(os.path.join(wx_folder, 'ws*.tiff')))[0:num_time_steps]
wd_list = sorted(glob.glob(os.path.join(wx_folder, 'wd*.tiff')))[0:num_time_steps]
ffmc_list = sorted(glob.glob(os.path.join(wx_folder, 'hffmc*.tiff')))[0:num_time_steps]
bui_list = [bui] * num_time_steps
```

For **spatial weather layers**, inputs must also:
- Have the **same spatial extent and resolution** as the fuel layers
**Units:**
- Wind speed: km/h
- Wind direction: degrees (0–360, meteorological convention)
- FFMC: dimensionless
- BUI: dimensionless

```python
    'wind_speeds_list': ws_list,  # wind speed
    'wind_dirs_list': wd_list,    # wind direction
    'ffmc_list': ffmc_list,       # Fine Fuel Moisture Code
    'bui_list': bui_list,         # Buildup Index
```
> **Note:** To generate wind grids using **WindNinja**, the input DEM extent must be **rectangular**, regardless of the shape of the area of interest.
>
> For use in **PYroCell**, wind speed grids can be scaled **outside of PYroCell** by multiplying the WindNinja wind speed grid by the wind speed from the weather stream, divided by **10**. The divisor of 10 corresponds to the reference wind height used in WindNinja (e.g., a 10 m wind height parameter). In WindNinja, users must specify the height of the input domain averaged wind speed and direction.

***
## Model Input Parameters

Users can specify the following parameters:

- num_time_steps
- time_step
- interval_mins
- process_buffer
- ellipse_breadth
- use_breaching
- breach_type


***
### Number of Time Steps (**num_time_steps**)
- The total number of iterations the model runs to grow one fire

### Time Step Duration (**time_step**)

- Length (in minutes) of each simulation interval
- Defines how often inputs are updated (e.g., weather)
- Default: 60 minutes

### Interval duration (**interval_mins**)

- Time interval in minutes for arrival time contours
- Default: 60 minutes

### Total Run Time

**Total run time** is the total length of simulated fire spread (in minutes). It is calculated as:

> **Total run time = num_time_steps × time_step**

***

### Modeling Buffer (**process_buffer**)

A buffered window around ignition sources and in-simulation burn perimeters used to:

- Subset input data
- Reduce computer memory consumption
- Reduce processing time

A new buffer is applied at the beginning of each time step.

***

### Ellipse Breadth (**ellipse_breadth**)

The width (in number of cells) of the ellipse used to select neighboring target cells for fire spread.

- Larger breadth values:
  - Increase accuracy of fire polygon representation
  - Increase processing time

***

### Breaching Type (**breach_type**)

When breaching is enabled (`use_breaching = True`), three options are available:

- `default`
- `prometheus`
- `simple`

For **default** and **prometheus** breaching types, breaching occurs if the contiguous length of the non-fuel barrier is less than or equal to the **maximum breaching distance**, calculated as:

> **1.5 × flame length** of the fire attempting to breach adjacent non-fuel cells
> (Alexander et al. 2004)

#### 1. Default
- Breaching enabled for all possible spread directions
- Flame length modeling:
  - Surface and intermittent/passive crown fire:
    - Byram (1959) surface fire flame length equation
  - Active/continuous crown fire:
    - Butler et al. (2004) flame length equation
    - Calculated flame length is added to stand height

#### 2. Prometheus
- Breaching enabled for all possible spread directions
- Flame length modeling:
  - Surface and intermittent/passive crown fire:
    - Byram (1959) surface fire flame length equation
  - Active/continuous crown fire:
    - Flame length calculated as **2.5 × stand height**

#### 3. Simple
- Breaching enabled for **cell corners only**
- If two fuel cells are kitty-corner to each other and separated by two kitty-corner non-fuel cells (breaching distance = 0), fire is allowed to pass through the corner regardless of flame length

***

### Breaching Disabled

When breaching is disabled (`use_breaching = False`):

- Fires are **not allowed** to pass through non-fuel cells
- This includes kitty-corner fuel cells separated by kitty-corner non-fuel cells (breaching distance = 0)

This approach assumes non-fuel cells are connected at the corners.

### Assumptions and Constraints

- All spatial inputs must share the same:
  - Coordinate reference system (CRS)
  - Spatial extent
  - Cell size (resolution)
- Fuel, topography, and spatial weather layers must be perfectly aligned
- Fuel type rasters must contain **integer class codes**, not string labels
- Weather inputs must align temporally with the model time step


## Run PYroCell

> Note: The PYroCell model is instantiated via the `PYRO` class in the API.

```python

    # The fire behavior model kwargs
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
        'gpu_float_dtype': cp.float32
    }

    # Run PYro Cell
    pyro_model = pc.PYRO(
        fb_plugin='cffbps',
        ignition_path=ign_path,
        fire_date=wx_date,
        elevation_path=elv_path,
        slope_path=slope_path,
        aspect_path=aspect_path,
        out_folder=results_folder,
        out_request=out_request,
        fire_id=fire_id,
        ellipse_breadth=ellipse_breadth,
        process_buffer=process_buffer,
        use_accel=use_accel,
        float_dtype=float_dtype,
        use_jit=use_jit,
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
```
An example of how to run batch processing is provided in `Testing/pyro_cell_batch_testing.py`.

## Model Output Variables

These are all the available output options available in PYroCell:

```python
out_request = [
    'perim', 'cont_line', 'cont_poly', 'flow_paths', 'major_flow_paths',
    'tt', 'ros', 'fi', 'fire_type', 'fuel_type', 'ws', 'wd', 'm', 'fF',
    'fW', 'ffmc', 'bui', 'isi', 'a', 'b', 'c', 'rsz', 'sf', 'rsf', 'isf',
    'rsi', 'wse1', 'wse2', 'wse', 'wsx', 'wsy', 'wsv', 'raz', 'q', 'bui0',
    'be', 'be_max', 'ffc', 'wfc', 'sfc', 'latn', 'dj', 'd0', 'nd',
    'fmc', 'fme', 'csfi', 'rso', 'bfw', 'bisi', 'bros', 'sros', 'cros',
    'cbh', 'cfb', 'cfl', 'cfc', 'tfc', 'accel', 'fi_class'
]
```


### Final fire and fire behavior descriptor variables
- perim = Final burn perimeter
- cont_line, cont_poly = Arrival time contours in line and polygon formats
- flow_paths = Path of fire travel through all burned cells.
- major_flow_paths = Primary paths of fire spread through burned cells. Defined as paths with a Stahler stream order >= 3 (where three or more smaller upstream paths have joined together).
- ros = Rate of spread associated with the shortest travel time in each cell (m/min)
- fi = The fire intensity value associated with the shortest travel time in each cell (kW/m)
- fire_type = Type of fire predicted to occur (surface = 1, intermittent crown = 2, active crown = 3)

### Weather variables
- ws = Observed wind speed (km/h)
- wd = Wind azimuth/direction (degrees)
- m = Moisture content equivalent of the FFMC (%, value from 0-100+)
- fF = Fine fuel moisture function in the ISI
- fW = Wind function in the ISI
- isi = Final ISI, accounting for wind and slope

### Slope + wind effect variables
- a = Rate of spread equation coefficient
- b = Rate of spread equation coefficient
- c = Rate of spread equation coefficient
- RSZ = Surface spread rate with zero wind on level terrain (m/min)
- SF = Slope factor
- RSF = spread rate with zero wind, upslope (m/min)
- ISF = ISI, with zero wind upslope
- RSI = Initial spread rate without BUI effect (m/min)
- WSE1 = Original slope equivalent wind speed value
- WSE2 = New slope equivalent wind speed value for cases where WSE1 > 40 (capped at max of 112.45)
- WSE = Slope equivalent wind speed
- WSX = Net vectorized wind speed in the x-direction
- WSY = Net vectorized wind speed in the y-direction
- WSV = (aka: slope-adjusted wind speed) Net vectorized wind speed (km/h)
- RAZ = (aka: slope-adjusted wind direction) Net vectorized wind direction (degrees)

### BUI effect variables
- q = Proportion of maximum rate of spread at BUI equal to 50
- bui0 = Average BUI for each fuel type
- BE = Buildup effect on spread rate
- be_max = Maximum allowable BE value

### Surface fuel variables
- ffc = Estimated forest floor consumption (kg/m^2)
- wfc = Estimated woody fuel consumption (kg/m^2)
- sfc = Estimated total surface fuel consumption (kg/m^2)

### Foliar moisture content variables
- latn = Normalized latitude
- d0 = Julian date of minimum foliar moisture content
- nd = number of days between modelled fire date and d0
- fmc = foliar moisture content
- fme = foliar moisture effect

### Backing fire rate of spread variables
- bfW = backing fire wind speed component (km/h)
- brsi = backing fire spread rate without BUI effect (m/min)
- bisi = backing fire ISI without BUI effect
- bros = backing fire rate of spread (m/min)

### Critical crown fire threshold variables
csfi = critical intensity (kW/m)
rso = critical rate of spread (m/min)

### Crown fuel parameters
- cbh = Height to live crown base (m)
- cfb = Crown fraction burned (proportion, value ranging from 0-1)
- cfl = Crown fuel load (kg/m^2)
- cfc = Crown fuel consumed (kg/m^2)

### Final fuel parameters
- tfc = Total fuel consumed (kg/m^2)

### Acceleration parameter
- accel = Acceleration parameter for point source ignition

### Fire Intensity Class parameter
- fi_class = Fire intensity class (1-6)
