import os
import numpy as np
import rasterio
import fiona
import shapely


OUTPUT_RASTERS = [
    "cfb.tif",
    "fire_type.tif",
    "hfi.tif",
    "raz.tif",
    "ros.tif",
    "sfc.tif",
    "travel_times.tif",
    "wsv.tif",
]

OUTPUT_SHAPEFILES = [
    "arrival_contours.shp",
    "burn_perimeter.shp",
]


class PyrocellOutputLocation:
    def __init__(self, root, folder, prefix):
        self.folder = folder
        self.prefix = prefix
        self.root = root

    def __repr__(self):
        return f"PyrocellOutputLocation(folder={self.folder}, prefix={self.prefix})"

    def abs_path_to_file(self, filename):
        return os.path.join(self.root, self.folder, f"{self.prefix}_{filename}")


def assert_similar_rasters(
    test_loc,
    golden_loc,
    filename,
    rtol=1e-05,
    atol=1e-08,
    raster_pixel_tol=1e-04,
):
    test_path = test_loc.abs_path_to_file(filename)
    golden_path = golden_loc.abs_path_to_file(filename)
    try:
        with rasterio.open(test_path) as test_raster:
            try:
                with rasterio.open(golden_path) as golden_raster:
                    test_array = test_raster.read()
                    golden_array = golden_raster.read()
                    # Calculate the number of differing pixels
                    differing_pixels = np.sum(~np.isclose(test_array, golden_array, rtol=rtol, atol=atol))
                    total_pixels = test_array.size

                    # Calculate the proportion of differing pixels
                    proportion_differing = differing_pixels / total_pixels
                    assert proportion_differing <= raster_pixel_tol
            except IOError as e:
                raise AssertionError(f"Failed to open golden raster file '{golden_path}': {e}")
    except IOError as e:
        raise AssertionError(f"Failed to open test raster file '{test_path}': {e}")


def assert_similar_polygons(
    test_loc,
    golden_loc,
    filename,
    tolerance=1e-3,
):
    test_path = test_loc.abs_path_to_file(filename)
    golden_path = golden_loc.abs_path_to_file(filename)
    try:
        with fiona.open(test_path) as test_shapefile:
            try:
                with fiona.open(golden_path) as golden_shapefile:
                    assert test_shapefile.schema == golden_shapefile.schema
                    assert len(test_shapefile) == len(golden_shapefile)

                    for test_feature, golden_feature in zip(test_shapefile, golden_shapefile):
                        test_shape = shapely.geometry.shape(test_feature["geometry"])
                        golden_shape = shapely.geometry.shape(golden_feature["geometry"])

                        shape_difference = shapely.symmetric_difference(test_shape, golden_shape)

                        # Check if the area of the shape difference is within the tolerance
                        # The inputs currently should only be polygons so this should be fine
                        assert shapely.area(shape_difference) < tolerance

            except IOError as e:
                raise AssertionError(f"Failed to open golden shapefile '{golden_path}': {e}")
    except IOError as e:
        raise AssertionError(f"Failed to open test shapefile '{test_path}': {e}")


def assert_equal_results(
    test_loc: PyrocellOutputLocation,
    golden_loc: PyrocellOutputLocation,
    raster_rtol=1e-05,
    raster_atol=1e-08,
    raster_pixel_tol=1e-04,
    shapefile_area_tol=1e-3,
):
    # Check if the rasters are similar
    for raster in OUTPUT_RASTERS:
        assert_similar_rasters(
            test_loc,
            golden_loc,
            raster,
            rtol=raster_rtol,
            atol=raster_atol,
            raster_pixel_tol=raster_pixel_tol,
        )

    # Check if shapefiles are similar
    for shapefile in OUTPUT_SHAPEFILES:
        assert_similar_polygons(
            test_loc,
            golden_loc,
            shapefile,
            tolerance=shapefile_area_tol,
        )
