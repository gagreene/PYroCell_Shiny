#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from . import _arrival_contours as acont
from . import _flow_paths as fpaths


# Declare global variables
suppress_msgs = False


def process_arrival_contours(
        out_request: list,
        interval_mins: int,
        total_run_time: float,
        crs,
        process_transform,
        process_grid_shape: tuple,
        cell_width: float,
        cell_height: float,
        ignition_type: str,
        ignition_geom,
        land_cover: np.ndarray,
        travel_times: np.ndarray,
        fire_id: str,
        out_folder: str,
        remove_poly_ign: bool,
        suppress_messages: bool
    ) -> None:
    """
    Process arrival contours from travel time data.

    :param out_request: List of requested output types.
    :param interval_mins: Time interval in minutes for contour generation.
    :param total_run_time: Total run time in minutes.
    :param crs: Coordinate reference system.
    :param process_transform: Affine transform for the process grid.
    :param process_grid_shape: Shape of the process grid (rows, cols).
    :param cell_width: Cell width of the raster.
    :param cell_height: Cell height of the raster.
    :param ignition_type: Type of ignition ("point" or "polygon").
    :param ignition_geom: Geometry of the ignition.
    :param land_cover: 2D numpy array of land cover types.
    :param travel_times: 2D numpy array of travel times.
    :param fire_id: Unique identifier for the fire.
    :param out_folder: Output folder for saving results.
    :param remove_poly_ign: Whether to remove polygon ignition areas from contours.
    :param suppress_messages: Whether to suppress messages during processing.
    :return: None
    """
    acont.gen_arrival_contours(
        out_request=out_request,
        interval_mins=interval_mins,
        total_run_time=total_run_time,
        crs=crs,
        process_transform=process_transform,
        process_grid_shape=process_grid_shape,
        cell_width=cell_width,
        cell_height=cell_height,
        ignition_type=ignition_type,
        ignition_geom=ignition_geom,
        land_cover=land_cover,
        travel_times=travel_times,
        fire_id=fire_id,
        out_folder=out_folder,
        remove_poly_ign=remove_poly_ign,
        suppress_messages=suppress_messages
    )

    return


def process_flow_paths(
        flow_path_request: list,
        travel_times_path: str,
        flow_paths_out: str,
        major_flow_paths_out: str,
        temp_folder: str,
        suppress_messages: bool
    ) -> tuple[bool, bool]:
    """
    Process flow paths from elevation data within a specified window.

    :param flow_path_request: List of requested flow path outputs (e.g., ['flow_paths', 'major_flow_paths']).
    :param travel_times_path: Path to the elevation raster file.
    :param flow_paths_out: Path to save the flow paths shapefile.
    :param major_flow_paths_out: Path to save the major flow paths shapefile.
    :param temp_folder: Temporary folder for intermediate files.
    :param suppress_messages: Whether to suppress messages during processing.
    :return: None
    """
    global suppress_msgs
    suppress_msgs = suppress_messages

    # Define paths for intermediate and output files
    fdir = os.path.join(temp_folder, 'flow_dir_d8.tif')
    acc = os.path.join(temp_folder, 'flow_acc.tif')

    # Compute flow direction
    fpaths.compute_d8_flow_direction(
        elev_path=travel_times_path,
        out_fdir_path=fdir,
        suppress_messages=suppress_messages
    )
    if not os.path.exists(fdir):
        return False, False

    # Compute flow accumulation
    fpaths.flow_accumulation_from_direction(
        fdir_path=fdir,
        out_acc_path=acc
    )
    if not os.path.exists(acc):
        return False, False

    # Extract stream segments as flow paths
    fpaths.extract_stream_segments_from_acc(
        elev_path=travel_times_path,
        fdir_path=fdir,
        acc_path=acc,
        acc_threshold=0,
        out_shp=flow_paths_out
    )
    if not os.path.exists(flow_paths_out):
        return False, False

    # If major flow paths are requested, assign Strahler order and filter
    if 'major_flow_paths' in flow_path_request:
        fpaths.assign_strahler_order(
            flowpaths_path=flow_paths_out,
            out_path=major_flow_paths_out,
            stahler_cutoff=3,
            coord_decimals=3
        )
        if not os.path.exists(major_flow_paths_out):
            return True, False

    # Clean up intermediate files
    os.remove(fdir)
    os.remove(acc)
    if 'flow_paths' not in flow_path_request:
        os.remove(flow_paths_out)

    return True, True
