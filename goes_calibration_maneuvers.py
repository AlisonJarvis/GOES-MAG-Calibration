# Imports
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from utils import *
import deepdish as dd
from glob import glob
import netCDF4 as nc
import os
import csv
import numpy as np
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from lmfit import Model, Parameters

########################################################## MAIN CLASS ####################################################################

# PLT-001 Calibration Maneuver Class
class CalibrationManeuver:
    def __init__(self, l1b_file_path: str, lut_path: str, time_constraints = None):
        '''
        :param l1b_file_path: path to the directory containing the l1b files from the calibration maneuver
        :param lut_path: path to the calibration file used to extract certain transformation information
        '''
        self.data_directory = l1b_file_path # Directory used for maneuver data
        self.lut_directory = lut_path # Directory used for lut
        self.maneuver_data = self.parse_l1b_files(l1b_file_path, lut_path) # Combined data from maneuver
        if time_constraints is not None:
            self.maneuver_data = shorten_time_frame(self.maneuver_data, time_constraints)

    def parse_l1b_files(self, l1b_file_directory: str, lut_file_path: str, original_interp=False):
        '''
        :return: dictionary (one for inboard and outboard) containing:
                    - field data in sensor frame
                    - field data in epn
                    - sensor frame to epn transformation matrix
                    - times corresponding to time series data
        '''

        # Parse information from the LUT
        LUT_info = dd.io.load(lut_file_path)['MAG_CAL_INR']
        Transform_boom_to_acrf = LUT_info['boom_to_acrf_trans_matrix']
        Transform_MFOB_to_boom = LUT_info['mfob_to_boom_trans_matrix']
        Transform_MFIB_to_boom = LUT_info['mfib_to_boom_trans_matrix']
        Transform_ORF_to_EPN = LUT_info['orf_to_epn_trans_matrix']

        # Parse information from the files in the l1b directory
        mag_l1b_data_filenames = sorted(glob(os.path.join(l1b_file_directory, '*.nc')))

        # Assign initial variables
        ib_field_data_in_mfib = []
        ob_field_data_in_mfob = []
        ib_field_data_in_epn = []
        ob_field_data_in_epn = []
        ib_data_times = []
        ob_data_times = []
        quaternions_for_acrf_to_eci_transform_i = []
        quaternions_for_orf_to_eci_transform_i = []

        # Iterate through l1b files and read
        for i, l1b_filename in enumerate(mag_l1b_data_filenames):
            # NC dataset of file being read
            l1b_file = nc.Dataset(l1b_filename)

            # FIELD DATA TIMES
            # Read in from l1b file
            IB_data_times_raw_format_current_file = l1b_file['IB_time'][:].data  # 'seconds since 2000-01-01 12:00:00'
            OB_data_times_raw_format_current_file = l1b_file['OB_time'][:].data  # 'seconds since 2000-01-01 12:00:00'

            # Get shape of time data
            IB_field_time_shape = np.shape(IB_data_times_raw_format_current_file)
            OB_field_time_shape = np.shape(OB_data_times_raw_format_current_file)

            # Flatten to 1d array
            if IB_data_times_raw_format_current_file.ndim > 1:
                IB_data_times_current_file_1D = np.reshape(IB_data_times_raw_format_current_file, (
                            IB_field_time_shape[0] * IB_field_time_shape[1]))  # re-format into 1-D matrix
            else:
                IB_data_times_current_file_1D = IB_data_times_raw_format_current_file
            if OB_data_times_raw_format_current_file.ndim > 1:
                OB_data_times_current_file_1D = np.reshape(OB_data_times_raw_format_current_file, (
                            OB_field_time_shape[0] * OB_field_time_shape[1]))  # re-format into 1-D matrix
            else:
                OB_data_times_current_file_1D = OB_data_times_raw_format_current_file

            # Convert to datetime for each time stamp
            IB_data_times_current_file = [time_convert(IB_date) for IB_date in IB_data_times_current_file_1D]
            OB_data_times_current_file = [time_convert(OB_date) for OB_date in OB_data_times_current_file_1D]

            # Get data into overall variable
            ib_data_times_stacked = IB_data_times_current_file
            ob_data_times_stacked = OB_data_times_current_file
            ib_data_times.extend(ib_data_times_stacked)
            ob_data_times.extend(ob_data_times_stacked)

            # Original Quaternion Interpolation Method
            if original_interp:
                # QUATERNION TIMES
                if OB_data_times_raw_format_current_file.ndim > 1:
                    quaternion_timestamp_raw_format_current_file = OB_data_times_raw_format_current_file[:, 0]
                else:
                    quaternion_timestamp_raw_format_current_file = l1b_file['quat_timestamp'][:].data

                # ATTITUDE QUATERNION
                # Get attitude data from l1b and interpolate to match field data shape (60x1 to 600x1)
                f_acrf_0 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['attitude_quat_Q0'], 'previous',
                                    fill_value='extrapolate')
                f_acrf_1 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['attitude_quat_Q1'], 'previous',
                                    fill_value='extrapolate')
                f_acrf_2 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['attitude_quat_Q2'], 'previous',
                                    fill_value='extrapolate')
                f_acrf_3 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['attitude_quat_Q3'], 'previous',
                                    fill_value='extrapolate')
                # Add into a quaternion matrix - all quaternion data combined for file
                quaternions_for_acrf_to_eci_transform_i_tmp = np.stack([f_acrf_0(OB_data_times_current_file_1D),
                                                                        f_acrf_1(OB_data_times_current_file_1D),
                                                                        f_acrf_2(OB_data_times_current_file_1D),
                                                                        f_acrf_3(OB_data_times_current_file_1D)], axis=1)
                # Get attitude quaternion data into net variable
                quaternions_for_acrf_to_eci_transform_i.extend(quaternions_for_acrf_to_eci_transform_i_tmp)

                # ORBITAL QUATERNION
                # Get orbital data from l1b and interpolate to match field data shape (60x1 to 600x1)
                f_orf_0 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['orbit_quat_Q0'], 'previous',
                                   fill_value='extrapolate')
                f_orf_1 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['orbit_quat_Q1'], 'previous',
                                   fill_value='extrapolate')
                f_orf_2 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['orbit_quat_Q2'], 'previous',
                                   fill_value='extrapolate')
                f_orf_3 = interp1d(quaternion_timestamp_raw_format_current_file, l1b_file['orbit_quat_Q3'], 'previous',
                                   fill_value='extrapolate')
                # Add into a quaternion matrix - all quaternion data combined for file
                quaternions_for_orf_to_eci_transform_i_tmp = np.stack([f_orf_0(OB_data_times_current_file_1D),
                                                                       f_orf_1(OB_data_times_current_file_1D),
                                                                       f_orf_2(OB_data_times_current_file_1D),
                                                                       f_orf_3(OB_data_times_current_file_1D)], axis=1)
                # Get attitude quaternion data into net variable
                quaternions_for_orf_to_eci_transform_i.extend(quaternions_for_orf_to_eci_transform_i_tmp)
            # VERSON 6 CHANGES: New interpolation method courtesy of Derrick
            else:
                # QUATERNION TIMES
                if OB_data_times_raw_format_current_file.ndim > 1:
                    quaternion_timestamp_raw_format_current_file = OB_data_times_raw_format_current_file[:, 0]
                else:
                    quaternion_timestamp_raw_format_current_file = l1b_file['quat_timestamp'][:].data

                # Add into a quaternion matrix - all quaternion data combined for file
                att_quat_stacked = np.stack([l1b_file['attitude_quat_Q0'],
                                             l1b_file['attitude_quat_Q1'],
                                             l1b_file['attitude_quat_Q2'],
                                             l1b_file['attitude_quat_Q3']], axis=1)

                quaternions_for_acrf_to_eci_transform_i_tmp = quantinterp(quaternion_timestamp_raw_format_current_file,
                                                                          att_quat_stacked, OB_data_times_current_file_1D)
                # Get attitude quaternion data into net variable
                quaternions_for_acrf_to_eci_transform_i.extend(quaternions_for_acrf_to_eci_transform_i_tmp)

                # ORBITAL QUATERNION
                # Add into a quaternion matrix - all quaternion data combined for file
                orbit_quat_stacked = np.stack([l1b_file['orbit_quat_Q0'],
                                             l1b_file['orbit_quat_Q1'],
                                             l1b_file['orbit_quat_Q2'],
                                             l1b_file['orbit_quat_Q3']], axis=1)

                quaternions_for_orf_to_eci_transform_i_tmp = quantinterp(quaternion_timestamp_raw_format_current_file,
                                                                          orbit_quat_stacked, OB_data_times_current_file_1D)
                # Get attitude quaternion data into net variable
                quaternions_for_orf_to_eci_transform_i.extend(quaternions_for_orf_to_eci_transform_i_tmp)

            # FIELD DATA in sensor frame
            # IB/OB data in sensor frame - field data
            IB_data_in_MFIB_current_file_native_dimensions = l1b_file['IB_data']  # MFIB is the coordinate frame for the IB sensor
            OB_data_in_MFOB_current_file_native_dimensions = l1b_file['OB_data']  # MFOB is the coordinate frame for the OB sensor

            # Get shape of field data
            IB_field_data_shape = np.shape(IB_data_in_MFIB_current_file_native_dimensions)
            OB_field_data_shape = np.shape(OB_data_in_MFOB_current_file_native_dimensions)

            # Reshape the field data
            if IB_data_in_MFIB_current_file_native_dimensions.ndim > 2:
                IB_data_in_MFIB_current_file = np.reshape(IB_data_in_MFIB_current_file_native_dimensions, (
                IB_field_data_shape[0] * IB_field_data_shape[1], IB_field_data_shape[2]))  # re-format into a 3xN matrix
            else:
                IB_data_in_MFIB_current_file = IB_data_in_MFIB_current_file_native_dimensions
            if OB_data_in_MFOB_current_file_native_dimensions.ndim > 2:
                OB_data_in_MFOB_current_file = np.reshape(OB_data_in_MFOB_current_file_native_dimensions, (
                OB_field_data_shape[0] * OB_field_data_shape[1], OB_field_data_shape[2]))  # re-format into a 3xN matrix
            else:
                OB_data_in_MFOB_current_file = OB_data_in_MFOB_current_file_native_dimensions

            # Add current l1b field data to the net field data in that frame
            ib_field_data_in_mfib.extend(IB_data_in_MFIB_current_file)
            ob_field_data_in_mfob.extend(OB_data_in_MFOB_current_file)

            # FIELD DATA IN EPN
            IB_data_in_EPN_current_file_native_dimensions = l1b_file['IB_mag_EPN']
            OB_data_in_EPN_current_file_native_dimensions = l1b_file['OB_mag_EPN']

            # Get shape of field data in epn
            IB_field_data_shape_epn = np.shape(IB_data_in_EPN_current_file_native_dimensions)
            OB_field_data_shape_epn = np.shape(OB_data_in_EPN_current_file_native_dimensions)

            # Reshape the field data in epn
            if IB_data_in_EPN_current_file_native_dimensions.ndim > 2:
                IB_data_in_MFIB_current_file_epn = np.reshape(IB_data_in_EPN_current_file_native_dimensions, (
                IB_field_data_shape_epn[0] * IB_field_data_shape_epn[1],
                IB_field_data_shape_epn[2]))  # re-format into a 3xN matrix
            else:
                IB_data_in_MFIB_current_file_epn = IB_data_in_EPN_current_file_native_dimensions
            if OB_data_in_EPN_current_file_native_dimensions.ndim > 2:
                OB_data_in_MFOB_current_file_epn = np.reshape(OB_data_in_EPN_current_file_native_dimensions, (
                OB_field_data_shape_epn[0] * OB_field_data_shape_epn[1],
                OB_field_data_shape_epn[2]))  # re-format into a 3xN matrix
            else:
                OB_data_in_MFOB_current_file_epn = OB_data_in_EPN_current_file_native_dimensions

            # Add current l1b field data to the net field data in that frame
            ib_field_data_in_epn.extend(IB_data_in_MFIB_current_file_epn)
            ob_field_data_in_epn.extend(OB_data_in_MFOB_current_file_epn)

        # Get EPN to sensor frame and sensor frame to EPN transformation matrices
        # Outboard
        transform_EPN_to_MFOB = []
        transform_MFOB_to_EPN = []
        # Iterate through each time and add to MFOB to EPN matrix
        for ob_index, ob_data_time in enumerate(ob_data_times):
            A = Transform_MFOB_to_boom.T
            B = Transform_boom_to_acrf.T
            C = quaternion_to_rotation(np.asarray(quaternions_for_acrf_to_eci_transform_i)[ob_index, :])
            D = np.asarray(quaternion_to_rotation(np.asarray(quaternions_for_orf_to_eci_transform_i)[ob_index, :])).T
            E = Transform_ORF_to_EPN.T
            MFOB_to_EPN_index = np.matmul(np.matmul(np.matmul(np.matmul(A, B), C), D), E)
            EPN_to_MFOB_index = MFOB_to_EPN_index.T
            transform_MFOB_to_EPN.extend(MFOB_to_EPN_index)
            transform_EPN_to_MFOB.extend(EPN_to_MFOB_index)
        # Inboard
        transform_EPN_to_MFIB = []
        transform_MFIB_to_EPN = []
        # Iterate through each time and add to MFIB to EPN matrix
        for ib_index, ib_data_time in enumerate(ib_data_times):
            A = Transform_MFIB_to_boom.T
            B = Transform_boom_to_acrf.T
            C = quaternion_to_rotation(np.asarray(quaternions_for_acrf_to_eci_transform_i)[ib_index, :])
            D = np.asarray(quaternion_to_rotation(np.asarray(quaternions_for_orf_to_eci_transform_i)[ib_index, :])).T
            E = Transform_ORF_to_EPN.T
            MFIB_to_EPN_index = np.matmul(np.matmul(np.matmul(np.matmul(A, B), C), D), E)
            EPN_to_MFIB_index = MFIB_to_EPN_index.T
            transform_MFIB_to_EPN.extend(MFIB_to_EPN_index)
            transform_EPN_to_MFIB.extend(EPN_to_MFIB_index)

        # Store values in dictionary of data to return
        fit_data = dict()
        fit_data['IB'] = dict()
        fit_data['OB'] = dict()

        fit_data['IB']['transform_EPN_to_sensor_frame'] = transform_EPN_to_MFIB
        fit_data['IB']['transform_sensor_frame_to_EPN'] = transform_MFIB_to_EPN
        fit_data['IB']['field_data_in_sensor_frame'] = ib_field_data_in_mfib
        fit_data['IB']['field_data_in_epn'] = ib_field_data_in_epn
        fit_data['IB']['data_times'] = ib_data_times
        fit_data['IB']['initial_guess_for_background_field'] = np.mean(ib_field_data_in_epn, axis=0)

        fit_data['OB']['transform_EPN_to_sensor_frame'] = transform_EPN_to_MFOB
        fit_data['OB']['transform_sensor_frame_to_EPN'] = transform_MFOB_to_EPN
        fit_data['OB']['field_data_in_sensor_frame'] = ob_field_data_in_mfob
        fit_data['OB']['field_data_in_epn'] = ob_field_data_in_epn
        fit_data['OB']['data_times'] = ob_data_times
        fit_data['OB']['initial_guess_for_background_field'] = np.mean(ob_field_data_in_epn, axis=0)

        return fit_data

    ############ TBD ############
    def save(self, save_path: str):
        '''
        :param save_path: path to save the object to
        :return: nothing, saves the calibration maneuver object to a pickle file
        '''

        cal_maneuver = dict()
        cal_maneuver['data_directory'] = self.data_directory
        cal_maneuver['lut_directory'] = self.lut_directory
        cal_maneuver['maneuver_data'] = self.maneuver_data
        pickle.dump(cal_maneuver, open(save_path, 'wb'))
    ############ TBD #############


    def get_bias_alignment_spline(self, sensor: str, number_of_spline_segments: int):
        '''
        :param sensor: whether to get the parameters for inboard or outboard
        :param number_of_spline_segments: number of spline segments to use for interpolation
        :return: dictionary containing bias, alignment, and estimated field values for each spline segment
        '''

        # Define dictionary of bias and alignment to return
        bias_align = dict()

        # Extract maneuver data to be used in least squares fit
        data_times = self.maneuver_data[sensor]['data_times']
        field_data_in_sensor_frame = self.maneuver_data[sensor]['field_data_in_sensor_frame']
        transform_EPN_to_sensor_frame = self.maneuver_data[sensor]['transform_EPN_to_sensor_frame']
        initial_guess_for_background_field = self.maneuver_data[sensor]['initial_guess_for_background_field']

        # Get the segment data times for the number of spline segments and convert to seconds
        seconds_per_segment = ((data_times[-1] - data_times[0])).seconds / number_of_spline_segments
        segment_time_array = [data_times[0] + timedelta(seconds=seconds_per_segment * n) for n in
                              range(number_of_spline_segments + 1)]
        data_times_sec = [time_convert_rev(date_data) for date_data in data_times]
        segment_time_array_sec = [time_convert_rev(seg_time) for seg_time in segment_time_array]

        # Define function for least squares fit that utilizes estimated field generation function
        def estimated_field_func_for_lsq_fit(x, **params):
            fit_params = []
            for param_key in params.keys():
                fit_params.append(params[param_key])
            return generate_estimated_measurement(fit_params, x, segment_time_array_sec, transform_EPN_to_sensor_frame)

        # Define Model object for the customized function
        lsq_model = Model(estimated_field_func_for_lsq_fit)

        # Define values to initialize least squares fit based on # of spline segments
        values_to_initialize_lsq_fit = np.concatenate(
            (np.zeros((6,)), np.tile(np.asarray(initial_guess_for_background_field).T, len(segment_time_array_sec))))

        # Turn values to initialize into parameter object (necessary to pass to function call)
        params = Parameters()
        for i in range(len(values_to_initialize_lsq_fit)):
            coeff_name = 'C' + str(i + 1)
            params.add(coeff_name, value=values_to_initialize_lsq_fit[i])

        # Use the model fit function to get parameters
        fit_parameters = lsq_model.fit(field_data_in_sensor_frame, params, x=data_times_sec)

        # Extract bias and alignment from the fit parameter output and store in dictionary
        bias = [fit_parameters.params['C1'].value, fit_parameters.params['C2'].value, fit_parameters.params['C3'].value]
        align = [fit_parameters.params['C4'].value, fit_parameters.params['C5'].value,
                 fit_parameters.params['C6'].value]
        bias_align['bias'] = bias
        bias_align['alignment'] = align

        # Extract spline field epn estimates from fit parameters and store in dictionary
        spline_field_est = []
        for i in range(6, len(fit_parameters.params)):
            param_str = 'C' + str(i + 1)
            spline_field_est.append(fit_parameters.params[param_str].value)
        bias_align['spline_field_est'] = spline_field_est

        # Return the dictionary with bias and alignment
        return bias_align

    def write_csv_file_spline(self, sensor: str, spline_segment_range: Tuple[int], csv_save_path: str, return_vals: Optional[bool] = False):
        '''
        :param sensor: sensor to create text file for (IB or OB)
        :param spline_segment_range: range of number of spline segments to calculate bias and alignment for
        :param csv_save_path: path to save the text file to
        :param return_vals: whether to return dictionary of bias, alignment, and field estimate per spline segment
        :return: nominally nothing - saves the text file to the designated path, but can return dictionary of values
        '''

        # Define dictionary to store values in
        spline_bias_align = dict()
        # Iterate through range of fit segments and get bias and alignment
        for number_of_fit_segments in range(spline_segment_range[0], spline_segment_range[1] + 1):
            bias_align = self.get_bias_alignment_spline(sensor, number_of_fit_segments)
            spline_bias_align[number_of_fit_segments] = bias_align

        # Write a csv file for this dictionary
        with open(csv_save_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['nseg', 'biasx', 'biasy', 'biasz', 'align_q1', 'align_q2', 'align_q3'])
            for spline_key in spline_bias_align.keys():
                row = np.concatenate((np.concatenate(([spline_key], spline_bias_align[spline_key]['bias'])),
                                      spline_bias_align[spline_key]['alignment']))
                csv_writer.writerow(row)

        # Return values if desired
        if return_vals:
            return spline_bias_align

    def get_bias_alignment_junoish(self, sensor: str, lag_time_seconds: int):
        '''
        :param sensor: whether to get the parameters for inboard (IB) or outboard (OB)
        :param lag_time_seconds: number of seconds of lag to use for the juno-ish method
        :return: dictionary containing estimated bias and alignment
        '''
    
        # Define dictionary of bias and alignment to return
        bias_align = dict()
    
        # Extract maneuver data to be used in least squares fit
        field_data_in_sensor_frame = self.maneuver_data[sensor]['field_data_in_sensor_frame']
        transform_sensor_frame_to_EPN = self.maneuver_data[sensor]['transform_sensor_frame_to_EPN']
    
        # Get the lag time in terms of indices (specific to 10 Hz data)
        lag_time_num_samples = lag_time_seconds*10
    
        # Create the lagged and non-lagged data
        non_lagged_sensor_data = field_data_in_sensor_frame[:-lag_time_num_samples]
        lagged_sensor_data = field_data_in_sensor_frame[lag_time_num_samples:]
    
        # Create a placeholder "time" to serve as the x value, enumerates all the values along the time dimension
        placeholder_time = np.linspace(1, len(lagged_sensor_data), num=len(lagged_sensor_data))
    
        # Create the lagged and non-lagged transformation matrices
        non_lagged_transform_sensor_frame_to_EPN = transform_sensor_frame_to_EPN[:-(3*lag_time_num_samples)]
        lagged_transform_sensor_frame_to_EPN = transform_sensor_frame_to_EPN[(3*lag_time_num_samples):]
    
        # Create data to fit least squares to - zeros in the size of the lagged / non lagged sensor data
        ideal_field_sample_diff_epn = np.zeros(np.shape(lagged_sensor_data))

        # Define function for least squares fit that utilizes calculate data difference function
        def estimated_diff_func_for_lsq_fit(x, **params):
            fit_params = []
            for param_key in params.keys():
                fit_params.append(params[param_key])
            return calculate_data_difference(fit_params, x, lagged_sensor_data, non_lagged_sensor_data,
                                             lagged_transform_sensor_frame_to_EPN, non_lagged_transform_sensor_frame_to_EPN)
    
        # Define Model object for the customized function
        lsq_model = Model(estimated_diff_func_for_lsq_fit)
    
        # Define values to initialize least squares fit - 3 zeros for bias, 3 zeros for alignment
        values_to_initialize_lsq_fit = np.zeros((6,))
    
        # Turn values to initialize into parameter object (necessary to pass to function call)
        params = Parameters()
        for i in range(len(values_to_initialize_lsq_fit)):
            coeff_name = 'C' + str(i + 1)
            params.add(coeff_name, value=values_to_initialize_lsq_fit[i])
    
        # Use the model fit function to get parameters
        fit_parameters = lsq_model.fit(ideal_field_sample_diff_epn, params, x=placeholder_time)
    
        # Extract bias and alignment from the fit parameter output and store in dictionary
        bias = [fit_parameters.params['C1'].value, fit_parameters.params['C2'].value, fit_parameters.params['C3'].value]
        align = [fit_parameters.params['C4'].value, fit_parameters.params['C5'].value,
                 fit_parameters.params['C6'].value]
        bias_align['bias'] = bias
        bias_align['alignment'] = align
    
        return bias_align

    def write_csv_file_junoish(self, sensor: str, lag_increment: int, number_of_increments: int, csv_save_path: str, return_vals: Optional[bool] = False):
        '''
        :param sensor: sensor to create text file for (IB or OB)
        :param lag_increment: number of seconds added to the lag for each iteration
        :param number_of_increments: total number of iterations for which the lag is increased
        :param csv_save_path: path to save the text file to
        :param return_vals: whether to return dictionary of bias and alignment
        :return: nominally nothing - saves the text file to the designated path, but can return dictionary of values
        '''
    
        # Define dictionary to store values in
        junoish_bias_align = dict()
        # Iterate through range of lag times and get bias and alignment
        lag_time = lag_increment
        while lag_time <= lag_increment*number_of_increments:
            bias_align = self.get_bias_alignment_junoish(sensor, lag_time)
            junoish_bias_align[lag_time] = bias_align
            lag_time = lag_time + lag_increment
    
        # Write a csv file for this dictionary
        with open(csv_save_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['nseg', 'biasx', 'biasy', 'biasz', 'align_q1', 'align_q2', 'align_q3'])
            for lag_key in junoish_bias_align.keys():
                row = np.concatenate((np.concatenate(([lag_key], junoish_bias_align[lag_key]['bias'])),
                                      junoish_bias_align[lag_key]['alignment']))
                csv_writer.writerow(row)
    
        # Return values if desired
        if return_vals:
            return junoish_bias_align

    def generate_calibration_plot_spline(self, sensor: str, image_save_path: str, num_spline_segments: int = 10,
                                         frame: str = 'sensor', with_error: bool = True):
        '''
        :param sensor: whether to do this for inboard or outboard
        :param image_save_path: path to save the generated plot to
        :param num_spline_segments: number of spline segments to plot for
        :param frame: whether to plot comparison in sensor frame or in EPN
        :param with_error: plot the error in the estimate on a second subplot
        :return: nothing, saves a plot of the estimated vs actual sensor field data
        '''

        # Get bias, alignment, and spline estimate for the given number of segments
        spline_bias_align = self.get_bias_alignment_spline(sensor, num_spline_segments)
        bias = spline_bias_align['bias']
        align = spline_bias_align['alignment']
        spline_field_est = spline_bias_align['spline_field_est']

        # Combine these into format to get estimated field with function
        input_params_and_data = np.concatenate((np.concatenate((bias, align)), spline_field_est))
        # Get input data times and segment times
        data_times = self.maneuver_data[sensor]['data_times']
        seconds_per_segment = (data_times[-1] - data_times[0]).seconds / num_spline_segments
        segment_time_array = [data_times[0] + timedelta(seconds=seconds_per_segment * n) for n in
                              range(num_spline_segments + 1)]
        data_times_sec = [time_convert_rev(date_data) for date_data in data_times]
        segment_time_array_sec = [time_convert_rev(seg_time) for seg_time in segment_time_array]

        # Get estimated field data in sensor frame
        estimated_field = np.asarray(
            generate_estimated_measurement(input_params_and_data, data_times_sec, segment_time_array_sec,
                                     self.maneuver_data[sensor]['transform_EPN_to_sensor_frame']))
        # Get actual field data in sensor frame
        field_data = np.asarray(self.maneuver_data[sensor]['field_data_in_sensor_frame'])

        # Convert field and estimated data to epn if the comparison frame is epn
        if frame == 'epn':
            transform_sensor_frame_to_EPN_flat = self.maneuver_data[sensor]['transform_sensor_frame_to_EPN']
            transform_sensor_frame_to_EPN = np.reshape(np.asarray(transform_sensor_frame_to_EPN_flat),
                                                       (len(data_times), 3, 3))
            estimated_field = np.einsum('nm,nmb->nb', estimated_field, transform_sensor_frame_to_EPN)
            field_data = np.einsum('nm,nmb->nb', field_data, transform_sensor_frame_to_EPN)

        # Get standard deviation of error
        fit_residual_standard_dev = np.std(field_data - estimated_field, axis=0)

        # Define plotting parameters
        if frame == 'sensor':
            sensor_ylabels = {'OB': 'OB MFOB (nT)', 'IB': 'IB MFIB (nT)'}
            sensor_error_ylabels = {'OB': 'OB Actual - Estimate (MFOB nT)', 'IB': 'IB Actual - Estimate (MFIB nT)'}
            components = ['x', 'y', 'z']
        elif frame == 'epn':
            sensor_ylabels = {'OB': 'OB EPN (nT)', 'IB': 'IB EPN (nT)'}
            sensor_error_ylabels = {'OB': 'OB Actual - Estimate (EPN nT)', 'IB': 'IB Actual - Estimate (EPN nT)'}
            components = ['e', 'p', 'n']

        if with_error:
            # Define plot with two subplots, data and error
            fig, ax = plt.subplots(2, sharex=True, figsize=(15, 10))

            # Plot data on first subplot
            ax[0].plot(data_times, field_data[:, 0])
            ax[0].plot(data_times, field_data[:, 1])
            ax[0].plot(data_times, field_data[:, 2])
            ax[0].plot(data_times, estimated_field[:, 0])
            ax[0].plot(data_times, estimated_field[:, 1])
            ax[0].plot(data_times, estimated_field[:, 2])
            ax[0].set_ylabel(sensor_ylabels[sensor], fontsize=14)
            ax[0].legend([components[0], components[1], components[2], components[0] + ' est',
                          components[1] + ' est', components[2] + ' est'], loc='upper right')
            bias_round = [round_sig(bias[0]), round_sig(bias[1]), round_sig(bias[2])]
            align_round = [round_sig(align[0]), round_sig(align[1]), round_sig(align[2])]
            title_str = sensor + ' bias = ' + str(bias_round) + ';   q_align = ' + str(
                align_round) + ';   nseg = ' + str(round_sig(num_spline_segments))
            ax[0].set_title(title_str)

            # Plot error on second subplot
            ax[1].plot(data_times, field_data[:, 0] - estimated_field[:, 0])
            ax[1].plot(data_times, field_data[:, 1] - estimated_field[:, 1])
            ax[1].plot(data_times, field_data[:, 2] - estimated_field[:, 2])
            x_str = components[0] + ' std = ' + str(round_sig(fit_residual_standard_dev[0]))
            y_str = components[1] + ' std = ' + str(round_sig(fit_residual_standard_dev[1]))
            z_str = components[2] + ' std = ' + str(round_sig(fit_residual_standard_dev[2]))
            ax[1].legend([x_str, y_str, z_str])
            date_form = DateFormatter('%H:%M')
            ax[1].xaxis.set_major_formatter(date_form)
            xlabel_str = 'GMT on ' + data_times[0].strftime("%Y/%m/%d")
            ax[1].set_xlabel(xlabel_str, fontsize=14)
            ax[1].set_ylabel(sensor_error_ylabels[sensor], fontsize=14)
            ax[1].tick_params(direction='in', labelsize=12)

        else:
            # Define one plot of the data
            fig, ax = plt.subplots(figsize=(15, 10))

            # Plot estimated and actual data on same plot
            ax.plot(data_times, field_data[:, 0])
            ax.plot(data_times, field_data[:, 1])
            ax.plot(data_times, field_data[:, 2])
            ax.plot(data_times, estimated_field[:, 0])
            ax.plot(data_times, estimated_field[:, 1])
            ax.plot(data_times, estimated_field[:, 2])
            ax.legend([components[0], components[1], components[2], components[0] + ' est',
                          components[1] + ' est', components[2] + ' est'], loc='upper right')
            date_form = DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(date_form)
            xlabel_str = 'GMT on ' + data_times[0].strftime("%Y/%m/%d")
            ax.set_xlabel(xlabel_str, fontsize=14)
            ax.set_ylabel(sensor_ylabels[sensor], fontsize=14)
            bias_round = [round_sig(bias[0]), round_sig(bias[1]), round_sig(bias[2])]
            align_round = [round_sig(align[0]), round_sig(align[1]), round_sig(align[2])]
            title_str = sensor + ' bias = ' + str(bias_round) + ';   q_align = ' + str(
                align_round) + ';   nseg = ' + str(round_sig(num_spline_segments))
            ax.set_title(title_str)
            ax.tick_params(direction='in', labelsize=12)

        # Configure and save the figure
        fig.tight_layout()
        fig.savefig(image_save_path)

    def generate_calibration_plot_junoish(self, sensor: str, image_save_path: str, lag_time_seconds: int = 10, 
                                          with_error: bool = True):
        '''
        :param sensor: whether to do this for inboard or outboard
        :param image_save_path: path to save the generated plot to
        :param lag_time_seconds: lag time in seconds to get the bias and alignment for
        :param with_error: plot the error in the estimate on a second subplot
        :return: nothing, saves a plot of the lagged vs non-lagged EPN data
        '''
    
        # Get bias, alignment, and spline estimate for the given number of segments
        junoish_bias_align = self.get_bias_alignment_junoish(sensor, lag_time_seconds)
        bias = junoish_bias_align['bias']
        align = junoish_bias_align['alignment']
        fit_params = np.concatenate((bias, align))
    
        # Extract maneuver data to be used in least squares fit
        field_data_in_sensor_frame = self.maneuver_data[sensor]['field_data_in_sensor_frame']
        transform_sensor_frame_to_EPN = self.maneuver_data[sensor]['transform_sensor_frame_to_EPN']
    
        # Get the lag time in terms of indices (specific to 10 Hz data)
        lag_time_num_samples = lag_time_seconds*10
    
        # Create the lagged and non-lagged data
        non_lagged_sensor_data = field_data_in_sensor_frame[:-lag_time_num_samples]
        lagged_sensor_data = field_data_in_sensor_frame[lag_time_num_samples:]
    
        # Create a placeholder "time" to serve as the x value, enumerates all the values along the time dimension
        placeholder_time = np.linspace(1, len(lagged_sensor_data), num=len(lagged_sensor_data))
    
        # Create the lagged and non-lagged transformation matrices
        non_lagged_transform_sensor_frame_to_EPN = transform_sensor_frame_to_EPN[:-(3*lag_time_num_samples)]
        lagged_transform_sensor_frame_to_EPN = transform_sensor_frame_to_EPN[(3*lag_time_num_samples):]
    
        # Get the lagged and non-lagged epn data given the bias, alignment, sensor data, and transformations
        lagged_epn, non_lagged_epn = calculate_data_difference(fit_params, placeholder_time, lagged_sensor_data, non_lagged_sensor_data, 
                                                               lagged_transform_sensor_frame_to_EPN, non_lagged_transform_sensor_frame_to_EPN, 
                                                               for_plot=True)
        
        # Get standard deviation of error
        fit_residual = np.subtract(lagged_epn, non_lagged_epn)
        fit_residual_standard_dev = np.std(fit_residual, axis=0)
    
        # Define plotting parameters
        sensor_ylabels = {'OB': 'OB EPN (nT)', 'IB': 'IB EPN (nT)'}
        sensor_error_ylabels = {'OB': 'OB Lagged - Non-Lagged (EPN nT)', 'IB': 'IB Lagged - Non-Lagged (EPN nT)'}
        components = ['e', 'p', 'n']
    
        if with_error:
            # Define plot with two subplots, data and error
            fig, ax = plt.subplots(2, sharex=True, figsize=(15, 10))
    
            # Plot data on first subplot
            ax[0].plot(placeholder_time, lagged_epn[:, 0])
            ax[0].plot(placeholder_time, lagged_epn[:, 1])
            ax[0].plot(placeholder_time, lagged_epn[:, 2])
            ax[0].plot(placeholder_time, non_lagged_epn[:, 0])
            ax[0].plot(placeholder_time, non_lagged_epn[:, 1])
            ax[0].plot(placeholder_time, non_lagged_epn[:, 2])
            ax[0].set_ylabel(sensor_ylabels[sensor], fontsize=14)
            ax[0].legend([components[0] + ' lagged', components[1] + ' lagged', components[2] + ' lagged', 
                          components[0] + ' non-lagged', components[1] + ' non-lagged', components[2] + ' non-lagged'], loc='upper right')
            bias_round = [round_sig(bias[0]), round_sig(bias[1]), round_sig(bias[2])]
            align_round = [round_sig(align[0]), round_sig(align[1]), round_sig(align[2])]
            title_str = sensor + ' bias = ' + str(bias_round) + ';   q_align = ' + str(align_round) + ';   lag_time = ' + str(lag_time_seconds)
            ax[0].set_title(title_str)
    
            # Plot error on second subplot
            ax[1].plot(placeholder_time, fit_residual[:, 0])
            ax[1].plot(placeholder_time, fit_residual[:, 1])
            ax[1].plot(placeholder_time, fit_residual[:, 2])
            x_str = components[0] + ' std = ' + str(round_sig(fit_residual_standard_dev[0]))
            y_str = components[1] + ' std = ' + str(round_sig(fit_residual_standard_dev[1]))
            z_str = components[2] + ' std = ' + str(round_sig(fit_residual_standard_dev[2]))
            ax[1].legend([x_str, y_str, z_str])
            ax[1].set_xlabel('Lag Count', fontsize=14)
            ax[1].set_ylabel(sensor_error_ylabels[sensor], fontsize=14)
            ax[1].tick_params(direction='in', labelsize=12)
    
        else:
            # Define one plot of the data
            fig, ax = plt.subplots(figsize=(15, 10))
    
            # Plot estimated and actual data on same plot
            ax.plot(placeholder_time, lagged_epn[:, 0])
            ax.plot(placeholder_time, lagged_epn[:, 1])
            ax.plot(placeholder_time, lagged_epn[:, 2])
            ax.plot(placeholder_time, non_lagged_epn[:, 0])
            ax.plot(placeholder_time, non_lagged_epn[:, 1])
            ax.plot(placeholder_time, non_lagged_epn[:, 2])
            ax.legend([components[0] + ' lagged', components[1] + ' lagged', components[2] + ' lagged', 
                       components[0] + ' non-lagged', components[1] + ' non-lagged', components[2] + ' non-lagged'], loc='upper right')
            ax.set_xlabel('Lag Count', fontsize=14)
            ax.set_ylabel(sensor_ylabels[sensor], fontsize=14)
            bias_round = [round_sig(bias[0]), round_sig(bias[1]), round_sig(bias[2])]
            align_round = [round_sig(align[0]), round_sig(align[1]), round_sig(align[2])]
            title_str = sensor + ' bias = ' + str(bias_round) + ';   q_align = ' + str(align_round) + ';   lag_time = ' + str(lag_time_seconds)
            ax.set_title(title_str)
            ax.tick_params(direction='in', labelsize=12)
    
        # Configure and save the figure
        fig.tight_layout()
        fig.savefig(image_save_path)
