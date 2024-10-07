# Imports
from __future__ import annotations
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
import math

######################################################## Utility Functions ###############################################################

'''
Utility functions that support various code components
'''

# Function to convert j2000 seconds into datetime
def time_convert(seconds_2000):
    date_original = datetime(2000, 1, 1, 12, 0)
    return date_original + timedelta(seconds=int(seconds_2000))

# Reverse function to convert date into seconds since j2000
def time_convert_rev(date):
    date_original = datetime(2000, 1, 1, 12, 0)
    return (date - date_original).total_seconds()

# quaternion to rotation matrix function
def quaternion_to_rotation(q):
    q11 = q[0] * q[0]
    q22 = q[1] * q[1]
    q33 = q[2] * q[2]
    q44 = q[3] * q[3]

    q12 = 2 * q[0] * q[1]
    q13 = 2 * q[0] * q[2]
    q14 = 2 * q[0] * q[3]
    q23 = 2 * q[1] * q[2]
    q24 = 2 * q[1] * q[3]
    q34 = 2 * q[2] * q[3]

    matrix = [[q11 - q22 - q33 + q44, q12 + q34, q13 - q24],
              [q12 - q34, -q11 + q22 - q33 + q44, q23 + q14],
              [q13 + q24, q23 - q14, -q11 - q22 + q33 + q44]]

    return matrix

# Function to round values to a certain amount of significant digits
def round_sig(x, sig=4):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)


def generate_estimated_measurement(input_parameters_and_data, input_data_times, segment_times, 
                                   epn_to_sensor_frame_trans_matrix):
    '''
    :param input_parameters_and_data: 0-3 bias in instrument frame, 3-6 alignment in instrument frame,
    6-end segment field levels for background spline interpolation
    :param input_data_times: times in cadence original data was taken
    :param segment_times: times corresponding to the spline segments
    :param epn_to_sensor_frame_trans_matrix: matrix to transform epn to sensor frame at each point in time
    :return: estimated field in sensor frame corrected with the estimated bias and alignment
    '''

    # Define the number of data samples
    number_of_data_samples = len(input_data_times)

    # Separate bias, alignment, and spline field levels for x, y, and z out of input parameters and data
    bias_est = np.asarray(input_parameters_and_data[0:3])
    [da_IBx, da_IBy, da_IBz] = np.asarray(input_parameters_and_data[3:6])
    segment_field_levels_flat = np.asarray(input_parameters_and_data[6:])
    total_segs = int(len(segment_field_levels_flat) / 3)
    segment_field_levels_x = [segment_field_levels_flat[i * 3] for i in range(total_segs)]
    segment_field_levels_y = [segment_field_levels_flat[(1 + (i * 3))] for i in range(total_segs)]
    segment_field_levels_z = [segment_field_levels_flat[(2 + (i * 3))] for i in range(total_segs)]

    # If the spline fit is for just the mean over the entire time period (ie one spline segment)
    if len(input_parameters_and_data) == 9:
        estimated_background_field_in_epn = np.tile(segment_field_levels_flat.T, number_of_data_samples)
    # If there are multiple spline segments, use more complicated 1d interpolation for each segment
    else:
        # Assumes linear spline fit
        field_funcx = interp1d(segment_times, segment_field_levels_x, kind='slinear')
        field_funcy = interp1d(segment_times, segment_field_levels_y, kind='slinear')
        field_funcz = interp1d(segment_times, segment_field_levels_z, kind='slinear')
        # Estimated background field in epn should be shape 3xn (n=length data times)
        estimated_background_field_in_epn = np.stack([field_funcx(input_data_times), field_funcy(input_data_times),
                                                      field_funcz(input_data_times)], axis=1)

    # Generate alignment correction matrix
    alignment_correction_matrix = [[1, -da_IBz, da_IBy], [da_IBz, 1, -da_IBx], [-da_IBy, da_IBx, 1]]

    # Get epn field in sensor frame and correct for bias and alignment
    A = estimated_background_field_in_epn
    B = np.reshape(np.asarray(epn_to_sensor_frame_trans_matrix), (number_of_data_samples, 3, 3))
    C = np.asarray(alignment_correction_matrix).T
    D = np.asarray(bias_est).T
    AB_product = np.einsum('nm,nmb->nb', A, B)
    estimated_field = np.matmul(np.add(AB_product, D), C)

    # Return the bias and alignment corrected data in the sensor frame
    return estimated_field

def calculate_data_difference(input_parameters_and_data, data_times, lagged_sensor_data, non_lagged_sensor_data,
                              lagged_sensor_to_EPN_transform, non_lagged_sensor_to_EPN_transform, for_plot=False):
    '''
    :param input_parameters_and_data: 0-3 bias in instrument frame, 3-6 alignment in instrument frame
    :param data_times: simple count enumerating each of the values in time
    :param lagged_sensor_data: lagged field data in the sensor frame
    :param non_lagged_sensor_data: non-lagged field data in the sensor frame
    :param lagged_sensor_to_EPN_transform: lagged matrix to transform sensor frame to EPN at each point in time
    :param non_lagged_sensor_to_EPN_transform: non-lagged matrix to transform sensor frame to EPN at each point in time
    '''

    # Get the number of data points in the array
    number_of_data_samples = len(data_times)

    # Extract the bias and alignment from the input parameters and data
    bias_est = np.asarray(input_parameters_and_data[0:3])
    [da_IBx, da_IBy, da_IBz] = np.asarray(input_parameters_and_data[3:6])

    # Define the correction matrix from alignment
    alignment_correction_matrix = [[1, -da_IBz, da_IBy], [da_IBz, 1, -da_IBx], [-da_IBy, da_IBx, 1]]

    # Transform the lagged data into EPN
    # Process: First, multiply by alignment. Then subtract bias. Then transform from sensor to EPN
    A = lagged_sensor_data # Data in sensor frame
    B = np.reshape(np.asarray(lagged_sensor_to_EPN_transform), (number_of_data_samples, 3, 3)) # Sensor to EPN correction matrix
    C = np.asarray(alignment_correction_matrix) # Alignment correction matrix
    D = np.asarray(bias_est) # Bias estimate
    corrected_sensor_lagged = np.subtract(np.matmul(A, C), D)
    lagged_epn_data = np.einsum('nm,nmb->nb', corrected_sensor_lagged, B)

    # Transform the non-lagged data into EPN
    A = non_lagged_sensor_data # Data in sensor frame
    B = np.reshape(np.asarray(non_lagged_sensor_to_EPN_transform), (number_of_data_samples, 3, 3)) # Sensor to EPN correction matrix
    C = np.asarray(alignment_correction_matrix) # Alignment correction matrix
    D = np.asarray(bias_est) # Bias estimate
    corrected_sensor_non_lagged = np.subtract(np.matmul(A, C), D)
    non_lagged_epn_data = np.einsum('nm,nmb->nb', corrected_sensor_non_lagged, B)

    # Get the difference between the lagged and non-lagged data
    data_difference_in_epn = np.subtract(lagged_epn_data, non_lagged_epn_data)

    # Option to be used by the plotting function only
    if for_plot:
        return lagged_epn_data, non_lagged_epn_data
    # For all lmfit operations
    else:
        return data_difference_in_epn
    

def set_plot_params(ax, plot_params):
    """
    :param ax: the matplotlib axis object that will be modified and returned
    :param plot_params: a dictionary of plot parameters to set for the axis
    :return: ax, modified with plot params
    """

    # Iterates through list and sets each parameter
    for param in plot_params.keys():
        if param == 'title':
            ax.set_title(plot_params[param])
        if param == 'xlabel':
            ax.set_xlabel(plot_params[param])
        if param == 'ylabel':
            ax.set_ylabel(plot_params[param])

def shorten_time_frame(data_dict, time_constraints):
    # Shorten the time frame for all variables in the dictionary, for IB and OB
    ibob_keys = ['IB', 'OB']
    var_exclude = ['initial_guess_for_background_field']
    var_flat = ['transform_EPN_to_sensor_frame', 'transform_sensor_frame_to_EPN']
    data_dict_short = data_dict.copy()
    # Inboard variables
    for ibob_key in ibob_keys:
        time_var = data_dict[ibob_key]['data_times']
        indx_low = np.where(time_var > np.repeat(time_constraints[0], len(time_var)))[:][0][0]
        indx_high = np.where(time_var < np.repeat(time_constraints[1], len(time_var)))[:][0][-1]

        # Iterate through variables and shorten them
        for var in data_dict_short[ibob_key].keys():
            if not (var in var_exclude):
                # For 3d variables
                if var in var_flat:
                    i_low = indx_low*3
                    i_high = indx_high*3
                else:
                    i_low = indx_low
                    i_high = indx_high
                if np.shape(data_dict_short[ibob_key][var])[0] == 3:
                    data_dict_short[ibob_key][var] = np.stack((data_dict_short[ibob_key][var][0, :][i_low:i_high],
                                                    data_dict_short[ibob_key][var][1, :][i_low:i_high],
                                                    data_dict_short[ibob_key][var][2, :][i_low:i_high]))
                # For 1d variables
                else:
                    data_dict_short[ibob_key][var] = data_dict_short[ibob_key][var][i_low:i_high]

    return data_dict_short

# Quaternion interpolation function
def quantinterp(t, q, ti):
    nti = len(ti)
    qi = np.zeros((nti, 4))
    nt = len(t)
    for i in range(nti):
        r1 = np.where(ti[i] >= t)[0]  # Finds last index where interpolated time is > non-interpolated
        # Edge case - python version doesn't handle if no indices are found
        if r1.size == 0:
            r1 = 0
        else:
            r1 = r1[-1]
        t1 = t[r1]
        q1 = q[r1, :]
        if r1 < (nt - 1):
            t2 = t[(r1 + 1)]
            q2 = q[(r1 + 1), :]
            tf = (ti[i] - t1) / (t2 - t1)
            qi[i, :] = slerp(q1, q2, tf)
        else:
            qi[i, :] = q1

    # Return the interpolated quaternion
    return qi


# SLERP Python - takes in two quaternions and interpolates between using shortest great arc on unit sphere
# (shortest possible interpolation path while assuming constant angular velocity)

def slerp(qi, qn, t):
    # Saves calculation time, where qm = qi
    if t == 0:
        qm = qi
    # Saves calculation time, where qm = qn
    elif t == 1:
        qm = qn
    # Else it's the typical, more complex case
    else:
        C = np.dot(np.asarray(qi), np.asarray(qn))
        teta = np.arccos(C)  # This will output teta in radians, similar to matlab version

        # If angle teta is close by epsilon to 0 degrees -> calculate by linear interpolation
        if teta <= 1e-6:
            qmt = qi * (1 - t) + qn * t  # avoiding divisions by number close to 0
            qm = qmt / np.linalg.norm(qmt)  # Normalizes qmt

        # When teta is close to 180 degrees, unwrap the jump
        elif (teta * 180 / math.pi) > 175:
            theta = 2 * np.arccos(qn[3])
            axis = qn[0:3] / np.sin(theta / 2)
            thetaNew = (2 * math.pi) - theta
            qn180_temp = list(-np.asarray(axis) * np.sin(thetaNew / 2))
            qn180 = np.asarray(qn180_temp + [np.cos(thetaNew / 2)])
            teta180 = np.arccos(np.dot(np.asarray(qi), np.asarray(qn180)))
            qm = (qi * (np.sin((1 - t) * teta180)) / np.sin(teta180)) + (qn180 * np.sin(t * teta180) / np.sin(teta180))

        # Else, no edge case, use the main interpolation method
        else:
            qm = (qi * (np.sin((1 - t) * teta)) / np.sin(teta)) + (qn * np.sin(t * teta) / np.sin(teta))

    # Return qm
    return qm
