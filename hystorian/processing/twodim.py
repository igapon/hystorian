import h5py
import matplotlib.pyplot as plt
import numpy as np
from . import core as pt
import cv2
import time

from scipy.signal import medfilt, cspline2d
from scipy.optimize import curve_fit
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.measurements import label
from scipy import interpolate, ndimage

from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import itertools


def line_fit(line, order=1,box=[0]):
    """
    Do a nth order polynomial line flattening

    Parameters
    ----------
    line : 1d array-like
    order : integer

    Returns
    -------
    result : 1d array-like
        same shape as data
    """
    if order < 0:
        raise ValueError('expected deg >= 0')
    newline=line
    if len(box)==2:
        newline = line[box[0]:box[1]]
    x = np.arange(len(newline))
    k = np.isfinite((newline))
    if not np.isfinite(newline).any():
        return line
    coefficients = np.polyfit(x[k], newline[k], order)

    return line - np.polyval(coefficients, np.arange(len(line)))


def line_flatten_image(data, order=1, axis=0, box=[0]):
    """
    Do a line flattening

    Parameters
    ----------
    data : 2d array
    order : integer
    axis : integer
        axis perpendicular to lines

    Returns
    -------
    result : array-like
        same shape as data
    """

    if axis == 1:
        data = data.T

    ndata = np.zeros_like(data)

    for i, line in enumerate(data):
        ndata[i, :] = line_fit(line, order, box)

    if axis == 1:
        ndata = ndata.T

    return ndata


def plane_flatten_image(data, order=1,box=[]):
    """
    Do a plane flattening

    Parameters
    ----------
    data : 2d array
    order : integer

    Returns
    -------
    result : array-like
        same shape as data
    """
    fitdata = data
    if len(box)==4:
        fitdata = data[box[0]:box[1],box[2]:box[3]]
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xxfit, yyfit = np.meshgrid(np.arange(fitdata.shape[1]), np.arange(fitdata.shape[0]))
    m = polyfit2d(xxfit.ravel(), yyfit.ravel(), fitdata.ravel(), order=order)
    return data - polyval2d(xx, yy, m)


def polyfit2d(x, y, z, order=1):
    ncols = (order + 1) ** 2
    g = np.zeros((x.size, ncols))
    ij = itertools.product(range(order + 1), range(order + 1))
    for k, (i, j) in enumerate(ij):
        g[:, k] = x ** i * y ** j
    k = np.isfinite(z)
    m, _, _, _ = np.linalg.lstsq(g[k], z[k], rcond=None)
    return m


def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order + 1), range(order + 1))
    z = np.zeros_like(x, dtype=float)
    for a, (i, j) in zip(m, ij):
        z += a * x ** i * y ** j
    return z


def normalise(array, new_min=0, new_max=1):
    """
    Normalises an array of numbers so each number is within a range fo two numbers

    Parameters
    ----------
    array : array_like
        The array to be normalised
    new_min : int or float, optional
        Minimum value of new range (default: 0)
    new_max : int or float, optional
        Maximum value of new range (default: 1)
    Returns
    -------
    array : array_like
        the entry that has been converted
    """
    old_range = np.max(array) - np.min(array)
    new_range = new_max - new_min
    array = array - np.min(array)
    array = array * new_range / old_range
    array = array + new_min
    return array


def distortion_params_(filename, all_input_criteria, mode='SingleECC', read_offset=False,
                       cumulative=False, filterfunc=normalise, warp_mode=cv2.MOTION_TRANSLATION,
                       termination_eps=1e-5, number_of_iterations=10000, max_divisions=8,
                       warp_check_range = 8, divisor_init = 0.25, lim=50, speed=2):
    """
    Determine cumulative translation matrices for distortion correction and directly write it into
    an hdf5 file

    Parameters
    ----------
    filename : str
        name of hdf5 file containing data
    all_input_criteria : str
        criteria to identify paths to source files using pt.path_search. Should be
        height data to extract parameters from
    mode : str
        Determines mode of operation, and thus parameters used. Can be 'SingleECC', 'ManualECC',
        or 'MultiECC', in decreasing order of speed
    read_offset : bool, optional
        If set to True, attempts to read dataset for offset attributes to
        improve initial guess and thus overall accuracy (default is False).
    cumulative : bool, optional
        Determines if each image is compared to the previous image (default,
        False), or to the original image (True). Output format is identical.
    fitlerfunc : func, optional
        Function applied to image before identifying distortion params
    warp_mode : int
        Mode of warp identified. Relevant for SingleECC and MultiECC modes.
        Only cv2.MOTION_TRANSLATION is currently tested.
    termination_eps : float
        Termination precision for SingleECC and MultiECC modes
    number_of_iterations : int
        Termination number for SingleECC and MultiECC modes
    max_divisions : int
        Number of recursive checks for ManualECC mode
    warp_check_range : int
        Length and height of grid checked for ManualECC mode
    divisor_init : float
        Spacing between grid points in Manual ECCmode.
    lim : int
        External limits examined in ManualECC and MultiECC mode
    speed : int, optional
        Value between 1 and 4, which determines speed and accuracy of MultiECC mode. A higher
        number is faster, but assumes lower distortion and thus may be incorrect. Default value
        is 2.

    Returns
    -------
        None
    """
    
    if type(all_input_criteria) != list:
        all_input_criteria = [all_input_criteria]
    if type(all_input_criteria[0]) != list:
        all_input_criteria = [all_input_criteria]
    all_in_path_list = []
    for channel_type in all_input_criteria:
        in_path_list = pt.path_search(filename, channel_type)
        all_in_path_list.append(in_path_list[0])
    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_params',
                                                          all_in_path_list[0])
    eyes = np.eye(2, 3, dtype=np.float32)
    cumulative_tform21 = np.eye(2, 3, dtype=np.float32)
    with h5py.File(filename, "a") as f:
        recent_offsets = []
        for i in range(len(all_in_path_list[0])):
            if i == 0:
                start_time = time.time()
            else:
                print('---')
                print('Currently reading path ' + all_in_path_list[0][i])
                img1 = []
                img2 = []
                for channel_i in range(len(all_in_path_list)):
                    i1 = f[all_in_path_list[channel_i][0]]
                    if (i > 1) and (not cumulative):
                        i1 = f[all_in_path_list[channel_i][i-1]]
                    i2 = f[all_in_path_list[channel_i][i]]
                    if filterfunc is not None:
                        i1 = filterfunc(i1)
                        i2 = filterfunc(i2)
                    img1.append(img_as_ubyte(i1))
                    img2.append(img_as_ubyte(i2))

                if read_offset:
                    offset2 = (f[all_in_path_list[0][i]]).attrs['offset']
                    offset1 = (f[all_in_path_list[0][i - 1]]).attrs['offset']
                    scan_size = (f[all_in_path_list[0][i]]).attrs['size']
                    shape = (f[all_in_path_list[0][i]]).attrs['shape']
                    offset_px = m2px(offset2 - offset1, shape, scan_size)
                else:
                    offset_px = np.array([0, 0])
                if (offset_px[0] != 0) or (offset_px[1] != 0):
                    recent_offsets = []
                if mode == 'SingleECC':
                    tform21 = generate_transform_xy_single(img1[0], img2[0], offset_px, warp_mode, termination_eps, number_of_iterations)
                elif mode == 'ManualECC':
                    tform21 = generate_transform_xy_manual(img1, img2, offset_px, max_divisions, warp_check_range, divisor_init, lim)
                elif mode == 'MultiECC':
                    tform21 = generate_transform_xy_multi(img1[0], img2[0], offset_px, warp_mode, termination_eps, number_of_iterations, lim,
                                                          speed, recent_offsets, cumulative, cumulative_tform21)
                else:
                    print('Invalid mode requested. Defaulting to SingleECC')
                    mode='SingleECC'
                    tform21 = generate_transform_xy_single(img1[0], img2[0], offset_px, warp_mode, termination_eps, number_of_iterations)
                        
                if cumulative:
                    tform21[0, 2] = tform21[0, 2] - cumulative_tform21[0, 2]
                    tform21[1, 2] = tform21[1, 2] - cumulative_tform21[1, 2]
                cumulative_tform21[0, 2] = cumulative_tform21[0, 2] + tform21[0, 2]
                cumulative_tform21[1, 2] = cumulative_tform21[1, 2] + tform21[1, 2]
                print('Scan ' + str(i) + ' Complete. Cumulative Transform Matrix:')
                print(cumulative_tform21)
                
                if mode == 'multiECC':
                    if (offset_px[0] == 0) and (offset_px[1] == 0):
                        recent_offsets.append([tform21[0, 2], tform21[1, 2]] - offset_px)
                        if len(recent_offsets) > 3:
                            recent_offsets = recent_offsets[1:]

            data = pt.write_output_f(f, cumulative_tform21, out_folder_locations[i],
                                     all_in_path_list[0][i],  distortion_params_, locals())
            pt.progress_report(i + 1, len(all_in_path_list[0]), start_time, 'distortion_params',
                               all_in_path_list[0][i], clear=False)


def m2px(m, points, scan_size):
    """
    Converts length in metres to a length in pixels

    Parameters
    ----------
    m : int or float
        length in metres to be converted
    points : int
        number of lines or points per row
    scan_size : int or float
        total length of scan

    Returns
    -------
    px : float
        converted length in pixels
    """
    px = m * points / scan_size
    return px


def generate_transform_xy_single(img, img_orig, offset_guess=[0,0], warp_mode = cv2.MOTION_TRANSLATION, termination_eps = 1e-5,
                          number_of_iterations=10000):
    """
    Determines transformation matrices in x and y coordinates for SingleECC mode

    Parameters
    ----------
    img : cv2
        Currently used image (in cv2 format) to find transformation array of
    img_orig : cv2
        Image (in cv2 format) transformation array is based off of
    offset_guess : list of ints
        Estimated shift and offset between images
    warp_mode : see cv2 documentation
        warp_mode used in cv2's findTransformationECC function
    termination_eps : float
        eps used to terminate fit
    number_of_iterations : int
        number of iterations in fit before termination
    
    Returns
    -------
    warp_matrix : ndarray
        Transformation matrix used to convert img_orig into img
    """
    # Here we generate a MOTION_EUCLIDEAN matrix by doing a 
    # findTransformECC (OpenCV 3.0+ only).
    # Returns the transform matrix of the img with respect to img_orig
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    term_flags = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    criteria = (term_flags, number_of_iterations, termination_eps)
    warp_matrix[0, 2] = offset_guess[0]
    warp_matrix[1, 2] = offset_guess[1]
    (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode,
                                         criteria,None,1)
    return tform21


def generate_transform_xy_manual(img1, img2, offset_guess=[0,0], max_divisions = 8, warp_check_range = 8, divisor_init = 0.25, lim=50):
    """
    Determines transformation matrices in x and y coordinates for ManualECC mode

    Parameters
    ----------
    img1 : cv2
        Currently used image (in cv2 format) to find transformation array of
    img2 : cv2
        Image (in cv2 format) transformation array is based off of
    offset_guess : list of ints
        Estimated shift and offset between images
    max_divisions : int
        Number of recursive checks for ManualECC mode
    warp_check_range : int
        Length and height of grid checked for ManualECC mode
    divisor_init : float
        Spacing between grid points in Manual ECCmode.
    lim : int
        External limits examined in ManualECC and MultiECC mode
    
    Returns
    -------
    warp_matrix : ndarray
        Transformation matrix used to convert img_orig into img
    """
    offset_guess[1] = -offset_guess[1]
    divisor = divisor_init
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    for num in range(max_divisions):
        diff = np.Inf
        last_k = np.Inf
        last_j = np.Inf
        offset_guess = np.array(offset_guess)
        low_offset_check = offset_guess - (warp_check_range/2)/divisor
        high_offset_check = offset_guess + (warp_check_range/2)/divisor
        i_check = np.linspace(low_offset_check[0], high_offset_check[0], warp_check_range+1)
        j_check = np.linspace(low_offset_check[1], high_offset_check[1], warp_check_range+1)
        for k in i_check:
            for j in j_check:
                warp_matrix[0, 2] = k
                warp_matrix[1, 2] = j
                if k-offset_guess[0] < last_k*1.5:
                    if j-offset_guess[1] < last_j*1.5:
                        currDiff=0
                        for channel_i in range(len(img1)):
                            img_test = cv2.warpAffine(img1[channel_i], warp_matrix, (np.shape(img2)[2], np.shape(img2)[1]), flags=cv2.INTER_LINEAR +
                                                                                          cv2.WARP_INVERSE_MAP)
                            currDiff = currDiff+np.sum(np.square(img_test[lim:-lim, lim:-lim]
                                                            - img2[channel_i][lim:-lim, lim:-lim]))
                        if currDiff < diff:
                            diff = currDiff
                            offset1 = k
                            offset2 = j
        warp_matrix[0, 2] = offset1
        warp_matrix[1, 2] = offset2
        divisor = divisor*4
        tform21 = warp_matrix
        offset_guess = [offset1, offset2]
    return tform21


def generate_transform_xy_multi(img, img_orig, offset_px = [0,0], warp_mode = cv2.MOTION_TRANSLATION, termination_eps = 1e-5,
                          number_of_iterations=10000, lim=50, speed=2, recent_offsets = [], 
                          cumulative=False, cumulative_tform21=np.eye(2,3,dtype=np.float32)):
    """
    Determines transformation matrices in x and y coordinates

    Parameters
    ----------
    img : cv2
        Currently used image (in cv2 format) to find transformation array of
    img_orig : cv2
        Image (in cv2 format) transformation array is based off of
    offset_px : list of ints
        Actual shift estimated and taken from machine
    warp_mode : see cv2 documentation
        warp_mode used in cv2's findTransformationECC function
    termination_eps : float
        eps used to terminate fit
    number_of_iterations : int
        number of iterations in fit before termination
    lim : int
        External limits examined in ManualECC and MultiECC mode
    speed : int, optional
        Value between 1 and 4, which determines speed and accuracy of MultiECC mode. A higher
        number is faster, but assumes lower distortion and thus may be incorrect. Default value
        is 2.
    recent_offsets : list
        List of recent offsets used to increase speed
    cumulative : bool, optional
        Determines if each image is compared to the previous image (default, False), or to the original image (True).
        Output format is identical.
    cumulative_tform21 : ndarray, optional
        The transformation matrix, only used if cumulative is switched to True. (default: np.eye(2,3,dtype=np.float32))

    Returns
    -------
    warp_matrix : ndarray
        Transformation matrix used to convert img_orig into img
    """
    # Here we generate a MOTION_EUCLIDEAN matrix by doing a 
    # findTransformECC (OpenCV 3.0+ only).
    # Returns the transform matrix of the img with respect to img_orig
    
    # Redo using the same logic as the previous one; but only use the actual logic to optimise parameters just once

    offset_px[1] = -offset_px[1]
    if speed != 0 and speed != 1 and speed != 2 and speed != 3 and speed != 4:
        print('Error: Speed should be an integer between 1 (slowest) and 4 (fastest).\
                Speed now set to level 2.')
        speed = 2
    if len(recent_offsets) == 0:
        offset_guess = offset_px
        if speed == 1:
            warp_check_range = 16
        elif speed == 2:
            warp_check_range = 12
        elif speed == 3:
            warp_check_range = 10
        elif speed == 4:
            warp_check_range = 8
    elif len(recent_offsets) < 3:
        offset_guess = offset_px + recent_offsets[-1]
        if speed == 1:
            warp_check_range = 12
        elif speed == 2:
            warp_check_range = 8
        elif speed == 3:
            warp_check_range = 8
        elif speed == 4:
            warp_check_range = 6
    else:
        offset_guess = (offset_px + recent_offsets[2] / 2 + recent_offsets[1] / 3
                        + recent_offsets[0] / 6)
        if speed == 1:
            warp_check_range = 8
        elif speed == 2:
            warp_check_range = 6
        elif speed == 3:
            warp_check_range = 4
        elif speed == 4:
            warp_check_range = 2
    if (offset_px[0] != 0) or (offset_px[1] != 0):
        print('Offset found from file attributes: ' + str(offset_px))
        warp_check_range = warp_check_range + 8
        recent_offsets = []
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    term_flags = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT

    if cumulative:
        offset_guess[0] = offset_guess[0] + cumulative_tform21[0, 2]
        offset_guess[1] = offset_guess[1] + cumulative_tform21[1, 2]

    criteria = (term_flags, number_of_iterations, termination_eps)

    diff = np.Inf
    offset1 = 0
    offset2 = 0
    for i in range(-warp_check_range // 2, (warp_check_range // 2) + 1):
        for j in range(-warp_check_range // 2, (warp_check_range // 2) + 1):
            warp_matrix[0, 2] = 2 * i + offset_guess[0]
            warp_matrix[1, 2] = 2 * j + offset_guess[1]
            try:
                (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode,
                                                         criteria)
                img_test = cv2.warpAffine(img, tform21, (np.shape(img)[1], np.shape(img)[0]), flags=cv2.INTER_LINEAR +
                                                                              cv2.WARP_INVERSE_MAP)
                currDiff = np.sum(np.square(img_test[lim:-lim, lim:-lim]
                                                - img_orig[lim:-lim, lim:-lim]))
                if currDiff < diff:
                    diff = currDiff
                    offset1 = tform21[0, 2]
                    offset2 = tform21[1, 2]
            except:
                pass
            warp_matrix[0, 2] = offset1
            warp_matrix[1, 2] = offset2
    return warp_matrix


def distortion_correction_(filename, all_input_criteria, cropping=True):
    """
    Applies distortion correction parameters to an image. The distortion corrected data is then
    cropped to show only the common data, or expanded to show the maximum extent of all possible data.

    Parameters
    ----------
    filename : str
        Filename of hdf5 file containing data
    all_input_criteria : list
        Criteria to identify paths to source files using pt.path_search. First should
        be data to be corrected, second should be the distortion parameters.
    cropping : bool, optional
        If set to True, each dataset is cropped to show only the common area. If
        set to false, expands data shape to show all data points of all images. (default: True)

    Returns
    -------
    None
    """
    all_in_path_list = pt.path_search(filename, all_input_criteria, repeat='block')
    in_path_list = all_in_path_list[0]
    dm_path_list = all_in_path_list[1]

    distortion_matrices = []
    with h5py.File(filename, "a") as f:
        for path in dm_path_list[:]:
            distortion_matrices.append(np.copy(f[path]))
        xoffsets = []
        yoffsets = []
        for matrix in distortion_matrices:
            xoffsets.append(np.array(matrix[0, 2]))
            yoffsets.append(np.array(matrix[1, 2]))
    offset_caps = [np.max(xoffsets), np.min(xoffsets), np.max(yoffsets), np.min(yoffsets)]

    out_folder_locations = pt.find_output_folder_location(filename, 'distortion_correction',
                                                          in_path_list)

    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for i in range(len(in_path_list)):
            orig_image = f[in_path_list[i]]
            if cropping:
                final_image = array_cropped(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            else:
                final_image = array_expanded(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            data = pt.write_output_f(f, final_image, out_folder_locations[i], [in_path_list[i],
                                                                               dm_path_list[i]],
                                    distortion_correction_, locals())
            propagate_scale_attrs(data, f[in_path_list[i]])
            pt.progress_report(i + 1, len(in_path_list), start_time, 'distortion_correction',
                               in_path_list[i])


def propagate_scale_attrs(new_data, old_data):
    """
    Attempts to write the scale attributes to a new dataset. This is done by directly copying from
    an old dataset. If this is not possible, then it attempts to generate this from the old dataset
    by calculating from the 'size' and 'shape' attributes.

    Parameters
    ----------
    new_data : hdf5 file
        New dataset to write to
    old_data : hdf5 file
        Old dataset to read from

    Returns
    -------
    None
    """
    if 'scale_m_per_px' in old_data.attrs:
        new_data.attrs['scale_m_per_px'] = old_data.attrs['scale_m_per_px']
    else:
        if ('size' in old_data.attrs) and ('shape' in old_data.attrs):
            scan_size = old_data.attrs['size']
            shape = old_data.attrs['shape']
            new_data.attrs['scale_m_per_px'] = scan_size[0] / shape[0]

def array_cropped(array, xoffset, yoffset, offset_caps):
    """
    Crops a numpy_array given the offsets of the array, and the minimum and maximum offsets of a set,
    to include only valid data shared by all arrays

    Parameters
    ----------
    array : array_like
        The array to be cropped
    xoffset : int
        The x-offset ot the array
    yoffset : int
        The y-offset of the array
    offset_caps : list
        A list of four entries. In order, these entries are the xoffset maximum, xoffset
        minimum, yoffset maximum, and yoffset minimum for all arrays

    Returns
    -------
    cropped_array : array_like
    """
    if offset_caps != [0, 0, 0, 0]:
        left = int(np.ceil(offset_caps[0]) - np.floor(xoffset))
        right = int(np.floor(offset_caps[1]) - np.floor(xoffset))
        top = int(np.ceil(offset_caps[2]) - np.floor(yoffset))
        bottom = int(np.floor(offset_caps[3]) - np.floor(yoffset))
        if right == 0:
            right = np.shape(array)[1]
        if bottom == 0:
            bottom = np.shape(array)[0]
        cropped_array = array[top:bottom, left:right]
    else:
        cropped_array = array
    return cropped_array


def array_expanded(array, xoffset, yoffset, offset_caps):
    """
    Expands a numpy_array given the offsets of the array, and the minimum and maximum offsets of a
    set, to include all points of each array. Empty data is set to be NaN

    Parameters
    ----------
    array : array_like
        The array to be expanded
    xoffset : int
        The x-offset ot the array
    yoffset : int
        The y-offset of the array
    offset_caps : list
        A list of four entries. In order, these entries are the xoffset maximum, xoffset
        minimum, yoffset maximum, and yoffset minimum for all arrays

    Returns
    -------
    expanded_array : array_like
        The expanded array
    """
    height = int(np.shape(array)[0] + np.ceil(offset_caps[2]) - np.floor(offset_caps[3]))
    length = int(np.shape(array)[1] + np.ceil(offset_caps[0]) - np.floor(offset_caps[1]))
    expanded_array = np.empty([height, length])
    expanded_array[:] = np.nan
    left = int(-np.floor(offset_caps[1]) + xoffset)
    right = int(length - np.ceil(offset_caps[0]) + xoffset)
    top = int(-np.floor(offset_caps[3]) + yoffset)
    bottom = int(height - np.ceil(offset_caps[2]) + yoffset)
    expanded_array[top:bottom, left:right] = array
    return expanded_array


def phase_linearisation(image, min_separation=90, background=None,
                        flip_proportion=0.8, phase_range=None, show=False):
    """
    Converts each entry of a 2D phase channel (rotating 360 degrees with an arbitrary 0 point) into a
    float between 0 and 1.  The most common values become 0 or 1, and other values are a linear
    interpolation between these two values. 0 and 1 are chosen such that the mean of the entire
    channel does not become greater than a value defined by flip_proportion, and such that the
    edgemost pixels are more 0 than the centre.

    Parameters
    ----------
    image : array_like
        Array that contains the data
    min_separation : int, optional
        Minimum distance between the two peaks assigned as 0 and 1 (default: 90)
    background : int or flaot, optional
        Number to identify where background is to correctly attribute values.
        If positive, tries to make everything to the left of this value background; if negative, makes
        everything to the right background
    flip_proportion : int or float, optional
        Threshold, above which the data is flipped to (1-data)
    phase_range : int, optional
        Sets the range over which the linearisation will occur. If not set, calculates from max-min
    show : bool, optional
        If True, show the data prior to saving

    Returns
    -------
    hdf5_dict
        Contain linearised data and peak values as attributes
    """
    phase_flat = np.array(image).ravel()
    min_phase = int(np.floor(np.min(phase_flat)))
    if phase_range == None:
        max_phase = int(np.floor(np.max(phase_flat)))
        phase_range = max_phase-min_phase
    else:
        max_phase = min_phase+phase_range

    # Convert original data into histograms and find largest peak
    ydata, bin_edges = np.histogram(phase_flat, bins=360, range=[min_phase, max_phase])
    peak1_index = np.argmax(ydata)

    # Find next largest peak a distance away from original peak
    peak1_exclude_left = wrap(peak1_index - min_separation, 0, phase_range)
    peak1_exclude_right = wrap(peak1_index + min_separation, 0, phase_range)
    if peak1_exclude_left < peak1_exclude_right:
        peak2_search_region = np.delete(ydata,
                                        np.arange(peak1_exclude_left, peak1_exclude_right))
        peak2_index = np.argmax(peak2_search_region)
        if peak2_index < peak1_exclude_left:
            pass
        else:
            peak2_index = peak2_index + 2 * min_separation
    else:
        peak2_search_region = ydata[peak1_exclude_right:peak1_exclude_left]
        if peak2_search_region.size != 0:
            peak2_index = np.argmax(peak2_search_region) + peak1_exclude_right - 1
        else:
            peak2_index = np.mean([peak1_exclude_right, peak1_exclude_left])

    # Split wrapped dataset into two number lines; one going up and one down
    if peak1_index > peak2_index:
        peak1_index, peak2_index = peak2_index, peak1_index
    peak1_value = peak1_index + min_phase
    peak2_value = peak2_index + min_phase
    range_1to2 = peak2_value - peak1_value
    range_2to1 = phase_range - range_1to2

    # Create a new array whose values depend on their position on the number lines
    linearised_array = np.copy(image)
    linearise_map = np.vectorize(linearise)
    linearised_array = linearise_map(linearised_array, peak1_value, peak2_value,
                                     range_1to2, range_2to1)
    # Define which points are 0 or 1 based on relative magnitude
    if np.mean(linearised_array) > flip_proportion:
        linearised_array = 1 - linearised_array
    elif np.mean(linearised_array) > 1 - flip_proportion:
        if background is None:
            if (np.mean(linearised_array[:, :10]) + np.mean(linearised_array[:, -10:])) \
                    > 2 * np.mean(linearised_array):
                linearised_array = 1 - linearised_array
        elif background < 0:
            if np.mean(linearised_array[:, background:]) > np.mean(linearised_array):
                linearised_array = 1 - linearised_array
        elif background > 0:
            if np.mean(linearised_array[:, :background]) > np.mean(linearised_array):
                linearised_array = 1 - linearised_array
    pt.intermediate_plot(linearised_array, force_plot=show, text='Linearised Array')
          
    linearised = medfilt(cv2.blur(linearised_array, (7, 7)), 7)
    result = pt.hdf5_dict(linearised, peak_values=[peak1_value, peak2_value])
    return result


def linearise(entry, peak1_value, peak2_value, range_1to2, range_2to1):
    """
    Converts a phase entry (rotating 360 degrees with an arbitrary 0 point) into a float between 0
    and 1, given the values of the two extremes, and the ranges between them

    Parameters
    ----------
    entry : int or float
        The phase entry to be converted
    peak1_value : int or float
        The phase value that would be linearised to 0
    peak2_value : int or float
        The phase value that would be linearised to 1
    range_1to2 : int or float
        The distance in phase from peak1 to peak2
    range_2to1 : int or float
        The distance in phase from peak2 to peak1

    Returns
    -------
    entry : int or float
        The phase entry that has been converted
    """
    if (entry >= peak1_value) and (entry < peak2_value):
        entry = (entry - peak1_value) / range_1to2
    elif entry < peak1_value:
        entry = (peak1_value - entry) / range_2to1
    else:
        entry = 1 - ((entry - peak2_value) / range_2to1)
    return entry


def m_sum(*args):
    """
    Adds multiple channels together. The files are added in order, first by channel and then by
    sample. The amount of source files in each destination file defined by entry_count. Replaces sum_

    Parameters
    ----------
    *args : array_like
        Arrays to be summed

    Returns
    -------
    result : hdf5 dict
        Contains summed data and amount of inputs as attributes
    """
    if (args[0].dtype == 'uint8') or (args[0].dtype == 'bool'):
        convert = True
    else:
        convert = False
    total = np.zeros_like(args[0])
    if convert:
        total = total.astype(int)
    for arg in args:
        if convert:
            arg = arg.astype(int)
        total = total + arg
    input_count = len(args)
    result = pt.hdf5_dict(total, input_count=input_count)
    return result


#   FUNCTION phase_binarisation
# Converts each entry of an array that is between two values to either 0 or 1. Designed for use
# with linearised phase data, where peaks exist at endpoints.
#   INPUTS:
# phase: array of phase data to be binarised
# thresh_estimate (default: 2): initial guess for where the threshold should be placed
# thresh_search_range (default: 0.8): range of thresholds searched around the estimate
# blur (default: 7): blur applied to image to help assist in binarisation
# source_input_count: number of phases summed, used to estimate threshold
#    OUTPUTS
# result: hdf5_dict containing the binarised phase, as well as the threshold in attributes

def phase_binarisation(phase, thresh_estimate=None, thresh_search_range=None, blur = 7,
                       source_input_count=None):
    if thresh_estimate is None:
        if source_input_count is not None:
            thresh_estimate = source_input_count / 2
        else:
            thresh_estimate = 0.5
    if thresh_search_range is None:
        if source_input_count is not None:
            thresh_search_range = source_input_count / 10
        else:
            thresh_search_range = 0.1
    blurred_phase = cv2.blur(phase, (blur, blur))
    best_thresh = threshold_noise(blurred_phase, thresh_estimate, thresh_search_range / 2, 5)
    binary = blurred_phase > best_thresh

    if np.mean(binary) > 0.95:
        binary = 1 - binary
    result = pt.hdf5_dict(binary, threshold=best_thresh)
    return result


#   FUNCTION threshold_noise
# Iterative threshold function designed for phase_binarisation). Decides threshold based on what
# gives the "cleanest" image, with minimal high frequency noise.
#   INPUTS:
# image: data to be thresholded
# old_guess: initial estimate of threshold
# thresh_range: span searched (in both positive and negative directions) for optimal threshold
# iterations: number of times the function is run iteratively
# old_diff (default: None): number that represents the number of noise. Determines best threshold.
#   OUTPUTS
# best_guess: final guess for best threshold

def threshold_noise(image, old_guess, thresh_range, iterations, old_diff=None):
    if old_diff is None:
        binary = image > old_guess
        binary_filt = medfilt(binary, 3)
        # binary_filt = contour_closure(binary_filt)
        old_diff = np.sum(np.abs(binary_filt - binary))
    if iterations > 0:
        new_guesses = [old_guess - thresh_range, old_guess + thresh_range]
        diffs = []
        for thresh in new_guesses:
            binary = image > thresh
            binary_filt = medfilt(binary, 3)
            # binary_filt = contour_closure(binary_filt)
            diffs.append(np.sum(np.abs(binary_filt - binary)))
        best_i = np.argmin(diffs)
        best_guess = threshold_noise(image, new_guesses[best_i], thresh_range / 2, iterations - 1,
                                     diffs[best_i])
    else:
        best_guess = old_guess
    return best_guess


#   FUNCTION wrap
# Extended modulo function. Converts a number outside of a range between two numbers by continually
# adding or substracting the span between these numbers.
#   INPUTS:
# x: number to be wrapped
# low (default: 0): lowest value of the wrap
# high (default: 0): highest value of the wrap
#   OUTPUTS
# x: wrapped number

def wrap(x, low=0, high=360):
    angle_range = high - low
    x = ((x-low)%angle_range)+low
    return x


#   FUNCTION contour_closure
# Removes regions on a binarised image with an area less than a value defined by size_threshold.
# This is performed by finding the contours and the area of the contours, thus providing no change
# to the bulk of the image itself (as a morphological closure would)
#   INPUTS:
# source: image to be closed
# size_threshold (default: 100): area in pixels that a contour is compared to before being closed
# type_bool (default: True): sets data to a boolean type
#   OUTPUTS
# image: data with contours closed

def contour_closure(source, size_threshold=50, type_bool=True):
    source = np.array(source).astype('uint8')
    image = np.zeros_like(source)
    cv2_image, contours, hierarchy = cv2.findContours(source, cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > size_threshold:
            new_contours.append(contour)
    cv2.drawContours(image, new_contours, -1, 1, thickness=cv2.FILLED)
    if type_bool:
        image = image.astype(bool)
    return image


#   FUNCTION find_a_domains
# Determines the a-domains in an amplitude image, by looking for points of high second derivative.
# These points are then fit to lines, and these lines filtered to the most common lines that are 
# either parallel or perpendicular to one another. If given, the phase data can also be used to
# distinguish between a-domains and 180 degree walls.
#   INPUTS:
# amplitude: amplitude data containing the a-domains to be found
# binarised_phase: binarised phase data used to differentiate a-domain walls from c-domain walls
# direction (default: None): Direction of the a domains found:
#     None: Finds domain walls at any angle
#     'Vert': Finds vertical domain walls
#     'Horz': Finds horizontal domain walls
# filter_width (default: 15): total width of the filter, in pixels, around the domain-wall
#     boundaries. This is the total distance - so half this value is applied to each side.
# thresh_factor (default: 2): factor used by binarisation. A higher number gives fewer valid points.
# dilation (default: 2): amount of dilation steps to clean image
# erosion (default: 4): amount of erosion steps to clean image
# line_threshold (default: 80): minimum number of votes (intersections in Hough grid cell)
# min_line_length (default: 80): minimum number of pixels making up a line
# max_line_gap (default: 80): maximum gap in pixels between connectable line segments
# plots (default: [None]): option to plot intermediary steps. Plots if the following are in array:
#     'amp': Raw amplitude data that contains a-domains
#     'phase': Binarised phase data
#     'filter': Filter made from the domain walls visible in phase
#     'spline': Spline fit of original amplitude data
#     'first_deriv': First derivitave of amplitude
#     'second_deriv': Second derivitave of amplitude
#     'binary': Binarisation of second derivative
#     'erode': Binarisation data after an erosion filter is applied
#     'lines': Lines found, and should correspond to a-domains on original amplitude image
#     'clean': Lines found, after filtering to the most common angles
#   OUTPUTS
# result: hdf5_dict containing the predicted a-domains, and the binarisation threshold in attributes

def find_a_domains(amplitude, binarised_phase=None, direction=None, filter_width=15,
                   thresh_factor=2, dilation=2, erosion=4, line_threshold=50,
                   min_line_length=50, max_line_gap=10, plots=None):
    if binarised_phase is not None:
        domain_wall_filter = create_domain_wall_filter(binarised_phase,
                                                       filter_width=filter_width,
                                                       plots=plots)
    else:
        domain_wall_filter = np.zeros_like(amplitude) + 1
    a_estimate, bin_thresh = estimate_a_domains(amplitude, domain_wall_filter,
                                                direction=direction,
                                                plots=plots,
                                                thresh_factor=thresh_factor,
                                                dilation=dilation, erosion=erosion)

    # Find Lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    line_image = np.copy(a_estimate) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(a_estimate, rho, theta, line_threshold, np.array([]),
                            min_line_length, max_line_gap)

    if lines is not None:
        # Draw lines, filtering with phase filter if possible
        phase_filter_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if binarised_phase is not None:
                    blank = np.zeros_like(line_image)
                    one_line = cv2.line(blank, (x1, y1), (x2, y2), (255, 0, 0), 5)
                    points_outside_mask = one_line * domain_wall_filter
                    if np.sum(points_outside_mask) > 0.2 * np.sum(one_line):
                        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                        phase_filter_lines.append(line)
                else:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                    phase_filter_lines.append(line)
        lines_edges = cv2.addWeighted(a_estimate, 0.8, line_image, 1, 0)
        pt.intermediate_plot(line_image, 'lines', plots, 'Lines Found')

        # Find angles of each line
        angles = []
        for line in phase_filter_lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    angles.append(90)
                else:
                    angles.append((180 * np.arctan((y2 - y1) / (x2 - x1)) / np.pi))

        # Find first angle guess
        if direction == 'Vert':
            key_angles = [-90, 90]
        elif direction == 'Horz':
            key_angles = [0, 180]
        else:
            key_angles = find_desired_angles(angles)

        # Filter To Angle-Valid Lines
        angle_filter_lines = []
        i = 0
        for angle in angles:
            for key_angle in key_angles:
                if check_within_angle_range(angle, key_angle, 1):
                    angle_filter_lines.append(phase_filter_lines[i])
            i = i + 1

        # Draw Lines
        line_image = np.copy(a_estimate) * 0  # creating a blank to draw lines on
        for line in angle_filter_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
        lines_edges = cv2.addWeighted(a_estimate, 0.8, line_image, 1, 0)
        pt.intermediate_plot(line_image, 'clean', plots, 'Lines Found, after filtering')

    result = pt.hdf5_dict(line_image, binarisation_threshold=bin_thresh)
    return result


#   FUNCTION create_domain_wall_filter
# Creates a filter from the phase binarisation data, which can be used to find a-domains
#   INPUTS:
# phase: Binarised phase image from which images are to be obtained
# filter_width (default: 15): total width of the filter, in pixels, around the domain-wall
#     boundaries. This is the total distance - so half this value is applied to each side.
# plots (default: []): option to plot intermediary steps. Plots if the following are in array:
#     'phase': Binarised phase data
#     'filter': Filter made from the domain walls visible in phase
#   OUTPUTS
# domain_wall_filter: filter made from phase binarisation data

def create_domain_wall_filter(phase, filter_width=15, plots=[]):
    # Binarised_Phase
    binPhase = np.copy(phase)
    pt.intermediate_plot(binPhase, 'phase', plots, 'Binarised Phase')

    # Create Filter
    ite = filter_width // 2
    domain_wall_filter = ~(binary_dilation(binPhase, iterations=ite) ^
                           binary_erosion(binPhase, iterations=ite, border_value=1))
    pt.intermediate_plot(domain_wall_filter, 'filter', plots, 'Domain Wall Filter')
    return domain_wall_filter


#   FUNCTION estimate_a_domains
# Refines an amplitude image to points of higher second derivative, which are likely to correspond
# to domain walls. If phase is given, this is used to further the filter the lines to find only
# the a-domains. Function allows for optional viewing of intermediate steps.
#   INPUTS:
# amplitude: Amplitude image from which domain walls are to be emphasised
# domain_wall_filter (default: None): Filter used during row alignment to ignore effects of 180
#     degree walls
# direction (default: None): Direction of the derivative taken:
#     None: Takes both derivatives, adding together the values of the second derivative.
#     'Vert': Finds vertical domain walls (differentiates horizontally)
#     'Horz': Finds horizontal domain walls (differentiates vertically)
# plots (default: []): option to plot intermediary steps. Plots if the following are in array:
#     'amp': Raw amplitude data that contains a-domains
#     'row_align': Data after row-alignment (if any)
#     'spline': Spline fit of original amplitude data
#     'first_deriv': First derivitave of amplitude
#     'second_deriv': Second derivitave of amplitude
#     'binary': Binarisation of second derivative
#     'erode': Binarisation data after an erosion filter is applied
# thresh_factor (default: 2): factor used by binarisation. A higher number gives fewer valid points.
# dilation (default: 2): amount of dilation steps to clean image
# erosion (default: 4): amount of erosion steps to clean image
#   OUTPUTS
# filtered_deriv_amp: adjusted amplitude image made to highlight points of higher second derivative
# thresh: the threshold value used to find the a-domains

def estimate_a_domains(amplitude, domain_wall_filter=None, direction=None, plots=[],
                       thresh_factor=2, dilation=2, erosion=4):
    # Raw Data
    amp = np.copy(amplitude)
    pt.intermediate_plot(amp, 'amp', plots, 'Original Data')

    # Row Alignment, if direction set
    if direction == 'Vert':
        amp = align_rows(amp, domain_wall_filter)
    elif direction == 'Horz':
        amp = align_rows(amp, domain_wall_filter, cols=True)
    pt.intermediate_plot(amp, 'row_align', plots, 'Row Aligned Data')

    # Fit to a spline (reduce high frequency noise)
    spline_amp = cspline2d(amp, 2.0)
    pt.intermediate_plot(spline_amp, 'spline', plots, 'Spline Fitted Data')

    # Find derivatives to highlight peaks
    if direction == 'Vert':
        first_deriv = np.gradient(spline_amp)[1]
        pt.intermediate_plot(first_deriv, 'first_deriv', plots, 'First Derivatives')
        deriv_amp = (np.gradient(first_deriv))[1]
        pt.intermediate_plot(deriv_amp, 'second_deriv', plots, 'Second Derivatives')
    elif direction == 'Horz':
        first_deriv = np.gradient(spline_amp)[0]
        pt.intermediate_plot(first_deriv, 'first_deriv', plots, 'First Derivatives')
        deriv_amp = (np.gradient(first_deriv))[0]
        pt.intermediate_plot(deriv_amp, 'second_deriv', plots, 'Second Derivatives')
    else:
        if direction is not None:
            print('Direction should be set to either \'Vert\', \'Horz\' or None. Behaviour\
                    defaulting to None')
        first_deriv = np.gradient(spline_amp)
        pt.intermediate_plot(first_deriv[0] + first_deriv[1], 'first_deriv', plots,
                             'First Derivatives')
        second_deriv_y = np.gradient(first_deriv[0])[0]
        second_deriv_x = np.gradient(first_deriv[1])[1]
        deriv_amp = second_deriv_y + second_deriv_x
        pt.intermediate_plot(deriv_amp, 'second_deriv', plots, 'Second Derivatives')

    # Binarise second derivative
    thresh = threshold_after_peak(deriv_amp, thresh_factor)
    binary = (deriv_amp > thresh)
    pt.intermediate_plot(binary, 'binary', plots, 'Binarised Derivatives')

    # Remove Small Points
    filtered_deriv_amp = binary_erosion(binary_dilation(binary, iterations=dilation),
                                        iterations=erosion)
    pt.intermediate_plot(filtered_deriv_amp, 'erode', plots, 'Eroded Binary')
    return filtered_deriv_amp.astype(np.uint8), thresh


#   FUNCTION align_rows
# Aligns rows (or cols) of an array, with a mask provided
#   INPUTS:
# array: the array to be aligned
# mask (default: None): mask of data to be ignored when aligning rows
# cols (default: False): If set to true, the columns are instead aligned
#   OUTPUTS
# new_array: the row (or col) aligned array

def align_rows(array, mask=None, cols=False):
    if mask is None:
        mask = 1 + np.zeros_like(array)
    if cols:
        array = np.transpose(array)
        mask = np.transpose(mask)
    masked_array = np.copy(array)
    masked_array = np.where(mask == 0, np.nan, masked_array)
    new_array = np.zeros_like(array)
    for i in range(np.shape(array)[0]):
        if all(np.isnan(masked_array[i])) or (np.mean(mask[i]) < 0.05):
            new_array[i] = array[i] - np.nanmean(array[i])
        else:
            new_array[i] = array[i] - np.nanmean(masked_array[i])
    if cols:
        new_array = np.transpose(new_array)
    return new_array


#   FUNCTION threshold_after_peak
# Creates a threshold value, used when finding a-domains. This works by creatinga histogram of all
# valid values, and finding the maximum value of this histogram. The threshold is the point where
# the height of the maximum is less than the the maximum divided by the factor passed to this
# function.
#   INPUTS:
# deriv_amp: data passed in to obtain the threshold.
# thresh_factor (default: 2): The factor the maximum is divided by to find the threshold. A higher
#     number gives fewer valid points.
#   OUTPUTS
# thresh: the determined value of the optimal threshold

def threshold_after_peak(deriv_amp, factor=4):
    deriv_hist = np.histogram(deriv_amp.ravel(), bins=256)
    max_counts = np.max(deriv_hist[0])
    found_max = False
    i = 0
    for count in deriv_hist[0]:
        if count == max_counts:
            found_max = True
        if found_max == True and count < max_counts / factor:
            found_max = False
            thresh = deriv_hist[1][i]
        i = i + 1
    return thresh


#   FUNCTION find_desired_angles
# Finds best angles to find a-domains, by sorting all angles into a histogram and finding the most
# common angle. A list of this angle, its antiparallel, and its two perpendiculars are then 
# returned.
#   INPUTS:
# raw_data: a list of angles to be given
#   OUTPUTS
# angles: A list of the four angles that the a-domains should fit to

def find_desired_angles(raw_data):
    angles = np.zeros(4)
    ydata, bin_edges = np.histogram(raw_data, bins=360, range=[-180, 180])
    base_angle = (np.argmax(ydata)) - 180
    for i in range(4):
        angles[i] = wrap(base_angle + 90 * i, -180, 180)
    return angles


#   FUNCTION check_within_angle_range
# Checks if an angle is within a valid range around another angle given. This function uses the wrap
# functionality to search a full revolution.
#   INPUTS:
# angle: one of the angles to be searched
# key_angle: another angle to be compared to
# angle_range: the range that angle and key_angle must be within one another
# low (default: -180): the minimum value of the angle span
# high (default: 180): the maximum value of the angle span
#   OUTPUTS
# status: a bool stating whether the angles are in range (True) or not (False)

def check_within_angle_range(angle, key_angle, angle_range, low=-180, high=180):
    low_angle = wrap(key_angle - angle_range, low, high)
    high_angle = wrap(key_angle + angle_range, low, high)
    status = False
    if low_angle < angle:
        if angle < high_angle:
            status = True
    if low_angle > high_angle:
        if angle < high_angle:
            status = True
        elif angle > low_angle:
            status = True
    return status


#   FUNCTION find_a_domain_angle_
# Creates a transformation matrix that would rotate each image around the centre such that a-domains
# (or some other vertical feature) is oriented vertically and horizontally. Works by finding the
# a-domains (by looking for points of high second derivative), finding the most common angles in
# for these a-domains, and taking the median of these angles along all images
#   INPUTS:
# filename: name of hdf5 file containing data
# all_input_criteria: criteria to identify paths to source files using pt.path_search. First pass
#     amplitude data, then pass phase binarisation data.
# filter_width (default: 15): total width of the filter, in pixels, around the domain-wall
#     boundaries. This is the total distance - so half this value is applied to each side.
# thresh_factor (default: 2): factor used by binarisation. A higher number gives fewer valid points.
# dilation (default: 2): amount of dilation steps to clean image
# erosion (default: 4): amount of erosion steps to clean image
# line_threshold (default: 80): minimum number of votes (intersections in Hough grid cell)
# min_line_length (default: 80): minimum number of pixels making up a line
# max_line_gap (default: 80): maximum gap in pixels between connectable line segments
# plots (default: [None]): option to plot intermediary steps. Plots if the following are in array:
#     'amp': Raw amplitude data that contains a-domains
#     'phase': Binarised phase data
#     'filter': Filter made from the domain walls visible in phase
#     'spline': Spline fit of original amplitude data
#     'first_deriv': First derivitave of amplitude
#     'second_deriv': Second derivitave of amplitude
#     'binary': Binarisation of second derivative
#     'erode': Binarisation data after an erosion filter is applied
#     'lines': Lines found, and should correspond to a-domains on original amplitude image
#    OUTPUTS
# null

def find_a_domain_angle_(filename, all_input_criteria, filter_width=15, thresh_factor=2,
                         dilation=2, erosion=4, line_threshold=80, min_line_length=80,
                         max_line_gap=80, plots=None):
    all_in_path_list = pt.path_search(filename, all_input_criteria, repeat='block')
    in_path_list = all_in_path_list[0]
    if len(all_in_path_list) != 1:
        pb_path_list = all_in_path_list[1]
    else:
        pb_path_list = None

    out_folder_locations = pt.find_output_folder_location(filename, 'rotation_params', in_path_list)

    rotation_list = []
    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for index in range(len(in_path_list)):
            path = in_path_list[index]

            if pb_path_list is not None:
                domain_wall_filter = create_domain_wall_filter(f[pb_path_list[index]],
                                                               filter_width=filter_width,
                                                               plots=plots)
            else:
                domain_wall_filter = np.zeros_like(f[path]) + 1
            a_estimate, bin_thresh = estimate_a_domains(f[path], domain_wall_filter,
                                                        plots=plots,
                                                        thresh_factor=thresh_factor,
                                                        dilation=dilation,
                                                        erosion=erosion)

            # Find Lines
            rho = 1  # distance resolution in pixels of the Hough grid
            theta = np.pi / 180  # angular resolution in radians of the Hough grid
            line_image = np.copy(a_estimate) * 0  # creating a blank to draw lines on

            # Run Hough on edge detected image
            # Output "lines" is an array containing endpoints of detected line segments
            lines = cv2.HoughLinesP(a_estimate, rho, theta, line_threshold, np.array([]),
                                    min_line_length, max_line_gap)

            if lines is not None:
                # Draw lines, filtering with phase filter if possible
                valid_lines = []
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if pb_path_list is not None:
                            blank = np.zeros_like(line_image)
                            one_line = cv2.line(blank, (x1, y1), (x2, y2), (255, 0, 0), 5)
                            points_outside_mask = one_line * domain_wall_filter
                            if np.sum(points_outside_mask) > 0.2 * np.sum(one_line):
                                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                                if (x1 != x2) and (y1 != y2):
                                    valid_lines.append(line)
                        else:
                            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                            if (x1 != x2) and (y1 != y2):
                                valid_lines.append(line)
                lines_edges = cv2.addWeighted(a_estimate, 0.8, line_image, 1, 0)
                pt.intermediate_plot(line_image, 'lines', plots, 'Lines Found')

                # Find first angle guess
                angles = []
                for line in valid_lines:
                    for x1, y1, x2, y2 in line:
                        if x2 == x1:
                            angles.append(90)
                        else:
                            angles.append((180 * np.arctan((y2 - y1) / (x2 - x1)) / np.pi))
                key_angles = find_desired_angles(angles)

                # Refine angle estimate
                for repetitions in range(3):
                    valid_lines = []
                    angle_offsets = []
                    i = 0
                    for angle in angles:
                        for key_angle in key_angles:
                            if check_within_angle_range(angle, key_angle, 2.5):
                                valid_lines.append(lines[i])
                                angle_offsets.append(angles[i] - key_angle)
                        i = i + 1
                    key_angles = key_angles + np.mean(angle_offsets)
                rotation_deg = -key_angles[np.argmin(np.abs(key_angles))]
                rotation_list.append(rotation_deg)
                average_angle = np.mean(rotation_list)
            pt.progress_report(index + 1, len(in_path_list), start_time, 'a_angle',
                               in_path_list[index])

        rotation_array = np.array(rotation_list)
        rotation_array = sorted(rotation_array[~np.isnan(rotation_array)])
        average_angle = rotation_array[int(len(rotation_array) / 2)]
        orig_y, orig_x = (f[in_path_list[index]].attrs['shape'])
        warp_matrix = cv2.getRotationMatrix2D((orig_x / 2, orig_y / 2), average_angle, 1)
        data = pt.write_output_f(f, warp_matrix, out_folder_locations[0], in_path_list,
                                 find_a_domain_angle_, locals(), output_name = filename.split('.')[0])
        data.attrs['angle offset (degs)'] = average_angle
        data.attrs['binarisation_threshold'] = bin_thresh
        data.attrs['filter_width'] = filter_width
        data.attrs['line_threshold'] = line_threshold
        data.attrs['min_line_length'] = min_line_length
        data.attrs['max_line_gap'] = max_line_gap
        data.attrs['thresh_factor'] = thresh_factor


#   FUNCTION rotation_alignment_
# Applies a rotation matrix to an image. An option also allows for cropping to the largest common
# area, which is found via trial and error. If one rotation matrix is given, it is applied to all
# images given. If multiple rotation matrices are given, it applies each rotation matrix n times
# consecutively, where n is the amount of images divided by the number of rotation matrices. If this
# would not be a whole number, the program returns an error and ends without running.
#   INPUTS:
# filename: name of hdf5 file containing data
# all_input_criteria: criteria to identify paths to source files using pt.path_search. First should
#     be data to be corrected. Second should be rotation parameters.
# cropping (default: True): determines if the image should be cropped to the maximum common area.
#     If this value is set to False, the image will not be intentionally cropped and the image will
#     maintain consistent dimensions. This will often result in some cropping regardless.
#   OUTPUTS
# null
#   TO DO:
# Allow for true non-cropping, which would extend the border to the maximum possible limit.

def rotation_alignment_(filename, all_input_criteria, cropping=True):
    all_in_path_list = pt.path_search(filename, all_input_criteria, repeat='block')
    in_path_list = all_in_path_list[0]
    rm_path_list = all_in_path_list[1]

    rotation_matrices = []
    with h5py.File(filename, "a") as f:
        for path in rm_path_list[:]:
            rotation_matrices.append(np.copy(f[path]))

    out_folder_locations = pt.find_output_folder_location(filename, 'rotation_alignment',
                                                          in_path_list)
    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for i in range(len(in_path_list)):
            orig_img = np.copy(f[in_path_list[i]])

            orig_y, orig_x = (f[in_path_list[i]].attrs['shape'])

            if orig_img.dtype == bool:
                array_is_bool = True
                orig_img = orig_img.astype(float)
            else:
                array_is_bool = False

            new_img = cv2.warpAffine(orig_img, rotation_matrices[i], (orig_x, orig_y),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                     borderValue=np.nan)

            largest_area = 0
            if cropping:
                if i == 0:
                    for top in range(int(np.floor(orig_y / 2))):
                        width_left = 0
                        for x in range(len(new_img[top])):
                            if not np.isnan(new_img[top][x]):
                                if width_left == 0:
                                    width_left = (orig_x / 2) - x
                                width_right = x + 1 - (orig_x / 2)
                        height = (orig_y / 2) - top
                        width = np.min([width_left, width_right])
                        area = height * width
                        if area > largest_area:
                            largest_area = area
                            best_top = top
                            best_left = int((orig_x / 2) - width)
                new_img = new_img[best_top:orig_y - best_top, best_left:orig_x - best_left]
                while np.isnan(sum(new_img[-1])):
                    new_img = new_img[:-1]

            if array_is_bool:
                new_img = new_img.astype(bool)

            data = pt.write_output_f(f, new_img, out_folder_locations[i], [in_path_list[i],
                                                                           rm_path_list[i]],
                                    rotation_alignment_, locals())
            propagate_scale_attrs(data, f[in_path_list[i]])
            pt.progress_report(i + 1, len(in_path_list), start_time, 'a_alignment',
                               in_path_list[i])


#   FUNCTION threshold_ratio
# Thresholds an image by passing in a ratio between the minimum and maximum values of this image
#   INPUTS:
# image: image to be thresholded
# thresh_ratio (default: 0.5): ratio between the minimum and maximum of the image to threshold
#   OUTPUTS
# result: hdf5_dict thresholded image, and the threshold value in attrs

def threshold_ratio(image, thresh_ratio=0.5):
    max_level = np.nanmax(image)
    min_level = np.nanmin(image)
    real_threshold = min_level + (thresh_ratio * (max_level - min_level))
    thresh_data = image > real_threshold
    result = pt.hdf5_dict(thresh_data, threshold=real_threshold)
    return result


#   FUNCTION directional_skeletonize
# 'skeletonizes' a binary image either vertically or horizontally. This is done by finding the
# contours of each shape. The centre of each of these contours are then taken and extended either
# vertically or horizontally to the edge of each shape. If the edge of this shape is within 10
# pixels of the edge of the image, the line is further extended to the end of the image. Extra lines
# can also be removed via the false_positives variable
#   INPUTS:
# domain_guess: image showing the estimates for the a_domains
# direction (default: 'Vert'): Direction of the a skeletonization process:
#     'Vert': Draws vertical lines
#     'Horz': Draws horizontal lines
# false_positives (default: None): a list of ints that defines which lines to be ignored. Each line
#     is described by an int, starting from number 0, which is the left- or up-most line.
# max_edge (default: 10): the distance a line will stretch to read the edge or another line
#   OUTPUTS
# all_domains: image showing all domains found
# good_domains: image showing only good domains that have not been filtered

def directional_skeletonize(domain_guess, direction='Vert', false_positives=None, max_edge=10):
    if (direction != 'Vert') and (direction != 'Horz'):
        print('direction should be set to either \'Vert\' or \'Horz\'')
        return

    if type(false_positives) != list:
        false_positives = [false_positives]

    if domain_guess.dtype != 'uint8':
        domain_guess = domain_guess.astype('uint8')

    # find contours in the binary image
    image, contours, hierarchy = cv2.findContours(domain_guess, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
    x_centres = []
    y_centres = []
    bad_domains = np.zeros_like(domain_guess)
    good_domains = np.zeros_like(domain_guess)
    for contour in contours:
        # calculate moments for each contour
        moment = cv2.moments(contour)

        # calculate x,y coordinate of center
        if moment['m00'] != 0:
            x_centres.append(int(moment["m10"] / moment["m00"]))
            y_centres.append(int(moment["m01"] / moment["m00"]))

    if direction == 'Vert':
        x_centres = sorted(x_centres)
        for i in range(len(x_centres)):
            x = x_centres[i]
            whole_line = domain_guess[:, x]
            zero_count = 0
            for y in range(len(whole_line)):
                if whole_line[y] == 0:
                    zero_count = zero_count + 1
                else:
                    if zero_count <= max_edge:
                        whole_line[y - zero_count:y] = 1
                    zero_count = 0
            if (zero_count != 0) and (zero_count <= max_edge):
                whole_line[-zero_count:] = 1
            if i in false_positives:
                bad_domains[:, x] = whole_line
            else:
                good_domains[:, x] = whole_line
    elif direction == 'Horz':
        y_centres = sorted(y_centres)
        for i in range(len(y_centres)):
            y = y_centres[i]
            whole_line = domain_guess[y]
            zero_count = 0
            for x in range(len(whole_line)):
                if whole_line[x] == 0:
                    zero_count = zero_count + 1
                else:
                    if zero_count <= max_edge:
                        whole_line[x - zero_count:x] = 1
                    zero_count = 0
            if (zero_count != 0) and (zero_count <= max_edge):
                whole_line[-zero_count:] = 1
            if i in false_positives:
                bad_domains[y] = whole_line
            else:
                good_domains[y] = whole_line

    all_domains = bad_domains + good_domains
    return all_domains, good_domains


#   FUNCTION final_a_domains
# Creates a folder with all a-domain data from the directionally skeletonized binary images (both
# horz and vert). This includes the final cleaning steps, where both vertical and horizontal lines
# are overlayed and compared; if one line overlaps (or approaches) a perpendicular and ends at a
# distance less than that described by closing_distance, the extra line is cut (or the approaching
# line extended) such that the line terminates where it would coincide with the perpendicular. The
# final dataset folder contains images of both the horizontal and vertical a-domains, a composite
# image made from both the horizontal and vertical domains, and separate lists of both the
# horizontal and vertical domains that contain coordinates for a start and end of each domain
#   INPUTS:
# orig_vert: image showing the vertical a-domains
# orig_horz: image showing the horizontal a-domains
# closing distance (default: 50): extra distance a line is extended to (or cut by) if it approaches
#     a perpendicular
#   OUTPUTS
# new_all: all a-domains
# new_vert: vert a-domains
# new_horz: horz a-domains
# np.array(vert_list): coordinates of end points of vert a-domains
# np.array(horz_list): coordinates of end points of horz a-domains

def final_a_domains(orig_vert, orig_horz, closing_distance=50):
    new_vert = np.copy(orig_vert)
    # Lines defined by x1, y1, x2, y2
    vert_list = []
    for x in range(np.shape(new_vert)[1]):
        if np.sum(orig_vert[:, x]) != 0:
            if np.sum(orig_vert[:, x]) == np.shape(new_vert)[0]:
                vert_list.append([x, 0, x, np.shape(new_vert)[0] - 1])
            else:
                if np.sum(orig_horz[:, x]) != 0:
                    domains_list = np.where(orig_horz[:, x] == 1)[0]
                    for domain in domains_list:
                        min_index = np.max([0, domain - closing_distance])
                        max_index = np.min([domain + 1 + closing_distance, len(orig_horz[:, x])])
                        vert_top_segment = new_vert[min_index:domain + 1, x]
                        if (np.sum(vert_top_segment) != 0) and (np.sum(vert_top_segment) !=
                                                                np.shape(vert_top_segment)[0]):
                            if vert_top_segment[0] == 1:
                                new_vert[min_index:domain + 1, x] = np.zeros_like(vert_top_segment) + 1
                            else:
                                new_vert[min_index:domain + 1, x] = np.zeros_like(vert_top_segment)
                        vert_bot_segment = new_vert[domain:max_index, x]
                        if (np.sum(vert_bot_segment) != 0) and (np.sum(vert_bot_segment) !=
                                                                np.shape(vert_bot_segment)[0]):
                            if vert_bot_segment[-1] == 1:
                                new_vert[domain:max_index, x] = np.zeros_like(vert_bot_segment) + 1
                            else:
                                new_vert[domain:max_index, x] = np.zeros_like(vert_bot_segment)
                line_found = False
                for y in range(np.shape(new_vert)[0]):
                    if (new_vert[y, x] == 1) and (not line_found):
                        line_found = True
                        y1 = y
                    if (new_vert[y, x] == 0) and line_found:
                        vert_list.append([x, y1, x, y - 1])
                        line_found = False
                if line_found:
                    vert_list.append([x, y1, x, np.shape(new_vert)[0] - 1])

    new_horz = np.copy(orig_horz)
    horz_list = []
    for y in range(np.shape(new_horz)[0]):
        if np.sum(orig_horz[y, :]) != 0:
            if np.sum(orig_horz[y, :]) == np.shape(new_horz)[1]:
                horz_list.append([0, y, np.shape(new_horz)[1] - 1, y])
            else:
                if np.sum(orig_vert[y, :]) != 0:
                    domains_list = np.where(orig_vert[y, :] == 1)[0]
                    for domain in domains_list:
                        min_index = np.max([0, domain - closing_distance])
                        max_index = np.min([domain + 1 + closing_distance, len(orig_vert[y, :])])
                        horz_lft_segment = new_horz[y, min_index:domain + 1]
                        if (np.sum(horz_lft_segment) != 0) and (np.sum(horz_lft_segment) !=
                                                                np.shape(horz_lft_segment)[0]):
                            if horz_lft_segment[0] == 1:
                                new_horz[y, min_index:domain + 1] = np.zeros_like(horz_lft_segment) + 1
                            else:
                                new_horz[y, min_index:domain + 1] = np.zeros_like(horz_lft_segment)
                        horz_rgt_segment = new_horz[y, domain:max_index]
                        if (np.sum(horz_rgt_segment) != 0) and (np.sum(horz_rgt_segment) !=
                                                                np.shape(horz_rgt_segment)[0]):
                            if horz_rgt_segment[-1] == 1:
                                new_horz[y, domain:max_index] = np.zeros_like(horz_rgt_segment) + 1
                            else:
                                new_horz[y, domain:max_index] = np.zeros_like(horz_rgt_segment)
                line_found = False
                for x in range(np.shape(new_horz)[1]):
                    if (new_horz[y, x] == 1) and (not line_found):
                        line_found = True
                        x1 = x
                    if (new_horz[y, x] == 0) and line_found:
                        horz_list.append([x1, y, x - 1, y])
                        line_found = False
                if line_found:
                    horz_list.append([x1, y, np.shape(new_horz)[1] - 1, y])

    new_all = np.maximum(new_vert, new_horz)
    return new_all, new_vert, new_horz, np.array(vert_list), np.array(horz_list)


#   FUNCTION switchmap
# Generates a switchmap from binarised phase data. A switchmap is a 2D array, where the number
# at each coordinate corresponds to the 'time' it takes that coordinate to switch phase. If a
# coordinate did not switch, it is set to a NaN.
#   INPUTS:
# *phase_list: list of all binarised phases used to identify the switchmap
# method (default: 'total'): determines the method used to generate the switchmap:
#     'maximum': switching occurs at the final time the coordinate switches
#     'minimum': switching occurs at the first time the coordinate switches
#     'median': switching occurs at the median of all times the coordinate switches
#     'total': switching occurs at the number of total scans that the coordinate is not switched
# source_path (default: None): path name of first source, used to find initial voltage (in mV)
# voltage_increment (default: None): increment of voltage (in mV) on each step, used to find
#     subsequent voltages
#   OUTPUTS
# result: hdf5_dict showing the switchmap, as well as the relevant voltages in attributes

def switchmap(*phase_list, method='total', source_path=None, voltage_increment=None):
    if source_path is not None:
        mV = source_path.split('mV')[-2]
        mV = mV.split('_')[-1]
        mV = int(mV)

        if voltage_increment is not None:
            voltage = []
            for i in range(len(phase_list)):
                voltage.append(mV + voltage_increment * i)
        else:
            voltage = mV

    switchmap = np.zeros_like(phase_list[0].astype(float))
    start_time = time.time()

    for i in range(phase_list[0].shape[0]):
        for j in range(phase_list[0].shape[1]):
            switch_list = []
            for phase in phase_list:
                switch_list.append(phase[i, j])
            if switch_list[0] == switch_list[-1]:
                switchmap[i, j] = np.nan
            else:
                changes = wherechanged(switch_list)
                if method == 'maximum':
                    switch_scan = np.ceil(np.nanmax(changes))
                elif method == 'minimum':
                    if len(changes) == changes[-1]:
                        switch_scan = changes[-1]
                    else:
                        scan = 0
                        while changes[scan] == scan + 1:
                            scan = scan + 1
                        switch_scan = changes[scan - 1]
                elif method == 'median':
                    switch_scan = np.ceil(np.nanmedian(changes))
                elif method == 'total':
                    switch_scan = np.sum(switch_list) + 1
                else:
                    print('Error: Invalid method submitted')
                switchmap[i, j] = switch_scan
        pt.progress_report(i + 1, phase_list[0].shape[0], start_time, 'switchmap',
                           'Scanning Row ' + str(i + 1))

    if source_path is None:
        result = switchmap
    else:
        if voltage_increment is None:
            result = pt.hdf5_dict(switchmap, voltage_mV=voltage)
        else:
            result = pt.hdf5_dict(switchmap, voltage_list_mV=voltage)
    return result


#   FUNCTION wherechanged
# Returns the indices where a list changes values
#   INPUTS:
# arr: the array or list that may contain changes
#   OUTPUTS
# where: a list of indices that changes occur

def wherechanged(arr):
    where = []
    for i in range(len(arr) - 1):
        diff = arr[i + 1] ^ arr[i]
        if diff != 0:
            where.append(i + 1)
    return where


#   FUNCTION switch_type_
# Generates data regarding the type of switch. This data is placed into two folders. In switch_type,
# a folder is made for each scan. This folder contains an array, where the number defines the type
# of switch that had occurred at that coordinate by that point (no distinction is made in the
# recency of switches). NaN = no switch; 1 = nucleation; 2 = motion; 3 = merging; 4 = errors. These
# are defined by the amount of neighbours for a switch (nucleation has 0, motion has 1, merging has
# 2, errors have another value). The second folder, switch_type_general, holds information common to
# the entire switchmap. In the 'Centres' subfolder, two 2D arrays are given. NucleationCentres has 1 
# at the centre of a nucleation, and 0 elsewhere, while ClosureCentres has 1 at the centre of a
# closure (the opposite of nucleation; where no more motion or merging can continue off of it), and
# 0 elsewhere (note that as closure can be either motion or merging, switch_type does not contain
# any information on closure). In the "JumpTypes" subfolder, 6 arrays are stored, which contain
# information on the type of switch. These are the four types of switching used in switch_type
# (Nucleation; Motion; Merging; and Errors), as well as one for all switching (TotalJumps) and one
# for closure (Closure), which has redundancy with Motion and Merging. In these arrays, each row
# signifies scan. Each entry shows the size of a switch at that scan. As the length of each row
# is thus arbitirary, NaNs are made to fill each row to ensure constant length.
#   INPUTS:
# filename: name of hdf5 file containing data
# all_input_criteria: criteria to identify paths to source files using pt.path_search. Should lead
#     to switchmap only.
#   OUTPUTS
# null

def switch_type_(filename, all_input_criteria):
    in_path_list = pt.path_search(filename, all_input_criteria)[0]
    out_folder_locations = pt.find_output_folder_location(filename, 'switch_type', '')
    with h5py.File(filename, "a") as f:
        switchmap = np.copy(f[in_path_list[0]])

        if 'voltage_list_mV' in f[in_path_list[0]].attrs:
            voltage_array = f[in_path_list[0]].attrs['voltage_list_mV']
        else:
            voltage_array = None
        total_scans = len(f[in_path_list[0]].attrs['source'])

        totalmap = np.zeros_like(switchmap.astype(float))
        totalmap[:] = np.nan

        alljumps_tot = []
        alljumps_nucl = []
        alljumps_mot = []
        alljumps_merg = []
        alljumps_error = []
        alljumps_closure = []

        nucl_centres = np.zeros_like(switchmap.astype(bool))
        closure_centres = np.zeros_like(switchmap.astype(bool))

        switchmap = switchmap.astype(int)
        start_time = time.time()

        for i in range(total_scans):
            alljumps_tot_1img = []
            alljumps_nucl_1img = []
            alljumps_mot_1img = []
            alljumps_merg_1img = []
            alljumps_error_1img = []
            alljumps_closure_1img = []

            # Extract when switching occured
            prev_scan = (switchmap <= i).astype(int)
            curr_scan = (switchmap <= i + 1).astype(int)
            switched_regions = curr_scan - prev_scan

            # Label areas where switching occured
            structuring_element = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            labeled_switched_regions, total_features = label(switched_regions, structuring_element)

            for m in range(total_features):
                feature_region = (labeled_switched_regions == m + 1)

                # Get positions of all points of each label
                feature_positions = np.nonzero(feature_region)
                label_surface = np.sum(feature_region)

                # define box around each labelled pixel
                box = np.zeros_like(switchmap)
                box = binary_dilation(feature_region, iterations=1)
                box_border = box ^ feature_region

                if np.array_equal(np.logical_and(box_border, prev_scan), box_border):
                    closure = True
                else:
                    closure = False

                # Get switched areas up to the previous scan, that are in the box
                boxed_pha_bin = np.zeros_like(switchmap)
                boxed_pha_bin = (prev_scan + np.isnan(switchmap)) * box_border

                # Label the switched areas in the box
                labeled_boxed_pha_bin, num_connectors = label(boxed_pha_bin, structuring_element)

                # Convert the labeled array into a 1d-array of (unique) label values whose
                # length gives the number of domains connecting the newly switched area
                num_pix = np.shape(switchmap)[0] * np.shape(switchmap)[1]
                boxed_labels = np.reshape(labeled_boxed_pha_bin, (1, num_pix))
                boxed_labels = np.unique(boxed_labels[boxed_labels != 0])
                number_of_connecting_doms = len(boxed_labels)

                # Define event type and append all points of that event to the totalmap
                # nucleation
                alljumps_tot_1img.append(label_surface)
                if number_of_connecting_doms == 0:
                    eventtype = 2
                    alljumps_nucl_1img.append(label_surface)
                    event_centre = skeletonize(feature_region, method='lee').astype(bool)
                    nucl_centres = nucl_centres + event_centre
                    for j in range(len(feature_positions[0])):
                        x = feature_positions[0][j]
                        y = feature_positions[1][j]
                        totalmap[x, y] = 2.

                # motion
                elif number_of_connecting_doms == 1:
                    eventtype = 1
                    alljumps_mot_1img.append(label_surface)
                    for j in range(len(feature_positions[0])):
                        x = feature_positions[0][j]
                        y = feature_positions[1][j]
                        totalmap[x, y] = 1.
                # merging
                elif number_of_connecting_doms > 1:
                    eventtype = 3
                    alljumps_merg_1img.append(label_surface)
                    for j in range(len(feature_positions[0])):
                        x = feature_positions[0][j]
                        y = feature_positions[1][j]
                        totalmap[x, y] = 3.
                else:
                    eventtype = 4
                    alljumps_error_1img.append(label_surface)
                    for j in range(len(feature_positions[0])):
                        x = feature_positions[0][j]
                        y = feature_positions[1][j]
                        totalmap[x, y] = 4.
                # closure
                if closure:
                    alljumps_closure_1img.append(label_surface)
                    event_centre = skeletonize(feature_region, method='lee').astype(bool)
                    closure_centres = closure_centres + event_centre

            alljumps_tot.append(alljumps_tot_1img)
            alljumps_nucl.append(alljumps_nucl_1img)
            alljumps_mot.append(alljumps_mot_1img)
            alljumps_merg.append(alljumps_merg_1img)
            alljumps_error.append(alljumps_error_1img)
            alljumps_closure.append(alljumps_closure_1img)

            if voltage_array is not None:
                name = str(voltage_array[i]).zfill(4) + '_mV'
            else:
                name = 'Scan_' + str(i).zfill(3)
            current_folder_location = out_folder_locations[0] + name
            data = pt.write_output_f(f, totalmap, current_folder_location, in_path_list,
                                     switch_type_, locals(), output_name='Switchmap')
            propagate_scale_attrs(data, f[in_path_list[0]])
            pt.progress_report(i + 1, total_scans, start_time, 'switch_type_', '[' + name + ']')

        gen_loc = pt.find_output_folder_location(filename, 'switch_type_general', 'Centres')[0]
        data = pt.write_output_f(f, nucl_centres, gen_loc, in_path_list, switch_type_, locals(), output_name='NucleationCentres')
        data = pt.write_output_f(f, closure_centres, gen_loc, in_path_list, switch_type_, locals(), output_name='ClosureCentres')

        gen_loc = pt.find_output_folder_location(filename, 'switch_type_general', 'JumpTypes',
                                                 True)[0]
        data = pt.write_output_f(f, fill_blanks(alljumps_tot), gen_loc, in_path_list, switch_type_, locals(), output_name='TotalJumps')
        data = pt.write_output_f(f, fill_blanks(alljumps_nucl), gen_loc, in_path_list, switch_type_, locals(), output_name='Nucleation')
        data = pt.write_output_f(f, fill_blanks(alljumps_mot), gen_loc, in_path_list, switch_type_, locals(), output_name='Motion')
        data = pt.write_output_f(f, fill_blanks(alljumps_merg), gen_loc, in_path_list, switch_type_, locals(), output_name='Merging')
        data = pt.write_output_f(f, fill_blanks(alljumps_error), gen_loc, in_path_list, switch_type_, locals(), output_name='Errors')
        data = pt.write_output_f(f, fill_blanks(alljumps_closure), gen_loc, in_path_list, switch_type_, locals(), output_name='Closure')


#   FUNCTION fill_blanks
# Takes a list of lists, and extends each individual list such that each list is the same size. This
# is done by introducing NaNs. An list that is, for example, [[1,2,3],[],[1]], would then become
# [[1,2,3],[np.nan, np.nan, np.nan],[1, np.nan, np.nan]]
#   INPUTS:
# list_of_lists: list of lists to be altered so each component list is the same length
#   OUTPUTS
# list_of_lists: the new list of lists, where each list is extended to the same length

def fill_blanks(list_of_lists):
    longest_list_length = 0
    for one_list in list_of_lists:
        longest_list_length = max(longest_list_length, len(one_list))
    for one_list in list_of_lists:
        extra_nans = longest_list_length - len(one_list)
        one_list.extend([np.nan] * extra_nans)
    return list_of_lists


#   FUNCTION interpolated_features
# Creates isolines from a switchmap, then interpolates it
#   INPUTS:
# switchmap: the switchmap used as the base for key features
#   OUTPUTS
# interpolation: interpolated features

def interpolated_features(switchmap):
    isolines = find_isolines(switchmap)

    isoline_y = []
    isoline_x = []
    isoline_z = []
    for i in range(np.shape(isolines)[0]):
        for j in range(np.shape(isolines)[1]):
            if isolines[i, j] != 0:
                isoline_x.append(j)
                isoline_y.append(i)
                isoline_z.append(isolines[i, j])
    grid_x, grid_y = np.mgrid[0:np.shape(isolines)[0]:1, 0:np.shape(isolines)[1]:1]
    interpolation = interpolate.griddata(np.array([isoline_y, isoline_x]).T, np.array(isoline_z),
                                         (grid_x, grid_y), method='linear', fill_value=np.nan)
    return interpolation


#   FUNCTION find_isolines
# Finds key features on a switchmap (or similar structures), by finding edges and subtracting 1. If
# the point is at the bottom of an edge with height greater than two, and not otherwise defined, the
# point is set to the switchmap value. Borders are also set to nan.
#   INPUTS:
# switchmap: the switchmap used as the base for key features
# set_midpoints (default: True): sets the middle of each contour as 0.5 less than switchmap value
#   OUTPUTS
# isolines: the key features on the switchmap

def find_isolines(switchmap, set_midpoints=True):
    isolines = np.zeros_like(switchmap)
    for i in range(np.shape(switchmap)[0]):
        for j in range(np.shape(switchmap)[1]):
            if np.isnan(switchmap[i, j]):
                isolines[i, j] = np.nan
            if (i==0) or (i==np.shape(switchmap)[0]-1) or (j==0) or (j==np.shape(switchmap)[1]-1):
                isolines [i, j] = np.nan
            else:
                for i_del in [-1, +1]:
                    if switchmap[i,j] < switchmap[i+i_del, j]-1:
                        isolines[i,j] = switchmap[i,j]
                for j_del in [-1, +1]:
                    if switchmap[i,j] < switchmap[i, j+j_del]-1:
                        isolines[i,j] = switchmap[i,j]
                for i_del in [-1, +1]:
                    if switchmap[i,j] > switchmap[i+i_del, j]:
                        isolines[i,j] = switchmap[i,j] - 1
                for j_del in [-1, +1]:
                    if switchmap[i,j] > switchmap[i, j+j_del]:
                        isolines[i,j] = switchmap[i,j] - 1
    
    if set_midpoints:
        midpoints = np.zeros_like(switchmap)
        for i in range(int(np.nanmax(switchmap))+1):
            midpoints = (skeletonize(binary_erosion(switchmap==i, iterations=3),
                                     method='lee'))+midpoints
        midpoints = midpoints.astype(bool)

        for i in range(np.shape(switchmap)[0]):
            for j in range(np.shape(switchmap)[1]):
                if midpoints[i,j]:
                    isolines[i,j] = switchmap[i,j]-0.5
    return isolines


#   FUNCTION differentiate
# Differentiates an image. Creates files corresponding to the magnitude, of the derivative and
# optionally, derivatives along both axis separately.
#   INPUTS:
# image: data to be differentiated
# return_directions (default: True): If set to false, only the derivative magnitude is stored.
#   OUTPUTS
# mag_deriv (if return_directions == False): array showing magnitude of the derivative at all points
# result (if return_directions == True): A tuple. The first entry is mag_deriv as above. Subsequent
#    entries are different derivatives of from different directions

def differentiate(image, return_directions=True):
    deriv = np.gradient(image)
    abs_deriv = np.abs(deriv)
    mag_deriv = np.sqrt(abs_deriv[0] ** 2 + abs_deriv[1] ** 2)
    mag_deriv = pt.hdf5_dict(mag_deriv, dimension='Abs')
    if not return_directions:
        result = mag_deriv
    else:
        result = [mag_deriv]
        for i in range(len(deriv)):
            result.append(pt.hdf5_dict(deriv[i], dimension=[i]))
        result = tuple(result)
    return result


#   FUNCTION crop
# Crops first and last rows and columns to remove regions without any meaningful data. The columns
# that are removed are defined by that in background
#   INPUTS:
# array: array to be cropped
# background (default: 0): value of background data that can be cropped out
#   OUTPUTS
# cropped_array: the array after being cropped

def crop(array, background=0):
    empty_rows = []
    for i in range(np.shape(array)[0]):
        if len(set(array[i])) == 1:
            if np.isnan(background):
                if np.isnan(array[i,0]):
                    empty_rows.append(i)
            else:
                if array[i,0] == background:
                    empty_rows.append(i)
    empty_cols = []
    for j in range(np.shape(array)[1]):
        if len(set(array[:,j])) == 1:
            if np.isnan(background):
                if np.isnan(array[0,j]):
                    empty_cols.append(j)
            else:
                if array[0,j] == background:
                    empty_cols.append(j)

    starting_row = 0
    ending_row = np.shape(array)[0]
    starting_col = 0
    ending_col = np.shape(array)[1]
    if empty_rows:
        if empty_rows[0]==0:
            starting_row = starting_row+1
            for i in empty_rows:
                if i==starting_row:
                    starting_row = starting_row+1
        if empty_rows[-1]==np.shape(array)[0]-1:
            ending_row = ending_row-1
            for i in reversed(empty_rows):
                if i==ending_row:
                    ending_row = ending_row-1
    if empty_cols:
        if empty_cols[0]==0:
            starting_col = starting_col+1
            for j in empty_cols:
                if j==starting_col:
                    starting_col = starting_col+1
        if empty_cols[-1]==np.shape(array)[1]-1:
            ending_col = ending_col-1
            for j in reversed(empty_cols):
                if j==ending_col:
                    ending_col = ending_col-1
    cropped_array = array[starting_row:ending_row, starting_col:ending_col]
    return cropped_array


#   FUNCTION uncrop_to_multiple
# Adds additional columns and rows around an array. The values added are defined by background,
# while the amount of rows and columns are defined by an array. The cols and rows are added equally
# in all directions, preferentially to the right/bottom if an odd amount are added
#   INPUTS:
# array: array to be uncropped
# multiple (default: [50,50]): list of dimensions that the data will be uncropped by. Data will be
#     uncropped to the the next multiple of the values provided here. ie, given the default value,
#     rows and cols will be extended such that there are 60, 120, 180, ... or etc. rows or cols
# background (default: 0): the values granted to the uncropped regions
#   OUTPUTS
# extended_array: array after it is uncropped

def uncrop_to_multiple(array, multiple=[50,50], background=0):
    extended_rows = int(np.ceil(np.shape(array)[0]/multiple[0])*multiple[0])
    extra_rows = extended_rows-np.shape(array)[0]
    offset_rows = int(np.floor(extra_rows/2))
    extended_cols = int(np.ceil(np.shape(array)[1]/multiple[1])*multiple[1])
    extra_cols = extended_cols-np.shape(array)[1]
    offset_cols = int(np.floor(extra_cols/2))
    extended_array = np.zeros([extended_rows,extended_cols])+background
    extended_array[offset_rows:np.shape(array)[0]+offset_rows,
                   offset_cols:np.shape(array)[1]+offset_cols] = array
    return extended_array


#   FUNCTION compress_to_shape
# Compresses an array to a smaller size. This takes rectangular sections of the original array,
# takes the average of it, and averages it into a single pixel. The size of the compressed shape is
# defined by the shape argument. If the initial array is not a direct multiple of the desired shape,
# the function will call uncrop_to_multiple to extend it and ensure clean rectangles.
#   INPUTS:
# array: array to be compressed
# shape (default: [50,50]): dimensions of the output array.
#   OUTPUTS
# compressed_array: array after it is compressed

def compress_to_shape(array, shape=[50,50]):
    compressed_array = np.zeros(shape)
    extended_array = uncrop_to_multiple(array, shape)
    row_compression = int(np.shape(extended_array)[0]/shape[0])
    col_compression = int(np.shape(extended_array)[1]/shape[1])
    for i in range(np.shape(compressed_array)[0]):
        for j in range(np.shape(compressed_array)[1]):
            compressed_array[i,j] = np.average(extended_array[i*row_compression:
                                                              (i+1)*row_compression,
                                                              j*col_compression:
                                                              (j+1)*col_compression])
    return compressed_array


#   FUNCTION decompress_to_shape
# Decompresses a smaller array into a larger array. Each pixel of the initial, smaller array is
# expanded into a larger rectangle of pixels, with the same value, on the larger array.
#   INPUTS:
# array: array to be decompressed
# shape: shape of the final, larger array
#   OUTPUTS
# decompressed_array: array after it is decompressed

def decompress_to_shape(array, shape):
    decompressed_array = np.zeros(shape)
    row_compression = np.ceil(shape[0]/np.shape(array)[0])
    col_compression = np.ceil(shape[1]/np.shape(array)[1])
    for i in range(np.shape(decompressed_array)[0]):
        for j in range(np.shape(decompressed_array)[1]):
            decompressed_array[i,j] = array[int(np.floor(i/row_compression)),
                                            int(np.floor(j/col_compression))]
    return decompressed_array
    

#   FUNCTION sample_fraction
# Randomly samples a contiguous block and small fraction of a larger array. Works by taking a small
# point (or smaller shape) and randomly expanding in all directions. To ensure uniformity, a
# 'supershape' is constructed, which takes the initial array and adds 5 pixels on each direction
# to reduce edge effects during sampling.
#   INPUTS:
# array: array to be sampled
# shape_fraction (default: 0.1): the fraction of the array that is actually sampled. eg., by
#     default, an area of 0.1 will return a block of area 10% of the overall array
# start_shape (default: None): A shape used to start the sample. Allows sample_fraction to feed into
#     itself and generate larger samples from smaller ones.
#   OUTPUTS
# curr_shape: the sample generated from the main array

def sample_fraction(array, shape_fraction=0.1, start_shape=None):
    basic_array = np.zeros_like(array)
    total_shapes = 300000 
    target_shape_size = shape_fraction*np.sum(array)

    #Create Coordinate List:
    coord_list = []
    coord_list_index_array = np.zeros_like(basic_array)
    index=0
    for i in range(np.shape(basic_array)[0]):
        for j in range(np.shape(basic_array)[1]):
            coord_list.append([i,j])
            coord_list_index_array[i,j] = index
            index = index+1
    neighbour_array = np.zeros_like(basic_array)
    neighbour_scaling = np.zeros_like(basic_array)+4
    
    if start_shape is None:
        #Pick First Point
        np.random.seed()
        init_i, init_j = coord_list[int(np.random.choice(coord_list_index_array.flatten()))]
        new_i = init_i
        new_j = init_j
        curr_shape = np.zeros_like(basic_array)
        curr_shape[init_i, init_j]=1
    else:
        start_shape = np.copy(start_shape)
        new_i = None
        new_j = None
        curr_shape = start_shape
        expanded_point = np.zeros_like(basic_array)
        for i in range(np.shape(basic_array)[0]):
            for j in range(np.shape(basic_array)[1]):
                if curr_shape[i,j]!= 0:
                    if i != 0:
                        if i == 1:
                            expanded_point[i-1,j] = expanded_point[i-1,j]+2
                        else:
                            expanded_point[i-1,j] = expanded_point[i-1,j]+2
                    if j != 0:
                        if j == 1:
                            expanded_point[i,j-1] = expanded_point[i,j-1]+2
                        else:
                            expanded_point[i,j-1] = expanded_point[i,j-1]+2
                    if i != np.shape(curr_shape)[0]-1:
                        if i == np.shape(curr_shape)[0]-2:
                            expanded_point[i+1,j] = expanded_point[i+1,j]+2
                        else:
                            expanded_point[i+1,j] = expanded_point[i+1,j]+2
                    if j != np.shape(curr_shape)[1]-1:
                        if j == np.shape(curr_shape)[1]-2:
                            expanded_point[i,j+1] = expanded_point[i,j+1]+2
                        else:
                            expanded_point[i,j+1] = expanded_point[i,j+1]+2
                            
    curr_shape_size = np.sum(curr_shape*array)
    while curr_shape_size < target_shape_size:
        #Find possible growths from that point
        last_point = np.zeros_like(basic_array)
        last_point[new_i, new_j]=1
        #Use mirror boundary conditions
        if new_i != None:
            expanded_point = np.copy(last_point)
            if new_i > 1:
                expanded_point[new_i-1, new_j]=1
            elif new_i == 1:
                expanded_point[new_i-1, new_j]=2
            if new_j > 1:
                expanded_point[new_i, new_j-1]=1
            elif new_j == 1:
                expanded_point[new_i, new_j-1]=2
            if new_i < np.shape(basic_array)[0]-2:
                expanded_point[new_i+1, new_j]=1
            elif new_i == np.shape(basic_array)[0]-2:
                expanded_point[new_i+1, new_j]=2
            if new_j < np.shape(basic_array)[1]-2:
                expanded_point[new_i, new_j+1]=1
            elif new_j == np.shape(basic_array)[1]-2:
                expanded_point[new_i, new_j+1]=2
        neighbour_array = neighbour_array+(expanded_point/neighbour_scaling)
        neighbour_array = neighbour_array*(1-curr_shape)

        #Grow from one of the growth coords
        thresh=0.75
        if np.any(neighbour_array>=thresh):
            #Grow if too surrounded
            total_to_change = np.sum(neighbour_array>=thresh)
            #print(total_to_change)
            if total_to_change>1:
                growth_locations = np.where(neighbour_array>=thresh)
                rand_num=np.random.randint(0, len(growth_locations[0]))
                new_i = growth_locations[0][rand_num]
                new_j = growth_locations[1][rand_num]
            else:
                new_coords = np.where(neighbour_array>=thresh)
                new_i = new_coords[0]
                new_j = new_coords[1]
        else:
            #Randomly grow:
            probability_array = np.copy(neighbour_array)
            normalise_factor = np.sum(probability_array)
            new_i,new_j=coord_list[int(np.random.choice(coord_list_index_array.flatten(),
                                                p=(probability_array/normalise_factor).flatten()))]
        curr_shape[new_i, new_j]=1
        curr_shape_size = np.sum(curr_shape*array)
        #Should make it only consider largest area?
    return curr_shape


#   FUNCTION MLE
# Generates a list of cutoffs and scaling parameters of a power law, using maximum likelihood
# estimation
#   INPUTS:
# x_list: list of events/avalanches
# show (default: False): shows the MLE plot
#   OUTPUTS
# x0_list: list of all x0 (cutoffs)
# a_list: list of all a (scaling parameters)

def MLE(x_list, show=False):
    x_arr = np.array(x_list)
    max_x0_check = round(max(x_list)/2)
    x0_arr=np.linspace(1, max_x0_check, max_x0_check)
    x0_list = list(x0_arr)
    a_list = []
    for x0 in x0_arr:
        a_est = 1+(len(x_arr[x_arr>x0])/(np.sum(np.log(x_arr[x_arr>x0]/(x0)))))
        a_list.append(a_est)
    if show:
        plt.semilogx(x0_list, a_list)
        plt.ylabel('a')
        plt.xlabel('x0')
        plt.title('MLE')
        plt.show()
        plt.close()
    return x0_list, a_list


#   FUNCTION KS_statistic
# Generates a list of cutoffs and scaling parameters of a power law, using maximum likelihood
# estimation
#   INPUTS:
# x_list: list of events/avalanches
# show (default: False): shows the MLE plot
#   OUTPUTS
# x0_list: list of all x0 (cutoffs)
# a_list: list of all a (scaling parameters)

def KS_statistic(x_list, x0_list, a_list, show=False):
    x_ks_all = np.array(sorted(x_list))
    D_list = []
    for i in range(len(x0_list)):
        x0_est = x0_list[i]
        a_est = a_list[i]
        x_ks = x_ks_all[x_ks_all >= x0_est]
        S_ks = np.linspace(0, 1, len(x_ks))
        P_ks = []
        for x in x_ks:
            P_ks.append(power_law_CDF(x, a_est, x0_est))
        #P_ks = np.array(P_ks)
        D = np.max(abs(P_ks-S_ks))
        D_list.append(D)
    if show:
        plt.semilogx(x0_list, D_list)
        plt.title('KS Statistic with varying x0')
        plt.ylabel('D')
        plt.xlabel('x0')
        plt.show()
        plt.close()
    return D_list


#   FUNCTION power_law_CDF
# Returns the CDF of a power law fit, given an event size, scaling parameter, and power law cutoff
#   INPUTS:
# x: an event size to find the cumulative probability of
# a: scaling parameter
# x0: power law cutoff
#   OUTPUTS
# P: cumulative probability

def power_law_CDF(x, a, x0):
    P = 1-((x/x0)**(1-a))
    return P


#   FUNCTION power_law_params
# Generates important parameters of a power law fit, given MLE and KS results
#   INPUTS:
# x_list: list of events
# x0_list: list of minimum cutoffs
# a_list: list of scaling parameters
# D_list: list of KS statistics
# max_x0 (default: 20): maximum value of x0 that will be considered
#   OUTPUTS
# P: parameters as a list, including, in order: minimum cutoff; scaling parameter; KS statistic for
#     optimal cutoff and scaling parameter; number of events above cutoff; total number of events;
#     average event size considered; largest event size considered

def power_law_params(x_list, x0_list, a_list, D_list, max_x0 = 20):
    x_array = np.array(x_list)
    min_D_arg = np.argmin(D_list[0:max_x0])
    optimal_x0 = x0_list[min_D_arg]
    optimal_a = a_list[min_D_arg]
    min_D = np.min(D_list)
    valid_n = sum(np.array(x_list)>optimal_x0)
    total_n = len(x_list)
    mean_x_size = np.mean(x_array[x_array>optimal_x0])
    max_x_size = np.max(x_array)
    total_x_size_thresh = np.sum(x_array[x_array>optimal_x0])
    total_x_size = np.sum(x_array)
    P = [optimal_x0, optimal_a, min_D, valid_n, total_n, mean_x_size, max_x_size,
         total_x_size_thresh, total_x_size]
    return P


#   FUNCTION power_law_params_force_fit
# Generates important parameters of a power law fit, given the scaling parameters and minimum cutoff
#   INPUTS:
# x_list: list of events
# optimal_x0: minimum cutoff used
# optimal_a: scaling parameter used
#   OUTPUTS
# P: parameters as a list, including, in order: minimum cutoff; scaling parameter; KS statistic for
#     optimal cutoff and scalign parameter; number of events above cutoff; total number of events;
#     average event size considered; largest event size considered

def power_law_params_force_fit(x_list, optimal_x0, optimal_a):
    x_array = np.array(x_list)
    x_ks_all = np.array(sorted(x_list))
    x_ks = x_ks_all[x_ks_all >= optimal_x0]
    S_ks = np.linspace(0, 1, len(x_ks))
    P_ks = []
    for x in x_ks:
        P_ks.append(power_law_CDF(x, optimal_a, optimal_x0))
    P_ks = np.array(P_ks)
    D = np.max(abs(P_ks-S_ks))
    valid_n = sum(np.array(x_list)>optimal_x0)
    total_n = len(x_list)
    mean_x_size = np.mean(x_array[x_array>optimal_x0])
    max_x_size = np.max(x_array)
    total_x_size_thresh = np.sum(x_array[x_array>optimal_x0])
    total_x_size = np.sum(x_array)
    P = [optimal_x0, optimal_a, D, valid_n, total_n, mean_x_size, max_x_size,
         total_x_size_thresh, total_x_size]
    return P


#   FUNCTION all_sample_fractions
# Generates a 4D array of several samples; the first axis allows choice of the fraction of each
# sampling; the second axis allows for each iteration of this fraction. The remaining two axes
# are the 2D array of each individual sample.
#   INPUTS:
# array: the array to be sampled
# iterations (default: 100): number of samples to be taken of for each fraction
# fractions (default: [0.1, 0.15, 0.2, 0.25]): fractions to be sampled of the array
# compression (default: [50,50]): the size to which the array is compressed to during sampling
# background (default: np.nan): values of background areas that may be removed during compression
#   OUTPUTS
# sample_fractions_all_fractions: 4D array describing all samples extracted

def all_sample_fractions(array, iterations=100, fractions=[0.1,0.15,0.2,0.25], compression=[50,50],
                         background=np.nan):
    #Compress array and generate "supershape" template from which subshapes are drawn
    if np.isnan(background):
        bool_array = ~np.isnan(array)
    else:
        bool_array = ~(array==background)
    cropped_array = crop(bool_array)
    expanded_array = uncrop_to_multiple(cropped_array, compression)
    compressed_array = compress_to_shape(expanded_array, compression)
    array_supershape = uncrop_to_multiple(compressed_array, [compression[0]+10, compression[1]+10])
    
    #Generate first generation of all shapes
    if type(fractions)!= list:
        fractions = [fractions]
    sample_fractions_all_fractions = []
    sample_fractions_one_fraction = []
    for sample_count in range(iterations):
        shape = sample_fraction(array_supershape, fractions[0]).astype(bool)
        sample_fractions_one_fraction.append(shape)
       
    #If multiple fractions are provides, generate successive generations
    if len(fractions)>1:
        for i in range(1,len(fractions)):
            sample_fractions_all_fractions.append(sample_fractions_one_fraction)
            sample_fractions_one_fraction = []
            for sample_count in range(iterations):
                shape = sample_fraction(array_supershape, fractions[i], 
                                        sample_fractions_all_fractions[i-1]
                                        [sample_count]).astype(bool)
                sample_fractions_one_fraction.append(shape)
        sample_fractions_all_fractions.append(sample_fractions_one_fraction)
    else:
        sample_fractions_all_fractions = sample_fractions_one_fraction
        
    sample_fractions_all_fractions = np.array(sample_fractions_all_fractions)
    #axes = [(fraction_num)][sample_num][y][x]
    return sample_fractions_all_fractions


#   FUNCTION multi_power_law
# Applies a power law fit to a switchmap (or similar), multiple times according to samples extracted
# from sample_fractions, and extracts parameters
#   INPUTS:
# switchmap: the array to be sampled
# sample_fractions: 4D array showing all samples, extracted from function all_sample_fractions
# compression (default: [50,50]): the size to which the array is compressed to during sampling
# background (default: np.nan): values of background areas that may be removed during compression
#   OUTPUTS
# all_params: parameters in an 3D array. First axis represents each fraction value provided by
#     sample_fractions. Second axis represents data for each particular sample. What remains is a 1D
#     array, which contains the following parameters in order: optimal cutoff; scaling parameter;
#     KS statistic for the optimal cutoff and scaling parameter; number of events above cutoff;
#     total number of events; average event size considered; largest event size considered

def multi_power_law(switchmap, sample_fractions, compression=[50,50], background=np.nan):
    all_params = np.zeros([np.shape(sample_fractions)[0], np.shape(sample_fractions)[1],9])

    #Recenter initial array in same manner as supershape
    if np.isnan(background):
        bool_array = ~np.isnan(switchmap)
    else:
        bool_array = ~(array==background)
    cropped_array = crop(bool_array)
    expanded_array = uncrop_to_multiple(cropped_array, compression)
    compressed_array = compress_to_shape(expanded_array, compression)
    array_supershape = uncrop_to_multiple(compressed_array, [compression[0]+10, compression[1]+10])
    copied_array = np.copy(switchmap)
    copied_array[~bool_array]=0
    scaled_array = uncrop_to_multiple(crop(copied_array*bool_array),compression)
            
    for fraction_num in reversed(range(np.shape(sample_fractions)[0])):
    #for fraction_num in (range(np.shape(sample_fractions)[0])):
        for shape_num in range(np.shape(sample_fractions)[1]):
            print(shape_num)
            shape = sample_fractions[fraction_num][shape_num]
            
            #Prepare the samples of the switchmap
            avalanche_size_list = []
            sample=decompress_to_shape(np.ceil(shape*array_supershape)[5:np.shape(shape)[0]-5,
                                                                       5:np.shape(shape)[1]-5],
                                       np.shape(expanded_array))
            sample=sample*scaled_array

            #Find Avalanche Sizes
            for i in range(1,int(np.max(sample)+1)):
                avalanche_size = np.sum(sample==i)
                if avalanche_size>0:
                    avalanche_size_list.append(avalanche_size)
            if fraction_num == (np.shape(sample_fractions)[0]-1):
            #if fraction_num == 0:
            
                #Using MLE, generate list of x0s and as
                x0_list, a_list = MLE(avalanche_size_list)

                #Find KS_Statistic
                D_list = KS_statistic(avalanche_size_list, x0_list, a_list)

                #Report Parameters
                params = power_law_params(avalanche_size_list, x0_list, a_list, D_list, max_x0=20)
                
            else:
                #optimal_x0 = all_params[0, shape_num, 0]
                #optimal_a = all_params[0, shape_num, 1]
                optimal_x0 = all_params[np.shape(sample_fractions)[0]-1, shape_num, 0]
                optimal_a = all_params[np.shape(sample_fractions)[0]-1, shape_num, 1]
                params = power_law_params_force_fit(avalanche_size_list, optimal_x0, optimal_a)
                
                
            #optimal_x0, optimal_a, min_D, valid_n, total_n, mean_x_size, max_x_size,
            #   total_x_size_thresh, total_x_size
            all_params[fraction_num, shape_num] = params
    return all_params


# FUNCTION centre_peak
# Given a dataset containing a gaussian peak, transforms the data by a linear shift such that the
# peak is at a given coordinate
#   INPUTS:
# x: 2D dataset containing position data in one of the spacial dimensions
# y: 2D dataset containing position data in one of the spacial dimensions
# z: 2D dataset containing value at the spacial dimensions defined by x and y
# theo_x: the x-position of where the peak should be
# theo_y: the y-position of where the peak should be
# xc_range (default: []): x-range of data searched for peak. By default, searches entire span
# yc_range (default: []): y-range of data searched for peak. By default, searches entire span
# angle_correction (default: False): if set to True, will attempt to fit x and y as angles to a
#     a change in displacement
# plot_fit (default: False): if set to True, plots the fit
# plot_name (default: 'fit'): name of output plot if generated
# plot_axes (default: ['','']): axes labels (x and y) for the output plot
#   OUTPUTS:
# xc_fitted: x-coordinate of centre of fitted peak
# yc_fitted: y-coordinate of centre of fitted peak

def centre_peak(x, y, z, theo_x, theo_y, xc_range = [], yc_range = [], angle_correction_deg = False,
                plot_fit=False, plot_name = 'fit', plot_axes = ['','']):
    measured_x, measured_y = find_peak_position(x, y, z, xc_range, yc_range, plot_fit,
                                                plot_name, plot_axes)
    if angle_correction_deg:
        angle_factor_x = np.cos(np.pi*theo_x/180)/np.cos(np.pi*x/180)
        angle_factor_y = np.cos(np.pi*theo_y/180)/np.cos(np.pi*y/180)
    else:
        angle_factor_x = 1
        angle_factor_y = 1
    adj_x = x + (theo_x - measured_x)*angle_factor_x
    adj_y = y + (theo_y - measured_y)*angle_factor_y
    return adj_x, adj_y


# FUNCTION find_peak_position
# Fits, and optionally plots, 2-dimensional rotated gaussian to 3 sets of 2D arrays
#   INPUTS:
# x: 2D dataset containing position data in one of the spacial dimensions
# y: 2D dataset containing position data in one of the spacial dimensions
# z: 2D dataset containing value at the spacial dimensions defined by x and y
# xc_range (default: []): x-range of data searched for peak. By default, searches entire span
# yc_range (default: []): y-range of data searched for peak. By default, searches entire span
# plot_fit (default: False): if set to True, plots the fit
# plot_name (default: 'fit'): name of output plot if generated
# plot_axes (default: ['','']): axes labels (x and y) for the output plot
#   OUTPUTS:
# xc_fitted: x-coordinate of centre of fitted peak
# yc_fitted: y-coordinate of centre of fitted peak

def find_peak_position(x,y,z, xc_range = [], yc_range = [], plot_fit=False, plot_name = 'fit',
                       plot_axes = ['','']):
    if xc_range and yc_range:
        xc_min = xc_range[0]
        xc_est = np.mean([xc_range])
        xc_max = xc_range[1]
        yc_min = yc_range[0]
        yc_est = np.mean([yc_range])
        yc_max = yc_range[1]
        ilow = 0
        ihigh = np.shape(y)[0]-1
        for i in range(np.shape(y)[0]):
            if all(y[i,:]<yc_min):
                ilow = i
            if all(y[i,:]>yc_max):
                ihigh = i
        jlow = 0
        jhigh = np.shape(x)[1]-1
        for j in range(np.shape(x)[1]):
            if all(x[:,j]<xc_min):
                jlow = j
            if all(x[:,j]>xc_max):
                jhigh = j
    else:
        min_z = np.min(z)
        max_z = np.max(z)
        ratio = max_z/min_z
        threshold = ratio**0.75
        
        ijmax = np.argmax(z)
        imax = int(np.floor(ijmax/np.shape(z)[1]))
        jmax = ijmax%np.shape(z)[1]

        xc_est = x[imax,jmax]
        ilow = imax
        while (z[ilow, jmax] >= z[imax,jmax]/threshold) and ilow > 0:
            ilow -= 1
        ihigh = imax
        while (z[ihigh, jmax] >= z[imax,jmax]/threshold) and ihigh < np.shape(z)[0]-1:
            ihigh += 1
        xc_min = np.min([x[ilow, jmax],x[ihigh,jmax]])
        xc_max = np.max([x[ilow, jmax],x[ihigh,jmax]])

        yc_est = y[imax,jmax]
        jlow = jmax
        while (z[imax, jlow] >= z[imax,jmax]/threshold) and jlow > 0:
            jlow -= 1
        jhigh = jmax
        while (z[imax, jhigh] >= z[imax,jmax]/threshold) and jhigh < np.shape(z)[1]-1:
            jhigh += 1
        yc_min = np.min([y[imax, jlow],y[imax,jhigh]])
        yc_max = np.max([y[imax, jlow],y[imax,jhigh]])
    
    if ilow > ihigh:
        ilow, ihigh = ihigh, ilow
    if jlow > jhigh:
        jlow, jhigh = jhigh, jlow
    
    z0_est = np.min(z[ilow:ihigh, jlow:jhigh])
    z0_min = z0_est/2
    z0_max = z0_est*2
    
    amp_est = np.max(z[ilow:ihigh, jlow:jhigh])-z0_est
    amp_min = amp_est/2
    amp_max = amp_est*2
    
    theta_est = 0
    theta_min = -np.pi
    theta_max = np.pi
    
    w1_est = 0.1
    w1_min = 0
    w1_max = 10
    
    w2_est = 0.1
    w2_min = 0
    w2_max = 10
    
    p_est = [amp_est, xc_est, w1_est, yc_est, w2_est, theta_est, z0_est]
    p_min = [amp_min, xc_min, w1_min, yc_min, w2_min, theta_min, z0_min]
    p_max = [amp_max, xc_max, w1_max, yc_max, w2_max, theta_max, z0_max]
    
    xyz = np.vstack((x.ravel(),y.ravel(),z.ravel()))
    if xc_range and yc_range:
        for_deletion = []
        for i in range(np.shape(xyz)[1]):
            if ((xyz[0,i] < xc_range[0]) or (xyz[0,i] > xc_range[1]) or
                    (xyz[1,i] < yc_range[0]) or (xyz[1,i] > yc_range[1])):
                for_deletion.append(i)
        xyz = np.delete(xyz, for_deletion, axis=1)

    xy_crop = xyz[0:2]
    z_crop = xyz[2]
    
    popt, pcov = curve_fit(gauss_curve_rotation, xy_crop, z_crop, p0 = p_est,
                           bounds = [p_min, p_max])
    xc_fitted = popt[1]
    yc_fitted = popt[3]
    theta_fitted = popt[5]
    
    if plot_fit:
        m1 = np.tan(theta_fitted)
        m2 = np.tan(theta_fitted+np.pi/2)
        b1 = yc_fitted-m1*xc_fitted
        b2 = yc_fitted-m2*xc_fitted
        x_plot = np.array([xc_min, xc_max])
        y1 = m1*x_plot+b1
        y2 = m2*x_plot+b2
        plt.figure(figsize=(10,10))
        spacing=500
        x_spaced = np.linspace(xc_min, xc_max, spacing)
        y_spaced = np.linspace(yc_min, yc_max, spacing)
        x_arr = np.array([list(x_spaced),]*spacing)
        y_arr = np.array([list(y_spaced),]*spacing).T
        theo_intensity = np.log10(gauss_curve_rotation([x_arr, y_arr], *popt))
        plt.contour(x_spaced, y_spaced, np.log10(theo_intensity), levels=5, colors='k')
        plt.tricontourf(xy_crop[0], xy_crop[1], np.log10(z_crop), levels=100, cmap='jet')
        plt.tick_params(labelsize=14)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel('log10(Intensity)', fontsize=18)
        plt.plot(x_plot, y1, 'k', linewidth=2)
        plt.plot(x_plot, y2, 'k', linewidth=2)
        plt.xlim(xc_min, xc_max)
        plt.ylim(yc_min, yc_max)
        plt.xlabel(plot_axes[0], fontsize=18)
        plt.ylabel(plot_axes[1], fontsize=18)
        plt.title('Peak Fit', fontsize=22)
        plt.savefig(plot_name+'.png')
        plt.show()
        plt.close()
    return [xc_fitted, yc_fitted]


# FUNCTION gauss_curve_rotation
# Calculates the value at the surface of a rotated gaussian, given the spacial coordinates and fit
# parameters
#   INPUTS:
# data: a list, containing the spacial coordinates
# amp: amplitude of gassuain curve
# xc: x-coordinate of centre of gaussian 
# w1: width of gaussian (1)
# yc: y-coordinate of centre of gaussian 
# w2: width of gaussian (2)
# theta: angle of rotation of gaussian
# z0: baseline of curve fit
#   OUTPUTS:
# z: value of the surface

def gauss_curve_rotation(data, amp, xc, w1, yc, w2, theta, z0):
    x, y = data
    z=z0+amp*np.exp(-0.5*((x*np.cos(theta)+y*np.sin(theta)
                           -xc*np.cos(theta)-yc*np.sin(theta))/w1)**2
                    -0.5*((-x*np.sin(theta)+y*np.cos(theta)
                           +xc*np.sin(theta)-yc*np.cos(theta))/w2)**2)
    return z


# FUNCTION qvector
# Converts 2-theta and omega 2-axes measurement into reciprocal space data, in units of 1/angstrom
#   INPUTS:
# twtheta: array of 2-theta angle data
# omega: array of omega angle data
# wavelength (default: 1.5409580): wavelength of incident beam
# source_unit (default: 'deg'): unit of angles
#   OUTPUTS:
# qx: in-plane reciprocal space vector
# qz: out-of-plane reciprocal space vector

def qvector(twtheta, omega, wavelength = 1.5405980, source_unit='deg'):
    if source_unit == 'deg' or source_unit == 'degree':
        twtheta = twtheta*np.pi/180
        omega = omega*np.pi/180
    qx = 2*np.pi/wavelength*(np.cos(omega)-np.cos(twtheta-omega))
    qz = 2*np.pi/wavelength*(np.sin(omega)+np.sin(twtheta-omega))
    return qx,qz


# FUNCTION add_lattice_param_attributes_
# Adds lattice-parameter related attributes to a dataset containing q-vectors
#   INPUTS:
# filename: name of hdf5 file containing data
# all_input_criteria: criteria to identify paths to source files using pt.path_search. Should be
#        q-vector data to write attributes to
# out_index: out-of-plane lattice index
# in_index (default: 0): in-plane lattice index
#   OUTPUTS:
# NULL

def add_lattice_param_attributes_(filename, all_input_criteria, out_index, in_index=0):
    in_path_list = pt.path_search(filename, all_input_criteria)[0]
    with h5py.File(filename, "a") as f:
        for i in range(len(in_path_list)):
            path = in_path_list[i]
            q_out = np.array(f[path])[1]
            q_in = np.array(f[path])[0]
            f[path].attrs['out_index'] = out_index
            f[path].attrs['in_index'] = in_index
            q_magnitude = np.sqrt(np.sum(np.array(f[path])**2))
            f[path].attrs['q_magnitude'] = q_magnitude
            d_spacing = 2*np.pi/q_magnitude
            f[path].attrs['d_spacing'] = d_spacing
            if in_index==0:
                out_param = out_index*2*np.pi/q_magnitude
                f[path].attrs['out_param'] = out_param
                angle = 180*np.arctan(q_in/q_out)/np.pi
                f[path].attrs['angle_deg'] = angle
            else:
                out_param = out_index*2*np.pi/q_out
                f[path].attrs['out_param'] = out_param
                in_param = in_index*2*np.pi/q_in
                f[path].attrs['in_param'] = in_param
                

def find_contours(data):
    data = data.astype(np.uint8)
    (cnts, _) = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    img_contours = np.zeros(data.shape)
    cv2.drawContours(img_contours, c, -1, (1), 2)
    return c, img_contours


def morphological_interpolation_filled(c1, c2, f1, f2):
    MaxVal=np.max([c1, c2])+1

    Fmin = np.minimum(c1, c2)
    Fmax = np.maximum(c1, c2)

    Fw = np.where(c1+c2 == 0, f1 ^ f2, 0)*MaxVal
    Fmin = np.where(Fmin == 0, np.maximum(Fmin, Fw), Fmin)

    i = 0
    while np.max(Fmin) == MaxVal:
        i = i+1
        Fw = np.maximum(Fmin, Fmax)
        while True:
            tmp = ndimage.grey_erosion(Fw, footprint=1.0*ndimage.generate_binary_structure(2, 1))
            mask = Fw == MaxVal
            Fw = np.where(mask, tmp, Fw)
            if np.max(Fw) < MaxVal:
                break

        Ft = np.where(Fw != 0, ndimage.grey_dilation(Fw, footprint=ndimage.generate_binary_structure(2, 1)), 0)
        Fw = np.where(np.logical_and(Fw != 0, Fw != Ft), Fw, 0)
        Fw = np.where(Fw != 0, (Fw+Ft)/2, 0)
        Fmin = np.where(Fw != 0, np.minimum(Fmin, Fw), Fmin)
        Fmax = np.where(Fw != 0, np.maximum(Fmax, Fw), Fmax)
        if i > 20:
            Fmin = np.where(Fmin == MaxVal, 0, Fmin)
            Fmax = np.where(Fmin == MaxVal, 0, Fmax)

    return Fmin, Fmax


def dpad8(angle):
    '''
    given an angle, return the direction to go (x,y)
    '''
    step = np.pi / 8.0
    if (angle < step) and (angle >= -step):
        return (1, 0)
    if np.pi / 4.0 + step > angle >= np.pi / 4.0 - step:
        return (1, 1)
    if np.pi / 2.0 + step > angle >= np.pi / 2.0 - step:
        return (0, 1)
    if 3 * np.pi / 4.0 + step > angle >= 3 * np.pi / 4.0 - step:
        return (-1, 1)
    if (-np.pi <= angle < -3 * np.pi / 4.0 - step) or (np.pi >= angle >= 3 * np.pi / 4.0 + step):
        return (-1, 0)
    if -3 * np.pi / 4.0 + step > angle >= -3 * np.pi / 4.0 - step:
        return (-1, -1)
    if -np.pi / 2.0 + step > angle >= -np.pi / 2.0 - step:
        return (0, -1)
    if -np.pi / 4.0 + step > angle >= -np.pi / 4.0 - step:
        return (1, -1)

    print(angle)
    print("ERROR NO ORIENTATION WAS FOUND")
    return (0, 0)


def dpad8_val(angle):
    '''
    given an angle, return the direction to go (x,y)
    '''
    step = np.pi / 8.0
    if (angle < step) and (angle >= -step):
        return 0
    if np.pi / 4.0 + step > angle >= np.pi / 4.0 - step:
        return 1
    if np.pi / 2.0 + step > angle >= np.pi / 2.0 - step:
        return 2
    if 3 * np.pi / 4.0 + step > angle >= 3 * np.pi / 4.0 - step:
        return 3
    if (-np.pi <= angle < -3 * np.pi / 4.0 - step) or (np.pi >= angle >= 3 * np.pi / 4.0 + step):
        return 4
    if -3 * np.pi / 4.0 + step > angle >= -3 * np.pi / 4.0 - step:
        return 5
    if -np.pi / 2.0 + step > angle >= -np.pi / 2.0 - step:
        return 6
    if -np.pi / 4.0 + step > angle >= -np.pi / 4.0 - step:
        return 7

    print("ERROR NO ORIENTATION WAS FOUND")
    return -1


def nan_gradient(array):
    # Calculates gradient of array, using the central difference method of np.gradient
    # If this generates nans, attempts to replace nans with backwards and forwards difference
    # calculated using np.diff
    array = np.array(array)
    y_grad, x_grad = np.gradient(array)

    # y_direction
    y_diff = np.diff(array.T)
    col1 = y_diff[:, 0]
    y_bwd_diff = np.insert(y_diff, 0, col1, axis=1).T
    col2 = y_diff[:, -1]
    y_fwd_diff = np.insert(y_diff, -1, col2, axis=1).T

    final_y_diff = np.where(np.isnan(y_grad), y_bwd_diff, y_grad)
    final_y_diff = np.where(np.isnan(final_y_diff), y_fwd_diff, final_y_diff)

    # x_direction
    x_diff = np.diff(array)
    col1 = x_diff[:, 0]
    x_bwd_diff = np.insert(x_diff, 0, col1, axis=1)
    col2 = x_diff[:, -1]
    x_fwd_diff = np.insert(x_diff, -1, col2, axis=1)

    final_x_diff = np.where(np.isnan(x_grad), x_bwd_diff, x_grad)
    final_x_diff = np.where(np.isnan(final_x_diff), x_fwd_diff, final_x_diff)

    return final_y_diff, final_x_diff


def find_path_growth(interpolation, contour_init, surf2, loop_exit=100):
    # Compute the gradient
    gradx, grady = nan_gradient(interpolation)
    angle_map = np.arctan2(grady, gradx)
    direction_map_x = angle_map.copy()
    direction_map_y = angle_map.copy()

    # For each gradient value choose which direction the gradient is pointing using dpad8
    for x in range(np.shape(angle_map)[0]):
        for y in range(np.shape(angle_map)[1]):
            if np.isnan(angle_map[x][y]):
                pass
            else:
                dirx, diry = dpad8(angle_map[x][y])
                direction_map_x[x][y] = dirx
                direction_map_y[x][y] = diry

    tmp_cnt = 0
    paths = []

    contour_end, img_contour = find_contours(surf2)

    # Generate a list containing all the initial pixels of the growing paths
    for elem in contour_init:
        init_y = elem[0][0]
        init_x = elem[0][1]
        tmp_path = [[init_y, init_x]]
        paths.append(tmp_path)

    path_check = [False for i in paths]  # Used to check if the path is complete
    path_cnt = [0 for i in paths]  # Count the number of loop to exit if it take too long
    path_warning = [False for i in paths]  # Used to know if the loop was exited because it took too loog

    while True:
        for idx, path in enumerate(paths):  # Loop through all the paths
            if not path_check[idx]:

                # Check if the path finding is taking too long
                path_cnt[idx] += 1
                if loop_exit is not None:
                    if path_cnt[idx] > loop_exit:
                        print(
                            'more than ' + str(loop_exit) + ' steps where taken, breaking out of the loop. IDX: ' + str(
                                idx))
                        print(direction_map_x[current_x][current_y])
                        print(direction_map_y[current_x][current_y])
                        print(path[-3:])
                        path_check[idx] = True
                        path_warning[idx] = True
                        break

                # Get the current x,y position of the growing path
                current = path[-1]
                current_x = current[1]
                current_y = current[0]

                # Check if an element of the growing path is ON the final path
                for element in contour_end:
                    if current_x == element[0][1] and current_y == element[0][0]:
                        path_check[idx] = True
                        break

                # compute the next step of the growing path
                stepx = direction_map_x[current_x][current_y]
                stepy = direction_map_y[current_x][current_y]
                newx = current_x + int(stepx)
                newy = current_y + int(stepy)

                # Check if the new position is outside of the external surface
                if not path_check[idx]:
                    if not surf2[newx][newy]:
                        path_check[idx] = True
                        break

                if not path_check[idx]:
                    for p in path:
                        if p[0] == newy and p[1] == newx:
                            path_check[idx] = True
                            break

                if not path_check[idx]:
                    path.append([newy, newx])

        # Once all paths are completed, exit the infinite loop
        if all(path_check):
            break

    return paths, path_warning


def find_grown_element(contour_init, contour_end, path1, path2, img_contour_end):
    idx_1 = 0
    idx_2 = -1
    for i, px in enumerate(contour_init):
        if path1[0][0] == px[0][0] and path1[0][1] == px[0][1]:
            idx_1 = i
            break

    for i, px in enumerate(contour_init):
        if path2[0][0] == px[0][0] and path2[0][1] == px[0][1]:
            idx_2 = i
            break

    if idx_2 < idx_1:
        contour_init = contour_init[idx_2:idx_1]
    # Check if the first element is one of the index, to see if it is the last one, which need to be handle differently
    # (since we are linking [-1] to [0])
    if idx_1 == 0 or idx_2 == 0:
        if idx_1 < idx_2:
            if len(contour_init[idx_1:idx_2]) < len(contour_init[idx_2:]):
                contour_init = contour_init[idx_1:idx_2]
            else:

                contour_init = contour_init[idx_2:]
    else:
        contour_init = contour_init[idx_1:idx_2]

    img_bin = np.zeros(np.shape(img_contour_end))

    for px in contour_init:
        px = px[0]
        img_bin[px[1], px[0]] = 1

    for px in contour_end:
        px = px[0]
        img_bin[px[1], px[0]] = 1

    for px in path1:
        img_bin[px[1], px[0]] = 1

    for px in path2:
        img_bin[px[1], px[0]] = 1

    data = (img_bin).astype(np.uint8)
    (cnts, img) = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    return cnts