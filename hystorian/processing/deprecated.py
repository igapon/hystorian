import h5py

import numpy as np
import cv2
import time
from . import core
from . import twodim
from skimage import img_as_ubyte

import functools
import inspect
import warnings

string_types = (type(b""), type(""))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):
        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):
            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason), category=DeprecationWarning, stacklevel=2
                )
                warnings.simplefilter("default", DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):
        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(fmt2.format(name=func2.__name__), category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter("default", DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))



@deprecated("use distortion_params_ in twodim")
def distortion_params_(
    filename,
    all_input_criteria,
    mode="SingleECC",
    read_offset=False,
    cumulative=False,
    filterfunc=twodim.normalise,
    warp_mode=cv2.MOTION_TRANSLATION,
    termination_eps=1e-5,
    number_of_iterations=10000,
    max_divisions=8,
    warp_check_range=8,
    divisor_init=0.25,
    lim=50,
    speed=2,
):
    """
    Determine cumulative translation matrices for distortion correction and directly write it into
    an hdf5 file

    Parameters
    ----------
    filename : str
        name of hdf5 file containing data
    all_input_criteria : str
        criteria to identify paths to source files using core.path_search. Should be
        height data to extract parameters from
    mode : str, optional
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
    warp_mode : int, optional
        Mode of warp identified. Relevant for SingleECC and MultiECC modes.
        Only cv2.MOTION_TRANSLATION is currently tested.
    termination_eps : float, optional
        Termination precision for SingleECC and MultiECC modes
    number_of_iterations : int, optional
        Termination number for SingleECC and MultiECC modes
    max_divisions : int, optional
        Number of recursive checks for ManualECC mode
    warp_check_range : int, optional
        Length and height of grid checked for ManualECC mode
    divisor_init : float, optional
        Spacing between grid points in Manual ECCmode.
    lim : int, optional
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
        in_path_list = core.path_search(filename, channel_type)
        all_in_path_list.append(in_path_list[0])
    out_folder_locations = core.find_output_folder_location(filename, "distortion_params", all_in_path_list[0])
    eyes = np.eye(2, 3, dtype=np.float32)
    cumulative_tform21 = np.eye(2, 3, dtype=np.float32)
    with h5py.File(filename, "a") as f:
        recent_offsets = []
        for i in range(len(all_in_path_list[0])):
            if i == 0:
                start_time = time.time()
            else:
                print("---")
                print("Currently reading path " + all_in_path_list[0][i])
                img1 = []
                img2 = []
                for channel_i in range(len(all_in_path_list)):
                    i1 = f[all_in_path_list[channel_i][0]]
                    if (i > 1) and (not cumulative):
                        i1 = f[all_in_path_list[channel_i][i - 1]]
                    i2 = f[all_in_path_list[channel_i][i]]
                    if filterfunc is not None:
                        i1 = filterfunc(i1)
                        i2 = filterfunc(i2)
                    img1.append(img_as_ubyte(i1))
                    img2.append(img_as_ubyte(i2))

                if read_offset:
                    offset2 = (f[all_in_path_list[0][i]]).attrs["offset"]
                    offset1 = (f[all_in_path_list[0][i - 1]]).attrs["offset"]
                    scan_size = (f[all_in_path_list[0][i]]).attrs["size"]
                    shape = (f[all_in_path_list[0][i]]).attrs["shape"]
                    offset_px = m2px(offset2 - offset1, shape, scan_size)
                else:
                    offset_px = np.array([0, 0])
                if (offset_px[0] != 0) or (offset_px[1] != 0):
                    recent_offsets = []
                if mode == "SingleECC":
                    tform21 = generate_transform_xy_single(
                        img1[0], img2[0], offset_px, warp_mode, termination_eps, number_of_iterations
                    )
                elif mode == "ManualECC":
                    tform21 = generate_transform_xy_manual(
                        img1, img2, offset_px, max_divisions, warp_check_range, divisor_init, lim
                    )
                elif mode == "MultiECC":
                    tform21 = generate_transform_xy_multi(
                        img1[0],
                        img2[0],
                        offset_px,
                        warp_mode,
                        termination_eps,
                        number_of_iterations,
                        lim,
                        speed,
                        recent_offsets,
                        cumulative,
                        cumulative_tform21,
                    )
                else:
                    print("Invalid mode requested. Defaulting to SingleECC")
                    mode = "SingleECC"
                    tform21 = generate_transform_xy_single(
                        img1[0], img2[0], offset_px, warp_mode, termination_eps, number_of_iterations
                    )

                if cumulative:
                    tform21[0, 2] = tform21[0, 2] - cumulative_tform21[0, 2]
                    tform21[1, 2] = tform21[1, 2] - cumulative_tform21[1, 2]
                cumulative_tform21[0, 2] = cumulative_tform21[0, 2] + tform21[0, 2]
                cumulative_tform21[1, 2] = cumulative_tform21[1, 2] + tform21[1, 2]
                print("Scan " + str(i) + " Complete. Cumulative Transform Matrix:")
                print(cumulative_tform21)

                if mode == "multiECC":
                    if (offset_px[0] == 0) and (offset_px[1] == 0):
                        recent_offsets.append([tform21[0, 2], tform21[1, 2]] - offset_px)
                        if len(recent_offsets) > 3:
                            recent_offsets = recent_offsets[1:]

            data = core.write_output_f(
                f, cumulative_tform21, out_folder_locations[i], all_in_path_list[0][i], distortion_params_, locals()
            )
            core.progress_report(
                i + 1, len(all_in_path_list[0]), start_time, "distortion_params", all_in_path_list[0][i], clear=False
            )


@deprecated("use distortion_correction_ in twodim")
def distortion_correction_(filename, all_input_criteria, cropping=True):
    """
    Applies distortion correction parameters to an image. The distortion corrected data is then
    cropped to show only the common data, or expanded to show the maximum extent of all possible
    data.

    Parameters
    ----------
    filename : str
        Filename of hdf5 file containing data
    all_input_criteria : list
        Criteria to identify paths to source files using core.path_search. First should
        be data to be corrected, second should be the distortion parameters.
    cropping : bool, optional
        If set to True, each dataset is cropped to show only the common area. If
        set to false, expands data shape to show all data points of all images. (default: True)

    Returns
    -------
    None
    """
    all_in_path_list = core.path_search(filename, all_input_criteria, repeat="block")
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

    out_folder_locations = core.find_output_folder_location(filename, "distortion_correction", in_path_list)

    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for i in range(len(in_path_list)):
            orig_image = f[in_path_list[i]]
            if cropping:
                final_image = array_cropped(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            else:
                final_image = array_expanded(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            data = core.write_output_f(
                f,
                final_image,
                out_folder_locations[i],
                [in_path_list[i], dm_path_list[i]],
                distortion_correction_,
                locals(),
            )
            propagate_scale_attrs(data, f[in_path_list[i]])
            core.progress_report(i + 1, len(in_path_list), start_time, "distortion_correction", in_path_list[i])


@deprecated("use m_apply or l_apply in core")
def l_apply_classic(
    filename,
    function,
    all_input_criteria,
    output_names=None,
    folder_names=None,
    use_attrs=None,
    prop_attrs=None,
    repeat=None,
    **kwargs
):
    #   FUNCTION l_apply_classic
    # Runs m_apply multiple times successively, intended to operate on an entire process or dataset
    # folder
    #   INPUTS:
    # filename : name of the hdf5 file where the datas are stored
    # function : Custom function that you want to call
    # all_input_criteria : Regex expression to describe the inputs searched for. Can be composed as a
    #     list of a list of strings, with extra list parenthesis automatically generated. Eg:
    #         'process*Trace1*' would pass to m_apply all files that contain 'process*Trace1*'.
    #         ['process*Trace1*'] as above
    #         [['process*Trace1*']] as above
    #         [['process*Trace1*', 'process*Trace2*']] would pass to m_apply all files that contain
    #             'process*Trace1*' and 'process*Trace2*' in a single list.
    #         [['process*Trace1*'], ['process*Trace2*']] would pass to m_apply all files that contain
    #             'process*Trace1*' and 'process*Trace2*' in two different lists; and thus will operate
    #             differently on each of these lists.
    # outputs_names (default: None): list of the names of the channels for the writting of the results.
    #     By default, copies names from the first of the in_paths
    # folder_names (default: None): list of the names of the folder containing results data channels.
    #     By default, copies names from the first of the in_paths
    # use_attrs (default: None): string, or list of strings, that are the names of attributes that will
    #     be copied from in_paths, and passed into the function as a kwarg for use.
    # prop_attrs (default: None): string, or list of strings, that are the names of attributes that will
    #     be copied from in_paths, into each output file. If the same attribute name is in multiple
    #     in_paths, the first in_path with the attribute name will be copied from.
    # repeat (default: None): determines what to do if path_lists generated are of different lengths.
    #     None: Default, no special action is taken, and extra entries are removed. ie, given lists
    #         IJKL and AB, IJKL -> IJ.
    #     'alt': The shorter lists of path names are repeated to be equal in length to the longest list.
    #         ie, given IJKL and AB, AB -> ABAB
    #     'block': Each entry of the shorter list of path names is repeated to be equal in length to the
    #         longest list. ie, given IJKL and AB, AB -> AABB.
    # **kwargs : All the non-data inputs to give to the function
    #    OUTPUTS:
    # null
    #    TO DO:
    # Can we force a None to be passed?
    print("This function is deprecated, please try to make use of l_apply instead.")
    all_in_path_list = core.path_search(filename, all_input_criteria, repeat)
    all_in_path_list = list(map(list, zip(*all_in_path_list)))
    increment_proc = True
    start_time = time.time()
    for path_num in range(len(all_in_path_list)):
        core.m_apply(
            filename,
            function,
            all_in_path_list[path_num],
            output_names=output_names,
            folder_names=folder_names,
            increment_proc=increment_proc,
            use_attrs=use_attrs,
            prop_attrs=prop_attrs,
            **kwargs
        )
        core.progress_report(
            path_num + 1, len(all_in_path_list), start_time, function.__name__, all_in_path_list[path_num]
        )
        increment_proc = False


@deprecated("use distortion_params_ in twodim")
def distortion_params_classic(
    filename, all_input_criteria, speed=2, read_offset=False, cumulative=False, filterfunc=twodim.normalise
):
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
    speed : int, optional
        Value between 1 and 4, which determines speed and accuracy of function. A higher number is
        faster, but assumes lower distortion and thus may be incorrect. Default value is 2.
    read_offset : bool, optional
        If set to True, attempts to read dataset for offset attributes to
        improve initial guess and thus overall accuracy (default is False).
    cumulative : bool, optional
        Determines if each image is compared to the previous image (default,
        False), or to the original image (True). Output format is identical.
    fitlerfunc : func, optional
        Function applied to image before identifying distortion params

    Returns
    -------
        None
    """

    print("This function is deprecated, please try to make use of distortion_params_ instead.")

    in_path_list = core.path_search(filename, all_input_criteria)[0]
    out_folder_locations = core.find_output_folder_location(filename, "distortion_params", in_path_list)
    tform21 = np.eye(2, 3, dtype=np.float32)
    cumulative_tform21 = np.eye(2, 3, dtype=np.float32)
    with h5py.File(filename, "a") as f:
        recent_offsets = []
        for i in range(len(in_path_list)):
            if i == 0:
                start_time = time.time()
            else:
                print("---")
                print("Currently reading path " + in_path_list[i])
                i1 = f[in_path_list[0]]
                if (i > 1) and (not cumulative):
                    i1 = f[in_path_list[i - 1]]
                i2 = f[in_path_list[i]]
                if filterfunc is not None:
                    i1 = filterfunc(i1)
                    i2 = filterfunc(i2)
                img1 = img_as_ubyte(i1)
                img2 = img_as_ubyte(i2)

                # try estimate offset change from attribs of img1 and img2
                if read_offset:
                    offset2 = (f[in_path_list[i]]).attrs["offset"]
                    offset1 = (f[in_path_list[i - 1]]).attrs["offset"]
                    scan_size = (f[in_path_list[i]]).attrs["size"]
                    shape = (f[in_path_list[i]]).attrs["shape"]
                    offset_px = twodim.m2px(offset2 - offset1, shape, scan_size)
                else:
                    offset_px = np.array([0, 0])
                if speed != 0 and speed != 1 and speed != 2 and speed != 3 and speed != 4:
                    print(
                        "Error: Speed should be an integer between 1 (slowest) and 4 (fastest).\
                            Speed now set to level 2."
                    )
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
                    offset_guess = offset_px + recent_offsets[2] / 2 + recent_offsets[1] / 3 + recent_offsets[0] / 6
                    # if i == 9:
                    #    offset_guess = offset_guess-np.array([20,20])
                    #    print(offset_guess)
                    if speed == 1:
                        warp_check_range = 8
                    elif speed == 2:
                        warp_check_range = 6
                    elif speed == 3:
                        warp_check_range = 4
                    elif speed == 4:
                        warp_check_range = 2
                if (offset_px[0] != 0) or (offset_px[1] != 0):
                    print("Offset found from file attributes: " + str(offset_px))
                    warp_check_range = warp_check_range + 8
                    recent_offsets = []
                tform21 = generate_transform_xy_classic(
                    img1, img2, tform21, offset_guess, warp_check_range, cumulative, cumulative_tform21
                )
                if cumulative:
                    tform21[0, 2] = tform21[0, 2] - cumulative_tform21[0, 2]
                    tform21[1, 2] = tform21[1, 2] - cumulative_tform21[1, 2]
                cumulative_tform21[0, 2] = cumulative_tform21[0, 2] + tform21[0, 2]
                cumulative_tform21[1, 2] = cumulative_tform21[1, 2] + tform21[1, 2]
                print("Scan " + str(i) + " Complete. Cumulative Transform Matrix:")
                print(cumulative_tform21)
                if (offset_px[0] == 0) and (offset_px[1] == 0):
                    recent_offsets.append([tform21[0, 2], tform21[1, 2]] - offset_px)
                    if len(recent_offsets) > 3:
                        recent_offsets = recent_offsets[1:]
            data = core.write_output_f(f, cumulative_tform21, out_folder_locations[i], in_path_list[i])
            core.progress_report(
                i + 1, len(in_path_list), start_time, "distortion_params", in_path_list[i], clear=False
            )


@deprecated("use distortion_correction_ in twodim")
def distortion_correction_classic(filename, all_input_criteria, cropping=True):
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
    print("This function is deprecated, please try to make use of distortion_correction_ instead.")

    all_in_path_list = core.path_search(filename, all_input_criteria, repeat="block")
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

    out_folder_locations = core.find_output_folder_location(filename, "distortion_correction", in_path_list)

    with h5py.File(filename, "a") as f:
        start_time = time.time()
        for i in range(len(in_path_list)):
            orig_image = f[in_path_list[i]]
            if cropping:
                final_image = twodim.array_cropped(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            else:
                final_image = twodim.array_expanded(orig_image, xoffsets[i], yoffsets[i], offset_caps)
            data = core.write_output_f(f, final_image, out_folder_locations[i], [in_path_list[i], dm_path_list[i]])
            twodim.propagate_scale_attrs(data, f[in_path_list[i]])
            core.progress_report(i + 1, len(in_path_list), start_time, "distortion_correction", in_path_list[i])


@deprecated("use generate_transform_xy in twodim")
def generate_transform_xy_classic(
    img,
    img_orig,
    tfinit=None,
    offset_guess=[0, 0],
    warp_check_range=10,
    cumulative=False,
    cumulative_tform21=np.eye(2, 3, dtype=np.float32),
):
    """
    Determines transformation matrices in x and y coordinates

    Parameters
    ----------
    img : cv2
        Currently used image (in cv2 format) to find transformation array of
    img_orig : cv2
        Image (in cv2 format) transformation array is based off of
    tfinit : array_like or None, optional
        Base array passed into function
    offset_guess : list, optional
        Array showing initial estimate of distortion, in pixels (default: [0,0])
    warp_check_range : int, optional
        Distance (in pixels) that the function will search to find the optimal transform matrix.
        Number of iterations = (warp_check_range+1)**2. (default: 10)
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

    print("This function is deprecated, please try to make use of gernerate_transform_xy_ instead.")

    # Here we generate a MOTION_EUCLIDEAN matrix by doing a
    # findTransformECC (OpenCV 3.0+ only).
    # Returns the transform matrix of the img with respect to img_orig
    warp_mode = cv2.MOTION_TRANSLATION
    if tfinit is not None:
        warp_matrix = tfinit
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    number_of_iterations = 10000
    termination_eps = 1e-3
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
                (cc, tform21) = cv2.findTransformECC(img_orig, img, warp_matrix, warp_mode, criteria)
                img_test = cv2.warpAffine(img, tform21, (512, 512), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                currDiff = np.sum(np.square(img_test[150:-150, 150:-150] - img_orig[150:-150, 150:-150]))
                if currDiff < diff:
                    diff = currDiff
                    offset1 = tform21[0, 2]
                    offset2 = tform21[1, 2]
            except:
                pass
            warp_matrix[0, 2] = offset1
            warp_matrix[1, 2] = offset2
    return warp_matrix


@deprecated("")
def sample_fraction(array, shape_fraction=0.1, start_shape=None):
    """
    Randomly samples a contiguous block and small fraction of a larger array. Works by taking a
    small point (or smaller shape) and randomly expanding in all directions. To ensure uniformity,
    a 'supershape' is constructed, which takes the initial array and adds 5 pixels on each
    direction to reduce edge effects during sampling.

    Parameters
    ----------
    array : 2d array
        array to be sampled
    shape_fraction : int or float, optional
        the fraction of the array that is actually sampled. eg., by
        default, an area of 0.1 will return a block of area 10% of the overall array
    start_shape : 2d array, optional
        A shape used to start the sample. Allows sample_fraction to feed into
        itself and generate larger samples from smaller ones.

    Returns
    ------
    curr_shape :
        the sample generated from the main array
    """
    basic_array = np.zeros_like(array)
    total_shapes = 300000
    target_shape_size = shape_fraction * np.sum(array)

    # Create Coordinate List:
    coord_list = []
    coord_list_index_array = np.zeros_like(basic_array)
    index = 0
    for i in range(np.shape(basic_array)[0]):
        for j in range(np.shape(basic_array)[1]):
            coord_list.append([i, j])
            coord_list_index_array[i, j] = index
            index = index + 1
    neighbour_array = np.zeros_like(basic_array)
    neighbour_scaling = np.zeros_like(basic_array) + 4

    if start_shape is None:
        # Pick First Point
        np.random.seed()
        init_i, init_j = coord_list[int(np.random.choice(coord_list_index_array.flatten()))]
        new_i = init_i
        new_j = init_j
        curr_shape = np.zeros_like(basic_array)
        curr_shape[init_i, init_j] = 1
    else:
        start_shape = np.copy(start_shape)
        new_i = None
        new_j = None
        curr_shape = start_shape
        expanded_point = np.zeros_like(basic_array)
        for i in range(np.shape(basic_array)[0]):
            for j in range(np.shape(basic_array)[1]):
                if curr_shape[i, j] != 0:
                    if i != 0:
                        if i == 1:
                            expanded_point[i - 1, j] = expanded_point[i - 1, j] + 2
                        else:
                            expanded_point[i - 1, j] = expanded_point[i - 1, j] + 2
                    if j != 0:
                        if j == 1:
                            expanded_point[i, j - 1] = expanded_point[i, j - 1] + 2
                        else:
                            expanded_point[i, j - 1] = expanded_point[i, j - 1] + 2
                    if i != np.shape(curr_shape)[0] - 1:
                        if i == np.shape(curr_shape)[0] - 2:
                            expanded_point[i + 1, j] = expanded_point[i + 1, j] + 2
                        else:
                            expanded_point[i + 1, j] = expanded_point[i + 1, j] + 2
                    if j != np.shape(curr_shape)[1] - 1:
                        if j == np.shape(curr_shape)[1] - 2:
                            expanded_point[i, j + 1] = expanded_point[i, j + 1] + 2
                        else:
                            expanded_point[i, j + 1] = expanded_point[i, j + 1] + 2

    curr_shape_size = np.sum(curr_shape * array)
    while curr_shape_size < target_shape_size:
        # Find possible growths from that point
        last_point = np.zeros_like(basic_array)
        last_point[new_i, new_j] = 1
        # Use mirror boundary conditions
        if new_i is not None:
            expanded_point = np.copy(last_point)
            if new_i > 1:
                expanded_point[new_i - 1, new_j] = 1
            elif new_i == 1:
                expanded_point[new_i - 1, new_j] = 2
            if new_j > 1:
                expanded_point[new_i, new_j - 1] = 1
            elif new_j == 1:
                expanded_point[new_i, new_j - 1] = 2
            if new_i < np.shape(basic_array)[0] - 2:
                expanded_point[new_i + 1, new_j] = 1
            elif new_i == np.shape(basic_array)[0] - 2:
                expanded_point[new_i + 1, new_j] = 2
            if new_j < np.shape(basic_array)[1] - 2:
                expanded_point[new_i, new_j + 1] = 1
            elif new_j == np.shape(basic_array)[1] - 2:
                expanded_point[new_i, new_j + 1] = 2
        neighbour_array = neighbour_array + (expanded_point / neighbour_scaling)
        neighbour_array = neighbour_array * (1 - curr_shape)

        # Grow from one of the growth coords
        thresh = 0.75
        if np.any(neighbour_array >= thresh):
            # Grow if too surrounded
            total_to_change = np.sum(neighbour_array >= thresh)
            # print(total_to_change)
            if total_to_change > 1:
                growth_locations = np.where(neighbour_array >= thresh)
                rand_num = np.random.randint(0, len(growth_locations[0]))
                new_i = growth_locations[0][rand_num]
                new_j = growth_locations[1][rand_num]
            else:
                new_coords = np.where(neighbour_array >= thresh)
                new_i = new_coords[0]
                new_j = new_coords[1]
        else:
            # Randomly grow:
            probability_array = np.copy(neighbour_array)
            normalise_factor = np.sum(probability_array)
            new_i, new_j = coord_list[
                int(
                    np.random.choice(
                        coord_list_index_array.flatten(), p=(probability_array / normalise_factor).flatten()
                    )
                )
            ]
        curr_shape[new_i, new_j] = 1
        curr_shape_size = np.sum(curr_shape * array)
        # Should make it only consider largest area?
    return curr_shape


@deprecated("")
def all_sample_fractions(
    array, iterations=100, fractions=[0.1, 0.15, 0.2, 0.25], compression=[50, 50], background=np.nan
):
    """
    Generates a 4D array of several samples; the first axis allows choice of the fraction of each
    sampling; the second axis allows for each iteration of this fraction. The remaining two axes
    are the 2D array of each individual sample.

    Parameters
    ----------
    array : 2d array
        the array to be sampled
    iterations : int or float, optional
        number of samples to be taken of for each fraction
    fractions : list, optional
        fractions to be sampled of the array
    compression : list, optional
        the size to which the array is compressed to during sampling
    background : int or float, optional
        values of background areas that may be removed during compression

    Returns
    ------
    sample_fractions_all_fractions :
        4D array describing all samples extracted
    """
    # Compress array and generate "supershape" template from which subshapes are drawn
    if np.isnan(background):
        bool_array = ~np.isnan(array)
    else:
        bool_array = ~(array == background)
    cropped_array = crop(bool_array)
    expanded_array = uncrop_to_multiple(cropped_array, compression)
    compressed_array = compress_to_shape(expanded_array, compression)
    array_supershape = uncrop_to_multiple(compressed_array, [compression[0] + 10, compression[1] + 10])

    # Generate first generation of all shapes
    if type(fractions) != list:
        fractions = [fractions]
    sample_fractions_all_fractions = []
    sample_fractions_one_fraction = []
    for sample_count in range(iterations):
        shape = sample_fraction(array_supershape, fractions[0]).astype(bool)
        sample_fractions_one_fraction.append(shape)

    # If multiple fractions are provides, generate successive generations
    if len(fractions) > 1:
        for i in range(1, len(fractions)):
            sample_fractions_all_fractions.append(sample_fractions_one_fraction)
            sample_fractions_one_fraction = []
            for sample_count in range(iterations):
                shape = sample_fraction(
                    array_supershape, fractions[i], sample_fractions_all_fractions[i - 1][sample_count]
                ).astype(bool)
                sample_fractions_one_fraction.append(shape)
        sample_fractions_all_fractions.append(sample_fractions_one_fraction)
    else:
        sample_fractions_all_fractions = sample_fractions_one_fraction

    sample_fractions_all_fractions = np.array(sample_fractions_all_fractions)
    # axes = [(fraction_num)][sample_num][y][x]
    return sample_fractions_all_fractions


@deprecated("")
def interpolated_features(switchmap):
    """
    Creates isolines from a switchmap, then interpolates it

    Parameters
    ----------
    switchmap : 2d array
        the switchmap used as the base for key features

    Returns
    ------
    interpolation :
        interpolated features
    """
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
    grid_x, grid_y = np.mgrid[0 : np.shape(isolines)[0] : 1, 0 : np.shape(isolines)[1] : 1]
    interpolation = interpolate.griddata(
        np.array([isoline_y, isoline_x]).T, np.array(isoline_z), (grid_x, grid_y), method="linear", fill_value=np.nan
    )
    return interpolation
