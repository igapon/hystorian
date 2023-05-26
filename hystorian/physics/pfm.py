import numpy as np
import numpy.typing as npt

from ..processing.twodim import gauss_area


def extract_hist(
    *chans: list[npt.NDArray],
    len_bias: int,
    waveform_pulsetime: int,
    waveform_dutycycle: float = 0.5,
    num_pts_per_sec: float
) -> tuple[npt.NDArray]:
    """extract_hist split data from an SSPFM measurement into part where the bias is off, and the bias is on.
    Therefore it can be used to build P-E loop.

    Parameters
    ----------
    len_bias : int
        len_bias is the lenghth of the bias applied during the SSPF.
        If the data is from an Asylum ARDF you can find it by, for example, doing f.read("datasets/.../bias/retrace")[0,0,:]
    waveform_pulsetime : int
        The length in time of the pulse.
        If from Asylum can be found in metadata under name "ARDoIVArg3". f.read("metadata/.../ArDoIVArg3")
    NumPtsPerSec : float
        The number of points per seconds in a pulse.
        If from Asylum can be computed from the metadata "NumPtsPerSec": float(f.read("metadata/.../NumPtsPerSec"))
    waveform_dutycycle : float, optional
        The length in time of the pulse.
        If from Asylum can be found in metadata under name "ARDoIVArg4", by default 0.5 (which is the default value of Asylum)

    Returns
    -------
    tuple[npt.NDArray]
        The tuple is twice the size of chans, containing chans[0] 'on', chans[0] 'off', chans[1] 'on', ...
        It splits each channel into on/off where the bias waveform was either 0 (off) or something else (on)
    """

    output = []
    waveform_delta = 1 / num_pts_per_sec
    waveform_numbiaspoints = int(np.floor(waveform_delta * len_bias / waveform_pulsetime))
    waveform_pulsepoints = int(waveform_pulsetime / waveform_delta)
    waveform_offpoints = int(waveform_pulsepoints * (1.0 - waveform_dutycycle))

    for chan in chans:
        result_on = np.ndarray(shape=(np.shape(chan)[0], np.shape(chan)[1], waveform_numbiaspoints))
        result_off = np.ndarray(shape=(np.shape(chan)[0], np.shape(chan)[1], waveform_numbiaspoints))
        for b in range(waveform_numbiaspoints):
            start = b * waveform_pulsepoints + waveform_offpoints
            stop = (b + 1) * waveform_pulsepoints

            var2 = stop - start + 1
            realstart = int(start + var2 * 0.25)
            realstop = int(stop - var2 * 0.25)
            result_on[:, :, b] = np.nanmean(chan[:, :, realstart:realstop], axis=2)
            start = stop
            stop = stop + waveform_pulsepoints * waveform_dutycycle

            var2 = stop - start + 1
            realstart = int(start + var2 * 0.25)
            realstop = int(stop - var2 * 0.25)
            result_off[:, :, b] = np.nanmean(chan[:, :, realstart:realstop], axis=2)
        output.append(result_on)
        output.append(result_off)

    output = tuple(output)

    output = tuple(output)
    return output


def get_phase_unwrapping_shift(phase, phase_step=1):
    """
    Finds the smallest shift that minimizes the phase jump in a phase series

    Parameters
    ----------
    phase : a 1D array of consecutive phase measurements
    phase_step : the algorithm will try shifts every phase_step. Increase this to speed up execution.

    Returns
    -------
    The phase shift. (phase + shift) % 360 will have the smallest jumps between consecutive phase points.
    """
    phase_step = 90
    jumps = []
    for shift in range(0, 360, phase_step):
        y = (phase + shift) % 360
        # what is the biggest phase jump?
        max_phase_jump = np.max(np.abs(np.diff(y)))
        jumps.append(max_phase_jump)

    return phase_step * np.argmin(np.array(jumps))


def PFM_params_map(
    bias: npt.NDArray, phase: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """PFM_params_map calculates physically relevant hysteresis parameters from bias and phase channels.

    Parameters
    ----------
    bias : npt.NDArray
        array containing the bias acquired during an SSPFM. This should be the "on" bias computed with extract_hist
    phase : npt.NDArray
        array containing the phase acquired during an SSPFM. This should be the "off" phase computed with extract_hist

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        coerc_pos: the positive coercive bias
        coerc_neg: the negative coercive bias
        step_left: The size of the phase jump (left side)
        step_right: The size of the phase jump (right side)
        imprint: The imprint of the switching loop
        phase_shift: The phase shift of the switching loop
    """
    x, y, z = np.shape(phase)
    coerc_pos = np.zeros((x, y), dtype=float)
    coerc_neg = np.zeros((x, y), dtype=float)
    step_left = np.zeros((x, y), dtype=float)
    step_right = np.zeros((x, y), dtype=float)
    imprint = np.zeros((x, y), dtype=float)
    phase_shift = np.zeros((x, y), dtype=float)
    for xi in range(x):
        for yi in range(y):
            hyst_matrix = _calc_hyst_params(bias[xi, yi, :], phase[xi, yi, :])
            coerc_pos[xi, yi] = hyst_matrix[0]
            coerc_neg[xi, yi] = hyst_matrix[1]
            step_left[xi, yi] = hyst_matrix[2]
            step_right[xi, yi] = hyst_matrix[3]
            imprint[xi, yi] = (hyst_matrix[0] + hyst_matrix[1]) / 2.0
            phase_shift[xi, yi] = hyst_matrix[3] - hyst_matrix[2]

    return coerc_pos, coerc_neg, step_left, step_right, imprint, phase_shift


def _calc_hyst_params(bias: npt.NDArray, phase: npt.NDArray) -> list[npt.NDArray]:
    """
    Calculate hysteresis parameters from bias and phase channels.
    Used in PFM_params_map. Would recommend using the function instead.

    Parameters
    ----------
    bias : npt.NDArray
        array containing the bias acquired during an SSPFM. This should be the "on" bias computed with extract_hist
    phase :npt.NDArray
        array containing the phase acquired during an SSPFM. This should be the "off" phase computed with extract_hist

    Returns
    -------
        list[npt.NDArray]
        list where the elements correspond to :
            the positive coercive bias
            the negative coercive bias
            The size of the phase jump (left side)
            The size of the phase jump (right side)
    """
    biasdiff = np.diff(bias)
    up = np.sort(np.unique(np.hstack((np.where(biasdiff > 0)[0], np.where(biasdiff > 0)[0] + 1))))
    dn = np.sort(np.unique(np.hstack((np.where(biasdiff < 0)[0], np.where(biasdiff < 0)[0] + 1))))
    phase_shift = get_phase_unwrapping_shift(phase)

    # UP leg calculations
    if up.size == 0:
        step_left_up = np.nan
        step_right_up = np.nan
        coercive_volt_up = np.nan
    else:
        x = np.array(bias[up])
        y = (np.array(phase[up]) + phase_shift) % 360
        step_left_up = np.median(y[np.where(x == np.min(x))[0]])
        step_right_up = np.median(y[np.where(x == np.max(x))[0]])

        avg_x = []
        avg_y = []
        for v in np.unique(x):
            avg_x.append(v)
            avg_y.append(np.mean(y[np.where(x == v)[0]]))

        my_x = np.array(avg_x)[1:]
        my_y = np.abs(np.diff(avg_y))

        coercive_volt_up = my_x[np.nanargmax(my_y)]

    # DOWN leg calculations
    if dn.size == 0:
        step_left_dn = np.nan
        step_right_dn = np.nan
        coercive_volt_dn = np.nan
    else:
        x = np.array(bias[dn])
        y = (np.array(phase[dn]) + phase_shift) % 360
        step_left_dn = np.median(y[np.where(x == np.min(x))[0]])
        step_right_dn = np.median(y[np.where(x == np.max(x))[0]])

        avg_x = []
        avg_y = []
        for v in np.unique(x):
            avg_x.append(v)
            avg_y.append(np.mean(y[np.where(x == v)[0]]))

        my_x = np.array(avg_x)[1:]
        my_y = np.abs(np.diff(avg_y))

        coercive_volt_dn = my_x[np.nanargmax(my_y)]

    return [
        coercive_volt_up,
        coercive_volt_dn,
        np.nanmean([step_left_dn, step_left_up]),
        np.nanmean([step_right_dn, step_right_up]),
    ]


def clean_loop(bias, phase, amp, threshold=None):
    """
    Used to determine if a SSPFM loop is good or not by calculating the area encompassed by the
    hysteresis curve and comparing it to a threshold

    Parameters
    ----------
    bias: nd-array
        3d-array containing the bias applied to each point of the grid
    phase: nd-array
        3d-array containing the phase applied to each point of the grid
    amp: nd-array
        3d-array containing the amplitude applied to each point of the grid
    threshold: int or float, optional
        minimal value of the loop area to be considered a good loop (default: None)
        if set to None will use threshold = np.mean(area_grid_full) - 2 * np.std(area_grid_full)

    Returns
    -------
        good_bias: list
            list of the bias corresponding to good loops
        good_phase: list
            list of the phase corresponding to good loops
        good_amp: list
            list of the amplitudes corresponding to good loops
        mask: list
            a 2D mask where a 1 correspond to a good loop and 0 to a bad loop, can be used to mask
            the input data.
    """
    good_bias = []
    good_phase = []
    good_amp = []

    mask = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))

    if threshold is None:
        area_grid_full = np.ndarray((np.shape(bias)[0], np.shape(bias)[1]))
        for xi in range(np.shape(bias)[0]):
            for yi in range(np.shape(bias)[1]):
                area_grid_full[xi, yi] = gauss_area(bias[xi, yi, :], phase[xi, yi, :])
            threshold = np.mean(area_grid_full) - 2 * np.std(area_grid_full)

    for xi in range(np.shape(bias)[0]):
        for yi in range(np.shape(bias)[1]):
            if gauss_area(bias[xi, yi, :], phase[xi, yi, :]) > threshold:
                good_bias.append(bias[xi, yi, :])
                good_phase.append(phase[xi, yi, :])
                good_amp.append(amp[xi, yi, :])
                mask[xi, yi] = True
            else:
                mask[xi, yi] = False

    return good_bias, good_phase, good_amp, mask
