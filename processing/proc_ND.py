from processing import proc_tools as pt
import numpy as np
import h5py


def extract_hyst(filename, data_folder = 'datasets', selection = None):
    with h5py.File(filename, 'a') as f:

        notedict = {}
        for file in f['metadata'].keys(): 
            for k in f['metadata'][file]:
                tmp = k.decode('utf-8').split(":",1)
                try:
                    notedict[tmp[0]] = tmp[1]
                except:
                    pass

        if selection == None:
            keylist = f[data_folder].keys()
        elif selection == list:
            keylist = selection
        elif selection == str:
            keylist = [selection]
        else:
            print('selection input is incorrect, looking at the keys automatically')
            keylist = f[data_folder].keys()
            
        print(keylist)
        for file in keylist:

            inout_list = []
            inout_list = pt.initialise_process(filename, 
                      process_name = 'Extracted_PFM', 
                      data_folder = data_folder, 
                      selection = file, 
                      selection_depth = 1, 
                      create_groups = True)
            try:
                bias = f[data_folder][file]['Bias']
            except:
                print('Impossible to find the bias in this file, skipping it.')
                continue
                
            waveform_amp = notedict["ARDoIVAmp"]
            waveform_freq = float(notedict["ARDoIVFrequency"])
            waveform_phase = float(notedict["ARDoIVArg2"])
            waveform_pulsetime = float(notedict["ARDoIVArg3"])
            if 'ARDoIVArg4' in notedict.keys():
                waveform_dutycycle = float(notedict["ARDoIVArg4"])
            else:
                waveform_dutycycle = 0.5

            waveform_delta = 1/float(notedict["NumPtsPerSec"])
            waveform_numbiaspoints = int(np.floor(waveform_delta*len(bias[1,1,:])/waveform_pulsetime))
            waveform_pulsepoints = int(waveform_pulsetime/waveform_delta)
            waveform_offpoints = int(waveform_pulsepoints*(1.0-waveform_dutycycle))

            for chan in inout_list:
                print(chan[2] + ' started...')
                x,y,z = np.shape(f[data_folder][file][chan[2]])
                b = waveform_numbiaspoints
                # dgroup = f.require_group('process/' + file + '/' + 'SSPFM_parameters/')
                
                dset_on = f.require_dataset(chan[1] + chan[2]+ '_on', (x,y,b), dtype=float)
                PATH = [chan[0], chan[1], chan[2] + '_on']
                pt.generic_write(f, path=PATH)
                
                dset_off = f.require_dataset(chan[1] + chan[2]+ '_off', (x,y,b), dtype=float)
                PATH = [chan[0], chan[1], chan[2] + '_off']
                pt.generic_write(f, path=PATH)
                
                for b in range(waveform_numbiaspoints):
                    start = b*waveform_pulsepoints+waveform_offpoints
                    stop = (b+1)*waveform_pulsepoints

                    var2 = stop-start+1
                    realstart = int(start+var2*.25)
                    realstop = int(stop-var2*.25)
                    f[chan[1] + chan[2]+ '_on'][:,:,b] = \
                        np.nanmean(f[chan[0]][:,:,realstart:realstop], axis=2)

                    start = stop
                    stop = stop + waveform_pulsepoints*waveform_dutycycle

                    var2 = stop-start+1
                    realstart = int(start+var2*.25)
                    realstop = int(stop-var2*.25)
                    f[chan[1] + chan[2]+ '_off'][:,:,b] = \
                        np.nanmean(f[chan[0]][:,:,realstart:realstop], axis=2)
                print(chan[2] + ' done.')


def calc_hyst_params(bias, phase):
    biasdiff = np.diff(bias)
    up = np.sort(np.unique(np.hstack((np.where(biasdiff>0)[0],np.where(biasdiff>0)[0]+1))))
    dn = np.sort(np.unique(np.hstack((np.where(biasdiff<0)[0],np.where(biasdiff<0)[0]+1))))


    #UP leg calculations
    x = np.array(bias[up])
    y = np.array((phase[up]+360)%360)
    step_left_up = np.median(y[np.where(x==np.min(x))[0]])
    step_right_up = np.median(y[np.where(x==np.max(x))[0]])



    avg_x = []
    avg_y = []
    for v in np.unique(x):
        avg_x.append(v)
        avg_y.append(np.mean(y[np.where(x==v)[0]]))

    my_x = np.array(avg_x)[1:]+0*(avg_x[0]+avg_x[1])/2.0
    my_y = np.abs(np.diff(avg_y))

    coercive_volt_up = my_x[np.nanargmax(my_y)]

    #DOWN leg calculations
    x = np.array(bias[dn])
    y = np.array((phase[dn]+360)%360)
    step_left_dn = np.median(y[np.where(x==np.min(x))[0]])
    step_right_dn = np.median(y[np.where(x==np.max(x))[0]])

    avg_x = []
    avg_y = []
    for v in np.unique(x):
        avg_x.append(v)
        avg_y.append(np.mean(y[np.where(x==v)[0]]))

    my_x = np.array(avg_x)[1:]+0*(avg_x[0]+avg_x[1])/2.0
    my_y = np.abs(np.diff(avg_y))

    coercive_volt_dn = my_x[np.nanargmax(my_y)]

    return [coercive_volt_up, coercive_volt_dn, 0.5*step_left_dn+0.5*step_left_up, 0.5*step_right_dn+0.5*step_right_up]


def PFM_params_map(filename, phase = 1):
    
    with h5py.File(filename, 'a') as f:
        for proc in f['process/'].keys():
            print(proc)
            if len(proc.split('-')) < 2:
                print(proc + '. This process name was not formatted correctly, skipping it')
                continue
            if proc.split('-')[1] != 'Extracted_PFM':
                continue
            else:
                operation_number = str(len(f['process'])+1)    
                for file in f['process/' + proc].keys():
                    bias = f['process/' + proc + '/' + file + '/Bias_on']
                    if phase == 1:
                        path_bias = 'process/' + proc + '/' + file + '/Phase_on'
                        phase = f[path_bias]
                    else:
                        path_bias = 'process/' + proc + '/' + file + '/Phas2_on'
                        phase = f[path_bias]
                    x,y,z = np.shape(phase)
                    
                    list_values = ['coerc_pos', 'coerc_neg', 'step_left', 'step_right', 'imprint', 'phase_shift']
                    f['process/'].require_group(operation_number + '-PFM_physical_parameters/' + file)
                    for val in list_values:
                        f['process/' + operation_number + '-PFM_physical_parameters/' + file].require_dataset(val, (x,y), dtype=float)
                        PATH = [path_bias, 'process/' + operation_number + '-PFM_physical_parameters/' + file, val]
                        pt.generic_write(f, path=PATH)
                    for xi in range(x):
                        for yi in range(y):    
                            hyst_matrix = calc_hyst_params(bias[xi,yi,:],phase[xi,yi,:])
                            f['process/' + operation_number + '-PFM_physical_parameters/' + file + '/coerc_pos'][xi,yi] = hyst_matrix[0] 
                            f['process/' + operation_number + '-PFM_physical_parameters/' + file + '/coerc_neg'][xi,yi] = hyst_matrix[1]
                            f['process/' + operation_number + '-PFM_physical_parameters/' + file + '/step_left'][xi,yi] = hyst_matrix[2]
                            f['process/' + operation_number + '-PFM_physical_parameters/' + file + '/step_right'][xi,yi] = hyst_matrix[3]
                            f['process/' + operation_number + '-PFM_physical_parameters/' + file + '/imprint'][xi,yi] = (hyst_matrix[0]+hyst_matrix[1])/2.0
                            f['process/' + operation_number + '-PFM_physical_parameters/' + file + '/phase_shift'][xi,yi] = (hyst_matrix[3]-hyst_matrix[2])
