from os import listdir
from os.path import isfile, join
from datetime import timedelta
import time

from scipy.signal import decimate
import pandas as pd
# import matplotlib.pyplot as plt
import pyedflib
import numpy as np


def load_ESM(path, filename):
    '''
    Loads .dta file, (re)names the columns and returns
    the dataframe (N x Features)

    in:
        path: path containing ESM data
        filename

    out:
        pd.DataFrame [N_subjects x Features]
    '''
    esm = pd.read_stata(path + filename, convert_categoricals=False)
    esm = esm[['subjno', 'mood_well', 'mood_down', 'mood_fright', 'mood_tense', 'phy_sleepy', 'phy_tired',
               'mood_cheerf', 'mood_relax', 'thou_concent', 'pat_hallu', 'loc_where',
               'soc_who', 'soc_who02', 'soc_who03', 'act_what', 'act_what02',
               'act_what03', 'act_norpob', 'sanpar_been', 'sanpar_stil',
               'sanpar_spreken', 'sanpar_lopen', 'sanpar_tremor', 'sanpar_traag',
               'sanpar_stijf', 'sanpar_spann', 'sanpar_beweeg', 'sanpar_onoff',
               'sanpar_medic', 'beep_disturb', '_datetime', '_datetime_e', 'dayno_n', 'beepno_n']]
    esm['duration'] = esm['_datetime_e'] - esm['_datetime']

    # rename to english
    esm = esm.rename(index=str,
                     columns={'sanpar_been': 'prob_mobility',
                              'sanpar_stil': 'prob_stillness',
                              'sanpar_spreken': 'prob_speech',
                              'sanpar_lopen': 'prob_walking',
                              'sanpar_tremor': 'tremor',
                              'sanpar_traag': 'slowness',
                              'sanpar_stijf': 'stiffness',
                              'sanpar_spann': 'tension',
                              'sanpar_beweeg': 'dyskinesia',
                              'sanpar_onoff': 'onoff',
                              'sanpar_medic': 'medic'})

    # Add correct subject_id labels
    map_names = {}
    for i in range(25):
        map_names[9009989 + i] = 110001 + i

    esm['castorID'] = [map_names[e] for e in esm['subjno']]

    return esm


def get_file_lists(local_path, subject):
    '''
    Creates a list of files for each sensor (Left/right/chest) located
    in local_path.

    in:
        local_path: Path containing sensor files
        subject: subject to retrieves files from

    out:
        (left/right/chest)_files: lists of filepaths per sensor location

    '''
    local_path = join(local_path, subject)

    # sensor names
    left_sensors = ['13797', '13799', '13794', '13806']
    right_sensors = ['13805', '13801', '13793', '13795']
    chest_sensors = ['13804', '13792', '13803', '13796']

    bdf_files = [f for f in listdir(local_path) if (isfile(join(local_path, f))) and (f[0] != '_' and f[-3:] == 'edf')]
    # bdffiles are the files in mypath, not directories

    left_files = []
    right_files = []
    chest_files = []

    for f in bdf_files:
        if f[0:5] in left_sensors:
            left_files.append(join(local_path, f))
        elif f[0:5] in right_sensors:
            right_files.append(join(local_path, f))
        elif f[0:5] in chest_sensors:
            chest_files.append(join(local_path, f))

    left_files = sorted(left_files)
    right_files = sorted(right_files)
    chest_files = sorted(chest_files)

    return left_files, right_files, chest_files


def extract_raw_trials(left_files, right_files, chest_files, esm_frame,
                       esm_window_length=15, feature_window_length=60):
    '''
    Reads the sensor files and aligns the data with the esm data. Then
    data quality is checked.

    Returns cleaned and synced trial data and ESM beeps
    '''
    # Read in the three list of files
    # Process leftWristData
    # left_wrist_df = []
    # right_wrist_df = []
    # chest_df = []
    # # trials = [[[] for _ in range(esm_frame.shape[0])], [[] for _ in range(esm_frame.shape[0])], [[] for _ in range(esm_frame.shape[0])]]

    # identifiers = ['l', 'r', 'c']

    files = [left_files, right_files, chest_files]

    n_beeps = esm_frame.shape[0]
    n_sensors = len(files)
    trials = [[[]] * n_beeps] * n_sensors

    found_trials = np.zeros((n_beeps, n_sensors))
    for i, f in enumerate(files):
        for file in f:
            print(file)

            # Read the data from the filepath, raise error if file is not in the
            # correct format.
            try:
                labels, timestamps, data, fs = read_edf_data(file)  # as input instead: leftFiles
                if data.shape[1] < fs * feature_window_length:
                    raise ValueError('File too short to proceed.')
            except Exception:
                print('%s is broken' % file)
                continue

            data = pd.DataFrame(data.T, index=timestamps)
            for beep in range(n_beeps):
                if found_trials[beep, i] == 1:
                    continue

                # Get the corresponding time
                beep_time = esm_frame['_datetime'].iloc[beep]
                timediff = np.min(np.abs(data.index - beep_time))

                # Find corresponding moment for beep time in the sensor data by
                # calculating the difference in sensor and beep timestamps and
                # select the index with the smallest difference.
                if timediff > timedelta(minutes=esm_window_length):
                    # If the time difference is larger than the window length,
                    # remove beep
                    continue
                pos = np.argmin(np.abs(data.index - beep_time))

                # For the smallest time difference, find the position in the sensor data
                if pos > esm_window_length * WINDOW_LENGTH * fs:
                    trials[i][beep] = data.iloc[pos - (int(esm_window_length * WINDOW_LENGTH * fs)):pos]
                    found_trials[beep, i] = 1

    keep = np.sum(found_trials, axis=1) == n_sensors  # Keep trials if all three sensors contain values
    trialData = np.zeros((np.sum(keep), int(esm_window_length * WINDOW_LENGTH * fs), 3 * 6))  # What are 3 and 6? n_sensors and ...?
    counter = 0
    for beep in range(n_beeps):
        if keep[beep]:
            temp = np.concatenate((trials[0][beep], trials[1][beep], trials[2][beep]), axis=1)
            trialData[counter, :, :] = temp
            counter += 1
    foundESM = esm_frame.iloc[keep, :]

    return trialData, foundESM


def read_edf_data(filename):
    '''
    Reads and .edf file and returns labels, timestamps, signal buffers and samplefrequency
    '''

    # Extract data
    f = pyedflib.EdfReader(filename)
    fs = f.getSampleFrequencies()[0]
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sig_bufs = np.zeros((n, f.getNSamples()[0]))

    for i in np.arange(n):
        sig_bufs[i, :] = f.readSignal(i)

    # Get starting time
    # startingTime=f.getStartdatetime() #needs to be tested
    starting_time = filename[-19:-4]
    starting_time = pd.to_datetime(starting_time, format='%Y%m%d_%H%M%S', errors='ignore')

    sig_bufs = decimate(sig_bufs, DOWNSAMPLING, axis=1)
    fs = fs / DOWNSAMPLING
    freq = '%d ms' % (1000 / fs)
    timestamps = pd.date_range(start=starting_time, periods=sig_bufs.shape[1], freq=freq)

    return signal_labels, timestamps, sig_bufs, fs


def align_features_esm(list_of_dfs, esm_frame, esm_columns, esm_window_length=15):
    '''
    not used

    '''

    combined_columns = esm_columns
    for feature_frame in list_of_dfs:
        combined_columns = combined_columns + feature_frame.keys().tolist()
    esm_features = pd.DataFrame(columns=combined_columns)  # Create new empty dataframe with feature and esm columns

    hop = np.mean(np.diff(list_of_dfs[0].index))
    for beep in range(esm_frame.shape[0]):  # Loop through all the ESM Beeps
        beep_time = esm_frame['_datetime'].iloc[beep]  # Get the corresponding time

        esm_data = np.matlib.repmat(esm_frame.iloc[beep][esm_columns], esm_window_length, 1)
        combined = esm_data

        sub_index = [beep_time - hop * t for t in range(esm_window_length)][::-1]

        for feature_frame in list_of_dfs:

            time_diff = np.min(np.abs(feature_frame.index - beep_time))
            # Find corresponding moment for beep time in the sensor data
            # print(timediff)
            if time_diff > timedelta(minutes=esm_window_length):
                # If corresponding time is too far off, remove beep
                # print("Couldn't find corresponding sensor data")
                continue
            pos = np.argmin(np.abs(feature_frame.index - beep_time))
            # For the smallest time difference find the position in the sensor data
            if pos > esm_window_length:
                feat_columns = feature_frame.keys().tolist()  # The names of the features
                feat_data = feature_frame.iloc[pos - esm_window_length:pos][feat_columns].values
                # Get corresponding timestamps
                # Repeat ESM data for each data point in the window
                combined = np.concatenate((combined, feat_data), axis=1)
                # Combine ESM & feature data
        if combined.shape[1] == len(combined_columns):
            esm_features = esm_features.append(pd.DataFrame(combined, columns=combined_columns, index=sub_index))
            # Append combined data to the dataframe
    return esm_features


if __name__ == '__main__':
    # constants
    WINDOW_LENGTH = 60
    DOWNSAMPLING = 2
    FEATURE_WINDOW_LENGTH = 60
    ESM_WINDOW_LENGTH = 15

    # path = "Y:/ADBS"
    # out_path = 'C:/data/processed/ESM_pilot/'
    # local_path = 'C:/data/raw/MOX/'
    path = './data/'
    out_path = './output'
    local_path = './data/MOX'

    esm_path = 'PRDB_20190227T102701/'
    esm_filename = 'SANPAR_BE.dta'

    # all_subjs = ['110001','110002','110003','110004','110005','110006','110007','110008','110009','110010',
    #            '110011','110013','110014','110015','110016','110017','110018','110019','110020','110021']
    # all_subjs = ['110015','110017','110018','110019','110020','110021']
    # all_subjs = ['110020'] #'110020',
    all_subjs = ['110004']

    esm = load_ESM(path + esm_path, esm_filename)

    for subject in all_subjs:
        leftFiles, rightFiles, chestFiles = get_file_lists(local_path, subject)
        t = time.time()
        trial_data, selected_esm = extract_raw_trials(leftFiles, rightFiles, chestFiles, esm[esm['castorID'] == int(subject)])
        print(time.time() - t)
        print(trial_data.shape)
        print(selected_esm.shape)
        np.save(join(out_path, subject + '_trials.npy'), trial_data.astype(np.float32))
        # np.save(join(outPath,subject + '_trials64.npy'),trialData)
        selected_esm.to_csv(join(out_path, subject + '_esm.csv'), index=False)


'''
### Transform Gyro data into orientation estimation
from madgwickahrs import MadgwickAHRS
mw = MadgwickAHRS(sampleperiod=1/sr)
euler = np.zeros((3,sigbufs.shape[1]))
for sample in range(sigbufs.shape[1]):
    mw.update_imu(sigbufs[6:,sample],sigbufs[3:6,sample])
    euler[:,sample] = mw.quaternion.to_euler123()


# In[ ]:


plt.matshow(euler,aspect='auto')
plt.yticks([0,1,2],['Roll', 'Pitch', 'Yaw'])
plt.xlabel('Time in samples')
plt.show()

'''
