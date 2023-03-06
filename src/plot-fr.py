import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.logs.log import get_logger

logger = get_logger('Generate FR')

sig_name = 1
file_mame = "b001"
data_path = f'ECG_IEEE_Access_2022/dat/{file_mame}'
fr_path = f'ECG_IEEE_Access_2022/fr/{file_mame}'

if __name__ == '__main__':
    logger.info("Read physionet file")
    logger.info(data_path)

    signals, fileds = wfdb.rdsamp(data_path)

    sampling_rate = fileds['fs']
    logger.info(f'Fileds: {fileds["sig_name"]}')
    logger.info(f'Sampling rate: {sampling_rate}')

    signals_T = np.array([[signals[j][i] for j in range(len(signals))] for i in range(len(signals[0]))])

    ecg_fr = pd.read_csv(f'{fr_path}_{fileds["sig_name"][sig_name]}.csv')
    D_c = ecg_fr["D_c"]
    D_z = ecg_fr["D_z"][:-1]

    matrix_passivity = []
    matrix_activity = []
    
    for i in range(len(D_z)):
        start = int((D_c[i]) * sampling_rate)
        end = int((D_z[i]) * sampling_rate)
        matrix_passivity.append(signals_T[sig_name][start:end])
        start = int((D_z[i]) * sampling_rate)
        end = int((D_c[i + 1]) * sampling_rate)
        matrix_activity.append(signals_T[sig_name][start:end])

    plt.clf()
    f, axis = plt.subplots(1)
    f.tight_layout()
    f.set_size_inches(17, 4)
    axis.grid(True)
    for i in matrix_activity:
        axis.plot(i, linewidth=2)
    plt.savefig(f'fr-img/a_{file_mame}_{fileds["sig_name"][sig_name]}.png', dpi=300)

    plt.clf()
    f, axis = plt.subplots(1)
    f.tight_layout()
    f.set_size_inches(17, 4)
    axis.grid(True)
    for i in matrix_passivity:
        axis.plot(i, linewidth=2)
    plt.savefig(f'fr-img/p_{file_mame}_{fileds["sig_name"][sig_name]}.png', dpi=300)

    B_i = len(D_z) -1

    T1_D_c = []
    T1_D_z = []
    T1_X = []
    T1_Y = []

    for i in range(len(D_c)-1):
        T1_D_c.append(round(D_c[i+1] - D_c[i], 2))

    for i in range(len(D_z)-1):
        T1_D_z.append(round(D_z[i+1] - D_z[i], 2))

    for i in range(len(D_c)):
        T1_X.append(D_c[i])
        if i == B_i :
            break
        # print(i)
        T1_X.append(D_z[i])

    for i in range(len(T1_D_c)):
        T1_Y.append(T1_D_c[i])
        if i == B_i :
            break
        T1_Y.append(T1_D_z[i])

    plt.clf()
    plt.rcParams.update({'font.size': 14})
    f, axis = plt.subplots(1)
    f.tight_layout()
    f.set_size_inches(10, 6)
    axis.grid(True)
    axis.plot(T1_X, T1_Y, linewidth=3)
    axis.set_xlabel("$t, s$", loc = 'right')
    axis.legend(['$T(t, 1), s$'])
    axis.axis(ymin = 0, ymax = 1.2)
    axis.axis(xmin = 0, xmax = 272)
    plt.savefig(f'fr-img/FR_{file_mame}_{fileds["sig_name"][sig_name]}.png', dpi=300)