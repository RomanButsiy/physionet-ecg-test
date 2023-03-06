import wfdb
import pandas as pd
import neurokit2 as nk
import numpy as np
from common.logs.log import get_logger

logger = get_logger('Generate FR')

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

    logger.info("Get ECG Peaks")

    for i in range(2):
        _, rpeaks = nk.ecg_peaks(signals_T[i], sampling_rate=sampling_rate)
        signals, waves = nk.ecg_delineate(signals_T[i], rpeaks, sampling_rate=sampling_rate)
        ECG_T_Peaks = list(np.array(waves["ECG_T_Peaks"]) / sampling_rate)
        ECG_P_Peaks = list(np.array(waves["ECG_P_Peaks"]) / sampling_rate)
        del ECG_P_Peaks[0]
        ECG_P_Peaks.append(None)
        ecg_fr = pd.DataFrame({"D_c" : ECG_T_Peaks, "D_z" : ECG_P_Peaks})
        nk.write_csv(ecg_fr, f'{fr_path}_{fileds["sig_name"][i]}.csv')
