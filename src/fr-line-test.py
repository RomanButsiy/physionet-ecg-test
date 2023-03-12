import pandas as pd
import neurokit2 as nk
import numpy as np
from common.logs.log import get_logger

logger = get_logger('Generate Liner FR')

if __name__ == '__main__':
    logger.info("Test")
    D_c = np.arange(0.9334, 271, 0.9104)
    D_z = np.arange(1.4124, 271, 0.9104)
    ecg_fr = pd.DataFrame({"D_c" : np.round(D_c, 6), "D_z" : np.round(D_z, 6)})
    nk.write_csv(ecg_fr, f'test.csv')