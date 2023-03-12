import pandas as pd
import neurokit2 as nk
from pathlib import Path

class To_CSV:

    def __init__(self, path, fileds, sig_name):
        self.path = f'{path}_{fileds["sig_name"][sig_name]}'
        Path(self.path).mkdir(parents=True, exist_ok=True)
        
      
    def toFile(self, raw_data, name):
        path = f'{self.path}/{name}.csv'
        ecg_raw = pd.DataFrame({"Raw_data" : raw_data})
        nk.write_csv(ecg_raw, path)
        return 

      