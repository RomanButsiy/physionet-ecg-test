import numpy as np
import pandas as pd
import neurokit2 as nk
from plot_to_file import Plot_To_File as plotToFile

arr = ["20s Математичне сподівання", "20s Початковий момент другого порядку", "20s Початковий момент третього порядку", "20s Початковий момент четвертого порядку", "20s Центральний момент другого порядку", "20s Центральний момент четвертого порядку", "Математичне сподівання", "Початковий момент другого порядку", "Початковий момент третього порядку", "Початковий момент четвертого порядку", "Центральний момент другого порядку", "Центральний момент четвертого порядку"]
y_text = [r"$m_{{\xi}} (t), mV$", r"$d_{{\xi}} (t), mV^2$", r"$d_{{\xi}} (t), mV^3$", r"$d_{{\xi}} (t), mV^4$", r"$d_{{\xi}} (t), mV^2$", r"$d_{{\xi}} (t), mV^4$", r"$m_{{\xi}} (t), mV$", r"$d_{{\xi}} (t), mV^2$", r"$d_{{\xi}} (t), mV^3$", r"$d_{{\xi}} (t), mV^4$", r"$d_{{\xi}} (t), mV^2$", r"$d_{{\xi}} (t), mV^4$"]


def plot(name, ytext):
    path = f"m-file/b001_II/{name}"
    path_c = f"m-file/b001_II_const/{name}"
    ecg_raw = pd.read_csv(f'{path}.csv')
    ecg_raw_cons = pd.read_csv(f'{path_c}.csv')

    data_new = ecg_raw["Raw_data"]
    data_const = ecg_raw_cons["Raw_data"]

    data = np.abs(data_new - data_const)

    ptf = plotToFile()

    size = (19, 6)
    xlim = (0, 6)
    xtext = "$t, s$"

    ptf.plot_to_file(data, f'diff {name}', xtext=xtext, ytext=ytext)


if __name__ == '__main__':
    for i, y in zip(arr, y_text):
        plot(i, y)