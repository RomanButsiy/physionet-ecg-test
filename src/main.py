import itertools
import wfdb
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy
from common.logs.log import get_logger
from toCSV import To_CSV as toCSV
from plot_to_file import Plot_To_File as plotToFile
from my_helpers import Helpers as myHelp

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

logger = get_logger('Main')

sig_name = 1
file_mame = "b001"
data_path = f'ECG_IEEE_Access_2022/dat/{file_mame}'
fr_path = f'ECG_IEEE_Access_2022/fr/{file_mame}'
m_path = f'm-file/{file_mame}'

multiplier = 0.2

def get_new_matrix(matrix, k):
    n = 0
    for i in range(len(matrix)):
        n = n + len(matrix[i])
    n = int((n / len(matrix)) * k)
    return n

def get_all_matrix(input_matrix, matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate):
    activity = input_matrix[:matrix_activity_size]
    passivity = input_matrix[- matrix_passivity_size:]
    res = []
    for i in range(len(D_z)):
        passivity_len = int((D_z[i] - D_c[i]) * sampling_rate)
        activity_len = int((D_c[i + 1] - D_z[i]) * sampling_rate)
        arr = np.array(activity)
        arr_interp = interp.interp1d(np.arange(arr.size), arr)
        arr_stretch = arr_interp(np.linspace(0, arr.size - 1, activity_len))
        res.append(arr_stretch)
        arr = np.array(passivity)
        arr_interp = interp.interp1d(np.arange(arr.size), arr)
        arr_stretch = arr_interp(np.linspace(0, arr.size - 1, passivity_len))
        res.append(arr_stretch)
    return np.concatenate(res)

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

    interp_matrix_passivity = []
    interp_matrix_activity = []

    sampling_rate = sampling_rate * multiplier

    matrix_passivity_size = get_new_matrix(matrix_passivity, multiplier)
    matrix_activity_size = get_new_matrix(matrix_activity, multiplier)
    # matrix_passivity_size = int(len(matrix_passivity[0]) * multiplier)
    # matrix_activity_size = int(len(matrix_activity[0]) * multiplier)

    for i in range(len(matrix_passivity)):
        arr = np.array(matrix_passivity[i])
        arr_interp = interp.interp1d(np.arange(arr.size), arr)
        arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_passivity_size))
        interp_matrix_passivity.append(arr_stretch)
    for i in range(len(matrix_activity)):
        arr = np.array(matrix_activity[i])
        arr_interp = interp.interp1d(np.arange(arr.size), arr)
        arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_activity_size))
        interp_matrix_activity.append(arr_stretch)

    interp_matrix_all = np.concatenate((interp_matrix_activity[:-1], interp_matrix_passivity[1:]), axis=1)

    # plt.clf()
    # f, axis = plt.subplots(1)
    # f.tight_layout()
    # f.set_size_inches(17, 4)
    # axis.grid(True)
    # for i in interp_matrix_all:
    #     axis.plot(i, linewidth=2)
    # plt.savefig(f'img/2a_{file_mame}_{fileds["sig_name"][sig_name]}.png', dpi=300)

    #Математичне сподівання
    m_ = []
    #Початковий момент другого порядку
    m_2_ = []
    #Початковий момент третього порядку
    m_3_ = []
    #Початковий момент четвертого порядку
    m_4_ = []
    #Початковий момент другого порядку
    m__2 = []
    #Початковий момент четвертого порядку
    m__4 = []

    interp_matrix_T = interp_matrix_all.transpose()

    #Математичне сподівання
    m_.append([np.mean(i) for i in interp_matrix_T])

    # #Початковий момент другого порядку
    # m_2_.append([np.sum(np.array(i)**2) / len(i) for i in interp_matrix_T])

    # #Початковий момент третього порядку
    # m_3_.append([np.sum(np.array(i)**3) / len(i) for i in interp_matrix_T])
            
    # #Початковий момент четвертого порядку
    # m_4_.append([np.sum(np.array(i)**4) / len(i) for i in interp_matrix_T])

    # #Центральний момент другого порядку
    # m__2.append([sum((interp_matrix_T[i] - m_[0][i])**2) / len(interp_matrix_T[i]) for i in range(len(m_[0]))])

    # #Центральний момент четвертого порядку
    # m__4.append([sum((interp_matrix_T[i] - m_[0][i])**4) / len(interp_matrix_T[i]) for i in range(len(m_[0]))])

    # #Математичне сподівання
    # m_all = get_all_matrix(m_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    # #Початковий момент другого порядку
    # m_2_all = get_all_matrix(m_2_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    # #Початковий момент третього порядку
    # m_3_all = get_all_matrix(m_3_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    # #Початковий момент четвертого порядку
    # m_4_all = get_all_matrix(m_4_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    # #Початковий момент другого порядку
    # m__2_all = get_all_matrix(m__2[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    # #Початковий момент четвертого порядку
    # m__4_all = get_all_matrix(m__4[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)

    hlp = myHelp(interp_matrix_all, m_, sampling_rate = sampling_rate)
    ptf = plotToFile(sampling_rate = sampling_rate)

    # Коваріація
    # c1 = hlp.getCorrelation(correlation = False, deep = 3, multiply = True)
    # ptf._3d_plot_to_file(c1, "Autocorrelation function", path = "3d-img", size=(10, 10, 10), correlation = False, v = (-0.02, 0.05), ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), mV^2$')

    # Кореляція
    # c2 = hlp.getCorrelation(correlation = True, deep = 3, multiply = True)
    # ptf._3d_plot_to_file(c2, "Autocovariation function", path = "3d-img", size=(10, 10, 10), correlation = True, v = (-0.0017, 0.001), ztext=r'$1    \hat{C}_{2_{\xi}} (t_1, t_2), mV^2$')

    # Коваріація
    # c1 = hlp.getCorrelation(correlation = False, deep = 3, multiply = True)
    # ptf._3d_plot_to_file(c1, "Autocorrelation function multiply", path = "3d-img", size=(10, 10, 10), correlation = False, v = (-0.002, 0.006), ztext=r'$\hat{R}_{2_{\xi_{\hat{T}_{av}}}} (t_1, t_2), mV^2$')

    # Кореляція
    # c2 = hlp.getCorrelation(correlation = True, deep = 3, multiply = True)
    # ptf._3d_plot_to_file(c2, "Autocovariation function multiply", path = "3d-img", size=(10, 10, 10), correlation = True, v = (-0.002, 0.006), ztext=r'$\hat{C}_{2_{\xi_{\hat{T}_{av}}}} (t_1, t_2), mV^2$')
    

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # fig.set_size_inches(10, 10, 10)

    # ft = np.fft.ifftshift(c2)
    # ft = np.fft.fft2(ft)
    # ft = np.fft.fftshift(ft)
    # ft = abs(ft) / sampling_rate

    # X = np.arange(0, len(ft[0]), 1)
    # Y = np.arange(0, len(ft), 1)
    # X, Y = np.meshgrid(X, Y)

    # ax.set_zlim(-0.005, 0.1)
    # surf = ax.plot_surface(X, Y, ft, rstride=5,cstride=5,cmap=cm.coolwarm,linewidth=0)

    # plt.gca().invert_xaxis()
    # plt.savefig("{}/{}.png".format("3d-img", "fft12"), dpi=300)


    size = (19, 6)
    xlim = (0, 20)
    xtext = "$f, Hz$"

    # ptf.fft_plot_to_file(*hlp.fft(m_all[:100000], sampling_rate), "Математичне сподівання", xtext=xtext, ytext=r"$S_{m_{{\xi}}} (f), mV / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m_2_all[:100000], sampling_rate), "Початковий момент другого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^2 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m_3_all[:100000], sampling_rate), "Початковий момент третього порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^3 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m_4_all[:100000], sampling_rate), "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^4 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m__2_all[:100000], sampling_rate), "Центральний момент другого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^2 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m__4_all[:100000], sampling_rate), "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^4 / Hz$", size=size, xlim=xlim)

    size = (19, 6)
    xlim = (0, 80)
    
    # ptf.fft_plot_to_file(*hlp.fft(m_[0], sampling_rate), "Математичне сподівання", xtext=xtext, ytext=r"$S_{m_{{\xi}}} (f), mV / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m_2_[0], sampling_rate), "Початковий момент другого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^2 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m_3_[0], sampling_rate), "Початковий момент третього порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^3 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m_4_[0], sampling_rate), "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^4 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m__2[0], sampling_rate), "Центральний момент другого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^2 / Hz$", size=size, xlim=xlim)

    # ptf.fft_plot_to_file(*hlp.fft(m__4[0], sampling_rate), "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^4 / Hz$", size=size, xlim=xlim)


    size = (19, 6)
    xlim = (0, 4.5)
    xtext = "$t, s$"

    # ptf.plot_to_file(m_all[:100000], "Mathematical expectation", xtext=xtext, ytext=r"$\hat{m}_{\xi} (t), mV$", size=size, xlim=xlim)

    # ptf.plot_to_file(m_2_all[:100000], "Initial moment of the 2nd order", xtext=xtext, ytext=r"$\hat{m}_{2_{\xi}} (t), mV^2$", size=size, xlim=xlim)

    # ptf.plot_to_file(m_3_all[:100000], "Initial moment of the 3nd order", xtext=xtext, ytext=r"$\hat{m}_{3_{\xi}} (t), mV^3$", size=size, xlim=xlim)

    # ptf.plot_to_file(m_4_all[:100000], "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$", size=size, xlim=xlim)

    # ptf.plot_to_file(m__2_all[:100000], "Central moment of the 2nd order", xtext=xtext, ytext=r"$\hat{d}_{2_{\xi}} (t), mV^2$", size=size, xlim=xlim)

    # ptf.plot_to_file(m__4_all[:100000], "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$", size=size, xlim=xlim)

    # ptf.plot_to_file(m_all[:100000], "Mathematical expectation", xtext=xtext, ytext=r"$\hat{m}_{\xi_{\hat{T}_{av}}} (t), mV$", size=size, xlim=xlim)

    # ptf.plot_to_file(m_2_all[:100000], "Initial moment of the 2nd order", xtext=xtext, ytext=r"$\hat{m}_{2_{\xi_{\hat{T}_{av}}}} (t), mV^2$", size=size, xlim=xlim)

    # ptf.plot_to_file(m_3_all[:100000], "Initial moment of the 3nd order", xtext=xtext, ytext=r"$\hat{m}_{3_{\xi_{\hat{T}_{av}}}} (t), mV^3$", size=size, xlim=xlim)

    # ptf.plot_to_file(m__2_all[:100000], "Central moment of the 2nd order", xtext=xtext, ytext=r"$\hat{d}_{2_{\xi_{\hat{T}_{av}}}} (t), mV^2$", size=size, xlim=xlim)

    ptf.plot_to_file(m_[0], "Математичне сподівання", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")

    # ptf.plot_to_file(m_2_[0], "Початковий момент другого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$")

    # ptf.plot_to_file(m_3_[0], "Початковий момент третього порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^3$")

    # ptf.plot_to_file(m_4_[0], "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$")

    # ptf.plot_to_file(m__2[0], "Центральний момент другого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$")

    # ptf.plot_to_file(m__4[0], "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$")

    # dataToFile = toCSV(m_path, fileds, sig_name)

    # dataToFile.toFile(m_[0], "Математичне сподівання")
    # dataToFile.toFile(m_2_[0], "Початковий момент другого порядку")
    # dataToFile.toFile(m_3_[0], "Початковий момент третього порядку")
    # dataToFile.toFile(m_4_[0], "Початковий момент четвертого порядку")
    # dataToFile.toFile(m__2[0], "Центральний момент другого порядку")
    # dataToFile.toFile(m__4[0], "Центральний момент четвертого порядку")

    # dataToFile.toFile(m_all[:100000], "20s Математичне сподівання")
    # dataToFile.toFile(m_2_all[:100000], "20s Початковий момент другого порядку")
    # dataToFile.toFile(m_3_all[:100000], "20s Початковий момент третього порядку")
    # dataToFile.toFile(m_4_all[:100000], "20s Початковий момент четвертого порядку")
    # dataToFile.toFile(m__2_all[:100000], "20s Центральний момент другого порядку")
    # dataToFile.toFile(m__4_all[:100000], "20s Центральний момент четвертого порядку")


    # time = np.arange(0, len(signals_T[sig_name]), 1) / sampling_rate
    # plt.clf()
    # plt.rcParams.update({'font.size': 16})
    # f, axis = plt.subplots(1)
    # f.tight_layout()
    # f.set_size_inches(25, 6)
    # axis.grid(True)
    # axis.plot(time, signals_T[sig_name], linewidth=3)
    # # for x1, x2 in zip(D_c[:10], D_z[:10]):
    # #     axis.axvline(x = x1, linewidth=3, color = '#ff7f0e')
    # #     axis.axvline(x = x2, linewidth=3, color = '#2ca02c')
    # axis.set_xlabel("$t, s$", loc = 'right')
    # axis.axis(ymin = -0.6)
    # axis.axis(xmin = 0, xmax = 10)
    # # axis.legend(['$\\xi_{\\omega} (t), mV$', '$T \\: peaks$', '$P \\: peaks$'])
    # axis.set_title("$\\xi_{\\omega} (t), mV$", loc = 'left', fontsize=16, position=(-0.05, 0))
    # plt.savefig(f'img/fragment_{file_mame}_{fileds["sig_name"][sig_name]}.png', dpi=300)