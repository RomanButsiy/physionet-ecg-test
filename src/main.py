import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from common.logs.log import get_logger

logger = get_logger('Main')

sig_name = 1
file_mame = "b001"
data_path = f'ECG_IEEE_Access_2022/dat/{file_mame}'
fr_path = f'ECG_IEEE_Access_2022/fr/{file_mame}'

def get_new_matrix(matrix, k = 1):
    n = 0
    for i in range(len(matrix)):
        n = n + len(matrix[i])
    n = int((n / len(matrix)) * k)
    return n

def plot_to_file(plot1, name, xlim = None, ylim = None, path = "img", ytext="", xtext="", sampling_rate=5000, size=(10, 6)):
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    f, axis = plt.subplots(1)
    f.tight_layout()
    f.set_size_inches(size[0], size[1])
    axis.grid(True)
    time = np.arange(0, len(plot1), 1) / sampling_rate
    axis.plot(time, plot1, linewidth=3)
    axis.set_xlabel(xtext, loc = 'right')
    axis.legend([ytext], loc='lower right')
    if xlim is not None:
        axis.axis(xmin = xlim[0], xmax = xlim[1])
    if ylim is not None:
        axis.axis(ymin = ylim[0], ymax = ylim[1])
    plt.savefig("{}/{}.png".format(path, name), dpi=300)
    return

def fft_plot_to_file(plot1, plot2, name, xlim = None, ylim = None, path = "img/fft", ytext="", xtext="", size=(10, 6)):
    plt.clf()
    plt.rcParams.update({'font.size': 14})
    f, axis = plt.subplots(1)
    f.tight_layout()
    f.set_size_inches(size[0], size[1])
    axis.grid(True)
    axis.stem(plot1, plot2, markerfmt=" ")
    axis.set_xlabel(xtext, loc = 'right')
    axis.set_title(ytext, loc = 'left', fontsize=10, position=(-0.07, 0))
    # axis.legend([ytext], loc='right')
    if xlim is not None:
        axis.axis(xmin = xlim[0], xmax = xlim[1])
    if ylim is not None:
        axis.axis(ymin = ylim[0], ymax = ylim[1])
    plt.savefig("{}/{}.png".format(path, name), dpi=300)
    return

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

def fft(i, fs):
    L = len(i)
    freq = np.linspace(0.0, 1.0 / (2.0 * fs **-1), L // 2)
    yi = np.fft.fft(i)
    y = yi[range(int(L / 2))]
    return freq, abs(y) / fs
    

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

    matrix_passivity_size = get_new_matrix(matrix_passivity)
    matrix_activity_size = get_new_matrix(matrix_activity)
    # matrix_passivity_size = len(matrix_passivity[0])
    # matrix_activity_size = len(matrix_activity[0])

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

    interp_matrix_T = [[interp_matrix_all[j][i] for j in range(len(interp_matrix_all))] for i in range(len(interp_matrix_all[0]))]

    # #Математичне сподівання
    m_.append([sum(i) / len(i) for i in interp_matrix_T])

    # #Початковий момент другого порядку
    m_2_.append([np.sum(np.array(i)**2) / len(i) for i in interp_matrix_T])

    # #Початковий момент третього порядку
    m_3_.append([np.sum(np.array(i)**3) / len(i) for i in interp_matrix_T])
            
    # #Початковий момент четвертого порядку
    m_4_.append([np.sum(np.array(i)**4) / len(i) for i in interp_matrix_T])

    #Центральний момент другого порядку
    m__2.append([sum((interp_matrix_T[i] - m_[0][i])**2) / len(interp_matrix_T[i]) for i in range(len(m_[0]))])

    # #Центральний момент четвертого порядку
    m__4.append([sum((interp_matrix_T[i] - m_[0][i])**4) / len(interp_matrix_T[i]) for i in range(len(m_[0]))])

    #Математичне сподівання
    m_all = get_all_matrix(m_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    #Початковий момент другого порядку
    m_2_all = get_all_matrix(m_2_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    #Початковий момент третього порядку
    m_3_all = get_all_matrix(m_3_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    #Початковий момент четвертого порядку
    m_4_all = get_all_matrix(m_4_[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    #Початковий момент другого порядку
    m__2_all = get_all_matrix(m__2[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)
    #Початковий момент четвертого порядку
    m__4_all = get_all_matrix(m__4[0], matrix_activity_size, matrix_passivity_size, D_c, D_z, sampling_rate)

    size = (19, 6)
    xlim = (0, 20)
    xtext = "$f, Hz$"

    fft_plot_to_file(*fft(m_all[:100000], sampling_rate), "Математичне сподівання", xtext=xtext, ytext=r"$S_{m_{{\xi}}} (f), mV / Hz$", size=size, xlim=xlim)

    fft_plot_to_file(*fft(m_2_all[:100000], sampling_rate), "Початковий момент другого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^2 / Hz$", size=size, xlim=xlim)

    fft_plot_to_file(*fft(m_3_all[:100000], sampling_rate), "Початковий момент третього порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^3 / Hz$", size=size, xlim=xlim)

    fft_plot_to_file(*fft(m_4_all[:100000], sampling_rate), "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^4 / Hz$", size=size, xlim=xlim)

    fft_plot_to_file(*fft(m__2_all[:100000], sampling_rate), "Центральний момент другого порядку", xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^2 / Hz$", size=size, xlim=xlim)

    fft_plot_to_file(*fft(m__4_all[:100000], sampling_rate), "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$S_{d_{{\xi}}} (f), mV^4 / Hz$", size=size, xlim=xlim)


    size = (19, 6)
    xlim = (0, 6)
    xtext = "$t, s$"

    # plot_to_file(m_all[:100000], sampling_rate), "Математичне сподівання", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$", size=size, xlim=xlim)

    # plot_to_file(m_2_all[:100000], sampling_rate), "Початковий момент другого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$", size=size, xlim=xlim)

    # plot_to_file(m_3_all[:100000], sampling_rate), "Початковий момент третього порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^3$", size=size, xlim=xlim)

    # plot_to_file(m_4_all[:100000], sampling_rate), "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$", size=size, xlim=xlim)

    # plot_to_file(m__2_all[:100000], sampling_rate), "Центральний момент другого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$", size=size, xlim=xlim)

    # plot_to_file(m__4_all[:100000], sampling_rate), "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$", size=size, xlim=xlim)


    # plot_to_file(m_[0], "Математичне сподівання", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")

    # plot_to_file(m_2_[0], "Початковий момент другого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$")

    # plot_to_file(m_3_[0], "Початковий момент третього порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^3$")

    # plot_to_file(m_4_[0], "Початковий момент четвертого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$")

    # plot_to_file(m__2[0], "Центральний момент другого порядку", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$")

    # plot_to_file(m__4[0], "Центральний момент четвертого порядку",  xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$")



    # time = np.arange(0, len(signals_T[sig_name]), 1) / sampling_rate
    # plt.clf()
    # plt.rcParams.update({'font.size': 14})
    # f, axis = plt.subplots(1)
    # f.tight_layout()
    # f.set_size_inches(19, 6)
    # axis.grid(True)
    # axis.plot(time, signals_T[sig_name], linewidth=2)
    # for x1, x2 in zip(D_c[:10], D_z[:10]):
    #     axis.axvline(x = x1, linewidth=3, color = '#ff7f0e')
    #     axis.axvline(x = x2, linewidth=3, color = '#2ca02c')
    # axis.set_xlabel("$t, s$", loc = 'right')
    # axis.axis(ymin = -0.6)
    # axis.axis(xmin = 1, xmax = 7)
    # axis.legend(['$\\xi_{\\omega} (t), mV$', '$T \\: peaks$', '$P \\: peaks$'])
    # plt.savefig(f'img/fragment_{file_mame}_{fileds["sig_name"][sig_name]}.png', dpi=300)