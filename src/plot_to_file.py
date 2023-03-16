import numpy as np
import matplotlib.pyplot as plt

class Plot_To_File:
    def __init__(self, sampling_rate=5000):
        self.sampling_rate = sampling_rate

    def plot_to_file(self, plot1, name, xlim = None, ylim = None, path = "img", ytext="", xtext="", size=(10, 6)):
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        time = np.arange(0, len(plot1), 1) / self.sampling_rate
        axis.plot(time, plot1, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.legend([ytext], loc='lower right')
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig("{}/{}.png".format(path, name), dpi=300)
        return

    def fft_plot_to_file(self, plot1, plot2, name, xlim = None, ylim = None, path = "img/fft", ytext="", xtext="", size=(10, 6)):
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
    
    def _3d_plot_to_file(self, plot1, name, xlim = None, ylim = None, path = "3d-img", ytext="", xtext="", size=(10, 6)):
        #TODO:
        return