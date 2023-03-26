import numpy as np

class Helpers:
    def __init__(self, interp_matrix_all, mathematical_expectation, sampling_rate=5000):
        self.sampling_rate = sampling_rate
        self.interp_matrix_all = interp_matrix_all
        self.m_ = mathematical_expectation

    def getCorrelation(self, correlation = False, deep = 1, multiply = False):
        tmp = None

        interp_matrix_all_len = len(self.interp_matrix_all)

        for i in range(interp_matrix_all_len - deep + 1):
            concated = np.ravel(self.interp_matrix_all[i: i + deep])
            one = self.interp_matrix_all[i]
            if correlation:
                one = one - self.m_[0]
                concated = concated - np.tile(self.m_[0], deep)

            r1, r2 = np.meshgrid(one, concated)
            r = r1 * r2
            if tmp is None:
                tmp = r
            else:
                tmp = tmp + r

        res2 = tmp / (interp_matrix_all_len - deep + 1)

        if (deep == 3) and multiply:
            res_len = len(res2[0])
            r1 = res2[0:res_len]
            r2 = res2[res_len:(2 * res_len)]
            r3 = res2[(2 * res_len):(3 * res_len)]
            w_1_1 = r1
            w_1_2 = r2
            w_1_3 = r3
            w_2_1 = np.rot90(np.fliplr(r2))
            w_2_2 = r1
            w_2_3 = r2
            w_3_1 = np.rot90(np.fliplr(r3))
            w_3_2 = np.rot90(np.fliplr(r2))
            w_3_3 = r1

            w1 = np.concatenate((w_1_1, w_1_2, w_1_3))
            w2 = np.concatenate((w_2_1, w_2_2, w_2_3))
            w3 = np.concatenate((w_3_1, w_3_2, w_3_3))

            res2 = np.concatenate((w1, w2, w3), axis=1)
        
        return res2

    def fft(self, i, sampling_rate):
        L = len(i)
        freq = np.linspace(0.0, 1.0 / (2.0 * sampling_rate **-1), L // 2)
        yi = np.fft.fft(i)
        y = yi[range(int(L / 2))]
        return freq, (abs(y) / sampling_rate)
    
    def tfft(self, i, sampling_rate):
        L = len(i)
        yi = np.fft.fft(i)
        y = yi[range(int(L / 2))]
        return (abs(y))