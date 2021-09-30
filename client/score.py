import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


class Score:
    """
    Class used to score the performance of the model.
    """

    def __init__(self, tolerance):
        '''
        parameters:
        true: true value
        predict: predict value i.e. output of the prediction model
        tolerance: under this tolerance the predict value can be seen as the right prediction of the true value.
                   This parameter is only used when calculate the accuracy.
        '''

        self.tolerance = tolerance

    def cal_accuracy(self, true, predict):
        true_array = np.array(true).astype(float)
        predict_array = np.array(predict).astype(float)
        error = true_array.reshape((len(true_array), 1)) - predict_array.reshape((len(predict_array), 1))
        error_rate = np.abs(error / true_array.reshape((len(true_array), 1)))
        x = list(error_rate < self.tolerance)
        num_true = x.count(True)
        accuracy = num_true / len(x)
        return accuracy, np.mean(error_rate)



    def cal_score(self, true, predict):
        accuracy, error_rate = self.cal_accuracy(true, predict)
        MSE_score = mean_squared_error(predict, true)
        RMSE_score = np.sqrt(MSE_score)
        R2_score = r2_score(predict, true)
        return accuracy, error_rate, MSE_score, RMSE_score, R2_score

    def plot_results(self, length, true, predict):
        plt.plot(predict[-length:], 'r', label='prediction')
        plt.plot(true[-length:], 'b', label='true')
        # plt.plot((len(predict), len(true)), (0, 1), 'g--')
        plt.legend(loc='best')
        plt.show()
