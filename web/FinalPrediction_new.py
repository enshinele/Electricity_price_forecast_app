import numpy as np
import pandas as pd
import copy
import torch
import time
import datetime
from Dataset_predict import Dataset
from GetData import GetData


def get_current_time():
    current_hour = time.strftime("%H", time.localtime())
    current_date = time.strftime("%Y-%m-%d", time.localtime()) + 'T00:00:00.0000000'
    current_timespan = current_hour + ':00' + '-' + str(int(current_hour) + 1) + ':00'
    return current_date, current_timespan


def get_next_timespan():
    current_hour = time.strftime("%H", time.localtime())
    if (int(current_hour) + 1) % 24 < 8:
        next_timespan = '0' + str((int(current_hour) + 1) % 24) + ':00' + '-' + '0' + str(
            (int(current_hour) + 2) % 24) + ':00'
    elif (int(current_hour) + 1) % 24 == 8:
        next_timespan = '0' + str((int(current_hour) + 1) % 24) + ':00' + '-' + str(
            (int(current_hour) + 2) % 24) + ':00'
    else:
        next_timespan = str((int(current_hour) + 1) % 24) + ':00' + '-' + str((int(current_hour) + 2) % 24) + ':00'
    return next_timespan

# read the data used for prediction in
from_date = (datetime.datetime.now()+datetime.timedelta(days=-365*2)).strftime("%Y-%m-%d")
to_date = datetime.datetime.now().strftime("%Y-%m-%d")

try:
    current_date, current_timespan = get_current_time()
    getData = GetData(from_date, to_date)
    dataset_prices = getData.getdata_price()
    dataset_prices.reset_index(drop=True)
    try:
        current_timestamp_index = dataset_prices[(dataset_prices.Date == current_date) &
                                                 (dataset_prices.Timespan == current_timespan)].index.tolist()[0]
        dataset_prices_ = dataset_prices.head(current_timestamp_index)
    except:
        dataset_prices_ = dataset_prices
except:
    dataset_prices = pd.read_csv('./data/prices_Phelix.csv',  index_col=0)
    dataset_prices_ = dataset_prices

# dataset_prices=pd.read_csv('../data/prices/prices_Phelix.csv',index_col = 0)#,index_col = 0
dataset_volumes=pd.read_csv('./data/volumes_Phelix.csv', index_col = 0)
# dataset_prices: data until the current date
# dataset_prices_: data until the current timespan


dataset_hour = Dataset(dataset_prices_, dataset_volumes, 'hour', 'predict', 72, 0.8, 'd2d', 64, 'Standard', 3)
data_hour, scaler_hour = dataset_hour.dataloader()
dataset_day = Dataset(dataset_prices_, dataset_volumes, 'day', 'predict', 12, 0.8, 'd2d', 64, 'Standard', 3)
data_day, scaler_day = dataset_day.dataloader()
dataset_week = Dataset(dataset_prices_, dataset_volumes, 'week', 'predict', 12, 0.8, 'd2d', 64, 'Standard', 3)
data_week, scaler_week = dataset_week.dataloader()

class FinalPrediction:
    def __init__(self, price, data_hour, data_day, data_week,
                 scaler_hour, scaler_day, scaler_week,
                 model_hour_path, model_day_path, model_week_path):
        '''
        :param get_model_type: string 'MLP'for moment. later other model types
        :param get_predict_data: csv for moment. later real-time data source from api
        :param get_predict_interval: string 'hour' for moment. later 'day','week'
        '''
        #self.get_model_type=get_model_type
        self.price = price
        self.data_hour = data_hour
        self.data_day = data_day
        self.data_week = data_week
        self.scaler_hour = scaler_hour
        self.scaler_day = scaler_day
        self.scaler_week = scaler_week
        self.model_hour_path = model_hour_path
        self.model_day_path = model_day_path
        self.model_week_path = model_week_path


    def average(self, data):
        # calculate the mean of the mittel part(25%~75%) of the data
        data_sorted = sorted(data)
        lower_bound = int(len(data)*0.25)
        upper_bound = int(len(data)*0.75)
        data_ = data_sorted[lower_bound:upper_bound]
        mean_ = np.mean(data_)
        return mean_


    def predict_hour(self):
        current_date, current_timespan = get_current_time()
        try:
            next_timespan = get_next_timespan()
            if next_timespan == '00:00-01:00':
                current_date = (datetime.datetime.now()+datetime.timedelta(days=1)).strftime("%Y-%m-%d") + 'T00:00:00.0000000'
            next_timestamp = current_date + next_timespan
            data_copy = copy.deepcopy(self.price)
            data_copy['Timestamp'] = data_copy['Date'].str.cat(data_copy['Timespan'])
            data_copy.set_index(["Timestamp"], inplace=True)
            predict_hour = data_copy.loc[next_timestamp, 'Value']
        except:
            predict_hour = []
            for hour_path in self.model_hour_path:
                model_hour = torch.load(hour_path)
                prediction = model_hour(self.data_hour).detach().numpy()
                predict_hour_ = self.scaler_hour.inverse_transform(prediction.repeat(3*72, axis=1)).flatten()
                price_value_index = [x*3 for x in range(72)]  # 3: num_feature 72: num_hdw
                predict_hour.append(self.average(predict_hour_[price_value_index]))

                # predict_hour = predict_hour_[:, 0]
                # predict_hour = self.scaler_hour.inverse_transform(prediction.repeat(5*240, axis=1))[:, 0]
                # self.predicted_price=self.prediction[-1]
                # dic = {'preticted price for one ' + args.mode: self.prediction[-1]}
        return np.mean(predict_hour)

    def predict_day(self):
        predict_day = []
        for day_path in self.model_day_path:
            model_day = torch.load(day_path)
            prediction = model_day(self.data_day).detach().numpy()
            predict_day_ = self.scaler_day.inverse_transform(prediction.repeat(3*12, axis=1)).flatten()
            price_value_index = [x * 3 for x in range(12)]
            predict_day.append(self.average(predict_day_[price_value_index]))

        # self.predicted_price=self.prediction[-1]
        # dic = {'preticted price for one ' + args.mode: self.prediction[-1]}
        return np.mean(predict_day)

    def predict_week(self):
        predict_week = []
        for week_path in self.model_week_path:
            model_week = torch.load(week_path)
            prediction = model_week(self.data_week).detach().numpy()
            predict_week_ = self.scaler_week.inverse_transform(prediction.repeat(3*12, axis=1)).flatten()
            price_value_index = [x * 3 for x in range(12)]
            predict_week.append(self.average(predict_week_[price_value_index]))
            # self.predicted_price=self.prediction[-1]
            # dic = {'preticted price for one ' + args.mode: self.prediction[-1]}
        return np.mean(predict_week)


    def predict(self):
        predict_hour = self.predict_hour()
        predict_day = self.predict_day()
        predict_week = self.predict_week()
        result_list = list([predict_hour, predict_day, predict_week])
        return result_list


model_hour_path = ['./model/hour_LSTM_ps1.pkl', './model/hour_MLP_ps2.pkl']  # ['MLP_hour.pkl']
model_day_path = ['./model/day_MLP_ps1.pkl', './model/day_LSTM_ps1.pkl', './model/day_LSTM_ps3.pkl'] # ['MLP_day.pkl']
model_week_path = ['./model/week_LSTM_ps5.pkl', './model/week_MLP_ps3.pkl']  # ['MLP_week.pkl']


final=FinalPrediction(dataset_prices, data_hour, data_day, data_week,
                      scaler_hour, scaler_day, scaler_week,
                      model_hour_path, model_day_path, model_week_path)
result_list = final.predict()
# print(result_list)






