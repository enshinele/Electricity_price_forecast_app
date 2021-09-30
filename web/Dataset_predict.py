import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.utils.data as Data
import pickle


class Dataset:
    """
    This is a class used to build the dataset for training and prediction.

    For training we need to do the following steps:
    1. preprocessing the raw data includes drop duplicates and missing value interpolation
    2. merge the price data and the volume data, if merge_ is True
    3. transform the data into the desired form.
    4. split the data into train set and test set according to the train_size
    5. scale the data with MinMax-or-Standardscaler

    For prediction we don't need to do the 4th step, other steps are the same as for training.

    Attention: the used scaler will also be return as it is needed to inverse transform the scaled predicted price.
    """

    def __init__(self, price_df, volume_df, mode, purpose, num_hdw, train_size,
                 day_type, batch_size, scaler_type, num_feature):
        '''
        :param price_df: raw price data type: dataframe
        :param volume_df: raw volume data type: dataframe
        :param mode: 'hour' or 'day' or 'week' type: string
        :param purpose: 'train' or 'predict' type: string
        :param num_hdw:  how many past hours/days/weeek used to predict the future price type: int
        :param train_size: how many percent of the data will be used for training type: float
        :param merge_: whether merge price-and-volume data type: bool
        :param day_type: the way to transformthe data for day ahead prediction training. 'd2d' or 'h2d' type: string
        :param batch_size: type: int
        :param scaler_type: which scaler will be used 'MinMax' or 'Standard' type: string
        '''
        self.price_df = price_df
        self.volume_df = volume_df
        self.mode = mode
        self.purpose = purpose
        self.num_hdw = num_hdw
        self.train_size = train_size
        #         self.merge_ = merge_
        self.day_type = day_type
        self.batch_size = batch_size
        self.scaler_type = scaler_type
        self.num_feature = num_feature

    def merge(self, preprocess_price, preprocess_volume):
        """
        return: merged_data dataframe
                columns:['Date','Timespan','price_Value','price_Base','price_Peak','volume_Value','volume_Base']
        """
        # deepcopy() garantees that price_df and volume_df will not be changed
        price = copy.deepcopy(preprocess_price)
        volume = copy.deepcopy(preprocess_volume)
        # Some of the timestamps between price data and volume data are different.
        price['Timestamp'] = price['Date'].str.cat(price['Timespan'])
        volume['Timestamp'] = volume['Date'].str.cat(volume['Timespan'])
        # price.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
        # volume.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
        price = price.drop(columns=['Date', 'Timespan'], axis=1)
        merged_data = pd.merge(price, volume, how='left', on='Timestamp')
        merged_data.rename(columns={'Value_x': 'price_Value', 'Base_x': 'price_Base', 'Peak': 'price_Peak',
                                    'Value_y': 'volume_Value', 'Base_y': 'volume_Base'}, inplace=True)
        # merged_data.columns = ['price_Value', 'price_Base', 'price_Peak',
        #                        'Timestamp', 'volume_Value', 'volume_Base', 'Date', 'Timespan']
        merged_data = merged_data[['Date', 'Timespan', 'price_Value',
                                   'price_Base', 'price_Peak', 'volume_Value', 'volume_Base']]
        return merged_data.dropna()

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def transform_hour(self, data):
        if self.purpose == 'train':
            data.drop(columns=['Date', 'Timespan'], inplace=True)
            n_in = self.num_hdw - 1
            n_out = 2
            data_hour = self.series_to_supervised(data, n_in, n_out)
            if self.num_feature == 1:
                drop_cols = []
            else:
                drop_cols = [x for x in data_hour.columns[-self.num_feature + 1:]]

            return data_hour.drop(columns=drop_cols, axis=1)
        elif self.purpose == 'predict':
            data.drop(columns=['Date', 'Timespan'], inplace=True)
            n_in = self.num_hdw - 1
            n_out = 1
            data_hour = self.series_to_supervised(data, n_in, n_out)
            return data_hour.reset_index(drop=True).loc[[len(data_hour) - 1]]#, data_hour
            # return data_hour.drop(columns=drop_cols, axis=1).reset_index(drop=True).loc[[len(data_hour) - 1]], data_hour

    def transform_day(self, data):
        #         num_features = len(data.columns) - 2  # except Date and Timespan
        if self.purpose == 'train':
            data_day = pd.DataFrame()
            if self.day_type == 'd2d':
                n_in = self.num_hdw - 1
                n_out = 2
                ts_group = data.groupby('Timespan')
                for key, item in ts_group:
                    group = ts_group.get_group(key)
                    data_day = pd.concat(
                        [data_day, self.series_to_supervised(group.drop(columns=['Date', 'Timespan']), n_in, n_out)],
                        ignore_index=True)
                if self.num_feature == 1:
                    return data_day
                drop_cols = [x for x in data_day.columns[-self.num_feature + 1:]]
                return data_day.drop(columns=drop_cols, axis=1)
            elif self.day_type == 'h2d':
                n_in = self.num_hdw - 1
                n_out = 25
                data_ = self.series_to_supervised(data.drop(columns=['Date', 'Timespan']), n_in, n_out)
                if self.num_feature == 1:
                    drop_cols = [x for x in data_.columns[-(n_out - 1) * self.num_feature:-self.num_feature]]
                else:
                    drop_cols = [x for x in data_.columns[-(n_out - 1) * self.num_feature:-self.num_feature]] + [x for x
                                                                                                                 in
                                                                                                                 data_.columns[
                                                                                                                 -self.num_feature + 1:]]
                data_day = data_.drop(columns=drop_cols, axis=1)
            return data_day

        if self.purpose == 'predict':
            if self.day_type == 'd2d':
                current_timespan = data.reset_index(drop=True).loc[len(data)-1, 'Timespan']
                data_predict = data.loc[data['Timespan'].isin([current_timespan])]
                n_in = self.num_hdw - 1
                n_out = 1
                data_day = self.series_to_supervised(data_predict.drop(columns=['Date', 'Timespan']), n_in, n_out)
                # if self.num_feature != 1:
                #     drop_cols = [x for x in data_day.columns[-self.num_feature + 1:]]
                #     data_day.drop(columns=drop_cols, inplace=True, axis=1)
            elif self.day_type == 'h2d':
                n_in = self.num_hdw - 1
                n_out = 1
                data_day = self.series_to_supervised(data.drop(columns=['Date', 'Timespan']), n_in, n_out)
                # if self.num_feature == 1:
                #     drop_cols = [x for x in data_.columns[-(n_out - 1) * self.num_feature:-self.num_feature]]
                # else:
                #     drop_cols = [x for x in data_.columns[-(n_out - 2) * self.num_feature:-self.num_feature]] + [x for x
                #                                                                                                  in
                #                                                                                                  data_.columns[
                #                                                                                                  -self.num_feature + 1:]]
                # data_day = data_.drop(columns=drop_cols, axis=1)

            return data_day.reset_index(drop=True).loc[[len(data_day) - 1]]#, data_day

    def transform_week(self, data):

        data_ = data.reset_index(drop=True)
        data_['Date'] = data_['Date'].apply(lambda x: str(x).replace('T00:00:00.0000000', ''))
        data_['Date'] = data_['Date'].apply(lambda x: str(x).replace('-', ''))
        data_['Date'] = pd.to_datetime(data_['Date'])
        data_['weekday'] = data_.Date.dt.day_name()
        weekday_ = data_.drop(columns=['Date'])
        weekday_['weektime'] = weekday_['weekday'].str.cat(weekday_['Timespan'])
        weekday_df = weekday_.drop(columns=['weekday', 'Timespan'])

        if self.purpose == 'train':
            n_in = self.num_hdw - 1
            n_out = 2
            data_week = pd.DataFrame()
            wt_group = weekday_df.groupby('weektime')
            for key, item in wt_group:
                group = wt_group.get_group(key)
                data_week = pd.concat(
                    [data_week, self.series_to_supervised(group.drop(columns=['weektime']), n_in, n_out)],
                    ignore_index=True)
            if self.num_feature == 1:
                return data_week
            drop_cols = [x for x in data_week.columns[-self.num_feature + 1:]]
            data_week.drop(columns=drop_cols, inplace=True, axis=1)
            return data_week
        if self.purpose == 'predict':
            n_in = self.num_hdw - 1
            n_out = 1
            current_weektime = weekday_df.loc[weekday_df.index[-1]].weektime
            data_predict = weekday_df.loc[weekday_df['weektime'].isin([current_weektime])]
            assert len(data_predict) >= self.num_hdw
            data_week = self.series_to_supervised(data_predict.drop(columns=['weektime']), n_in, n_out)
            # if self.num_feature != 1:
            #     drop_cols = [x for x in data_week.columns[-self.num_feature + 1:]]
            #     data_week.drop(columns=drop_cols, inplace=True, axis=1)
        return data_week.reset_index(drop=True).loc[[len(data_week) - 1]]#, data_week

    def transform(self, data):
        if self.purpose == 'train':
            if self.mode == 'hour':
                data = self.transform_hour(data)
            elif self.mode == 'day':
                data = self.transform_day(data)
            elif self.mode == 'week':
                data = self.transform_week(data)
            return data.reset_index(drop=True)
        elif self.purpose == 'predict':
            if self.mode == 'hour':
                data = self.transform_hour(data)
            elif self.mode == 'day':
                data = self.transform_day(data)
            elif self.mode == 'week':
                data = self.transform_week(data)
            return data  # , data_.reset_index(drop=True)

            # if self.mode == 'all':
            #     data_hour = self.transform_hour(data)
            #     data_day = self.transform_day(data)
            #     dat_week = self.transform_week(data)
            # return data_hour, data_day, dat_week


    def preprocess(self, data):
        # ensure that original data will not be changed
        data_copy = copy.deepcopy(data)
        # drop duplicates
        data_copy['Timestamp'] = data_copy['Date'].str.cat(data_copy['Timespan'])
        data_copy.drop_duplicates(subset=['Timestamp'], keep='first', inplace=True)
        data_copy.drop(columns=['Timestamp'], inplace=True)
        # missing value interpolate
        data_ = data_copy.drop(columns=['Date', 'Timespan']).astype(float)
        data_ = data_.interpolate()
        data_[['Date', 'Timespan']] = data_copy[['Date', 'Timespan']]
        return data_

    def split(self, data):
        train_index = int(len(data) * self.train_size)
        train_data = data[0:train_index]
        test_data = data[train_index:len(data)]
        return train_data, test_data

    ### 最后要删掉这个子函数 ###
    def all_data(self):
        # drop duplicates and missing value processing
        preprocess_price = self.preprocess(self.price_df)
        preprocess_volume = self.preprocess(self.volume_df)
        # drop some columns according to the number of features
        assert self.num_feature == 1 or self.num_feature == 3 or self.num_feature == 5
        if self.num_feature == 1:
            merged_data = preprocess_price.drop(columns=['Base', 'Peak'])
        if self.num_feature == 3:
            merged_data = preprocess_price
        if self.num_feature == 5:
            merged_data = self.merge(preprocess_price, preprocess_volume)
        transform_data = self.transform(merged_data)
        return transform_data

    def scaler(self, train_data, test_data):
        if self.scaler_type == 'MinMax':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'Standard':
            scaler = StandardScaler()

        if self.purpose == 'train':
            assert (train_data is not None) and (test_data is not None)
            scaler_train = scaler
            train_scaled = scaler_train.fit_transform(train_data)
            test_scaled = scaler_train.transform(test_data)
            return train_scaled, test_scaled, scaler_train
        elif self.purpose == 'predict':
            # assert (train_data is not None) and (test_data is None), 'For predict purpose, we only need one dataset.'
            # scaler_ = scaler
            # all_data = self.all_data()
            # data_scaled_ = scaler_.fit_transform(all_data)
            # data_scaled = scaler_.transform(train_data)

            scalerfile = 'scaler1_' + self.scaler_type + '_' + self.mode + '.sav'
            scaler_ = pickle.load(open(scalerfile, 'rb'))
            data_scaled = scaler_.transform(train_data)
            return data_scaled, scaler_

    def dataloader_train(self):
        # drop duplicates and missing value processing
        preprocess_price = self.preprocess(self.price_df)
        preprocess_volume = self.preprocess(self.volume_df)
        # drop some columns according to the number of features
        assert self.num_feature == 1 or self.num_feature == 3 or self.num_feature == 5
        if self.num_feature == 1:
            merged_data = preprocess_price.drop(columns=['Base', 'Peak'])
        if self.num_feature == 3:
            merged_data = preprocess_price
        if self.num_feature == 5:
            merged_data = self.merge(preprocess_price, preprocess_volume)

        transform_data = self.transform(merged_data)

        # train_test_split
        train, test = self.split(transform_data)
        # scale the data
        train_scaled, test_scaled, scaler_ = self.scaler(train, test)
        # build dataloader for train
        data_loader = Data.DataLoader(train_scaled, self.batch_size, shuffle=False)

        return data_loader, test_scaled, scaler_

    def dataloader_predict(self):
        # drop duplicates and missing value processing
        preprocess_price = self.preprocess(self.price_df)
        preprocess_volume = self.preprocess(self.volume_df)
        assert self.num_feature == 1 or self.num_feature == 3 or self.num_feature == 5
        if self.num_feature == 1:
            merged_data = preprocess_price.drop(columns=['Base', 'Peak'])
        if self.num_feature == 3:
            merged_data = preprocess_price
        if self.num_feature == 5:
            merged_data = self.merge(preprocess_price, preprocess_volume)

        if self.mode == 'hour':
            transform_data = self.transform(merged_data)
        elif self.mode == 'day':
             transform_data = self.transform(merged_data)
        elif self.mode == 'week':
            transform_data = self.transform(merged_data)
        # elif self.mode == 'all':
        #     transform_data_hour = self.transform(merged_data.drop(columns=['Date', 'Timespan']))
        #     transform_data_day = self.transform(merged_data)
        data_scaled, scaler_ = self.scaler(transform_data, None)
        return data_scaled, scaler_

    def dataloader(self):
        if self.purpose == 'train':
            data_loader, test_scaled, scaler_ = self.dataloader_train()
            # num_features = test_scaled.shape[1]
            return data_loader, torch.from_numpy(test_scaled).float(), scaler_  # , num_features
        if self.purpose == 'predict':
            data_scaled, scaler_ = self.dataloader_predict()
            return torch.from_numpy(data_scaled).float(), scaler_
