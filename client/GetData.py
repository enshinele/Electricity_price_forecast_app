import requests
import pandas as pd
import time



class GetData:
    def __init__(self, fromDate, toDate=None):
        self.fromDate = fromDate
        self.toDate = time.strftime("%Y-%m-%d", time.localtime()) if toDate is None else toDate
        try:
            respond = requests.get('https://coop.eikon.tum.de/mbt/mbt.json')
            print('Success access the page')
            self.token = respond
            self.headers = {"Authorization": 'bearer  ' + self.token.json()['access_token']}
        except:
            print('This token is not automative obtained by requsts and need to be updated by hand.')
            self.token = {
                "access_token": "LeN547JO6x01rH-jVWvAqLw4m9QSNOGYaScbARhcZe6Pqc3XW9e539yLXnZ6rP4laWxFNT96S6TyVR52FBqyNRHYscqpLcTgJ-dYv1UOuA1CaqpVayUQOvAxsYCs5uqJst0CNTeoC4dTlL-JyaUe4whs7OO4GBWuWLK55KHpGXTzIqLDB5epIjwmZfWzZEjTYlWcxm4XjiEj46EXN0aRNtaFjDr5td5FOa_daR2rb9XJqge1A5u0-k-HJG4DkJgwB1-Bgr2HeqD-WTTYA63-Mmho-RCbHCfTi-SXuxaj_CFJAcvu36R3Z-3Yx9GsvmrW_GUXLU6k-HnMarjLVbQHpmaTKWxJ1ZKH3lD6rRoPgl9nXQYE4mk9opKkm4NRRGltLjjGvHE17WOZlO0Of-EhNi8P9DLquwViWsptAV4IAVg",
                "token_type": "bearer", "expires_in": 2678399}
            self.headers = {"Authorization": 'bearer  ' + self.token['access_token']}
            
        # except:
        #     print('This token is not automative obtained by requsts and need to be updated by hand.')
        #     self.token = {"access_token": "QWSSh0GC0QHH54IOMKJtRaBd8qTsO3MR6BIQe0vZOoXqjojVBojNnEqjPVYh1zSDrQc2lmsyH4GsGlp3H3MsIT9edmI00SNg3tfkTUWCe8AY-mBEv0QtSAGeCBiYxE_yDJBPx5fRNCvnwtKc_Sr0eamKjyNkAk8-r0ULdxOVO7iF8VhiUKZoUhfEfG6E6lWj_a2b7eRS52Qeko-8RECezE8FEaO8xyK5-4WsvBoAiLVYQGqQuj3xh14250K8SrEC-nOtX6bHocX8gKuoMFNKCCZkpwfG77i4f7tTgV-sHUNPMNO5u_JvbyDko3kjwH14LuDG0zedtXVXtjdl5FH1jJCyYKuZIGliDZy1UElS-QB_Y8UnJSHReuK6zw-hEoE9zbPA7w6K5KdG9xo9Kbj3EJNFDcQU97yVYFdjaVKA1zg", "token_type": "bearer", "expires_in": 2678399}

        self.url_price = 'http://api.montelnews.com/spot/getprices'
        self.url_volume = 'http://api.montelnews.com/spot/getvolumes'
        self.params_price = {
            'spotKey': '14',
            'fields': ['Base', 'Peak', 'Hours'],
            'fromDate': self.fromDate,
            'toDate': self.toDate,
            'currency': 'eur',
            'sortType': 'Ascending',
            'Country ': 'Germany'
        }
        self.params_volume = {
            'spotKey': '14',
            'fields': ['Base', 'Hours'],
            'fromDate': self.fromDate,
            'toDate': self.toDate,
            # 'currency': 'eur',
            'sortType': 'Ascending',
            'Country ': 'Germany'
        }

    def drop_duplicate(self, data):
        data['Timestamp'] = data['Date'].str.cat(data['Timespan'])
        data_without_duplicates = data.drop_duplicates(subset=['Timestamp'], keep='first')
        return data_without_duplicates.drop(columns=['Timestamp'])

    def get_current_time(self):
        current_hour = time.strftime("%H", time.localtime())
        current_date = time.strftime("%Y-%m-%d", time.localtime()) + 'T00:00:00.0000000'
        current_timespan = current_hour + ':00' + '-' + str(int(current_hour) + 1) + ':00'
        return current_date, current_timespan

    def getdata_json_price(self):
        'get price data in json format from the Montel API'
        response = requests.get(self.url_price, headers=self.headers, params=self.params_price)
        data_json_price = response.json()
        return data_json_price

    def getdata_price(self):
        'retrive the useful information from the json data and transform it to dataframe'
        value = []
        Timespan = []
        date = []
        base = []
        peak = []
        data_json = self.getdata_json_price()
        for parts in data_json['Elements']:
            date.append(parts['Date'])
            base.append(parts['Base'])
            peak.append(parts['Peak'])
            for df in parts['TimeSpans']:
                value.append(df['Value'])
                Timespan.append(df['TimeSpan'])

        def repeatlist(list_before, i):
            list_after = [val for val in list_before for k in range(i)]
            return list_after

        date = repeatlist(date, 24)
        base = repeatlist(base, 24)
        peak = repeatlist(peak, 24)
        price_df = pd.DataFrame(list(zip(date, Timespan, value, base, peak)),
                                columns=['Date', 'Timespan', 'Value', 'Base', 'Peak'])
        return self.drop_duplicate(price_df)

    def getdata_json_volume(self):
        'get volume data in json format from the Montel API'
        response = requests.get(self.url_volume, headers=self.headers, params=self.params_volume)
        data_json_volume = response.json()
        return data_json_volume

    def getdata_volume(self):
        'retrive the useful information from the json data and transform it to dataframe'
        value = []
        Timespan = []
        date = []
        base = []
        # peak = []
        data_json = self.getdata_json_volume()
        for parts in data_json['Elements']:
            date.append(parts['Date'])
            base.append(parts['Base'])
            # peak.append(parts['Peak'])
            for df in parts['TimeSpans']:
                value.append(df['Value'])
                Timespan.append(df['TimeSpan'])

        def repeatlist(list_before, i):
            list_after = [val for val in list_before for k in range(i)]
            return list_after

        date = repeatlist(date, 24)
        base = repeatlist(base, 24)
        # peak = repeatlist(peak,24)
        volume_df = pd.DataFrame(list(zip(date, Timespan, value, base)),
                                 columns=['Date', 'Timespan', 'Value', 'Base'])
        return self.drop_duplicate(volume_df)
