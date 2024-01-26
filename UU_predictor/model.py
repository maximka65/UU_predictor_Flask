import pandas as pd
import numpy as np
import utils
from utils import categories
# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


# data = pd.read_json('../find_query_big.json')


class MetricaModel():
    def __init__(self):
        pass

    def _preprocess(self, data):
        data['Visits'] = utils.visits(data)
        data['UniqueUsers'] = utils.unique_users(data)
        data['PagesPerVisit'] = utils.pages_per_visit(data)
        data['BounceRate'] = utils.bounce_rate(data)
        data['AvgVisitDuration'] = utils.avg_visit_duration(data)
        data['Alexa'] = utils.alexa_pop(data)
        data['Summary'], data['Summary_2'] = utils.category(data)

        aux_df = pd.DataFrame(np.zeros((len(categories), data.shape[0])).T)
        aux_df.columns = categories
        data = data.join(aux_df)

        for i in range(data.shape[0]):
            data[data['Summary'].iloc[i]].iloc[i] = 1
            data[data['Summary_2'].iloc[i]].iloc[i] = 1

        data = data[['UniqueUsers', 'Alexa', 'Visits', 'PagesPerVisit', 'BounceRate',
                     'AvgVisitDuration'] + categories]

        data = data.loc[~data['Alexa'].isnull()]
        data = data.loc[~(data['Alexa'] == -1)]
        data = data.loc[data['UniqueUsers'] != -1]
        data = data.loc[data['Visits'] >= 0.97*data['UniqueUsers']]
        return data


class MetricaForest(MetricaModel):
    def __init__(self):
        self.reg = RandomForestRegressor()
        self.reg_no_alexa = RandomForestRegressor()

    def fit(self, data, data2=None):
        data = self._preprocess(data)
        if data2 is not None:
            data2 = self._preprocess(data2)
            data = data.append(data2)
        X = data.drop(columns=['UniqueUsers'])
        y = data['UniqueUsers']
        X_no_alexa = X.copy()
        X_no_alexa.Alexa = -1
        self.reg.fit(X, y)
        self.reg_no_alexa.fit(X_no_alexa, y)

    def predict(self, data: list):
        string = utils.category_encoder(data)
        if string[0] != -1:
            prediction = self.reg.predict(np.array(string).reshape(-1, 1).T)
        else:
            prediction = self.reg_no_alexa.predict(
                np.array(string).reshape(-1, 1).T)
        return float(prediction)


class MetricaVisitsOnly():
    def __init__(self):
        self.reg = LGBMRegressor(random_state=19)

    def fit(self, data):
        data['Visits'] = utils.visits(data)
        data['UniqueUsers'] = utils.unique_users(data)
        data = data.drop(columns='engagement')
        data = data.loc[data['Visits'] != 0]
        data = data.loc[data['UniqueUsers'] != -1]
        data = data.loc[data['Visits'] >= 0.95 * data['UniqueUsers']]
        X = pd.DataFrame(data.Visits)
        y = data.UniqueUsers
        self.reg.fit(X, y)

    def predict(self, string: list):
        prediction = self.reg.predict(np.array(string).reshape(-1, 1).T)

        return float(prediction)


class MetricaAlexaVisits():
    def __init__(self):
        self.reg = RandomForestRegressor(max_depth=10)

    def fit(self, data):
        data['Visits'] = utils.visits(data)
        data['Alexa'] = utils.alexa_pop(data)
        data['UniqueUsers'] = utils.unique_users(data)
        data = data[['Alexa', 'Visits', 'UniqueUsers']]
        data = data.loc[data['Visits'] != 0]
        data = data.loc[data['UniqueUsers'] != -1]
        data = data.loc[data['Visits'] >= 0.95 * data['UniqueUsers']]
        X = data.drop(columns='UniqueUsers')
        y = data['UniqueUsers']
        self.reg.fit(X, y)

    def predict(self, string: list):
        prediction = self.reg.predict(np.array(string[0:2]).reshape(-1, 1).T)

        return float(prediction)