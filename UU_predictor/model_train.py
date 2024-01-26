from model import MetricaForest, MetricaVisitsOnly, MetricaAlexaVisits
import pandas as pd
import dill

data = pd.read_json('../data/find_query_big.json')
data_small = pd.read_json('../data/find_query (2).json')
model = MetricaForest()
model.fit(data=data, data2=data_small)

with open("forest_trained.dill", "wb") as f:
    dill.dump(model, f)

data2 = pd.read_json('../data/find_query_without_eng.json')
model2 = MetricaVisitsOnly()
model2.fit(data2)

with open("visits_only.dill", "wb") as f:
    dill.dump(model2, f)

data3 = pd.read_json('../data/find_query (6).json')
model3 = MetricaAlexaVisits()
model3.fit(data3)

with open("alexa_visits.dill", "wb") as f:
    dill.dump(model3, f)
