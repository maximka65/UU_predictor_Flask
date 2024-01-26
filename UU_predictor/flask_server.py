import numpy as np
from flask import Flask, request, jsonify
import dill
from utils import categories
from decimal import Decimal

# Загружаем обученные модели
with open('forest_trained.dill', 'rb') as f:
    forest = dill.load(f)

with open("visits_only.dill", "rb") as f:
    visits_only = dill.load(f)

with open("alexa_visits.dill", "rb") as f:
    alexa_visits = dill.load(f)

app = Flask(__name__)

@app.route('/predictor')
def hello():
    return "<p>Hello, it's Metrica UU_predictor_src</p>"

@app.route('/predictor/predict', methods=['GET', 'POST'])
def predict():
        string = []
        data = request.get_json()
        print(data)
        if data['Alexa']:
            string.append(data['Alexa'])
        else:
            string.append(-1)

        if data['Visits']:
            string.append(data['Visits'])
        else:
            string.append(-1)

        if data['PagesPerVisit']:
            string.append(data['PagesPerVisit'])
        else:
            string.append(-1)

        if data['BounceRate']:
            string.append(data['BounceRate'])
        else:
            string.append(-1)

        if data['AvgVisitDuration']:
            string.append(data['AvgVisitDuration'])
        else:
            string.append(-1)

        if data['Category']:
            try:
                cat1 = data['Category'][:data['Category'].index('/')]
                cat2 = data['Category'][data['Category'].index('/')+1:]
            except ValueError:
                cat1 = data['Category']
                cat2 = data['Category']
            if (cat1 in categories) and (cat2 in categories):
                string.append(data['Category'])
            else:
                print('Unknown category:', data['Category'])
                string.append('Unknown')
        else:
            string.append('Unknown')

        print(string)
        dct = {}
        if string[0] == string[1] == string[2] == string[3] == string[4]:
            dct['min'] = None
            dct['max'] = None
            return jsonify(dct)
        elif (string[1] == -1) and (string[0] != -1):
            stat = int((26.4 / ((string[0] / 1000000) ** 0.96)) * 1000)
            dct['min'] = int(stat * 0.75)
            dct['max'] = int(stat * 1.25)
        elif string[2] == -1:
            if string[0] != -1:
                dct['min'] = int(alexa_visits.predict(string[:2]) * 0.75)
                dct['max'] = int(alexa_visits.predict(string[:2]) * 1.25)
            else:
                dct['min'] = int(visits_only.predict(string[1]) * 0.75)
                dct['max'] = int(visits_only.predict(string[1]) * 1.25)
        else:
            dct['min'] = int(forest.predict(string) * 0.9)
            dct['max'] = int(forest.predict(string) * 1.1)

        return jsonify(dct)


if __name__ == '__main__':
    app.run(threaded=False, processes=2, host='0.0.0.0')
