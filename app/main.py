from flask import Flask, request, jsonify
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)
MODEL = "{}/dataset/svm.pkl".format(dir_path)
MODEL_COLUMNS = ['Age','EstimatedSalary']

@app.route('/predict', methods=['POST'])
def index():
    data = request.json
    query = pd.get_dummies(pd.DataFrame(data))
    query = query.reindex(columns=MODEL_COLUMNS, fill_value=0)

    sc = StandardScaler()
    query = sc.fit_transform(query)
    clf = joblib.load(MODEL)
    response = list(map(lambda i: bool(i), clf.predict(query)))
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
