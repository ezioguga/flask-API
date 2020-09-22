from flask import Flask, jsonify, request
import joblib;
import pandas as pd
import numpy as np
from sklearn import *
from flask_cors import CORS
# from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
     # vectorizer = TfidfVectorizer()
     json_ = request.json["test"]
     print("JSON String --->",json_[0])
     # X = vectorizer.fit_transform(json_)
     # print(X)
     # query_df = pd.DataFrame(json_)
     # query = pd.get_dummies(query_df)

     int_features = [float(x) for x in json_]
     final_features = [np.array(int_features)]

     # # reshaped = json_.reshape(1, -1)
     prediction = clf.predict(final_features)
     print("[prediction -->",prediction)
     output = round(prediction[0],2)
     return jsonify({'prediction': int(output) })
if __name__ == '__main__':
     clf = joblib.load('finalized_ckda_model.pkl')
     app.run(port=8080)