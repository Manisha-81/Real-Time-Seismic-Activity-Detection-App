import json
from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle

model = pickle.load(open("MCCDAQ_NO-Disturbance_Disturbance_SVM_Model.sav", 'rb'))

app = Flask(__name__)
count = 0
prediction = "No-Disturbance"


@app.route('/model_predict')
def model_predict():
    return jsonify(result=prediction)
    # if  predictions:
    #     return jsonify(result="No-Disturbance")
    #
    # else:
    #     result = predictions.pop(0)



@app.route("/", methods=['POST'])
def get_feature():
    global prediction
    if request.json:
        f = json.loads(request.json)
        f = np.array(f)
        predicted_value = model.predict(f.reshape(1, 28))
        if predicted_value == [0]:
            prediction="No-Disturbance"
        else:
            prediction="Disturbance"
        print(prediction)
    return 'received'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('model_predict.html')


if __name__ == '__main__':
    app.debug = True
    app.run() 
