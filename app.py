import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model4.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    # if1-fraudlent
    # if0-not fraudlent
    # else invalid

    if str(output) == '1':
        return render_template('index.html', prediction_text='This is a fradulent transaction'.format(output))
    elif str(output) == '0':
        return render_template('index.html', prediction_text='This is a non fradulent transaction '.format(output))
    else:
        return render_template('index.html', prediction_text='This is invalid '.format(output))


if __name__ == "__main__":
    app.run(debug=True)