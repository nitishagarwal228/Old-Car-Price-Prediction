import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('car.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods= ['POST'])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        print(int_features)
        prediction = model.predict(final_features)
        output = round(prediction[0],2)
        return render_template('index.html',prediction_text="Price of Car : {} lakhs".format(output))
    except:
        return render_template('index.html',prediction_text="Enter correct inputs!!!Please Note that all inputs should be numbers")
    
    

if __name__ == '__main__':
    app.run(debug=True)
