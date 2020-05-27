import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    feature_name = ['Credit_Amount','Repayment_Status_Jan','Repayment_Status_March',
                    'Repayment_Status_May','Repayment_Status_June','Jan_Bill_Amount',
                    'Feb_Bill_Amount','Previous_Payment_Jan','Previous_Payment_Feb',
                    'Previous_Payment_March','Previous_Payment_April','Previous_Payment_June',
                    'Gender_Male','Academic_Qualification_Others',
                    'Academic_Qualification_Professional','Marital_Single','My_Intercept']
    
    df = pd.DataFrame(final_features, columns = feature_name)
    prediction = model.predict(df)

    output = prediction
    
    if output == 0:
        status = 'Customer wil not default, it is a good loan'
    else:
        status ='Customer will default, it is a bad loan'

    return render_template('index.html', prediction_text='Staus of Application: {}'.format(status))


if __name__ == "__main__":
    app.run(debug=True)