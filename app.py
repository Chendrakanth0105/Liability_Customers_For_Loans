from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

knn = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('Age'))
    income = int(request.form.get('Income'))
    family = int(request.form.get('Family'))
    ccavg = float(request.form.get('CCAvg'))
    education = int(request.form.get('Education'))
    mortgage = int(request.form.get('Mortgage'))
    securities_account = int(request.form.get('Securities Account'))
    cd_account = int(request.form.get('CD Account'))
    online = int(request.form.get('Online'))
    credit_card = int(request.form.get('Credit Card'))

    result = knn.predict([[age, income, family, ccavg, education, mortgage, securities_account, cd_account, online, credit_card]])


    if result == 1:
         return render_template('index.html',label = 1)
    else:
         return render_template('index.html',label = -1)

if __name__ == '__main__':
    app.run(debug=True)
