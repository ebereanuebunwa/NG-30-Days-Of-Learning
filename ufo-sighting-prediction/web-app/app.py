import numpy as np
from flask import Flask, request, render_template
import pickle as pkl


# create a Flask app
app = Flask(__name__)  

# Load the model
model = pkl.load(open('../ufo-model.pkl', 'rb'))



# Create a route for the default URL, which is http://localhost:5000
@app.route('/')
# Create a function that takes a single argument, x, and returns the value of the model for that value
def home():
    return render_template('index.html')


# Create a route for the URL http://localhost:5000/predict
@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]  # request.form.values() returns a list of strings
    final_features = [np.array(int_features)]  # convert the list of strings to a list of numbers
    prediction = model.predict(final_features)

    # return the prediction
    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        'index.html', prediction_text="Likely country: {}".format(countries[output])
    )


# Run the application
if __name__ == "__main__":
    app.run(debug=True)