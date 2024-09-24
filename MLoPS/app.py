from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

results = {'0': 'less expensive',
           '1': 'economical',
           '2': 'moderate expensive',
           '3': 'very expensive'}

# Load the model from the local file system
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Inference function
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data as JSON
    record_df = pd.DataFrame([data])  # Convert input to DataFrame
    prediction = loaded_model.predict(record_df)  # Run the model
    res = str(prediction[0])
    # print(prediction[0])
    # print(type(prediction))
    
    # print(type(res))
    # print(results[res])
    # print(type(results[res]))
    #res = results[prediction[0]]
    return jsonify({'prediction': results[res]})  # Return the result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run on port 5000
