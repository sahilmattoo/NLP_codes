


"""# Import Libraries"""

import os
import pandas as pd
import pickle

import warnings
warnings.filterwarnings('ignore')

"""# Set Working Directory"""

# setFilepath ="/content/drive/MyDrive/[00] BITS/MLoPsAssignment"
# os.chdir(setFilepath)

"""# Load Model"""

# Load the model from the file
with open('./best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

"""# InferenceFunction"""

def predict_mobile_price_range(record, loaded_model):
  # Convert the record to a DataFrame (this is necessary to match the format the model expects)
  record_df = pd.DataFrame([record])
  # Run inference on the new record
  try:
    prediction = loaded_model.predict(record_df)
    return prediction[0]

  except:

    return "Error in prediction"

"""# Test the Inference with Example"""

# Define the new record for inference
record = {
    'battery_power': 1440,
    'blue': 1,
    'clock_speed': 1.5,
    'dual_sim': 0,
    'fc': 16,
    'four_g': 1,
    'int_memory': 35,
    'm_dep': 0.5,
    'mobile_wt': 150,
    'n_cores': 8,
    'pc': 20,
    'px_height': 720,
    'px_width': 1500,
    'ram': 4000,
    'sc_h': 14,
    'sc_w': 7,
    'talk_time': 18,
    'three_g': 1,
    'touch_screen': 0,
    'wifi': 1
}

response = predict_mobile_price_range(record, loaded_model)
print(response)

