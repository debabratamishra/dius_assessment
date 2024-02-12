import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import SGD
from flask import Flask, request
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the trained model
model_checkpoint_filepath = 'model_checkpoint.ckpt'

def model_init():
    #Initializing the network
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(1, 10))) #10 feats
    model.add(BatchNormalization())
    model.add(Dense(units=70, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=70, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=70, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=1))

    #Defining the optimizer
    opt = SGD(learning_rate=0.0005, momentum=0.45, nesterov=True)
    model.compile(optimizer=opt, loss='mean_absolute_error')
    return model


# Preprocess the input data
def preprocess_data(data):
    
    train_df = pd.read_csv("train.csv")
    train_df.X10=train_df.X10.astype('category').cat.codes
    X = train_df.drop("Y", axis=1).values
    y = train_df["Y"].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    # Initialize standard scaler instance for normalizing sequences received from user
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    
    # Transforming test dictionary received from user
    encoding_dict = {
        0: "ACV",
        1: "AED",
        2: "DAF",
        3: "DEB",
        4: "OIF",
        5: "QWE"
    }

    # Get the value from dictionary_a["X10"]
    value_X10 = data["X10"]

    # Find the corresponding encoding from encoding_dict
    encoded_value = None
    for key, value in encoding_dict.items():
        if value == value_X10:
            encoded_value = key
            break

    # Update the value in dictionary_a["X10"] with its encoded value
    if encoded_value is not None:
        data["X10"] = encoded_value
    
    # Extract values and convert them to float
    values = [float(value) for value in data.values()]

    # Convert to numpy array
    values_np_arr = np.array(values)
    test_predictors_scaled = scaler.transform(values_np_arr.reshape(1, -1))
    test_predictors_reshaped = np.reshape(test_predictors_scaled, (1, 1, 10))

    return test_predictors_reshaped

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    try:
        model = model_init()
        model.load_weights(model_checkpoint_filepath)
        print("Model loaded successfully.")
    except Exception as e:
        print(e)
        print("Checkpointed model not found")
        return {"prediction" : "Checkpointed model not found"}
    
    try:
        # Preprocess the input data
        transformed_data = preprocess_data(data)

        # Make predictions using the loaded model
        predictions = model.predict(transformed_data)
        pred_val = float(predictions[0][0])
        return {"prediction" : pred_val}
    except Exception as e:
        print(e)
        return {"prediction" : "Invalid data passed"}
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8178)
    # curl --json '{"X1": "14.59355382431799","X2": "41.18622465124143","X3": "33.9978936605386","X4": "64.4769023500858","X5": "108.81641054516794","X6" : "79.18062928342908", "X7" : "70.75284656795137", "X8" : "109.25455637899452", "X9" : "123.59616528814512", "X10" : "QWE"}' http://0.0.0.0:8178/predict