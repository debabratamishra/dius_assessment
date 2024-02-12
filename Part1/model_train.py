import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn import metrics

np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
tf.config.experimental.enable_op_determinism()

# Load and preprocess data
train_df = pd.read_csv("train.csv")
#Encoding categorical feature in train set
train_df.X10=train_df.X10.astype('category').cat.codes

train_features = train_df.drop("Y", axis=1).values
train_label = train_df["Y"].values

#Splitting the train dataset in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(train_features, train_label, test_size=0.2, random_state=42)

#Using Standard scaler to normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Defined the checkpoint filepath
checkpoint_filepath = 'model_checkpoint.ckpt'

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                 save_weights_only=True,
                                                 verbose=1)
def model_init():
    #Initializing the network
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    # Adding Batch Normalization for normalizing the inputs of each layer
    model.add(BatchNormalization())
    model.add(Dense(units=70, activation='relu'))
    # Adding Batch Normalization for normalizing the inputs of each layer
    model.add(BatchNormalization())
    model.add(Dense(units=70, activation='relu'))
    # Adding Batch Normalization for normalizing the inputs of each layer
    model.add(BatchNormalization())
    model.add(Dense(units=70, activation='relu'))
    # Adding Batch Normalization for normalizing the inputs of each layer
    model.add(BatchNormalization())
    model.add(Dense(units=1))

    #Defining the optimizer
    opt = SGD(learning_rate=0.0005, momentum=0.45, nesterov=True)
    model.compile(optimizer=opt, loss='mean_absolute_error')
    model.summary()
    
    return model

train_model = model_init()

print("\nBeginning model training\n")

# Train the model
history = train_model.fit(x=X_train_reshaped, y=y_train, epochs=400, batch_size=64, 
                    validation_data=(X_test_reshaped, y_test), 
                    callbacks=[checkpoint, EarlyStopping(monitor='val_loss', patience=10)])

print("Train MAE: ",train_model.evaluate(X_train_reshaped, y_train))
print("Validation MAE: ",train_model.evaluate(X_test_reshaped, y_test))

print("\nTraining complete\n")

#Initializing model for prediction
pred_model = model_init()
try:
    #Loading model weights from checkpoint stored in disk
    pred_model.load_weights(checkpoint_filepath)
    print("Model weights loaded in memory successfully.")
except Exception as e:
    print("Model checkpoint not found. Please train the model again.")
    print(e)
    exit(0)

#Reading unseen data from Disk
test_df = pd.read_csv("test.csv")
#Encoding categorical feature in test dataset
test_df.X10=test_df.X10.astype('category').cat.codes

actual = test_df.drop("Y", axis=1).values
actual_scaled = scaler.transform(actual)

# Reshape the input data for LSTM (samples, time steps, features)
actual_reshaped = np.reshape(actual_scaled, (actual_scaled.shape[0], 1, actual_scaled.shape[1]))

# Extract actual target values
y_actual = test_df["Y"].values

print("Beginning prediction")
predictions = pred_model.predict(actual_reshaped)

# Create a DataFrame for predictions
predictions_df = pd.DataFrame(predictions, columns=["Predictions"])

# Calculate evaluation metrics
mae = metrics.mean_absolute_error(y_actual, predictions)
mse = metrics.mean_squared_error(y_actual, predictions)
rmse = np.sqrt(mse)

print("MAE on test data: ", mae)
print("MSE on test data: ", mse)
print("RMSE on test data:", rmse)

#Reading test dataset from disk again to avoid any earlier changes to original data
test_df_2 = pd.read_csv("test.csv")
test_prediction_concat = pd.concat([test_df_2, predictions_df], axis=1)
test_prediction_concat.to_csv("test_pred.csv", index=False)