import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Read the CSV data
df = pd.read_csv("IRIS.csv")

# Label encode the target variable (optional)
# If your model can handle integer labels, comment out this section
encoder = LabelEncoder()
y = encoder.fit_transform(df['Species'].astype(str))

# One-hot encode the labels (if necessary)
# Use only once to avoid creating a shape mismatch
dummy_y = to_categorical(y)  # One-hot encode for multi-class classification

# Separate features and target variables
x = df.iloc[:, 0:4].values
# Ensure the target variable has the correct shape (number of samples, number of classes)
y = dummy_y  # Use the one-hot encoded labels (or integer labels if not encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create the neural network model
model = Sequential()
model.add(Dense(12, input_shape=(4,), activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model (adjust epochs and batch size as needed)
model.fit(X_train, y_train, batch_size=150, epochs=500, validation_split=0.2)