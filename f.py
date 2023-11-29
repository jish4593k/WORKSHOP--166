import tkinter as tk
from tkinter import Label, Entry, Button, messagebox
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model using Keras
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=0)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Create a Tkinter GUI
class RegressionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Linear Regression App')

        self.create_widgets()

    def create_widgets(self):
        Label(self.master, text='Enter a value for X:').pack(pady=10)
        self.entry_x = Entry(self.master)
        self.entry_x.pack(pady=10)

        Button(self.master, text='Predict', command=self.predict).pack(pady=10)

    def predict(self):
        try:
            x_value = float(self.entry_x.get())
            x_input = np.array([[x_value]])
            y_pred = model.predict(x_input)
            messagebox.showinfo('Prediction Result', f'The predicted y-value is: {y_pred[0, 0]:.2f}')
        except ValueError:
            messagebox.showerror('Error', 'Invalid input. Please enter a valid numeric value.')

# Create the Tkinter application
root = tk.Tk()
app = RegressionApp(root)
root.mainloop()
