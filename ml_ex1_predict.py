import joblib
import numpy as np


# Load model
model = joblib.load('mlp_model_e1.pkl') 

# Generate new data
x_new = np.array([
    0,  0,  0,  5,  9,  0,  0,  0,
    0,  0,  2, 14, 14, 22,  0,  0,
    0,  0,  6, 14, 11,  9,  0,  0,
    0,  3, 15,  3, 11,  5,  0,  0,
    0,  8, 11,  0, 13,  6,  2,  0,
    6, 10, 16, 16, 12, 15, 17,  0,
    0,  0,  4, 10, 15,  3,  0,  0,
    0,  0,  0, 11, 13,  2,  0,  0
  ]).astype('float64').reshape(1, -1)

# Predict new data
y_new_pred = model.predict(x_new)
print(f"Predicted digit = {y_new_pred[0]}")