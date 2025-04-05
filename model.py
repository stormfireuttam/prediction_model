import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2,4,6,8,10])

model = LinearRegression()
model.fit(x,y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as pkl file")