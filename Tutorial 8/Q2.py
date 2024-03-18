import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("government-expenditure-on-education.csv")
data['Normalized_Year'] = data['year'] / 2018  # Normalize year by dividing by largest year
data['Normalized_Expenditure'] = data['total_expenditure_on_education'] / data['total_expenditure_on_education'].max()  # Normalize expenditure
x = np.vstack((np.ones(len(data)), data['Normalized_Year'])).T  # Add bias term to x
y = data['Normalized_Expenditure'].values

def exponential_model(w, x):
    return np.exp(-np.dot(x, w))

def squared_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, learning_rate=0.01, epochs=2000000):
    w = np.zeros(x.shape[1])  # Initial guess for w
    losses = []  # List to store loss values for visualization
    
    for epoch in range(epochs):
        y_pred = exponential_model(w, x)
        loss = squared_loss(y, y_pred)
        losses.append(loss)
        
        gradient = -2 * np.dot(x.T, (y - y_pred) * y_pred) # Gradient of the loss function
        
        w -= learning_rate * gradient  # Update w using gradient descent
        
        if epoch % 10000 == 0:
            print(f'Epoch {epoch}: Loss = {loss}')
    
    return w, losses

# Step 4: Fit the exponential model using gradient descent
learning_rate = 0.03
epochs = 2000000
y_log = np.log(y + 1) 
estimated_w, losses = gradient_descent(x, y_log, learning_rate, epochs)

plt.plot(range(epochs), losses)
plt.xlabel('Iterations')
plt.ylabel('Cost Function C(w)')
plt.title('Cost Function vs Iterations')
plt.show()

print(f'Estimated w: {estimated_w}')