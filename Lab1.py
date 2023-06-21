import numpy as np
x = np.array([1, 3.5, 6])
y = np.array([4, 5.5, 9 ])
lr = 0.05
epochs = 200
w = 0
b = 0
def compute_loss(y_pred, y_true):
  return np.mean((y_pred - y_true) ** 2)
loss_history = []
w_history = []
b_history = []
for i in range(epochs):
  y_pred = w * x + b
  dw = np.mean((y_pred - y) * x)
  db = np.mean(y_pred - y)
  w = w - lr * dw
  b = b - lr * db
  loss = compute_loss(y_pred, y)
  loss_history.append(loss)
  w_history.append(w)
  b_history.append(b)
print(f"Final weights: {w}")
print(f"Final bias: {b}")
print(f"Final loss: {loss_history[-1]}")


# Momentum Based Gradient Descent
import numpy as np
X = np.array([1, 3.5, 6])
Y = np.array([4, 5.5, 9])
lr = 0.05
beta = 0.9
epochs = 200
w = 0
b = 0
vw = 0
vb = 0
def compute_loss(y_pred, y_true):
  return np.mean((y_pred - y_true) ** 2)
loss_history = []
w_history = []
b_history = []
for i in range(epochs):
  y_pred = w * X + b
  dw = np.mean((y_pred - Y) * X)
  db = np.mean(y_pred - Y)
  vw = beta * vw + (1 - beta) * dw
  vb = beta * vb + (1 - beta) * db
  w = w - lr * vw
  b = b - lr * vb
  loss = compute_loss(y_pred, Y)
  loss_history.append(loss)
  w_history.append(w)
  b_history.append(b)
print(f"Final weights: {w}")
print(f"Final bias: {b}")
print(f"Final loss: {loss_history[-1]}")