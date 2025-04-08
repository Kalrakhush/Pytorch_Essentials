import numpy as np

#f =w*x
#f =2*x

X=np.array([1, 2, 3,4], dtype=np.float32)
Y=np.array([2, 4, 6,8], dtype=np.float32)


w=0.0

#model prediction
def forward(x):
    return w*x

#loss function =MSE
#MSE=1/n * sum(y_pred - y)^2
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean() # mean squared error

#gradient
#MSE= 1/n * sum(y_pred(w*x) - y)^2
#dMSE/dw = 1/n * sum(2*x*(y_pred - y))
def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean() #derivative of loss function with respect to w


print("Prediction before training: ", forward(5)) #expected 10

#training
learning_rate=0.01
n_iters=20

for epoch in range(n_iters):
    #prediction
    y_pred=forward(X)

    #loss
    l=loss(Y, y_pred)

    #gradient
    dw=gradient(X, Y, y_pred)

    #update weights
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}: w={w:.3f}, loss={l:.8f}")
print("Prediction after training: ", forward(5)) #expected 10
