import numpy as np
import matplotlib.pyplot as plt

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape(len(X_t), 1)
Y_t = np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

#plt.plot(X_t, Y_t)
#plt.show()

class RNN:
    def __init__(self, X_t, n_neurons):
        self.T = max(X_t.shape)
        self.X_t = X_t
        self.Y_hat = np.zeros((self.T, 1))
        
        self.n_neurons = n_neurons

        self.Wx = 0.1*np.random.randn(n_neurons, 1) # Weights for inputs
        # This is a [n x n] matrix because [n x 1] * [n x n] = [n x 1] (so we can add it to X(t) * W(x))
        self.Wh = 0.1*np.random.randn(n_neurons, n_neurons) # Weights for previous state
        self.Wy =  0.1*np.random.randn(1, n_neurons) # Weights for the output 
        self.biases =  0.1*np.random.randn(n_neurons, 1) # biases each cell

        self.H = [np.zeros((n_neurons, 1)) for t in range(self.T + 1)] # states
    
    def forward(self, xt, ht_1):
        # out = W(x) * X(t) + W(h) * H(t - 1) + b
        out = np.dot(self.Wx, xt) + np.dot(self.Wh, ht_1) + self.biases
        # H(t) = tanh[W(x) * X(t) + W(h) * H(t - 1) + b]
        ht = np.tanh(out)
        # Y(t) = H(t) * W(y)
        y_hat_t = np.dot(self.Wy, ht)
        # Return H(t) for next cell, Y(t) for ... and out for ...
        return ht, y_hat_t, out
