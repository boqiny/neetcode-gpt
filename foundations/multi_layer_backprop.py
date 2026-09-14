import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        y_hat = W2 @ a1 + b2
        loss = np.mean((y_hat - y_true)**2)
        dy_hat = 2 * (y_hat - y_true) / y_hat.size
        d_w2 = np.outer(dy_hat, a1)
        d_b2 = dy_hat

        d_a1 = dy_hat @ W2
        d_z1 = d_a1 * (z1 > 0)

        d_w1 = np.outer(d_z1, x)
        d_b1 = d_z1
        return {
            'loss': float(np.round(loss, 4)),
            'dW1': np.round(d_w1, 4).tolist(),
            'db1': np.round(d_b1, 4).tolist(),
            'dW2': np.round(d_w2, 4).tolist(),
            'db2': np.round(d_b2, 4).tolist(),
        }
                
