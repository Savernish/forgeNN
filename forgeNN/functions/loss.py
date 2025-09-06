# Library to implement loss functions like MSE, CrossEntropy, etc.

class MSE:
    @staticmethod
    def forward(y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()

    @staticmethod
    def backward(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

class CrossEntropy:
    @staticmethod
    def forward(y_pred, y_true):
        import numpy as np
        m = y_true.shape[0]
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        log_likelihood = -np.log(p[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    @staticmethod
    def backward(y_pred, y_true):
        import numpy as np
        m = y_true.shape[0]
        grad = y_pred.copy()
        grad[range(m), y_true] -= 1
        grad = grad / m
        return grad
    
