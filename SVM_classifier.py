import numpy as np

class SVM_:
    def __init__(self, lr, epochs, lambda_par):
        self.lr = lr
        self.epochs = epochs
        self.lambda_par = lambda_par

    def fit(self, X, y):
        # m is the number of data points and n is the number of features
        self.m, self.n = X.shape

        # initialization of weights and bias
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.y = y

        for _ in range(self.epochs):  # Use _ for loop variable if it's not used
            self.updates_weights()

    def updates_weights(self):
        # label encoding
        y_label = np.where(self.y <= 0, -1, 1)

        # gradients dw, db
        dw = np.zeros(self.n)  # Initialize gradients
        db = 0

        for index, x_i in enumerate(self.X):
            condition = y_label[index] * np.dot(x_i, self.w) - self.b >= 1

            if condition:
                dw += 2 * self.lambda_par * self.w
                db += 0
            else:
                dw += 2 * self.lambda_par * self.w - np.dot(x_i, y_label[index])
                db += y_label[index]

        # Update weights and bias
        self.w -= self.lr * dw / self.m
        self.b -= self.lr * db / self.m

    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        predict_labels = np.sign(output)

        y_hats = np.where(predict_labels <= -1, 0, 1)
        return y_hats
