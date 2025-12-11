import numpy as np
import matplotlib.pyplot as plt


class LinearSVM:
    """SVM lineal simple usando descenso por subgradiente sobre hinge loss.

    Minimiza: 0.5 * lambda_param * ||w||^2 + (1/n) * sum(max(0, 1 - y*(w·x + b)))
    """
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # convertir etiquetas a -1 y 1
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b)
                if condition < 1:
                    # subgradiente cuando hinge loss > 0
                    self.w += self.lr * (y_[idx] * x_i - self.lambda_param * self.w)
                    self.b += self.lr * y_[idx]
                else:
                    # solo regularización
                    self.w += self.lr * (- self.lambda_param * self.w)

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))


def example():
    # datos sintéticos 2D
    np.random.seed(0)
    X1 = np.random.randn(50, 2) + [2, 2]
    X2 = np.random.randn(50, 2) + [-2, -2]
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(50), -np.ones(50)])

    # mezclar
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    # entrenar
    svm = LinearSVM(learning_rate=0.01, lambda_param=0.01, n_iters=500)
    svm.fit(X, y)

    # evaluar
    preds = svm.predict(X)
    acc = np.mean(preds == y)
    print(f"Accuracy (entrenamiento): {acc:.4f}")

    # graficar frontera
    plt.figure(figsize=(6, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Clase 1')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Clase -1')
    plt.title('SVM - frontera de decisión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('svm.png', dpi=100)
    plt.show()


if __name__ == '__main__':
    example()
