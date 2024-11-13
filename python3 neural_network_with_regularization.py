import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5, l2_lambda=0.01):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        # Dropout rate and L2 regularization lambda
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X, training=True):
        # Forward pass with dropout on the hidden layer
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        if training:
            self.dropout_mask = (np.random.rand(*self.a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.a1 *= self.dropout_mask  # Apply dropout mask
        else:
            self.dropout_mask = 1  # No dropout in testing
        
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihood) / m

        # Add L2 regularization term
        l2_loss = (self.l2_lambda / 2) * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return loss + l2_loss

    def backward(self, X, y_true, learning_rate):
        # Number of samples
        m = X.shape[0]

        # Gradient of the output layer
        y_true_one_hot = np.zeros_like(self.a2)
        y_true_one_hot[np.arange(m), y_true] = 1
        dz2 = self.a2 - y_true_one_hot

        # Gradient for W2 and b2
        dW2 = (self.a1.T).dot(dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Add L2 regularization to W2 gradient
        dW2 += self.l2_lambda * self.W2

        # Backpropagate to hidden layer
        da1 = dz2.dot(self.W2.T)
        da1 *= self.relu_derivative(self.z1)
        
        # Apply dropout during backpropagation
        da1 *= self.dropout_mask

        # Gradient for W1 and b1
        dW1 = (X.T).dot(da1) / m
        db1 = np.sum(da1, axis=0, keepdims=True) / m

        # Add L2 regularization to W1 gradient
        dW1 += self.l2_lambda * self.W1

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X, training=True)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        # Forward pass without dropout
        y_pred = self.forward(X, training=False)
        return np.argmax(y_pred, axis=1)


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.01
    epochs = 1000

    # Sample Data (e.g., random dataset)
    X_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 3, 100)  # 100 labels, 3 classes

    # Initialize and train neural network
    nn = NeuralNetwork(input_size=10, hidden_size=5, output_size=3, dropout_rate=0.5, l2_lambda=0.01)
    nn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

    # Make predictions
    predictions = nn.predict(X_train)
    print("Predictions:", predictions)
