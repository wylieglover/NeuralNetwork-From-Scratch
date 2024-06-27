import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical

class NeuralNetwork:
    def __init__(self, input_size=784):
        self.weights = []
        self.biases = []
        self.trainable = []
        self.activations = []
        self.input_size = input_size

    def add_layer(self, units, activation='relu', trainable=True):
        input_dim = self.input_size if len(self.weights) == 0 else self.weights[-1].shape[1]
        self.weights.append(np.random.randn(input_dim, units) * np.sqrt(2 / input_dim))
        self.biases.append(np.zeros((1, units)))
        self.activations.append(activation)
        self.trainable.append(trainable)

    def save_model(self, file_prefix):
        for i, (weight, bias, activation) in enumerate(zip(self.weights, self.biases, self.activations)):
            np.savez(f"{file_prefix}_layer_{i+1}", weights=weight, biases=bias, activations=activation)

    def load_model(self, file_prefix, freeze_layers=False, exclude_final_layer=False):
        i = 1
        while True:
            try:
                layer_data = np.load(f"{file_prefix}_layer_{i}.npz")
                self.weights.append(layer_data['weights'])
                self.biases.append(layer_data['biases'])
                self.activations.append(layer_data['activations'])
                if freeze_layers:
                    self.trainable.append(False)
                else:
                    self.trainable.append(True)
                i += 1
            except FileNotFoundError:
                break 
        if len(self.weights) != len(self.biases):
            raise ValueError("Mismatch between weights and biases - Unsuccesfully loaded model.")
        elif len(self.weights) == 0:
            raise FileNotFoundError(f"File not found {file_prefix}_layer_{i}.npz - Unsuccesfully loaded model.")
        else:
            if exclude_final_layer:
                self.weights.pop()
                self.biases.pop()
                self.trainable.pop()
                self.activations.pop()
            print("Succesffully loaded model!")     

    def forward(self, inputs):
        self.layers = [inputs]
        for i in range(len(self.weights)):
            z = np.dot(self.layers[-1], self.weights[i]) + self.biases[i]
            if self.activations[i] == 'relu':
                a = np.maximum(0, z)
            elif self.activations[i] == 'sigmoid':
                a = 1 / (1 + np.exp(-z))
            elif self.activations[i] == 'softmax':
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]}")
            self.layers.append(a)
        return self.layers[-1]

    def backward(self, y_true, learning_rate=0.001, l2_lambda=0.0):
        delta = self.layers[-1] - y_true
        for i in reversed(range(len(self.weights))):
            if self.trainable[i]:
                grad_w = np.dot(self.layers[i].T, delta) + l2_lambda * self.weights[i]
                grad_b = np.sum(delta, axis=0, keepdims=True)
                if i != 0:
                    delta = np.dot(delta, self.weights[i].T) * (self.layers[i] > 0)
                self.weights[i] -= learning_rate * grad_w
                self.biases[i] -= learning_rate * grad_b
                
    def train(self, x_train, y_train, epochs=10, batch_size=32,  initial_learning_rate=0.001, decay_rate=0.0, l2_lambda=0.0):
        learning_rate = initial_learning_rate
        num_samples = x_train.shape[0]
        print("Training model:")
        for epoch in range(epochs):
            accuracies = []
            losses = []
            for i in range(0, num_samples, batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                y_pred = self.forward(x_batch)
                
                losses.append(self.compute_loss(y_batch, y_pred))
                accuracies.append(self.compute_accuracy(y_batch, y_pred))
                
                self.backward(y_batch, learning_rate, l2_lambda)

                print(f"\rEpoch {epoch + 1}/{epochs}: batch {i//batch_size + 1}/{num_samples//batch_size + 1} - loss: {np.mean(losses):.4f} - accuracy: {np.mean(accuracies):.4f}", end="")
            if decay_rate:
                learning_rate = initial_learning_rate * np.exp(-decay_rate * (epoch + 1))
            print()

    def compute_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        if self.activations[-1] == 'sigmoid':
            loss = -(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        elif self.activations[-1] == 'softmax':
            loss = -np.sum(y_true * np.log(y_pred + 1e-8), axis=1)
        else:
            raise ValueError(f"Unsupported activation function: {self.activations[-1]}")
        return np.sum(loss) / n_samples

    def compute_accuracy(self, y_true, y_pred):
        if self.activations[-1] == 'sigmoid':
            predictions = (y_pred >= 0.5).astype(int)
            if y_true.any() == None:
                return predictions
            accuracy = np.mean(predictions == y_true)
        elif self.activations[-1] == 'softmax':
            predictions = np.argmax(y_pred, axis=1)
            if y_true.any() == None:
                return predictions
            accuracy = np.mean(predictions == y_true)
        else:
            raise ValueError(f"Unsupported activation function: {self.activations[-1]}")
        return accuracy

    def predict(self, x_test):
        y_pred = self.forward(x_test)
        if self.activations[-1] == 'sigmoid':
            return (y_pred >= 0.5).astype(int)
        elif self.activations[-1] == 'softmax':
            return np.argmax(y_pred, axis=1)
        else:
            raise ValueError(f"Unsupported activation function: {self.activations[-1]}")

def create_data(path):
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-20, 20)),  # random rotations
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),  # random translations
        iaa.Affine(scale=(0.8, 1.2)),  # random scaling
        iaa.AdditiveGaussianNoise(scale=0.05*255),  # random noise
        iaa.Multiply((0.8, 1.2))  # random brightness
    ])

    X = []
    y = []
    convert = lambda category : int(category == 'dog')
    for p in os.listdir(path):
        category = ''
        category = (p.split("."))[0]
        category = convert(category)

        img_array = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE) # Greyscale
        new_img_array = cv2.resize(img_array, dsize=(80, 80))  # Resize 
        new_img_array = augmenter(image=new_img_array)

        X.append(new_img_array)
        y.append(category)
    
    return np.array(X).reshape(-1, 80 * 80) / 255.0, np.array(y).reshape(-1, 1)

def load_data_without_labels(path):
    X = []
    file_names = []
    for p in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE) # Greyscale
        new_img_array = cv2.resize(img_array, dsize=(80, 80))  # Resize 
        X.append(new_img_array)
        file_names.append(p)
    return np.array(X).reshape(-1, 80 * 80) / 255.0, file_names

def main():
    # Training and test paths:
    train_path = 'input/train'
    test_path = 'input/test1'
    
    # Cat/Dog data and formatting:
    X, y = create_data(train_path)
    
    # Mnist data and formatting (optional example data):
    # (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # train_X = train_X.reshape(-1, 784) / 255.0
    # test_X = test_X.reshape(-1, 784) / 255.0
    # train_y = to_categorical(train_y, 10)
    # test_y = to_categorical(test_y, 10)
    
    # Initializing model:
    model = NeuralNetwork(input_size=80*80)

    # Loading model (loading pre-trained data (model state), only use this if you have saved data):
    # model.load_model('cat_dog_classifier', freeze_layers=False, exclude_final_layer=False)

    # Setting up layers:
    model.add_layer(units=512, activation='relu', trainable=True)
    model.add_layer(units=512, activation='relu', trainable=True)
    model.add_layer(units=512, activation='relu', trainable=True)
    model.add_layer(units=1, activation='sigmoid', trainable=True)
    
    # Training and saving model:
    model.train(X, y, epochs=50, initial_learning_rate=0.001, decay_rate=0.01, batch_size=32, l2_lambda=0.001)
    model.save_model('cat_dog_classifier')
    
    # Example of using model to predict without y_test (y_true):
    X_new, file_names = load_data_without_labels(test_path)
    predictions = model.predict(X_new)

    def plot_image_with_label(file_name, label):
        img = mpimg.imread(file_name)
        plt.imshow(img)
        plt.axis('off') 
        plt.title(label, fontsize=12)
        plt.show()

    i = 0
    for prediction, file in zip(predictions, file_names):
        i += 1
        if "sol" not in file and i < 5:
            label = "dog" if prediction == 1 else "cat"
            plot_image_with_label('input/test1/' + file, label)

if __name__ == "__main__":
    main()
    