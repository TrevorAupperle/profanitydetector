import json
import pandas as pd
import numpy as np
import scipy.sparse
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier



class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data, categories):
        super().__init__()
        self.train_data = train_data  # Make sure these are numpy arrays
        self.val_data = val_data  # Make sure these are numpy arrays
        self.categories = categories
        self.metrics_history = {category: {'precision': [], 'recall': [], 'f1_score': []} for category in categories}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_train_pred = (self.model.predict(self.train_data[0]) > 0.5).astype(int)
        y_val_pred = (self.model.predict(self.val_data[0]) > 0.5).astype(int)

        # Handle each category
        for i, category in enumerate(self.categories):
            # Ensure your true labels are correctly formatted as numpy arrays
            y_train_true = np.array(self.train_data[1])[:, i]
            y_val_true = np.array(self.val_data[1])[:, i]

            # Calculate metrics
            precision_train = precision_score(y_train_true, y_train_pred[:, i], zero_division=0)
            recall_train = recall_score(y_train_true, y_train_pred[:, i], zero_division=0)
            f1_train = f1_score(y_train_true, y_train_pred[:, i], zero_division=0)

            precision_val = precision_score(y_val_true, y_val_pred[:, i], zero_division=0)
            recall_val = recall_score(y_val_true, y_val_pred[:, i], zero_division=0)
            f1_val = f1_score(y_val_true, y_val_pred[:, i], zero_division=0)

            # Store and log metrics
            self.metrics_history[category]['precision'].append((precision_train, precision_val))
            self.metrics_history[category]['recall'].append((recall_train, recall_val))
            self.metrics_history[category]['f1_score'].append((f1_train, f1_val))

            logs[f'{category}_precision_train'] = precision_train
            logs[f'{category}_recall_train'] = recall_train
            logs[f'{category}_f1_train'] = f1_train
            logs[f'{category}_precision_val'] = precision_val
            logs[f'{category}_recall_val'] = recall_val
            logs[f'{category}_f1_val'] = f1_val


# Load data
train_data = pd.read_csv('train.csv')
validation_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')

# Handle NaN values in the text column
train_data['text'] = train_data['text'].fillna('')
validation_data['text'] = validation_data['text'].fillna('')
test_data['text'] = test_data['text'].fillna('')

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_data['text'])
X_validation = vectorizer.transform(validation_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Categories to predict
categories = ['Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate', 'not_profane']

# Initialize metrics dictionary
metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}


def sigmoid(z):
    """ Apply the sigmoid function to a scalar, vector, or matrix (sparse or dense). """
    if scipy.sparse.issparse(z):  # Check if z is a sparse matrix
        # Compute exp on the dense array of z directly to avoid full dense conversion if large
        return 1 / (1 + np.exp(-z.toarray()))
    else:
        return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights, lambda_):
    m = len(y)
    h = sigmoid(X.dot(weights))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    reg_cost = cost + (lambda_ / (2 * m)) * np.sum(weights[1:] ** 2)
    return reg_cost


def gradient_descent(X, y, weights, alpha, num_iterations, lambda_, print_progress=True):
    m = len(y)
    cost_history = []

    for i in tqdm(range(num_iterations), desc="Processing", leave=True):
        z = X.dot(weights)
        predictions = sigmoid(z)
        errors = predictions - y
        grad_reg = lambda_ / m * np.vstack([0, weights[1:]])
        gradients = (1 / m) * (X.T.dot(errors)) + grad_reg
        weights -= alpha * gradients
        cost = compute_cost(X, y, weights, lambda_)
        cost_history.append(cost)

    return weights, cost_history


# Prediction function:
def predict(X, weights):
    """ Predict binary labels using the trained logistic regression weights """
    z = X.dot(weights)
    y_pred = sigmoid(z) >= 0.5
    return y_pred.flatten()


def create_mlp_model(num_layers, num_neurons, activation_function, learning_rate):
    model = Sequential()
    model.add(Dense(num_neurons, activation=activation_function, input_dim=X_train.shape[1]))
    for _ in range(1, num_layers):
        model.add(Dense(num_neurons, activation=activation_function))
    model.add(Dense(len(categories), activation='sigmoid'))  # Multi-label classification
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Save metrics and settings
def save_metrics(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def train_custom_model(X_train, y_train, X_val, y_val, alpha, num_iterations, lambda_):
    n_features = X_train.shape[1]
    weights = np.zeros((n_features, 1), dtype=np.float32)
    train_loss_history, val_loss_history = [], []

    for i in range(num_iterations):
        train_pred = sigmoid(X_train.dot(weights))
        train_loss = compute_cost(X_train, y_train, weights, lambda_)
        train_loss_history.append(train_loss)

        val_pred = sigmoid(X_val.dot(weights))
        val_loss = compute_cost(X_val, y_val, weights, lambda_)
        val_loss_history.append(val_loss)

        # Gradient descent step
        errors = train_pred - y_train
        grad_reg = lambda_ / len(y_train) * np.vstack([0, weights[1:]])
        gradients = (1 / len(y_train)) * (X_train.T.dot(errors)) + grad_reg
        weights -= alpha * gradients

    y_pred = predict(X_val, weights)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Train Loss History': train_loss_history,
        'Validation Loss History': val_loss_history,
        'Model Settings': {
            'Alpha': alpha,
            'Lambda': lambda_,
            'Iterations': num_iterations
        }
    }


def train_mlp_model(model, X_train, y_train, X_val, y_val, categories, epochs=50, batch_size=32):
    metrics_callback = MetricsCallback(train_data=(X_train, y_train), val_data=(X_val, y_val), categories=categories)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1, callbacks=[metrics_callback])
    return history.history


def train_and_evaluate_dummy_classifier(X_train, y_train, X_test, y_test):
    # Initialize the dummy classifier to predict based on uniform random chance
    dummy_clf = DummyClassifier(strategy="uniform", random_state=42)
    dummy_clf.fit(X_train, y_train)
    y_pred_dummy = dummy_clf.predict(X_test)

    # Calculate and print the metrics
    dummy_metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_dummy),
        'Precision': precision_score(y_test, y_pred_dummy, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_dummy, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred_dummy, average='weighted')
    }

    return dummy_metrics


def visualize_metrics(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    categories = list(data.keys())
    accuracies = [data[cat]['Accuracy'] for cat in categories]
    precisions = [data[cat]['Precision'] for cat in categories]
    recalls = [data[cat]['Recall'] for cat in categories]
    f1_scores = [data[cat]['F1 Score'] for cat in categories]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Performance Metrics Across Categories')

    axs[0, 0].bar(categories, accuracies, color='blue')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_ylim([0, 1])

    axs[0, 1].bar(categories, precisions, color='green')
    axs[0, 1].set_title('Precision')
    axs[0, 1].set_ylim([0, 1])

    axs[1, 0].bar(categories, recalls, color='red')
    axs[1, 0].set_title('Recall')
    axs[1, 0].set_ylim([0, 1])

    axs[1, 1].bar(categories, f1_scores, color='purple')
    axs[1, 1].set_title('F1 Score')
    axs[1, 1].set_ylim([0, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def visualize_loss(train_loss, val_loss, title='Training and Validation Loss'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_mlp_metrics(filename):
    # Load the metrics data from the JSON file
    with open(filename, 'r') as file:
        data = json.load(file)

    # Number of epochs
    epochs = list(range(1, len(data['accuracy']) + 1))

    # Create figure and axes for subplots
    fig, axs = plt.subplots(3, 2, figsize=(40, 50))
    fig.suptitle('MLP Model Performance Evaluation', fontsize=16)

    # Loss over epochs
    axs[0, 0].plot(epochs, data['loss'], label='Training Loss')
    axs[0, 0].plot(epochs, data['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss over Epochs')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # Accuracy over epochs
    axs[0, 1].plot(epochs, data['accuracy'], label='Training Accuracy')
    axs[0, 1].plot(epochs, data['val_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy over Epochs')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # Evolution of metrics for each category over training epochs
    categories = ['Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate', 'not_profane']
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, category in enumerate(categories):
        if f'{category}_precision_train' in data:
            axs[1, 0].plot(epochs, data[f'{category}_f1_train'], label=f'{category} F1 Score', color=colors[i])

    axs[1, 0].set_title('Evolution of F1 Score for Each Category')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].legend()

    # Final metrics across all categories
    final_f1_scores = [data[f'{category}_f1_train'][-1] for category in categories if f'{category}_f1_train' in data]
    axs[1, 1].bar(categories, final_f1_scores, color=colors)
    axs[1, 1].set_title('Final F1 Scores Across All Categories')
    axs[1, 1].set_ylabel('F1 Score')

    # Visualize training vs validation for each metric (using accuracy as example)
    axs[2, 0].plot(epochs, data['accuracy'], label='Training Accuracy', color='b')
    axs[2, 0].plot(epochs, data['val_accuracy'], label='Validation Accuracy', color='g')
    axs[2, 0].set_title('Training vs Validation Accuracy')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].set_ylabel('Accuracy')
    axs[2, 0].legend()

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5, wspace=0.4)
    plt.show()


def choose_model():
    print("Select which model to train:")
    print("1: Scikit-Learn Logistic Regression")
    print("2: Custom Logistic Regression")
    print("3: TensorFlow MLP")  # New option
    choice = input("Enter choice (1, 2, or 3): ")
    return int(choice)


def get_hyperparameters(model_choice):
    if model_choice == 1:
        print("Adjust the hyperparameters for the Scikit-Learn Logistic Regression.")
        C = float(input("Enter the regularization strength (inverse of C, default 1.0): "))
        solver = input("Enter the solver ('liblinear', 'lbfgs', etc., default 'liblinear'): ")
        return {'C': C, 'solver': solver}
    elif model_choice == 2:
        print("Adjust the hyperparameters for the Custom Logistic Regression.")
        alpha = float(input("Enter the learning rate (default 0.1): "))
        num_iterations = int(input("Enter the number of iterations (default 5000): "))
        lambda_ = float(input("Enter the regularization strength (default 0.1): "))
        return {'alpha': alpha, 'num_iterations': num_iterations, 'lambda_': lambda_}
    elif model_choice == 3:
        print("Adjust the hyperparameters for the TensorFlow MLP.")
        num_layers = int(input("Enter the number of hidden layers: "))
        num_neurons = int(input("Enter the number of neurons per layer: "))
        activation_function = input("Enter the activation function (e.g., 'relu', 'sigmoid'): ")
        learning_rate = float(input("Enter the learning rate: "))
        return {'num_layers': num_layers, 'num_neurons': num_neurons,
                'activation_function': activation_function, 'learning_rate': learning_rate}


# Train and evaluate models
def train_and_evaluate_model(model_choice, hyperparameters):
    if model_choice == 1:
        model_name = 'Scikit-Learn Logistic Regression'
        model_metrics = {}
        for category in tqdm(categories, desc=f"Training {model_name}"):
            y_train = train_data[category]
            y_test = test_data[category]

            model = LogisticRegression(C=hyperparameters['C'], solver=hyperparameters['solver'])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'Model Settings': hyperparameters
            }
            model_metrics[category] = metrics

        save_metrics(model_metrics, 'sklearn_model_metrics.json')

    elif model_choice == 2:
        model_name = 'Custom Logistic Regression'
        custom_model_metrics = {}
        for category in tqdm(categories, desc=f"Training {model_name}"):
            y_train = train_data[category].values.reshape(-1, 1)
            y_test = test_data[category].values.reshape(-1, 1)

            metrics = train_custom_model(X_train, y_train, X_test, y_test,
                                         hyperparameters['alpha'], hyperparameters['num_iterations'],
                                         hyperparameters['lambda_'])
            custom_model_metrics[category] = metrics

        save_metrics(custom_model_metrics, 'custom_model_metrics.json')

    elif model_choice == 3:
        model_name = 'TensorFlow MLP'
        mlp_model = create_mlp_model(hyperparameters['num_layers'], hyperparameters['num_neurons'],
                                     hyperparameters['activation_function'], hyperparameters['learning_rate'])
        y_train_encoded = train_data[categories]
        y_validation_encoded = validation_data[categories]
        history = train_mlp_model(mlp_model, X_train, y_train_encoded, X_validation, y_validation_encoded, categories)
        save_metrics(history, 'mlp_model_metrics.json')


def main():
    action = input(
        "Do you want to train models or visualize existing data? Enter 'train' or 'visualize': ").strip().lower()

    if action == 'train' or action == 't':
        model_choice = choose_model()
        hyperparameters = get_hyperparameters(model_choice)
        train_and_evaluate_model(model_choice, hyperparameters)

        # Train and evaluate dummy classifier
        dummy_metrics = train_and_evaluate_dummy_classifier(X_train, train_data[categories], X_test,
                                                            test_data[categories])
        print("Dummy Classifier Metrics:")
        for metric, value in dummy_metrics.items():
            print(f"{metric}: {value}")

        visualize_prompt = input("Would you like to visualize the results now? (y/n): ").strip().lower()
        if visualize_prompt in ['y', 'yes']:
            if model_choice == 1:
                visualize_metrics('sklearn_model_metrics.json')
            elif model_choice == 2:
                visualize_metrics('custom_model_metrics.json')
                with open('custom_model_metrics.json', 'r') as file:
                    data = json.load(file)
                    for category in data:
                        if 'Train Loss History' in data[category]:
                            visualize_loss(data[category]['Train Loss History'],
                                           data[category]['Validation Loss History'],
                                           'Training and Validation Loss for ' + category)
            elif model_choice == 3:
                visualize_mlp_metrics('mlp_model_metrics.json')

    elif action == 'visualize' or action == 'v':
        print("Select which model data to visualize:")
        print("1: Scikit-Learn Logistic Regression")
        print("2: Custom Logistic Regression")
        print("3: TensorFlow MLP")
        print("4: All")
        visualization_choice = input("Enter choice (1, 2, 3, or 4): ").strip()

        if visualization_choice == '1':
            visualize_metrics('sklearn_model_metrics.json')
        elif visualization_choice == '2':
            visualize_metrics('custom_model_metrics.json')
            with open('custom_model_metrics.json', 'r') as file:
                data = json.load(file)
                for category in data:
                    if 'Train Loss History' in data[category]:
                        visualize_loss(data[category]['Train Loss History'],
                                       data[category]['Validation Loss History'],
                                       'Training and Validation Loss for ' + category)
        elif visualization_choice == '3':
            visualize_mlp_metrics('mlp_model_metrics.json')
        elif visualization_choice == '4':
            print("Visualizing Scikit-Learn Logistic Regression Data:")
            visualize_metrics('sklearn_model_metrics.json')
            print("Visualizing Custom Logistic Regression Data:")
            visualize_metrics('custom_model_metrics.json')
            with open('custom_model_metrics.json', 'r') as file:
                data = json.load(file)
                for category in data:
                    if 'Train Loss History' in data[category]:
                        visualize_loss(data[category]['Train Loss History'],
                                       data[category]['Validation Loss History'],
                                       'Training and Validation Loss for ' + category)
            print("Visualizing TensorFlow MLP Data:")
            visualize_mlp_metrics('mlp_model_metrics.json')
        else:
            print("Invalid choice. Exiting.")

    else:
        print("Invalid action. Please enter 'train' or 'visualize'. Exiting.")


if __name__ == "__main__":
    main()
