import json
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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


# Save metrics and settings
def save_metrics(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def train_custom_model(X_train, y_train, X_test, y_test, alpha, num_iterations, lambda_):
    n_features = X_train.shape[1]
    weights_init = np.zeros((n_features, 1), dtype=np.float32)
    weights, cost_history = gradient_descent(X_train, y_train, weights_init, alpha, num_iterations, lambda_)
    y_pred = predict(X_test, weights)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Cost History': cost_history,
        'Model Settings': {
            'Alpha': alpha,
            'Lambda': lambda_,
            'Iterations': num_iterations
        }
    }


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


def visualize_cost_history(cost_history, category):
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history, label='Cost Over Iterations')
    plt.title('Cost History for ' + category)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()


def choose_model():
    print("Select which model to train:")
    print("1: Scikit-Learn Logistic Regression")
    print("2: Custom Logistic Regression")
    choice = input("Enter choice (1 or 2): ")
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


def main():
    model_choice = choose_model()
    hyperparameters = get_hyperparameters(model_choice)
    train_and_evaluate_model(model_choice, hyperparameters)

    # Prompt for visualizations
    visualize_prompt = input("Would you like to visualize the results? (y/n): ").lower()
    if visualize_prompt in ['y', 'yes']:
        if model_choice == 1:
            visualize_metrics('sklearn_model_metrics.json')
        elif model_choice == 2:
            visualize_metrics('custom_model_metrics.json')
            with open('custom_model_metrics.json', 'r') as file:
                data = json.load(file)
                for category in data:
                    if 'Cost History' in data[category]:
                        visualize_cost_history(data[category]['Cost History'], category)


if __name__ == "__main__":
    main()
