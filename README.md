# Artificial Neural Network (ANN) – Complete Beginner-Friendly Notes

# Table of Contents

1. Introduction to Deep Learning
2. Machine Learning vs Deep Learning
3. What is an Artificial Neural Network?
4. Biological Neuron vs Artificial Neuron
5. Architecture of ANN
6. Layers in ANN
7. How Does an Artificial Neuron Work?
8. Weights and Biases
9. Activation Functions
10. Forward Propagation
11. Loss Function
12. Backward Propagation
13. Optimizers
14. Learning Rate
15. Gradient Descent
16. Epoch, Batch and Iteration
17. Overfitting and Underfitting
18. Dropout Regularization
19. Introduction to TensorFlow and Keras
20. ANN Workflow
21. ANN Implementation – Employee Attrition Prediction
22. Understanding Every Line of Code
23. Model Evaluation
24. Important Interview Questions
25. Quick Revision Notes

---

# 1. Introduction to Deep Learning

## What is Deep Learning?

Deep Learning is a subset of Machine Learning that uses Artificial Neural Networks with multiple layers to learn complex patterns from large amounts of data.

In simple words:

Deep Learning allows computers to learn patterns from data in a way that is inspired by the human brain.

### Relationship between AI, ML and DL

```text
Artificial Intelligence (AI)
          |
          |
    Machine Learning (ML)
          |
          |
      Deep Learning (DL)
          |
          |
 Artificial Neural Networks
```

- Artificial Intelligence: Making machines perform tasks that normally require human intelligence.
- Machine Learning: Teaching machines to learn patterns from data.
- Deep Learning: Using multi-layer neural networks to learn complex patterns.

## Machine Learning vs Deep Learning

| Machine Learning | Deep Learning |
|---|---|
| Uses algorithms such as Decision Trees, Random Forest and SVM. | Uses Artificial Neural Networks. |
| Often requires manual feature engineering. | Can automatically learn useful features. |
| Usually works well with smaller datasets. | Often benefits from large datasets. |
| Usually requires less computational power. | Can require significant computational power. |
| Examples: Linear Regression, Random Forest. | Examples: ANN, CNN, RNN, Transformers. |

---

# 2. What is an Artificial Neural Network (ANN)?

ANN stands for Artificial Neural Network.

Our human brain contains billions of neurons. These neurons communicate with each other and process information.

Artificial Neural Networks are inspired by this biological neuron system.

Similarly, an ANN contains artificial neurons, also called nodes.

Each artificial neuron:

1. Receives input information.
2. Performs mathematical calculations.
3. Applies an activation function.
4. Passes the result to the next layer.

### Simple Example

Suppose we want to predict whether an employee will leave a company.

Our input features could be:

- Age
- Monthly Income
- Job Satisfaction
- Years at Company
- Overtime
- Department

The ANN learns patterns from these features and predicts:

```text
Employee Information
         |
         |
    ANN Model
         |
         |
   Prediction
         |
    0 = No
    1 = Yes
```

The model learns the relationship between employee information and employee attrition.

---

# 3. Biological Neuron vs Artificial Neuron

| Biological Neuron | Artificial Neuron |
|---|---|
| Dendrites | Input features |
| Synapses | Weights |
| Cell body | Mathematical processing |
| Activation | Activation function |
| Axon | Output |

This is an inspiration-based comparison, not an exact biological equivalence.

---

# 4. Architecture of ANN

An Artificial Neural Network generally contains three types of layers:

1. Input Layer
2. Hidden Layer
3. Output Layer

### ANN Architecture

```text
       INPUT LAYER
           |
     x1    x2    x3
      \     |    /
       \    |   /
        HIDDEN LAYER
          h1
          h2
          h3
           |
           |
       OUTPUT LAYER
           |
           y
```

## 4.1 Input Layer

The input layer receives information from the dataset.

Example:

Suppose we want to predict house prices.

Input features:

- House Size
- Number of Bedrooms
- Location

These features are passed to the neural network.

Important:

The number of input features determines the number of input values the network receives. In Keras, the input shape is generally specified using the number of features.

## 4.2 Hidden Layer

The hidden layer is where the neural network learns patterns.

It performs mathematical operations using:

- Weights
- Biases
- Activation Functions

We can have one or multiple hidden layers.

For example:

```text
Input Layer
     |
     |
Hidden Layer 1 (128 neurons)
     |
     |
Hidden Layer 2 (64 neurons)
     |
     |
Output Layer (1 neuron)
```

Important:

The hidden layers are not simply deleting or hiding information. They transform input information into learned representations that help the model make predictions.

## 4.3 Output Layer

The output layer produces the final prediction.

The number of output neurons and activation function depend on the problem.

| Problem | Output Neurons | Activation |
|---|---:|---|
| Binary Classification | 1 | Sigmoid |
| Multiclass Classification | Number of classes | Softmax |
| Regression | 1 (usually) | Linear |

Examples:

- Employee Attrition: 1 neuron, Sigmoid.
- Cat/Dog/Mango classification: 3 neurons, Softmax.
- House Price Prediction: 1 neuron, Linear.

---

# 5. How Does an Artificial Neuron Work?

An artificial neuron performs a mathematical calculation.

The basic formula is:

**z = (x1 × w1) + (x2 × w2) + (x3 × w3) + b**

Where:

- x = Input features
- w = Weights
- b = Bias
- z = Weighted sum

After calculating z, the neuron passes it through an activation function.

**Output = Activation Function(z)**

### Example

Suppose:

- x1 = 2
- x2 = 3
- w1 = 0.5
- w2 = 0.2
- b = 1

Calculation:

```text
z = (2 × 0.5) + (3 × 0.2) + 1

z = 1 + 0.6 + 1

z = 2.6
```

Now the value 2.6 is passed through an activation function.

If we use ReLU:

```text
ReLU(2.6) = 2.6
```

Therefore, the output is 2.6.

---

# 6. Weights and Biases

## 6.1 What are Weights?

Weights determine how important an input feature is for making a prediction.

For example:

Suppose we want to predict employee attrition.

| Feature | Weight |
|---|---:|
| Monthly Income | 0.8 |
| Age | 0.3 |
| Overtime | 1.5 |

These numbers are only illustrative.

A higher absolute weight means the corresponding input has a stronger influence on that neuron's weighted sum, assuming comparable feature scales.

The neural network learns these weights during training.

## 6.2 What is Bias?

Bias is an additional trainable parameter added to the weighted sum.

Formula:

```text
z = (x1 × w1) + (x2 × w2) + b
```

Why do we need bias?

Bias allows the neuron to shift its activation threshold or decision boundary.

Without bias, the neuron would be more restricted because its weighted calculation would have to pass through the origin.

### Remember:

- Weight = Controls the influence of an input.
- Bias = Provides an additional adjustable value.

---

# 7. Activation Functions

## What is an Activation Function?

An activation function is a mathematical function applied to the output of a neuron.

It introduces non-linearity into the neural network.

Why do we need it?

Without non-linear activation functions, stacking multiple Dense layers would still behave like a single linear transformation.

Activation functions allow neural networks to learn complex patterns.

## Important Activation Functions

### 7.1 ReLU (Rectified Linear Unit)

Formula:

```text
f(x) = max(0, x)
```

Meaning:

- If input is negative, output is 0.
- If input is positive, output remains the same.

Example:

| Input | ReLU Output |
|---:|---:|
| -5 | 0 |
| -2 | 0 |
| 0 | 0 |
| 3 | 3 |
| 8 | 8 |

Usage:

Mostly used in hidden layers.

### 7.2 Sigmoid

Formula:

```text
f(x) = 1 / (1 + e^(-x))
```

It converts values into a range between 0 and 1.

| Input | Approximate Output |
|---:|---:|
| -5 | 0.0067 |
| -2 | 0.1192 |
| 0 | 0.5 |
| 2 | 0.8808 |
| 5 | 0.9933 |

Usage:

Mostly used in the output layer for binary classification.

Example:

```text
Prediction = 0.85

The model estimates an 85% probability
for the positive class.
```

### 7.3 Softmax

Softmax converts multiple output scores into probabilities that sum to 1.

Example:

```text
Cat = 0.70
Dog = 0.20
Horse = 0.10

Total = 1.00
```

Usage:

Multiclass classification.

### Activation Function Comparison

| Activation | Output Range | Common Usage |
|---|---|---|
| ReLU | 0 to infinity | Hidden layers |
| Sigmoid | 0 to 1 | Binary classification |
| Softmax | Probabilities summing to 1 | Multiclass classification |
| Tanh | -1 to 1 | Some hidden-layer architectures |
| Linear | Negative to positive infinity | Regression |

---

# 8. Forward Propagation

## What is Forward Propagation?

Forward Propagation is the process in which input data travels from the input layer through the hidden layers to the output layer to generate a prediction.

In simple words:

**Forward Propagation = Making a Prediction.**

### Forward Propagation Workflow

```text
Input Data
    |
    |
Input Layer
    |
    |
Weighted Sum
    |
    |
Activation Function
    |
    |
Hidden Layer
    |
    |
Output Layer
    |
    |
Prediction
```

### Mathematical Steps

Step 1: Receive input values.

```text
x1, x2, x3
```

Step 2: Multiply inputs with weights and add bias.

```text
z = x1w1 + x2w2 + x3w3 + b
```

Step 3: Apply activation function.

```text
a = ReLU(z)
```

Step 4: Pass the output to the next layer.

Step 5: Generate final prediction.

Step 6: Compare prediction with actual output using a loss function.

### Example

Suppose:

```text
Actual Output = 1
Predicted Output = 0.75
```

The model has made a prediction of 0.75.

Now we calculate the loss to understand how far the prediction is from the actual target.

---

# 9. Loss Function

## What is a Loss Function?

A loss function measures the difference between actual values and predicted values.

In simple words:

**Loss Function = How wrong is our model's prediction?**

Example:

```text
Actual = 1
Prediction = 0.8
```

The loss function calculates a numerical penalty based on this prediction.

The training process tries to minimize this loss.

## Important Loss Functions

### 9.1 Mean Squared Error (MSE)

Used mainly for regression problems.

Formula:

```text
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

Example:

```text
Actual House Price = 50
Predicted House Price = 45

Error = 50 - 45 = 5

Squared Error = 5² = 25
```

MSE calculates the average squared prediction error.

### 9.2 Binary Cross Entropy

Used for binary classification problems.

Examples:

- Yes / No
- 0 / 1
- Fraud / Not Fraud
- Employee Leaves / Employee Stays

In our Attrition dataset:

```text
Attrition = Yes or No
```

We use:

```python
loss_fn = tf.keras.losses.BinaryCrossentropy()
```

### 9.3 Sparse Categorical Cross Entropy

Used for multiclass classification when target labels are integers.

Example:

```text
Cat = 0
Dog = 1
Horse = 2
```

We use:

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
```

### 9.4 Categorical Cross Entropy

Used for multiclass classification when target labels are one-hot encoded.

Example:

```text
Cat   = [1, 0, 0]
Dog   = [0, 1, 0]
Horse = [0, 0, 1]
```

We use:

```python
loss_fn = tf.keras.losses.CategoricalCrossentropy()
```

### Loss Function Comparison

| Problem | Loss Function |
|---|---|
| Regression | Mean Squared Error |
| Binary Classification | Binary Cross Entropy |
| Multiclass with integer labels | Sparse Categorical Cross Entropy |
| Multiclass with one-hot labels | Categorical Cross Entropy |

---

# 10. Backward Propagation (Backpropagation)

## What is Backward Propagation?

Backward Propagation is the process of calculating how much each trainable weight contributes to the prediction error and calculating gradients that indicate how the weights should change to reduce the loss.

In simple words:

**Backpropagation = Learning from prediction errors.**

### Why do we need Backpropagation?

Suppose our model predicts:

```text
Actual Output = 1
Predicted Output = 0.2
```

Our model has made a poor prediction.

Now we need to adjust the weights and biases to improve future predictions.

Backpropagation helps calculate the gradients required for these adjustments.

### Backpropagation Workflow

```text
Forward Propagation
       |
       |
  Prediction
       |
       |
  Loss Calculation
       |
       |
 Backpropagation
       |
       |
 Calculate Gradients
       |
       |
 Optimizer Updates Weights
       |
       |
  Model Improves
```

## Understanding the Chain Rule

Backpropagation uses the chain rule of calculus to calculate gradients through multiple layers.

Suppose we have:

```text
Input Layer
    |
    |
Hidden Neuron 1 ----\
                     \
                      Output Neuron
                     /
Hidden Neuron 2 ----/
```

The output depends on multiple hidden neurons.

Therefore, the effect of an input on the final output can be calculated by adding the effects through all relevant paths.

For example:

```text
dy/dx1 =
(dy/dh1 × dh1/dx1)
+
(dy/dh2 × dh2/dx1)
```

This is a simplified chain-rule illustration.

In actual neural network training, we generally calculate gradients of the loss with respect to weights and biases.

For example:

```text
∂Loss / ∂Weight
```

This tells us how sensitive the loss is to a particular weight.

### Difference Between Forward and Backward Propagation

| Forward Propagation | Backward Propagation |
|---|---|
| Moves from input to output. | Calculates gradients from output toward earlier layers. |
| Generates predictions. | Calculates how parameters affect the loss. |
| Uses weights and biases. | Computes gradients with respect to weights and biases. |
| Calculates loss after prediction. | Provides information needed for parameter updates. |

Remember:

- Forward Propagation = Prediction.
- Backward Propagation = Learning through gradients.

---

# 11. Optimizers

## What is an Optimizer?

An optimizer is an algorithm that updates the weights and biases of a neural network using the gradients calculated during backpropagation.

In simple words:

The loss function tells us how wrong the prediction is.

Backpropagation calculates the gradients.

The optimizer uses those gradients to update the weights and biases to reduce the loss.

### Important Optimizers

| Optimizer | Simple Explanation |
|---|---|
| SGD | Updates parameters using gradients from batches. |
| SGD with Momentum | Uses previous update direction to help accelerate learning. |
| RMSprop | Adapts the learning rate using a moving average of squared gradients. |
| Adam | Combines momentum-like and adaptive learning-rate ideas. |
| Adagrad | Adapts learning rates based on accumulated squared gradients. |

## SGD (Stochastic Gradient Descent)

SGD updates model parameters using gradients calculated from a batch of training samples.

Basic formula:

```text
New Weight = Old Weight - Learning Rate × Gradient
```

## Momentum

Momentum uses information from previous gradients to help updates move more consistently.

It can help reduce oscillations and accelerate learning in useful directions.

## Nesterov Momentum

Nesterov Momentum calculates the gradient at an anticipated future parameter position rather than simply using the current position.

It is a variation of momentum that can improve optimization behavior.

### Example

```python
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True
)
```

Here:

- SGD = Stochastic Gradient Descent.
- learning_rate = 0.001.
- momentum = 0.9.
- nesterov = True.

---

# 12. Learning Rate

## What is Learning Rate?

Learning Rate determines how big a step the optimizer takes when updating model weights.

In simple words:

Learning Rate = How quickly the model learns.

### Example

Suppose:

```text
Old Weight = 5
Gradient = 2
Learning Rate = 0.1
```

Using the basic gradient descent update:

```text
New Weight = 5 - (0.1 × 2)

New Weight = 4.8
```

### What happens when we change the Learning Rate?

| Learning Rate | Behavior |
|---|---|
| Very High | May overshoot the minimum and fail to converge. |
| Very Low | Learning may become very slow. |
| Appropriate | Can help the model converge efficiently. |

Common starting values:

```text
0.1
0.01
0.001
0.0001
```

The best learning rate depends on the model, optimizer and dataset.

---

# 13. Epoch, Batch and Iteration

These three terms are very important in Deep Learning.

## 13.1 Epoch

An Epoch means one complete pass through the entire training dataset.

Example:

Suppose we have 1,000 training records.

```text
Epoch 1 = Model processes all 1,000 records once.

Epoch 2 = Model processes all 1,000 records again.

Epoch 3 = Model processes all 1,000 records again.
```

## 13.2 Batch

A Batch is a small group of training samples processed together before a parameter update.

Example:

```text
Total Records = 1,000

Batch Size = 100
```

The model processes 100 records at a time.

## 13.3 Iteration

An Iteration is one training step that processes a batch and performs a parameter update.

### Example

```text
Total Training Samples = 1,000
Batch Size = 100
Epochs = 10
```

Iterations per Epoch:

```text
1000 / 100 = 10
```

Total Iterations:

```text
10 × 10 = 100
```

### Difference

| Term | Meaning |
|---|---|
| Epoch | One complete pass through the training dataset. |
| Batch | A group of training samples. |
| Iteration | One batch-processing and parameter-update step. |

---

# 14. Overfitting and Underfitting

## 14.1 What is Overfitting?

Overfitting happens when a model learns the training data too specifically, including noise or accidental patterns, and performs poorly on unseen data.

Example:

```text
Training Accuracy = 99%
Testing Accuracy = 65%
```

The model performs very well on training data but poorly on testing data.

### How to reduce Overfitting?

1. Dropout
2. Regularization
3. Early Stopping
4. More training data
5. Data Augmentation (where applicable)
6. Simplifying the model

## 14.2 What is Underfitting?

Underfitting happens when a model is too simple or insufficiently trained to learn the patterns in the dataset.

Example:

```text
Training Accuracy = 60%
Testing Accuracy = 58%
```

Both training and testing performance are poor.

### How to reduce Underfitting?

1. Increase model complexity.
2. Add suitable hidden layers.
3. Increase the number of neurons when justified.
4. Train for more epochs.
5. Improve feature engineering.
6. Tune the optimizer and learning rate.

### Comparison

| Overfitting | Underfitting |
|---|---|
| Memorizes training patterns. | Fails to learn enough patterns. |
| Training performance is high. | Training performance is low. |
| Testing performance is relatively poor. | Testing performance is also poor. |
| Model may be too complex. | Model may be too simple or insufficiently trained. |

---

# 15. Dropout Regularization

## What is Dropout?

Dropout is a regularization technique used to reduce overfitting.

It randomly sets a percentage of neuron outputs to zero during training.

Example:

```python
tf.keras.layers.Dropout(0.3)
```

Meaning:

30% of the neuron outputs are randomly dropped during each training step.

Suppose we have 100 neurons:

```text
Total Neurons = 100

Dropout Rate = 0.3

Approximately 30 neuron outputs are dropped.

Approximately 70 remain active.
```

The selected neurons change across training steps.

### Why do we use Dropout?

Without dropout, a network may become overly dependent on particular neurons.

Dropout encourages the network to learn more distributed and robust representations.

Important:

- Dropout is active during training.
- Dropout is automatically disabled during normal evaluation and prediction.
- `0.3` means a 30% dropout rate.
- `0.5` means a 50% dropout rate.

---

# 16. TensorFlow and Keras

## What is TensorFlow?

TensorFlow is an open-source machine learning framework developed by Google.

It provides tools to build, train and deploy machine learning and deep learning models.

In simple words:

TensorFlow helps us build neural networks without manually implementing every mathematical operation.

### Why do we use TensorFlow?

TensorFlow provides tools for:

1. Mathematical calculations.
2. Forward propagation.
3. Loss calculation.
4. Gradient calculation.
5. Backpropagation.
6. Weight updates.
7. Model training and evaluation.

## What is Keras?

Keras is a high-level deep learning API.

It makes building neural networks easier through simple layers and model-building interfaces.

In TensorFlow, we commonly use:

```python
tf.keras
```

Example:

```python
tf.keras.layers.Dense(128, activation='relu')
```

This creates a Dense layer containing 128 neurons.

---

# 17. ANN Workflow

The general workflow of a Deep Learning project:

```text
1. Data Ingestion
        |
        v
2. Data Understanding
        |
        v
3. Data Cleaning
        |
        v
4. Data Preprocessing
        |
        v
5. Feature Encoding
        |
        v
6. Train-Test Split
        |
        v
7. Feature Scaling
        |
        v
8. Model Architecture
        |
        v
9. Model Compilation
        |
        v
10. Model Training
        |
        v
11. Model Evaluation
        |
        v
12. Model Prediction
```

Let's understand each step.

| Step | Meaning |
|---|---|
| Data Ingestion | Loading data from CSV, database, API, etc. |
| Data Understanding | Understanding columns, data types and distributions. |
| Data Cleaning | Handling missing values, duplicates and incorrect data. |
| Feature Encoding | Converting categorical features into numerical values. |
| Train-Test Split | Dividing data into training and testing datasets. |
| Feature Scaling | Bringing numerical features onto comparable scales. |
| Model Architecture | Creating input, hidden and output layers. |
| Model Compilation | Choosing optimizer, loss and metrics. |
| Model Training | Teaching the model using training data. |
| Model Evaluation | Checking performance on unseen testing data. |
| Model Prediction | Predicting results for new data. |

---

# 18. Practical Implementation: Employee Attrition Prediction Using ANN

## Project Objective

The objective is to build an Artificial Neural Network that predicts whether an employee will leave an organization.

Dataset:

`Attrition.csv`

Target Column:

`Attrition`

Possible Target Values:

```text
Yes
No
```

This is a Binary Classification problem because the target has two classes.

Our model will predict the probability of employee attrition.

## Step 1: Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn
```

## Step 2: Import Libraries

```python
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
```

Explanation:

| Library | Purpose |
|---|---|
| NumPy | Numerical calculations. |
| Pandas | Data loading and manipulation. |
| TensorFlow | Building and training ANN. |
| train_test_split | Splitting data into training and testing sets. |
| LabelEncoder | Encoding target labels. |
| StandardScaler | Scaling numerical features. |
| accuracy_score | Calculating classification accuracy. |

## Step 3: Load Dataset

```python
df = pd.read_csv("Attrition.csv")
```

Displays first four rows:

```python
df.head(4)
```

Check columns:

```python
df.columns
```

Check dataset information:

```python
df.info()
```

## Step 4: Handle Missing Values

```python
df = df.dropna()
```

Explanation:

`dropna()` removes rows containing missing values.

Important:

This is a simple approach. In real-world projects, we should inspect missing values first and decide whether dropping rows or imputing values is appropriate.

## Step 5: Separate Independent and Dependent Variables

```python
x = df.drop(columns=['Attrition'])

y = df['Attrition']
```

Explanation:

`x` contains independent variables (input features).

`y` contains the dependent variable (target).

Example:

```text
x = Age, Gender, MonthlyIncome, JobRole, OverTime, etc.

y = Attrition
```

## Step 6: Identify Categorical Features

```python
categorical_features = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
    'Over18',
    'OverTime'
]
```

Categorical features contain categories or labels instead of continuous numerical measurements.

Examples:

```text
Gender:
Male, Female

Department:
Sales, Research & Development, Human Resources

OverTime:
Yes, No
```

Neural networks require numerical inputs, so we need to encode categorical variables.

## Step 7: One-Hot Encoding

```python
x_encoded = pd.get_dummies(
    x,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)
```

### What is pd.get_dummies()?

It converts categorical features into numerical columns using One-Hot Encoding.

Example:

Before encoding:

| Gender |
|---|
| Male |
| Female |
| Male |

After encoding:

| Gender_Male |
|---:|
| 1 |
| 0 |
| 1 |

### Understanding Parameters

| Parameter | Meaning |
|---|---|
| x | Original input dataset. |
| columns | Categorical columns to encode. |
| drop_first=True | Drops the first category from each encoded feature. |
| dtype=int | Creates integer 0/1 columns. |

### What is drop_first=True?

Suppose:

```text
BusinessTravel:
Non-Travel
Travel_Frequently
Travel_Rarely
```

Without dropping the first category, we get three dummy columns.

With `drop_first=True`, one category is removed.

When all remaining category columns are zero, the dropped category is implied.

This helps avoid redundant dummy variables, particularly in linear models.

## Step 8: Convert Data Types

```python
x_encoded = x_encoded.astype(int)
```

Explanation:

`astype(int)` converts all columns in the DataFrame into integer data types.

Example:

```text
True  → 1

False → 0
```

Important:

Because we already used `dtype=int` inside `pd.get_dummies()`, this additional line is usually unnecessary for the dummy columns. However, it converts every column in the DataFrame.

## Step 9: Encode Target Variable

```python
lb = LabelEncoder()

y_encoded = lb.fit_transform(y)
```

Explanation:

The target variable contains text labels:

```text
Yes
No
```

LabelEncoder converts these labels into numerical values.

Example:

```text
No  → 0

Yes → 1
```

The exact mapping can be checked using:

```python
lb.classes_
```

To check encoded values:

```python
y_encoded
```

Why do we encode the target?

Because the neural network needs numerical target values for calculating the loss.

## Step 10: Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42
)
```

Explanation:

We divide the dataset into two parts.

Training Data:

Used to teach the neural network.

Testing Data:

Used to evaluate the trained model on unseen samples.

Parameters:

| Parameter | Meaning |
|---|---|
| x_encoded | Encoded input features. |
| y_encoded | Encoded target labels. |
| test_size=0.2 | 20% testing data. |
| random_state=42 | Makes the split reproducible. |

For classification, it is often better to use stratification:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
```

This helps maintain similar class proportions in training and testing data.

## Step 11: Feature Scaling

```python
sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)

x_test_scaled = sc.transform(x_test)
```

### What is StandardScaler?

StandardScaler is used to standardize numerical features.

It transforms features using:

```text
z = (x - mean) / standard_deviation
```

After scaling, each training feature generally has:

```text
Mean ≈ 0

Standard Deviation ≈ 1
```

Why do we use scaling?

Suppose:

```text
Age = 25

MonthlyIncome = 50000

YearsAtCompany = 5
```

These features have different numerical ranges.

Scaling helps prevent large-scale features from dominating optimization simply because of their units.

### Important Difference

```python
sc.fit_transform(x_train)
```

- fit() learns the mean and standard deviation from training data.
- transform() scales the training data.

```python
sc.transform(x_test)
```

Uses the same training mean and standard deviation to scale testing data.

Never fit the scaler separately on test data.

Important improvement:

In a production-quality preprocessing pipeline, scale numerical columns and encode categorical columns separately. It is generally unnecessary to standardize 0/1 one-hot encoded features.

For example, you can use `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder` for categorical features.

Also, fitting an encoder on the full dataset before splitting can expose information about category availability. A pipeline that fits preprocessing only on training data is preferable.

---

# 19. Building ANN Architecture Using TensorFlow

Now we build our Artificial Neural Network.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        units=128,
        activation='relu',
        input_dim=x_train_scaled.shape[1]
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        units=64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid'
    )
])
```

Let's understand every layer.

## What is Sequential?

Sequential is a Keras model-building API.

It allows us to create a neural network by stacking layers one after another.

Example:

```text
Input → Hidden Layer → Hidden Layer → Output
```

## First Dense Layer

```python
tf.keras.layers.Dense(
    units=128,
    activation='relu',
    input_dim=x_train_scaled.shape[1]
)
```

### What is Dense?

Dense means a fully connected layer.

Every neuron in this layer receives information from all outputs of the previous layer.

### units=128

It means the layer contains 128 neurons.

### activation='relu'

We use ReLU to introduce non-linearity.

### input_dim

It specifies the number of input features.

```python
x_train_scaled.shape[1]
```

Here:

- `shape[0]` = Number of rows (samples).
- `shape[1]` = Number of columns (features).

We need the number of features, not the number of samples.

## Dropout Layer

```python
tf.keras.layers.Dropout(0.3)
```

Meaning:

Randomly sets 30% of the layer's outputs to zero during training.

Purpose:

To reduce overfitting.

## Second Dense Layer

```python
tf.keras.layers.Dense(
    units=64,
    activation='relu'
)
```

Meaning:

- 64 neurons.
- ReLU activation.
- Learns additional patterns from previous layer outputs.

## Output Layer

```python
tf.keras.layers.Dense(
    units=1,
    activation='sigmoid'
)
```

Why units=1?

Because our problem is binary classification.

We need one output probability for the positive class.

Why sigmoid?

Because sigmoid converts the final output into a value between 0 and 1.

Example:

```text
Output = 0.85
```

The model predicts an 85% probability for the positive class.

The final class can be selected using a threshold, commonly 0.5:

```text
Probability >= 0.5 → Class 1

Probability < 0.5 → Class 0
```

The threshold can be adjusted based on the application and evaluation requirements.

---

# 20. Model Compilation

Before training, we compile the model.

```python
loss_fn = tf.keras.losses.BinaryCrossentropy()

metrics = ['accuracy']

learning_rate = 0.001

momentum = 0.9

optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=True
)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=metrics
)
```

There are three important components in model compilation.

## 1. Loss Function

```python
loss_fn = tf.keras.losses.BinaryCrossentropy()
```

We use Binary Cross Entropy because our target has two classes:

```text
Yes = 1

No = 0
```

It measures the prediction error for binary classification.

## 2. Optimizer

```python
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True
)
```

The optimizer updates weights and biases using calculated gradients.

Here:

- SGD = Stochastic Gradient Descent.
- Learning Rate = 0.001.
- Momentum = 0.9.
- Nesterov = True.

## 3. Metrics

```python
metrics = ['accuracy']
```

Metrics help us monitor model performance.

Accuracy measures the percentage of correct predictions.

Formula:

```text
Accuracy = Correct Predictions / Total Predictions
```

Example:

```text
Total Predictions = 100

Correct Predictions = 85

Accuracy = 85 / 100

Accuracy = 85%
```

Important:

Loss is used as the optimization objective.

Accuracy is used to monitor classification performance.

---

# 21. Model Training

Now we train our ANN.

```python
history = model.fit(
    x_train_scaled,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1
)
```

Let's understand each parameter.

## x_train_scaled

Scaled training input features.

## y_train

Training target labels.

## epochs=50

The model will pass through the entire training dataset 50 times.

## batch_size=8

The model processes 8 training samples per batch before updating its parameters.

## validation_split=0.1

10% of the training data is reserved for validation.

Important:

Validation data is used to monitor model performance during training.

It is NOT the same as testing data.

### Difference Between Training, Validation and Testing

| Dataset | Purpose |
|---|---|
| Training Data | Used to learn model parameters. |
| Validation Data | Used to monitor training and tune model choices. |
| Testing Data | Used for final evaluation on unseen data. |

In our implementation:

```text
Original Dataset
       |
       |
 Train-Test Split
       |
       |-------------------|
       |                   |
   80% Train           20% Test
       |
       |
 Validation Split
       |
       |-------------------|
       |                   |
   90% Train           10% Validation
```

The validation split takes 10% of the training data, not 10% of the original dataset.

---

# 22. Model Evaluation

After training, we evaluate our model.

```python
loss, accuracy = model.evaluate(
    x_test_scaled,
    y_test
)
```

Explanation:

- x_test_scaled = Scaled testing input features.
- y_test = Actual testing labels.
- loss = Testing loss.
- accuracy = Testing accuracy.

Display the results:

```python
print(f"Test Loss = {loss:.4f}")

print(f"Test Accuracy = {accuracy:.4f}")
```

Example output (illustrative only):

```text
Test Loss = 0.3500

Test Accuracy = 0.8600
```

This means:

```text
Test Loss = 0.35

Test Accuracy = 86%
```

The actual results depend on the dataset, model architecture and training process.

---

# 23. Model Prediction

We can use the trained model to predict new employee records.

```python
y_pred_prob = model.predict(x_test_scaled)
```

This returns probabilities.

Convert probabilities into binary predictions:

```python
y_pred = (y_pred_prob >= 0.5).astype(int).ravel()
```

Example:

```text
Probability = 0.85

Predicted Class = 1

Probability = 0.20

Predicted Class = 0
```

If we want to convert the numerical predictions back into original labels:

```python
y_pred_labels = lb.inverse_transform(y_pred)
```

Now:

```text
0 → No

1 → Yes
```

---

# 24. Important Evaluation Metrics for Classification

Accuracy alone may not be sufficient, especially for imbalanced datasets.

Employee attrition datasets often have more employees who stay than employees who leave.

Therefore, we should also understand:

## Confusion Matrix

A confusion matrix shows correct and incorrect predictions for each class.

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | True Negative (TN) | False Positive (FP) |
| Actual Yes | False Negative (FN) | True Positive (TP) |

## Precision

Out of all positive predictions, how many were actually positive?

```text
Precision = TP / (TP + FP)
```

## Recall

Out of all actual positive cases, how many did the model correctly identify?

```text
Recall = TP / (TP + FN)
```

## F1 Score

Harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall)
     / (Precision + Recall)
```

Python implementation:

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
```

For employee attrition prediction, recall for employees who leave can be particularly important because missing an employee who is likely to leave may be costly.

---

# 25. Complete ANN Implementation

Here is the complete version of the basic implementation using the concepts discussed above.

```python
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv("Attrition.csv")

# 2. Handle Missing Values
df = df.dropna()

# 3. Separate Input and Target
x = df.drop(columns=['Attrition'])
y = df['Attrition']

# 4. Identify Categorical Features
categorical_features = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
    'Over18',
    'OverTime'
]

# 5. Encode Target
lb = LabelEncoder()
y_encoded = lb.fit_transform(y)

# 6. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 7. Encode Categorical Features
x_train = pd.get_dummies(
    x_train,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

x_test = pd.get_dummies(
    x_test,
    columns=categorical_features,
    drop_first=True,
    dtype=int
)

# Align testing columns with training columns
x_test = x_test.reindex(
    columns=x_train.columns,
    fill_value=0
)

# 8. Scale Features
sc = StandardScaler()

x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# 9. Build ANN
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(x_train_scaled.shape[1],)),

    tf.keras.layers.Dense(
        units=128,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        units=64,
        activation='relu'
    ),

    tf.keras.layers.Dense(
        units=1,
        activation='sigmoid'
    )
])

# 10. Compile Model
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
    nesterov=True
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 11. Train Model
history = model.fit(
    x_train_scaled,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1
)

# 12. Evaluate Model
loss, accuracy = model.evaluate(
    x_test_scaled,
    y_test
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 13. Predict
y_pred_prob = model.predict(x_test_scaled)

y_pred = (y_pred_prob >= 0.5).astype(int).ravel()

# 14. Classification Report
print(classification_report(y_test, y_pred))

# 15. Confusion Matrix
print(confusion_matrix(y_test, y_pred))
```

Note: This complete example uses a straightforward train/test encoding workflow. For production use, fit preprocessing transformations only on training data using a `ColumnTransformer` or Pipeline, and consider handling class imbalance.

---

# 26. Important ANN Interview Questions

### Q1. What is ANN?

ANN stands for Artificial Neural Network. It is a machine learning model inspired by the biological neural network of the human brain. It consists of interconnected artificial neurons that learn patterns from data.

### Q2. What are the three layers of ANN?

1. Input Layer
2. Hidden Layer
3. Output Layer

### Q3. What is a neuron?

A neuron is the basic computational unit of an artificial neural network. It receives inputs, multiplies them by weights, adds bias, applies an activation function and produces an output.

### Q4. What are weights and biases?

Weights determine the influence of input features, while biases provide an additional trainable parameter that shifts the neuron's activation.

### Q5. What is an activation function?

An activation function introduces non-linearity into the neural network so that it can learn complex patterns.

### Q6. What is Forward Propagation?

Forward propagation is the process of passing input data through the neural network to generate predictions.

### Q7. What is Backpropagation?

Backpropagation calculates gradients of the loss with respect to the network's trainable parameters using the chain rule.

### Q8. What is a Loss Function?

A loss function measures the difference between actual and predicted values.

### Q9. What is an Optimizer?

An optimizer updates weights and biases using gradients to minimize the loss function.

### Q10. What is Learning Rate?

Learning rate controls the step size used by the optimizer when updating model parameters.

### Q11. What is an Epoch?

One complete pass through the entire training dataset.

### Q12. What is Batch Size?

The number of training samples processed together in one batch.

### Q13. What is Dropout?

Dropout is a regularization technique that randomly sets a percentage of neuron outputs to zero during training to reduce overfitting.

### Q14. Why do we use ReLU?

ReLU introduces non-linearity and is computationally simple. It is commonly used in hidden layers.

### Q15. Why do we use Sigmoid?

Sigmoid converts output values into probabilities between 0 and 1, making it useful for binary classification.

### Q16. Why do we use Binary Cross Entropy?

It measures the prediction error for binary classification problems.

### Q17. What is the difference between Validation and Testing Data?

Validation data helps monitor and tune the model during development. Testing data is reserved for final performance evaluation.

### Q18. Why do we use StandardScaler?

It standardizes numerical features using the training mean and standard deviation, helping optimization work more effectively when feature scales differ.

### Q19. What is TensorFlow?

TensorFlow is an open-source machine learning framework used to build, train and deploy machine learning and deep learning models.

### Q20. What is Keras?

Keras is a high-level API that makes building and training neural networks easier.

---

# 27. Quick Revision – Remember Forever

| Concept | Easy Definition |
|---|---|
| Deep Learning | Subset of ML using multi-layer neural networks. |
| ANN | Network of interconnected artificial neurons. |
| Neuron | Basic computational unit of ANN. |
| Input Layer | Receives input features. |
| Hidden Layer | Learns patterns from input information. |
| Output Layer | Produces final prediction. |
| Weight | Controls the influence of an input. |
| Bias | Additional trainable parameter. |
| Activation Function | Introduces non-linearity. |
| ReLU | Returns max(0, x). |
| Sigmoid | Converts output into a value between 0 and 1. |
| Softmax | Converts multiple scores into class probabilities. |
| Forward Propagation | Generates predictions. |
| Loss Function | Measures prediction error. |
| Backpropagation | Calculates gradients. |
| Optimizer | Updates weights and biases. |
| Learning Rate | Controls update step size. |
| Epoch | One complete pass through training data. |
| Batch | Small group of training samples. |
| Iteration | One batch-processing and parameter-update step. |
| Overfitting | Learns training data too specifically. |
| Underfitting | Fails to learn sufficient patterns. |
| Dropout | Randomly drops neuron outputs during training. |
| TensorFlow | Deep learning framework. |
| Keras | High-level neural network API. |
| StandardScaler | Standardizes numerical features. |
| One-Hot Encoding | Converts categorical values into binary columns. |

---

# 28. Final ANN Learning Workflow

Remember this complete process:

```text
              DATASET
                 |
                 v
          DATA PREPROCESSING
                 |
                 v
          FEATURE ENCODING
                 |
                 v
           TRAIN-TEST SPLIT
                 |
                 v
           FEATURE SCALING
                 |
                 v
          BUILD ANN MODEL
                 |
                 v
          FORWARD PROPAGATION
                 |
                 v
           LOSS CALCULATION
                 |
                 v
          BACKPROPAGATION
                 |
                 v
           OPTIMIZER UPDATE
                 |
                 v
            NEXT EPOCH
                 |
                 v
           MODEL TRAINING
                 |
                 v
          MODEL EVALUATION
                 |
                 v
            PREDICTION
```

## Most Important Concept to Understand

The entire ANN training process revolves around these four operations:

1. **Forward Propagation:** Make a prediction.
2. **Loss Function:** Calculate prediction error.
3. **Backward Propagation:** Calculate gradients.
4. **Optimizer:** Update weights and biases.

This cycle repeats during training to help the model learn better patterns.

**Final Remember Forever Definition:**

Artificial Neural Network is a computational model inspired by the human brain that learns patterns from data using interconnected neurons, weights, biases, activation functions, forward propagation and backpropagation.