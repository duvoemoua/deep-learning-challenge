# deep-learning-challenge
deep-learning-challenge
# Charity Donation Predictor

## Overview
This project uses a **deep neural network (DNN)** to predict whether charity donation applications will be successful. The dataset includes various details about organizations applying for funding, and the goal is to classify whether their requests will be approved.

## Dataset
The dataset used is `charity_data.csv`, retrieved from a cloud URL.

### Key Features:
- **APPLICATION_TYPE**: Type of application submitted
- **CLASSIFICATION**: Category of the organization
- **IS_SUCCESSFUL** (Target Variable): 1 if funding was approved, 0 otherwise

## Data Preprocessing
1. **Removed Unnecessary Columns**:
   - Dropped `EIN` and `NAME` since they don’t help with predictions.

2. **Handled Categorical Data**:
   - Grouped rare `APPLICATION_TYPE` values (less than 200 occurrences) into "Other".
   - Grouped rare `CLASSIFICATION` values (less than 100 occurrences) into "Other".
   - Used `pd.get_dummies()` for one-hot encoding.

3. **Scaled Features**:
   - Used `StandardScaler` to normalize numerical features.

4. **Split Data**:
   - 80% training data, 20% test data (`train_test_split()`).

## Model Architecture
Built using **TensorFlow/Keras**:

- **Input Layer**: Matches the number of input features.
- **Hidden Layers**:
  - First Layer: 8 neurons, ReLU activation
  - Second Layer: 16 neurons, ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation (for binary classification)

## Model Training
- **Loss Function**: `binary_crossentropy` (since it's a binary classification problem)
- **Optimizer**: `adam`
- **Metric**: `accuracy`
- **Epochs**: 50
- **Batch Size**: 32
- **Validation Data**: `X_test_scaled` and `y_test` to track progress.

## Model Performance
- **Accuracy**: ~72.83% on test data
- **Loss**: 0.5512

## Saving the Model
The trained model is saved in the recommended **Keras format**:
```python
nn.save("charity_model.keras")
```

## Possible Improvements
- **Add more layers/neurons** for better learning.
- **Tune hyperparameters** (learning rate, batch size, dropout, etc.).
- **Feature engineering** to extract more useful patterns.
- **Try different architectures** (CNNs, RNNs, etc., if relevant).

## How to Use
To train and test the model, run:
```python
python charity_dnn.py
```
To load a saved model and make predictions:
```python
from tensorflow.keras.models import load_model
loaded_model = load_model("charity_model.keras")
predictions = loaded_model.predict(X_test_scaled)
```

## Dependencies
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib (for visualizations, if needed)



