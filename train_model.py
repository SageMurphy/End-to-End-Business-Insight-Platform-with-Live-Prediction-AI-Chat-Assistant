import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import joblib

# 1. Generate some initial data for training
def generate_training_data(num_samples=1000):
    products = ['Electronics', 'Clothing', 'Books', 'Home Goods']
    data = []
    for _ in range(num_samples):
        product_type = np.random.choice(products)
        num_clicks = np.random.randint(1, 100)
        # Simulate price with some dependency on product and clicks
        if product_type == 'Electronics':
            price = round(100 + 2 * num_clicks + np.random.normal(0, 20), 2)
        elif product_type == 'Clothing':
            price = round(30 + 0.5 * num_clicks + np.random.normal(0, 10), 2)
        elif product_type == 'Books':
            price = round(10 + 0.1 * num_clicks + np.random.normal(0, 5), 2)
        else: # Home Goods
            price = round(50 + 1 * num_clicks + np.random.normal(0, 15), 2)
        data.append([product_type, num_clicks, price])
    df = pd.DataFrame(data, columns=['product_type', 'num_clicks', 'price'])
    return df

# 2. Load or generate training data
training_data = generate_training_data(num_samples=1000)

# 3. Separate features (X) and target (y)
X = training_data[['product_type', 'num_clicks']]
y = training_data['price']

# 4. Preprocessing for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['product_type'])],
    remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# Get the feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out(['product_type', 'num_clicks'])
X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Select the processed features for training
X_train = X_processed_df

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=42)

# 6. Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate the model (optional but good practice)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# 8. Save the trained model and the preprocessor
joblib.dump(model, 'price_prediction_model.joblib')
joblib.dump(preprocessor, 'price_preprocessor.joblib')

print("Trained Linear Regression model and preprocessor saved.")