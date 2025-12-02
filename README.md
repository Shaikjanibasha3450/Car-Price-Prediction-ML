# Car Price Prediction with Machine Learning

## Project Overview
This project is a machine learning implementation for predicting car prices using various regression algorithms. It is part of the **AICTE OASIS INFOBYTE internship Task 3**.

## Objective
The price of a car depends on many factors like brand, features, horsepower, mileage, and more. This project aims to build and train machine learning models that can accurately predict car prices based on these features.

## Dataset
- **Source**: Kaggle - Car Price Prediction (Used Cars)
- **Size**: 205 records with 26 features
- **Target Variable**: Price
- **Features Include**: Car ID, Symboling, Car Name, Fuel Type, Aspiration, Door Number, etc.

## Models Used

### 1. Linear Regression
- **R² Score**: 0.8407
- **RMSE**: 3546.16
- **MAE**: 2136.78
- **MSE**: 12,575,228.83

### 2. Random Forest Regressor (Best Model)
- **R² Score**: 0.9553
- **RMSE**: 1877.99
- **MAE**: 1300.25
- **MSE**: 3,526,840.27
- **Number of Trees**: 100
- **Max Depth**: 10

## Key Findings
- **Random Forest significantly outperforms Linear Regression**
- Random Forest achieves an R² score of 0.9553, explaining ~95.53% of variance in car prices
- Reduced prediction error by ~47% compared to Linear Regression (RMSE: 1877.99 vs 3546.16)

## Project Structure
```
Car-Price-Prediction-ML/
├── README.md
├── Task3_CarPricePrediction.py
└── notebook.ipynb (Google Colab)
```

## Technologies Used
- **Python 3**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib & Seaborn** - Data visualization
- **Google Colab** - Development environment

## Installation

```bash
# Install required libraries
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

```python
# Import libraries and load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load and preprocess data
df = pd.read_csv('CarPrice.csv')
# ... preprocessing steps ...

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## Results Summary
The Random Forest model achieved excellent performance with:
- **95.53% accuracy** (R² Score)
- **Mean Error**: ±$1,300 on average
- **Root Mean Squared Error**: $1,878

## Future Improvements
- Feature engineering and selection
- Hyperparameter tuning
- Ensemble methods combining multiple models
- Cross-validation for better generalization
- Feature importance analysis

## Author
Shaikjanibasha3450

## License
MIT License

## Acknowledgments
- AICTE OASIS INFOBYTE Internship Program
- Kaggle for the dataset
- Google Colab for computational resources
