# Predicting-Movie-Rental-Durations

# Overview/Introduction

A DVD rental company is seeking to optimise its inventory planning by predicting how many days a customer will rent a DVD. Accurate predictions will help the company manage its inventory more efficiently and improve customer satisfaction. This project uses regression models to predict rental durations based on features such as rental rate, movie length, release year, and special features. The goal is to develop a model with a Mean Squared Error (MSE) of 3 or less on the test set.

# Objectives

1. Build regression models to predict the number of days a customer will rent a DVD.
2. Evaluate model performance using Mean Squared Error (MSE).
3. Identify the best-performing model that meets the company's requirement of an MSE of 3 or less.
4. Provide actionable insights to help the company optimize inventory planning.

# Data Source

The dataset, rental_info.csv, contains the following columns:
  - rental_date: The date and time the customer rents the DVD.
  - return_date: The date and time the customer returns the DVD.
  - amount: The amount paid by the customer for renting the DVD.
  - amount_2: The square of the rental amount.
  - rental_rate: The rate at which the DVD is rented.
  - rental_rate_2: The square of the rental rate.
  - release_year: The year the movie was released.
  - length: Length of the movie in minutes.
  - length_2: The square of the movie length.
  - replacement_cost: The cost to replace the DVD.
  - special_features: Special features included with the DVD (e.g., trailers, deleted scenes).
  - NC-17, PG, PG-13, R: Dummy variables indicating the movie's rating.

# Tools Used

- Python Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Regression Models: Linear Regression, Lasso Regression, Random Forest Regression.
- Model Evaluation: Mean Squared Error (MSE).
- Hyperparameter Tuning: RandomizedSearchCV for optimizing Random Forest parameters.

# Insights

1. Feature Engineering:
    - The rental duration in days was calculated by subtracting the rental date from the return date.
    - Dummy variables were created for special features like "Deleted Scenes" and "Behind the Scenes."
2. Model Performance:
    - Random Forest Regression outperformed Linear Regression and Lasso Regression, achieving an MSE of 2.22, which meets the company's requirement.
    - Lasso Regression was used for feature selection, identifying the most important features for predicting rental duration.
3. Hyperparameter Tuning:
    - RandomizedSearchCV was used to find the optimal hyperparameters for the Random Forest model, resulting in 51 estimators and a max depth of 10.

# Key Findings

1. Best Model:
    - The Random Forest Regression model achieved the lowest MSE (2.22), making it the best model for predicting rental durations.
2. Feature Importance:
    - Features such as rental rate, movie length, and special features were found to be significant predictors of rental duration.
3. Model Comparison:
    - Linear Regression had an MSE of 4.81, while Random Forest Regression significantly improved performance with an MSE of 2.22.

# Recommendations

1. Deploy the Random Forest Model:
    - Use the Random Forest model for predicting rental durations, as it meets the company's MSE requirement.
2. Focus on Key Features:
    - Prioritize features like rental rate, movie length, and special features in future data collection and analysis.
3. Further Model Optimization:
    - Explore additional ensemble methods (e.g., Gradient Boosting) or neural networks to further reduce the MSE.
4. Inventory Planning:
    - Use the model's predictions to optimize inventory levels, reduce overstocking, and improve customer satisfaction.

# How to Use This Repository

1. Clone the repository.
2. Install the required Python libraries (pandas, numpy, scikit-learn, matplotlib, seaborn).
3. Run the Jupyter Notebook (Predicting Movie Rental Durations.ipynb) to reproduce the analysis.
4. Explore the dataset and modify the code to test additional models or features.
