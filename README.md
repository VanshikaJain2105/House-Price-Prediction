# 🏡 House Price Prediction Using Machine Learning

Predicting house prices using regression algorithms, exploring the factors that influence property values, and comparing different machine learning models to find the best-performing approach.

## 📌 Project Overview

The goal of this project is to predict median housing values in Boston-area neighborhoods using historical housing data.

Unlike classification problems, where the output belongs to a particular category, house price prediction is a **regression problem** because we predict a continuous numerical value.

In this project, I explored the dataset, analyzed relationships between housing features and prices, trained multiple regression models, performed hyperparameter tuning, and evaluated the final model on unseen test data.

## 📊 Dataset

**Dataset:** Boston Housing Dataset

The dataset contains:

- **506 observations**
- **13 original input features**
- **1 target variable (`medv`)**

The target variable, `medv`, represents the median value of owner-occupied homes in thousands of historical US dollars.

### Important Features

| Feature | Description |
|---|---|
| `crim` | Per capita crime rate |
| `zn` | Proportion of residential land zoned for large lots |
| `indus` | Proportion of non-retail business land |
| `chas` | Charles River dummy variable |
| `nox` | Nitric oxide concentration |
| `rm` | Average number of rooms per dwelling |
| `age` | Proportion of older owner-occupied buildings |
| `dis` | Weighted distance to employment centres |
| `rad` | Accessibility to radial highways |
| `tax` | Property tax rate |
| `ptratio` | Pupil–teacher ratio |
| `lstat` | Historical socioeconomic-status indicator |
| `medv` | Median housing value (target) |

**Note:** The original dataset also contains `b`, a race-derived variable. This feature was excluded from the analysis and model training.

The Boston Housing dataset is historical and contains outdated socioeconomic measures. It is used here for educational purposes and is not suitable for modern real-estate valuation.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## 🔍 Project Workflow

### 1. Data Understanding and Cleaning

Started by exploring the structure and quality of the dataset.

- Checked dataset dimensions and data types.
- Examined missing values.
- Checked for duplicate records.
- Analyzed the target variable.
- Identified 16 observations at the dataset's maximum reported housing value of 50.0.

**Findings:**

The dataset contains no missing values or duplicate rows.

### 2. Exploratory Data Analysis (EDA)

Used different visualizations to understand housing prices and their relationships with other variables.

**Histogram and Boxplot**

Explored the distribution of neighborhood housing values and identified potential outliers.

Most housing values were concentrated around $15,000–$25,000, with a noticeable spike at the historical reporting cap of $50,000.

**Scatterplots**

Compared housing values against three important features:

- **Rooms (`rm`):** More rooms generally corresponded to higher housing values.
- **LSTAT (`lstat`):** Higher values were generally associated with lower housing values.
- **Pupil–Teacher Ratio (`ptratio`):** Higher ratios showed a moderate negative relationship with housing values.

**Correlation Heatmap**

Examined how numerical features relate to the target and to each other.

Two particularly strong correlations with `medv` were:

| Feature | Correlation |
|---|---:|
| `lstat` | -0.738 |
| `rm` | +0.695 |

These patterns suggested that both features could be useful predictors.

### 3. Feature Selection and Data Splitting

Separated the dataset into:

- **X:** 12 input features
- **y:** Median housing value (`medv`)

Excluded `medv` from the input features to prevent target leakage and removed the race-derived `b` column.

Split the dataset into:

- **80% training data**
- **20% testing data**

The test set was reserved for evaluating the final selected model.

### 4. Model Training

Implemented and compared five regression approaches:

| Model | Purpose |
|---|---|
| Dummy Regressor | Establish a baseline using the median target value |
| Linear Regression | Learn linear relationships between inputs and price |
| Decision Tree Regressor | Predict using decision-based splits |
| Random Forest Regressor | Combine multiple decision trees |
| Gradient Boosting Regressor | Build trees sequentially to reduce prediction errors |

### 5. Cross-Validation and Model Comparison

Used **5-Fold Cross-Validation** on the training data to compare model performance.

Evaluated each model using:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger mistakes more heavily.
- **R² Score:** Measures how much variation in the target is explained by the model.

#### Cross-Validation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Gradient Boosting** | **2.318** | **3.556** | **0.847** |
| Random Forest | 2.380 | 3.718 | 0.833 |
| Linear Regression | 3.558 | 4.974 | 0.707 |
| Decision Tree | 3.071 | 5.252 | 0.642 |
| Dummy Regressor | 6.696 | 9.363 | -0.024 |

*MAE and RMSE are measured in thousands of historical US dollars.*

**Gradient Boosting achieved the lowest cross-validation RMSE**, making it the strongest untuned model in this comparison.

### 6. Hyperparameter Tuning

Used `GridSearchCV` to optimize the Random Forest model.

Tuned the following hyperparameters:

- `n_estimators`: Number of trees.
- `max_depth`: Maximum depth of individual trees.
- `min_samples_leaf`: Minimum number of observations required in a leaf.

The tuned Random Forest achieved a cross-validation RMSE of approximately **3.702**.

However, Gradient Boosting still performed better in the model comparison.

This demonstrated that hyperparameter tuning does not necessarily make one algorithm better than every alternative.

## 🏆 Final Model Performance

**Selected Model: Gradient Boosting Regressor**

After selecting the best model using cross-validation, it was evaluated on the held-out test set.

| Evaluation Metric | Result |
|---|---:|
| **MAE** | **1.905** |
| **RMSE** | **2.496** |
| **R² Score** | **0.915** |

The model achieved an R² score of **0.915**, explaining approximately 91.5% of the variation in housing values within the test dataset.

Its mean absolute prediction error was approximately **$1,905 in historical dataset units**.

These results indicate good performance on the selected test split.

## 📈 Prediction Error Analysis

To better understand the final model, I analyzed its predictions using:

**Actual vs Predicted Values**

Compared predicted housing values against actual values. Most predictions were reasonably close to the ideal diagonal line, indicating that the model captured many of the relationships in the dataset.

**Residual Analysis**

Calculated residuals using:

`Residual = Actual Value - Predicted Value`

Positive residuals indicate underprediction, while negative residuals indicate overprediction.

**Largest Prediction Errors**

Examined the five observations with the highest absolute prediction errors to understand where the model struggled.

Some of the largest errors involved observations near the dataset's upper reporting cap.

## 🔎 Feature Importance

Used permutation importance to investigate which features the final model relied on most.

This method measures how much prediction error increases when an individual feature is randomly shuffled.

The most influential features were:

| Feature | RMSE Increase When Shuffled |
|---|---:|
| `lstat` | 4.546 |
| `rm` | 3.371 |
| `dis` | 1.373 |
| `nox` | 0.847 |
| `ptratio` | 0.526 |

The analysis suggests that `lstat` and `rm` were particularly important to the final model.

Feature importance describes model reliance, not causal relationships.

## 🚀 How to Run the Project

1. Download or clone this repository.
2. Open the Jupyter Notebook in Google Colab or Jupyter.
3. Download the [Boston Housing dataset](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv).
4. Update the dataset path in the notebook if necessary.
5. Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

6. Run the notebook cells sequentially to reproduce the analysis, train the models, and inspect the results.

## 🧠 What I Learned

Through this project, I explored several important machine learning concepts:

- How regression differs from classification.
- How to understand numerical target distributions.
- How scatterplots and correlation heatmaps reveal relationships between features.
- Why a baseline model is useful when evaluating machine learning algorithms.
- How MAE, RMSE, and R² measure regression performance.
- How K-Fold Cross-Validation helps compare different models.
- How GridSearchCV helps optimize hyperparameters.
- Why a tuned model may not outperform other algorithms.
- How residual analysis helps identify prediction errors.
- How permutation importance helps interpret a trained model.

Most importantly, I learned that building a machine learning project is not just about training models. Understanding the dataset, choosing appropriate evaluation methods, and interpreting the results are equally important.

## ⚠️ Limitations

- The dataset contains only 506 observations.
- Housing values are historical and not adjusted for inflation.
- The target variable has an upper reporting cap.
- Some original features reflect outdated socioeconomic measurements.
- Results may vary with different data splits and model settings.

## 📚 Project Series

This project is part of my **100 Days, 100 Machine Learning Projects** challenge, where I'm building projects to strengthen my understanding of machine learning through practical implementation.
