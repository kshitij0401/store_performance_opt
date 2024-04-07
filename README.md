**Store Performance Classification Model**

This repository contains code for a store performance classification model built using Python and various machine learning techniques. The model aims to predict the performance of stores based on different features such as staff count, location, competition score, etc.

**Project Structure**

**Copy of Store_performance_classification.ipynb**: Jupyter Notebook containing the code for data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment.
**storedata.csv**: Dataset used for training and testing the model.
**transformers_and_model.pickle**: Pickle file containing transformers and the trained model for deployment.
**app.py**: Streamlit web app for interacting with the trained model.

**Dependencies/Requirements**

numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
streamlit

**Instructions for Use**

**Clone the repository**:

git clone https://github.com/kshitij0401/store_performance_opt.git

**Navigate to the project directory:**

cd kshitij0401/store_performance_opt

**Install dependencies**:

pip install -r requirements.txt

Run the Jupyter Notebook Copy of Store_performance_classification.ipynb to explore the code, train the model, and evaluate its performance.

**To deploy the model, run the Streamlit web app**:

streamlit run app.py

**About the Model**

The classification model in this project is built using logistic regression, decision tree, and random forest classifiers. It predicts the performance of stores based on features such as staff count, location, competition score, etc. The model has been trained, evaluated, and optimized using various techniques such as hypothesis testing, outlier detection, feature selection, model optimization, and model evaluation.

**Detailed Analysis**

**Hypothesis Testing**: The code performs an independent t-test to compare the means of two groups, specifically the competition number and competition score. It checks if there's a significant difference between these two groups.

**Outlier Detection using Box-plot & IQR**: The Interquartile Range (IQR) method is used to detect outliers in the dataset. Any data points falling below the lower bound or above the upper bound are considered outliers and removed from the dataset.

**Feature Selection using Lasso**: Lasso regression is employed for feature selection. It helps in identifying the most important features that contribute to the prediction of store performance.

**Model Optimization using GridSearchCV**: GridSearchCV is utilized for hyperparameter tuning of the logistic regression model. It systematically searches through a grid of hyperparameters to find the best combination that yields the highest accuracy.

**Model Evaluation using Confusion Matrix**: After training the model, it is evaluated using a confusion matrix. The confusion matrix helps in assessing the performance of the model by showing the number of true positives, true negatives, false positives, and false negatives.

**Using Pickle to Load Model and Transformations**: The trained model along with any necessary transformations (such as label encoders, scalers) are saved using Pickle. This allows for easy loading of the model and transformations for deployment or further use.

**Frontend Web App Creation using Streamlit**: A simple frontend web application is created using Streamlit to interact with the trained model. Users can input data and get predictions for store performance directly from the web app interface.

**Dataset**
The dataset (storedata.csv) contains information about different stores including features like staff count, location, competition score, etc. The target variable is the performance of the stores.

