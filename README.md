# Sentiment Analysis of Financial News Articles

## Classifying Articles into Positive and Negative Sentiments



---

## Objective

- **Classify** financial news articles based on sentiment.
- **Distinguish** between positive and negative sentiments.
- **Compare** neural network and boosting models for accurate classification.

---

## Dataset Overview

- **Dataset:** Collection of financial news articles.
- **Columns:**
  - `Sentiment`: Label indicating sentiment (`positive` or `negative`).
  - `Sentence`: The textual content of the news article.
- **Total Samples:** Approximately 2000.
- **Preprocessing:** Cleaning, tokenization, and encoding of text data.
- **Imbalanced Data:** Addressed to ensure balanced class distribution.

---

## Methodology

1. **Data Cleaning and Preprocessing**
   - Removal of HTML tags, special characters, and stopwords.
   - Tokenization and normalization of text.
   - Encoding textual data for model compatibility.

2. **Handling Class Imbalance via Oversampling**
   - Upsampling the minority class to balance the dataset.
   - Ensuring equal representation of positive and negative sentiments.

3. **Feature Extraction using Tokenization and TF-IDF**
   - Converting text data into numerical features.
   - Utilizing Tokenization and Term Frequency-Inverse Document Frequency (TF-IDF) for feature representation.

4. **Model Building**
   - **Convolutional Neural Network (CNN)**
   - **XGBoost Classifier**
   - **Random Forest Classifier**

5. **Hyperparameter Tuning**
   - **CNN:** Tuned dropout rates using Keras-Tuner's RandomSearch.
   - **XGBoost:** Employed RandomizedSearchCV for optimizing learning rate, max-depth, n_estimators, etc.
   - **Random Forest:** Optimized parameters like n_estimators, max_depth, etc.
   - **Objective:** Enhance model performance and prevent overfitting.

6. **Evaluation Metrics**
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1 Score**
   - **AUC (Area Under the Curve)**

---

## CNN Architecture

- **Embedding Layer:** Converts words into dense vectors of fixed size.
- **Convolutional Layer:** Captures local features in the text.
- **MaxPooling Layer:** Reduces dimensionality and captures the most significant features.
- **Dropout Layers:** Prevent overfitting by randomly dropping neurons during training.
- **Dense Layers:** Learns complex patterns in the data.
- **Global Max Pooling:** Aggregates feature maps into a single vector.
- **Output Layer:** Sigmoid activation for binary classification.

---

## Hyperparameter Tuning

- **CNN Model:**
  - Tuned dropout rates using Keras-Tuner's RandomSearch to find optimal values that balance model complexity and generalization.
  
- **XGBoost:**
  - Utilized RandomizedSearchCV to optimize hyperparameters such as learning rate, max-depth, number of estimators, and minimum child weight.
  
- **Random Forest:**
  - Optimized parameters like the number of estimators and maximum depth to enhance performance and reduce overfitting.
  
- **Objective:**
  - Enhance overall model performance.
  - Prevent overfitting through appropriate hyperparameter settings.

---

## Model Performance Comparison

| Model          | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score | AUC  |
|----------------|----------------|---------------|-----------|--------|----------|------|
| Random Forest  | 0.72           | 0.67          | 0.67      | 1.00   | 0.80     | 0.88 |
| XGBoost        | 0.99           | 0.86          | 0.88      | 0.94   | 0.91     | 0.80 |
| CNN            | 0.99           | 0.85          | 0.83      | 0.96   | 0.89     | 0.92 |


## Evaluation Metrics for Models

Detailed evaluation metrics including Accuracy, Precision, Recall, F1 Score, and AUC for each model are provided in the **Model Performance Comparison** table above.

---

## Novelties of the Approach

- **Combined Deep Learning with Traditional ML Models:**
  - Integrated Convolutional Neural Networks (CNN) with boosting models like XGBoost and ensemble methods like Random Forest.
  
- **Advanced Hyperparameter Tuning:**
  - Implemented sophisticated hyperparameter tuning techniques using Keras-Tuner and RandomizedSearchCV to achieve optimal model performance.
  
- **Comprehensive Evaluation Metrics:**
  - Applied a wide range of evaluation metrics to thoroughly assess model robustness and effectiveness.
  
- **Effective Handling of Class Imbalance:**
  - Addressed class imbalance through oversampling techniques to improve classification accuracy and model fairness.

---

## Conclusion

- **Performance Achievement:**
  - Achieved over 80% accuracy in sentiment classification of financial news articles.
  
- **Model Comparison:**
  - Performances of CNN and XGBoost are comparable, with XGBoost performing marginally better.
  
- **Hyperparameter Tuning Impact:**
  - Effective hyperparameter tuning significantly enhanced model performance. Further tuning could enable CNN to outperform XGBoost.

---
