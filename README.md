# Kidney-Disease-Prediction-ML
This code performs an analysis of a chronic kidney disease dataset, followed by training a Random Forest classifier for predicting the presence of the disease (represented by the 'Class' column). Here's a step-by-step explanation of what the code does:

Data Exploration and Cleaning:

    The dataset is loaded using pd.read_csv.
    Basic information about the dataset is displayed using .info(), .describe(), and .shape methods to inspect the data types, statistics, and dimensions.
    The first 10 rows (df.head(10)) and the last 10 rows (df.tail(10)) are displayed for a quick overview.
    Missing values are checked with df.isnull().sum(), and duplicate entries are checked with df.duplicated().sum().

Class Distribution and Visualization:

    The distribution of the 'Class' column (which indicates the presence of kidney disease: 1 for healthy, 0 for diseased) is visualized using a countplot from Seaborn.
    A histplot of the 'Bp' (blood pressure) feature is shown, broken down by the 'Class' label to observe the distribution of blood pressure across both classes.

Handling Imbalanced Classes:

    The dataset is imbalanced (with fewer diseased cases). To address this, the minority class (0, representing diseased cases) is upsampled using resample from sklearn.utils.
    The number of samples in the minority class is increased to 250, and the majority and upsampled minority class are combined to create a balanced dataset (df_final).
    A countplot is shown again to confirm the class balance after upsampling.

Outlier Detection:

    Outliers are detected using the Z-score method. For each feature, the Z-scores are calculated and stored in z.
    Any rows with Z-scores greater than 3 (considered outliers) are removed from the dataset (df_clean).

Correlation Analysis:

    The correlation matrix of the features is visualized using a heatmap to identify potential relationships between variables.

Feature Selection and Model Preparation:

    The feature set (X) is created by dropping the target variable 'Class', while the target variable (y) is selected as the 'Class' column.
    The dataset is split into training and test sets (80/20 split) using train_test_split.

Random Forest Classifier:

    A RandomForestClassifier is trained on the training set and evaluated on the test set.
    The accuracy score of the model is printed along with the confusion matrix and classification report, which includes precision, recall, F1-score, and support for each class.

Key Outputs:

    Accuracy Score: Indicates the percentage of correct predictions.
    Confusion Matrix: Provides insights into false positives, false negatives, true positives, and true negatives.
    Classification Report: Shows detailed classification metrics (precision, recall, F1-score) for both classes.

Possible Improvements:

    Hyperparameter Tuning: You can improve model performance by tuning hyperparameters (e.g., the number of trees, max depth).
    Cross-Validation: To get more reliable results, cross-validation could be implemented to assess the model's generalization ability.

