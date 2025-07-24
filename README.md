Mobile Phone Price Range Prediction
Introduction
This project aims to develop a machine learning model to predict the price range of mobile phones based on their technical specifications. In today's competitive mobile market, accurately classifying devices into appropriate price categories is crucial for manufacturers, retailers, and consumers. This project utilizes a dataset containing various features of mobile phones, such as battery power, RAM, screen dimensions, and other technical attributes, to build a predictive model. The core problem addressed is multiclass classification: assigning each mobile phone to one of several predefined price ranges (e.g., low cost, medium cost, high cost, very high cost) based on its characteristics. The insights gained from this project can be valuable for market analysis, product positioning, and informed decision-making.
Data Loading and Exploration
The dataset containing the technical specifications and price ranges of mobile phones was loaded from a CSV file named dataset.csv.

The dataset consists of 2000 entries (rows) and 21 columns. As shown by df.info(), the dataset contains predominantly integer (int64) data types, with two columns (clock_speed and m_dep) being floating-point (float64). There are no missing values in any of the columns.

The first few rows of the dataset, as shown by df.head(), provide a glimpse into the data structure and content:

battery_power	blue	clock_speed	dual_sim	fc	four_g	int_memory	m_dep	mobile_wt	n_cores	pc	px_height	px_width	ram	sc_h	sc_w	talk_time	three_g	touch_screen	wifi	price_range
842	0	2.2	0	1	0	7	0.6	188	2	10	20	756	2549	9	7	19	0	0	1	1
1021	1	0.5	1	0	1	53	0.7	136	3	15	905	1988	2631	17	3	7	1	1	0	2
563	1	0.5	1	2	1	41	0.9	145	5	11	1263	1716	2603	11	2	9	1	1	0	2
615	1	2.5	0	0	0	10	0.8	131	6	9	1216	1786	2769	16	8	11	1	0	0	2
1821	1	1.2	0	13	1	44	0.6	141	2	14	1208	1212	1411	8	2	15	1	1	0	1
The distribution of the target variable, price_range, shows that the dataset is balanced across the four price categories:
Reasoning: I need to include the markdown representation of the plots generated in the executed code cell: the target distribution plot, the box plots for selected features vs price range, and the correlation heatmap.

Distribution of Mobile Phone Price Range

Box plots illustrate the relationship between key features and the price range:

RAM vs Price Range: RAM vs Price Range

Battery Power vs Price Range: battery_power vs Price Range

Pixel Height vs Price Range: px_height vs Price Range

Pixel Width vs Price Range: px_width vs Price Range

Internal Memory vs Price Range: int_memory vs Price Range

Finally, a correlation heatmap provides insights into the relationships between different features:

Feature Correlation Heatmap

These visualizations help in understanding the data distribution, the relationship between features and the target variable, and the correlations among features, which is crucial for subsequent model building.

Feature Engineering
Based on the provided code, no explicit feature engineering steps were performed. The models were trained directly on the raw features available in the dataset.

Model Selection and Training
For this classification task, two popular and powerful ensemble learning models were selected:

Random Forest Classifier: An ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. It is known for its robustness and ability to handle non-linear relationships.
XGBoost Classifier (Extreme Gradient Boosting): An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework and is often a top choice for structured data problems due to its performance.
Before training the models, the dataset was split into features (X) and the target variable (y), which is the price_range. The data was then partitioned into training and testing sets using the train_test_split function from sklearn.model_selection. An 80/20 split was used, with 80% of the data allocated for training and 20% for testing. To ensure that the distribution of price ranges was similar in both sets, the stratify=y parameter was used. A random_state of 42 was set for reproducibility of the split.

The selected models were then trained (fit) on the training data (X_train, y_train).

To obtain a more reliable estimate of the models' performance and assess their generalization capabilities, 5-fold cross-validation was performed on the entire dataset (X, y) for both the Random Forest and XGBoost classifiers using cross_val_score. This process involves splitting the data into 5 folds, training the model on 4 folds, and evaluating on the remaining fold, repeating this 5 times with each fold serving as the test set once. The average cross-validation score provides a robust measure of the model's expected performance on unseen data.

Model Evaluation
Random Forest Classifier
The Random Forest model achieved an accuracy of 88% on the test set.

Classification Report:

Price Range	Precision	Recall	F1-Score	Support
0	0.96	0.96	0.96	100
1	0.82	0.84	0.83	100
2	0.81	0.79	0.80	100
3	0.93	0.93	0.93	100
Accuracy			0.88	400
Macro Avg	0.88	0.88	0.88	400
Weighted Avg	0.88	0.88	0.88	400
Confusion Matrix:

Model Evaluation
Random Forest Classifier
The Random Forest model achieved an accuracy of 88% on the test set.

Reasoning: I will generate and display the confusion matrix for the Random Forest model as an image and then include the evaluation metrics for the XGBoost classifier and the cross-validation scores.

0	1	2	3
0	96	4	0	0
1	5	84	11	0
2	0	14	79	7
3	0	0	7	93
XGBoost Classifier
The XGBoost model achieved a higher accuracy of 93.5% on the test set.

Classification Report:

Price Range	Precision	Recall	F1-Score	Support
0	0.97	1.00	0.99	100
1	0.94	0.93	0.93	100
2	0.87	0.90	0.89	100
3	0.96	0.91	0.93	100
Accuracy			0.94	400
Macro Avg	0.94	0.94	0.93	400
Weighted Avg	0.94	0.94	0.93	400
Confusion Matrix:

Reasoning: I will display the confusion matrix for the XGBoost model as an image and then include the cross-validation scores for both models to complete the model evaluation section.


	0	1	2	3
0	100	0	0	0
1	3	93	4	0
2	0	6	90	4
3	0	0	9	91
Cross-Validation Scores
To further assess the models' performance and generalization ability, 5-fold cross-validation was conducted:

Random Forest CV Score: 0.8780
XGBoost CV Score: 0.9075
The cross-validation scores generally align with the test set accuracies, indicating that both models perform reasonably well and generalize to unseen data, with XGBoost showing a slightly better average performance across different folds.

Feature Importance (Random Forest)
Feature importance in a Random Forest model indicates the relative contribution of each feature to the model's predictive accuracy. It is calculated based on how much each feature decreases the Gini impurity (for classification) or the mean squared error (for regression) across all trees in the forest. Features that lead to a greater reduction in impurity/error are considered more important.

Based on the importances variable calculated from the trained Random Forest model, the features have varying degrees of importance in predicting the mobile phone price range:

Feature	Importance
ram	0.4808
battery_power	0.0730
px_height	0.0560
px_width	0.0561
int_memory	0.0348
talk_time	0.0319
pc	0.0292
clock_speed	0.0289
sc_w	0.0278
sc_h	0.0277
fc	0.0261
m_dep	0.0250
n_cores	0.0238
mobile_wt	0.0390
blue	0.0073
touch_screen	0.0073
dual_sim	0.0069
four_g	0.0068
wifi	0.0065
three_g	0.0052
The most important feature by a significant margin is RAM, with an importance score of approximately 0.48. This suggests that the amount of Random Access Memory is the strongest predictor of a mobile phone's price range in this dataset. Other relatively important features include battery_power, px_height, px_width, and int_memory.

Conversely, features like three_g, wifi, four_g, dual_sim, touch_screen, and blue have very low importance scores, indicating they contribute minimally to the price range prediction according to this model.

The bar plot below visually represents the feature importance scores, sorted from least to most important:

Random Forest Feature Importance

As visually confirmed by the plot, RAM stands out as the most influential feature, reinforcing its critical role in determining the price range. The features with very low importance are clustered at the bottom of the plot, clearly showing their limited impact on the model's predictions. This analysis of feature importance is valuable for understanding which aspects of a mobile phone's specifications are most indicative of its price.

Best Model
Based on the evaluation metrics, the XGBoost classifier is the best-performing model for this mobile phone price range prediction task.

Here's a comparison of the key performance metrics:

Metric	Random Forest	XGBoost
Test Set Accuracy	88.0%	93.5%
CV Score (5-fold)	0.8780	0.9075
The XGBoost model demonstrates superior performance in both test set accuracy and cross-validation score. Its higher accuracy on the test set indicates that it makes more correct predictions on unseen data. Furthermore, its higher cross-validation score suggests better generalization capabilities, meaning it is more likely to perform well on new, unseen data compared to the Random Forest model. The combination of higher predictive power and better generalization makes the XGBoost classifier the clear choice for this problem.

Conclusion
This project successfully developed and evaluated machine learning models for predicting mobile phone price ranges based on technical specifications. The dataset, comprising 2000 entries and 20 features, was explored to understand the distribution of the target variable and the relationships between features, revealing that RAM is a particularly strong predictor of price range.

Two classification models, Random Forest and XGBoost, were trained and evaluated using a train-test split and cross-validation. The evaluation metrics, including accuracy, classification reports, and confusion matrices, demonstrated that the XGBoost model outperformed the Random Forest model, achieving a higher test set accuracy of 93.5% and a better 5-fold cross-validation score of 0.9075. This indicates that XGBoost is more effective in accurately classifying mobile phones into their respective price ranges based on the provided data.

Future Work and Improvements
Several avenues can be explored to further enhance the model and the project:

Hyperparameter Tuning: While default parameters were used for the models, extensive hyperparameter tuning for both Random Forest and XGBoost could potentially improve performance further. Techniques like GridSearchCV or RandomizedSearchCV could be employed.
Feature Engineering: Exploring the creation of new features from existing ones (e.g., pixel density from px_height and px_width, or battery life estimates from battery_power and talk_time) might provide additional predictive power.
Exploring Other Models: Investigating other classification algorithms such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), or neural networks could reveal models that perform even better.
Ensemble Methods: Building more sophisticated ensemble models, perhaps stacking or blending the predictions of multiple high-performing models, could lead to marginal improvements.
Data Augmentation/Collection: While the current dataset is balanced, collecting more data or using data augmentation techniques (if applicable and meaningful for this type of data) could potentially improve robustness.
Model Deployment: Deploying the trained XGBoost model as a web service or within an application would allow for real-world predictions on new mobile phone specifications.
Interpretability: While XGBoost performs well, exploring techniques for model interpretability (e.g., SHAP values) could provide deeper insights into how specific features influence the predicted price range for individual instances.

Summary:
Data Analysis Key Findings
The project's goal is to predict the price range of mobile phones using their technical specifications, which is a multiclass classification problem.
The dataset contains 2000 entries and 21 columns, with no missing values. The target variable, price_range, is balanced across its four categories.
Feature exploration reveals that RAM is the most significant predictor of price range, with an importance score of approximately 0.48 from the Random Forest model. Other influential features include battery_power, px_height, and px_width.
Two models were evaluated: Random Forest and XGBoost. The XGBoost classifier performed better, achieving a test set accuracy of 93.5% and a 5-fold cross-validation score of 0.9075, compared to the Random Forest's 88% accuracy and 0.8780 CV score.
Features such as three_g, wifi, four_g, dual_sim, and touch_screen have minimal impact on the price prediction, according to the feature importance analysis.
Insights or Next Steps
Focus on Key Features: Given the overwhelming importance of RAM and other key hardware specifications, future data collection or feature engineering efforts should prioritize these areas to potentially enhance model performance further.
Model Optimization and Deployment: The high-performing XGBoost model should be fine-tuned using hyperparameter optimization techniques like GridSearchCV and then deployed as a web service or API for real-world application in price estimation tools.

