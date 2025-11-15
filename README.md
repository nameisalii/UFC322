# UFC 322 Fight Outcome Predictor 🥊  
**Predicting Islam Makhachev vs Jack Della Maddalena and Valentina Shevchenko vs Zhang Weili**

## Overview
This project uses historical UFC fight data to train a binary classification model that predicts the winner of upcoming fights. We paid special attention to avoiding data leakage by creating a true temporal held-out test set – something rarely done properly in public UFC ML projects.

### Final Predictions – UFC 322 (November 15, 2025)
| Fighter 1              | Fighter 2           | Weight Class      | P(F1 Win) | P(F2 Win) | Predicted Winner         |
|------------------------|---------------------|-------------------|-----------|-----------|--------------------------|
| Jack Della Maddalena   | Islam Makhachev     | Welterweight      | 0.517     | 0.483     | **Jack Della Maddalena** |
| Valentina Shevchenko   | Zhang Weili         | Women's Flyweight | 0.580     | 0.420     | **Valentina Shevchenko** |

## 1. Data Loading and API Integration
- Started with the most complete public UFC dataset available on Kaggle.
- Manually split the original dataset into two completely separate CSV files:
  - `train_data.csv` → all fights up to mid-2024 (used for training + validation)
  - `real_test_set.csv` → only fights from late 2024 – 2025 (including UFC 322 matchups)
- This temporal split prevents data leakage and gives honest out-of-sample performance.

## 2. Cleaning and Merging Historical Data
- Removed cancelled bouts, exhibitions, and draws
- Standardized fighter names
- Handled missing values with fight-specific imputation
- Merged additional statistics scraped from UFCStats.com and Sherdog (career striking accuracy trends, recent form, etc.)

## 3. Exploratory Data Analysis (EDA) & Feature Engineering
- Visualized win rates by age, reach, height, stance, weight class
- Created high-impact features by combining columns:
  - Strike differential per minute
  - Takedown success differential
  - Significant strike absorption ratio
  - Recent win streak momentum
  - Title fight experience flag
  - Age difference impact
- Correlation heatmaps and feature importance analysis guided final selection

## 4. Model Training
Tested 5 algorithms with proper cross-validation:

| Model               | Train Acc | Train F1 | Val Acc | Val Precision | Val Recall | Val F1   |
|---------------------|-----------|----------|---------|---------------|------------|----------|
| XGBoost + Optuna    | 0.677     | 0.764    | 0.648   | 0.630         | 0.918      | **0.747** |
| Random Forest       | 0.812     | 0.848    | 0.656   | 0.652         | 0.842      | 0.735    |
| Logistic Regression | 0.659     | 0.719    | 0.645   | 0.662         | 0.765      | 0.710    |
| Gradient Boosting   | 0.808     | 0.840    | 0.642   | 0.664         | 0.747      | 0.703    |
| Decision Tree       | 0.795     | 0.826    | 0.594   | 0.630         | 0.692      | 0.660    |

**Winner: XGBoost with Optuna hyperparameter optimization**

## 5. Evaluation and Model Comparison
- Used stratified 5-fold CV on the training period
- Primary metric: F1-score (handles slight class imbalance)
- XGBoost + Optuna achieved the best balance between precision and recall

## 6. External Test Set Evaluation
The real held-out test set (fights the model has never seen) gave realistic performance and was used to generate the final UFC 322 predictions above.

Stat category predictions:
Category	Jack Della Maddalena	Islam Makhachev
SLpM (Strikes Landed/Min)	81.73%	18.27%
Str. Acc. (Striking Accuracy)	54.24%	45.76%
SApM (Strikes Absorbed/Min)	18.86%	81.14%
Str. Def. (Striking Defense)	54.24%	45.76%
TD Avg. (Takedown Average)	1.89%	98.11%
TD Acc. (Takedown Accuracy)	4.24%	95.76%
TD Def. (Takedown Defense)	4.24%	95.76%
Sub. Avg. (Submission Average)	8.22%	91.78%
Analysis: Jack Della Maddalena has advantages in striking volume, accuracy, and defense, plus physical advantages. Islam Makhachev has a large advantage in takedowns and submissions. The model slightly favors Jack Della Maddalena based on striking metrics.

Stat category predictions:
Category	Valentina Shevchenko	Zhang Weili
SLpM (Strikes Landed/Min)	43.36%	56.64%
Str. Acc. (Striking Accuracy)	80.35%	19.65%
SApM (Strikes Absorbed/Min)	56.96%	43.04%
Str. Def. (Striking Defense)	80.35%	19.65%
TD Avg. (Takedown Average)	41.00%	59.00%
TD Acc. (Takedown Accuracy)	43.60%	56.40%
TD Def. (Takedown Defense)	43.60%	56.40%
Sub. Avg. (Submission Average)	13.73%	86.27%
Analysis: Valentina Shevchenko has large advantages in striking accuracy and defense, plus physical advantages. Zhang Weili has advantages in striking volume, takedowns, and submissions. The model favors Valentina Shevchenko based on accuracy/defense and physical attributes.

## 7. Main Execution
Run `notebook.ipynb` end-to-end:
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
