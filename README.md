Logistic Regression: 

Accuracy: 

The model achieved an accuracy of 0.684 (68.4%), meaning it correctly classified 68.4% of the total data points. 

 

Precision, Recall, and F1-score: 

Class 0 (Casual Users) 

Precision: 0.56 → When the model predicts "Casual," it is correct 56% of the time. 

Recall: 0.06 → The model correctly identifies only 6% of actual casual users. 

F1-score: 0.11 → A very low F1-score indicates poor performance in predicting casual users. 

 

Class 1 (Members) 

Precision: 0.69 → When the model predicts "Member," it is correct 69% of the time. 

Recall: 0.98 → The model correctly identifies 98% of actual members. 

F1-score: 0.81 → A high F1-score indicates strong performance in predicting members. 

 

Confusion Matrix: 

False Negatives (Casual users misclassified as Members): 75,050 

False Positives (Members misclassified as Casual users): 4,043 

True Positives (Correctly classified Members): 166,144 

True Negatives (Correctly classified Casual users): 5,091 

 

The model is heavily biased toward predicting "Member," as seen in the high recall for class 1 (98%) but very low recall for class 0 (6%). 

 

Conclusion: 

The model has 68.4% accuracy, but it struggles with predicting Casual users (Class 0). It correctly identifies only 6% of actual casual users, while it performs well on Members (Class 1) with 98% recall and an F1-score of 0.81. The confusion matrix shows a strong bias toward predicting Members, misclassifying most Casual users as Members (75,050 false negatives). 

 

Key Takeaways: 

Recall is crucial if missing a certain class is costly, Precision is important when false positives are costly, F1-score is a balance between precision and recall. 

F1-score is the best metric to assess overall performance. 

The model is poor at identifying Casual users (low recall: 6%). 

Bias toward predicting Members needs correction. 

 

Random Forest: 

The Random Forest model achieved 69.5% accuracy, which is slightly better than the Logistic Regression model (68.4%). 

 

Key Observations: 

Casual users (Class 0) 

Recall improved (41% vs. 6% in Logistic Regression) → The model is better at identifying casual users. 

F1-score is 0.46, much better than 0.11 in Logistic Regression. 

False negatives (Casual misclassified as Members) reduced from 75,050 to 47,014. 

 
Members (Class 1) 

Recall decreased (83% vs. 98% in Logistic Regression) but is still high. 

F1-score is 0.79, slightly lower than 0.81 in Logistic Regression. 

 
Confusion Matrix Insights: 

Casuals correctly classified: 33,127 (Much better than 5,091 in Logistic Regression). 

Members correctly classified: 140,776 (Lower than 166,144 in Logistic Regression). 

False negatives (Casuals misclassified as Members) dropped significantly. 

 

Best Metric to Consider: 

F1-score is still the best metric, as it balances precision and recall. 

Improved Recall for Class 0 (Casual users) makes this model more balanced than Logistic Regression. 

 

Conclusion: 

Random Forest is a better model overall because: 

It significantly improves recall for casual users (from 6% → 41%). 

It still maintains good performance for members. 

The trade-off in member classification is acceptable for better balance. 

 

XGBoost Classification: 

The XGBoost model achieved 68.6% accuracy, which is slightly better than Logistic Regression (68.4%) but slightly worse than Random Forest (69.5%). 

Key Observations: 

Casual users (Class 0) 

Recall improved from 6% (Logistic Regression) to 13%, but it's still worse than 41% in Random Forest. 

F1-score is 0.20, better than 0.11 in Logistic Regression but much worse than 0.46 in Random Forest. 

False negatives (Casual misclassified as Members): 70,021 (Better than 75,050 in Logistic Regression, worse than 47,014 in Random Forest). 

 

Members (Class 1) 

Recall is 95%, higher than 83% in Random Forest, similar to 98% in Logistic Regression. 

F1-score is 0.80, slightly better than 0.79 in Random Forest. 

 

Confusion Matrix Insights: 

Casual users correctly classified: 10,120 (Better than 5,091 in Logistic Regression, worse than 33,127 in Random Forest). 

Members correctly classified: 161,534 (Better than Random Forest, worse than Logistic Regression). 

The model is still biased toward predicting Members, though slightly better than Logistic Regression. 

 

Best Metric to Consider: 

F1-score remains the best metric for overall balance. 

Recall for Class 0 is still low (13%), meaning it still struggles to classify casual users. 

 

Conclusion: 

XGBoost is not the best choice because: 

It still struggles with Class 0 (Casual users), just like Logistic Regression. 

Random Forest is more balanced in predicting both classes. 

It only improves slightly on Logistic Regression but does not outperform Random Forest. 

 

KNN Classification: 
The KNN model achieved 67.7% accuracy, which is: 

Better than Logistic Regression (68.4%) in some aspects but slightly worse in overall accuracy. 

Worse than Random Forest (69.5%) and XGBoost (68.6%). 

 

Key Observations: 

Casual users (Class 0) 

Recall is 15%, higher than 6% in Logistic Regression but worse than 41% in Random Forest. 

F1-score is 0.23, which is slightly better than 0.20 in XGBoost but much lower than 0.46 in Random Forest. 

False negatives (Casual misclassified as Members): 67,993, slightly better than 75,050 in Logistic Regression but worse than 47,014 in Random Forest. 

 

Members (Class 1) 

Recall is 92%, slightly lower than 95% in XGBoost but still quite high. 

F1-score is 0.80, the same as XGBoost and slightly below Random Forest. 

 

Confusion Matrix Insights: 

Casual users correctly classified: 12,148 (Better than Logistic Regression, worse than Random Forest). 

Members correctly classified: 157,349 (Decent but lower than XGBoost and Random Forest). 

Still biased towards predicting Members, similar to Logistic Regression and XGBoost. 

 

Best Metric to Consider: 

F1-score remains the most relevant to balance precision and recall. 

Random Forest still outperforms KNN in classifying both groups more effectively. 

 

Conclusion: 

KNN is not the best choice because: 

It struggles with Casual users (low recall, low F1-score). 

Random Forest is still the best model, offering a better balance. 

KNN might need feature scaling or tuning to perform better. 

 

Insights and Key Takeaways: 

Accuracy Ranking: 

Random Forest (69.5%) > XGBoost (68.6%) > Logistic Regression (68.4%) > KNN (67.7%). 

Random Forest has the highest accuracy, meaning it makes the fewest total errors. 

 

Class 0 (Casual users) - Performance Issues Across Models: 

Logistic Regression performs the worst (Recall = 6%). 

XGBoost & KNN slightly improve (Recall = 13-15%). 

Random Forest does best (Recall = 41%), meaning it recognizes casual users better than others. 

 

Class 1 (Members) - Best Recognition in All Models: 

Logistic Regression has highest recall (98%), meaning almost all members are correctly classified. 

XGBoost (95%) and KNN (92%) are slightly lower but still strong. 

Random Forest is most balanced (83%), ensuring better recognition for both classes. 

 

F1-Score Considerations: 

The F1-score for Casual users (0) is worst in Logistic Regression (0.11), improving slightly in KNN (0.23) and XGBoost (0.20), but Random Forest is best (0.46). 

The F1-score for Members (1) is highest across all models (~0.79-0.81). 

Random Forest gives the most balanced F1-score across both classes. 

 

Conclusion: 

Random Forest is the best model overall. 

It performs the best in balancing both Casual and Member users. 

Higher Recall for Casual users (41%) compared to others (6%-15%). 

While XGBoost & KNN have high recall for Members, they struggle with Casual users. 

Logistic Regression is the worst model for this classification task. 

Extremely poor at identifying Casual users (Recall = 6%). 

It over-predicts Members (1), meaning Casuals are often misclassified. 

XGBoost & KNN are similar, but Random Forest still wins. 

Both models improve over Logistic Regression but still struggle with Class 0 (Casuals). 

XGBoost is better than Logistic Regression, but not as strong as Random Forest. 

 
