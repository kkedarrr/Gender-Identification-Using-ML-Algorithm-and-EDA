# Gender-Identification-Using-ML-Algorithm-and-EDA

Objective:-
  Identification of Gender whether it’s Male or
  Female based on some parameters.

Study of Existing System:-
  Reference has been taken from Kaggle Website –
  Name of the Dataset – Gender Classification 
  Dataset By Solved By DANIEL FOURIE
  
  Gaps in Existing System:-
• He has performed a few Eda(Exploratory Data 
Analysis) visualization on Dataset.
• And he has just performed KNN(K-Nearest 
Algorithm) to Predict.

Proposed Solution:-
• Performed more Machine learning Algorithms to 
get more predictability.
• Performed more Eda(Exploratory Data Analysis) 
visualization on Dataset.

Features & Predictor:-
This dataset contains 7 features and a label 
column.
1.longhair - This column contains 0's and 1's 
where 1 is "long hair" and 0 is "not long hair".
2.
foreheadwidthcm - This column is in Cm. This 
is the width of the forehead.
3.
foreheadheightcm - This is the height of the 
forehead and it's in Cm.
4.
nosewide - This column contains 0's and 1's 
where 1 is "wide nose" and 0 is "not wide 
nose".
5.
noselong - This column contains 0's and 1's 
where 1 is "Long nose" and 0 is "not long 
nose".
6.
lipsthin - This column contains 0's and 1's 
where 1 represents the "thin lips" while 0 is 
"Not thin lips".
7.
distancenosetoliplong - This column contains 
0's and 1's where 1 represents the "long 
distance between nose and lips" while 0 is 
"short distance between nose and lips".
8.gender - This is either "Male" or "Female"

Tools/Technology used to implement Proposed 
Solution:-
• Python
• Pandas
• Numpy
• Matplotlib
• Seaborn
• Excel

In Machine learning Algorithms Following are Used:-
1.Logistic Regression:
Logistic regression is often used a lot of times in 
machine learning for predicting the likelihood of 
response attributes when a set of explanatory 
independent attributes are given. It is used when the 
target attribute is also known as a dependent variable 
with categorical values like yes/no, true/false, etc. It’s 
widely used for solving classification problems. It falls 
under the category of supervised machine learning. It 
efficiently solves linear and 12 binary classification 
problems. It is one of the most commonly used and 
easy-to-implement algorithms. It’s a statistical 
technique to predict binary classes. When the target 
variable has two possible classes, it predicts the 
likelihood of the event's occurrence. In our dataset, the 
target variable is categorical as it has only two classesyes/no.

2.Decision Tree: 
A decision tree is a non-parametric supervised learning 
algorithm utilized for classification and regression 
tasks. It has a hierarchical, tree structure, which 
consists of a root node, branches, internal nodes, and 
leaf nodes.

3.Random Forest : 
Random Forest is the most famous and it is considered 
the best algorithm for machine learning. It is a 
supervised learning algorithm. To achieve more 
accurate and consistent prediction, a random forest 
creates several decision trees and combines them. The 
major benefit of using it is its ability to solve both 
regression and classification issues. When building 
each tree, it employs bagging and feature randomness 
to produce an uncorrelated tree forest whose 
collective forecast has much better accuracy than any 
individual tree’s prediction. Bagging enhances the 
accuracy of machine learning methods by grouping 
them. In this algorithm, during the splitting of nodes, it 
takes only a random subset of nodes into an account. 
When splitting a node, it looks for the best feature 
from a random group of features rather than the most 
significant feature. This results in getting better 
accuracy. It efficiently deals with huge datasets. It also 
solves the issue of overfitting in datasets. It works as 
follows: First, it’ll select random samples from the 
provided dataset. Next, for every selected sample it’ll 
create a decision tree and it’ll receive a forecasted 
result from every created decision tree. Then for each 
result that was predicted, it’ll perform voting and 
through voting, it will select the best-predicted result.

4.K Nearest Neighbor (KNN) : 
KNN is a supervised machine learning algorithm. It 
assumes similar objects are nearer to one another. 
When the parameters are continuous in that case knn 
is preferred. This algorithm classifies objects by 
predicting their nearest neighbor. It’s simple and easy 
to implement and also has high speed because of 
which it is preferred over the other algorithms when it 
comes to solving classification problems.

5.Naive Bayes : 
It is a probabilistic machine learning algorithm that is 
mainly used in classification problems. 11 | Page It’s 
based on the Bayes theorem. It is simple and easy to 
build. It deals with huge datasets efficiently. It can 
solve complicated classification problems. The 
existence of a specific feature in a class is assumed to 
be independent of the presence of any other feature 
according to naïve Bayes theorem. Its formula is as 
follows : P(S|T) = P(T|S) * P(S) / P(T) Here, T is the 
event to be predicted, and S is the class value for an 
event. This equation. will find out the class in which the 
expected feature is for classification.


Observation of ML Algorithms:-
So, here the Accuracies’ are as follows:-
1. Logistic Regression:- 96.7%
2.Decision Tree:- 96.7%
3. Random Forest:- 97.3%
4. KNeighboursClassifier:- 96.9%
5.Naïve Bayes:- 95.7%
So, as Random Forest has the Highest Accuracy 
amongst all Algorithms therefore we use Random 
Forest for Predictin

Measuring Model Performance:-
While there are other ways of measuring model performance 
(precision, recall, F1 Score, ROC Curve, etc), let's keep this 
simple and use accuracy as our metric. To do this are going to 
see how the model performs on new data (test set) Accuracy 
is defined as (a fraction of correct predictions): correct 
predictions / total number of data points
Here, 502 is the number of True Positives in our data, while 
472 is the number of True Negatives. 13 & 14 are the number 
of errors. There are 13 type-1 error (False Positives)- You 
predicted positive and it’s false. There are 14 type-2 error 
(False Negatives)- You predicted negative and it’s false. 
Hence, if we calculate the accuracy it’s #Correct Predicted/ # 
Total. In other words, where TP, FN, FP, and TN represent the 
number of true positives, false negatives, false positives, and 
true negatives. (TP + TN)/(TP + TN + FP + FN). (502 +472)/( 
502 +472 +13 +14) = 0.80 = 97.3% accuracy.
Note: A good rule of thumb is that any accuracy above 70% is 
considered good, but be careful because if your accuracy is 
extremely high, it may be too good to be true (an example of 
Overfitting). Thus, 97.3% is the ideal accuracy!

Conclusions:-
Our Random Forest algorithm yields the highest 
accuracy, 97%. Any accuracy above 70% is considered 
good.
We can Identify Gender either Male or Female by the 
given Features
