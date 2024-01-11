# Optimizing-Variable-Selection-in-Predictive-Analytics-MIQP-vs-LASSO

Table of Contents

1. Introduction	2

2. Methodology	3

2.1. Data	3

2.2. Direct Variable Selection: The MIQP Approach	3

2.3. Indirect Variable Selection: The LASSO Approach	4

2.4. Cross-Validation	5

2.4.1 Manual Cross Validation for MIQP	5

2.4.2 Automated K-Fold Cross Validation for LASSO	5

3. Code and Results	6

3.1. MIQP	6

3.2. LASSO	14

4. MIQP vs LASSO - A Comparison	17

4.1 Accuracy	17

4.2 Variable Selection	18

4.3 Comparison of Coefficients	19

4.4 Computational Efficiency	20

4.5 Summary of Comparison	22

5. Conclusion and Recommendations	22




1. Introduction
In the field of predictive analytics, variable selection plays a pivotal role in enhancing the performance of regression models. This project is centered around comparing two distinct approaches to variable selection: Mixed Integer Quadratic Programming (MIQP) and the Least Absolute Shrinkage and Selection Operator (LASSO).

MIQP is a mathematical optimization technique used for selecting variables in a model to optimize a quadratic objective under linear constraints.
LASSO was originally developed in response to the computational challenges associated with direct variable selection methods. It is a regression analysis method that performs both variable selection and regularization to enhance the prediction accuracy and interpretability of the resulting statistical model.
Its appeal lies in its computational efficiency, making it particularly suitable for handling large datasets or scenarios with constrained computing resources. In addition, its shrinkage component, beneficial for managing overfitting through regularization, adds to its appeal. 

However, advancements in computational technology have renewed interest in direct variable selection methods like MIQP. This study aims to compare the effectiveness of MIQP and LASSO, evaluating the contexts where each method excels based on computational efficiency and the advantages of shrinkage in variable selection.

The methodology involves applying both MIQP and LASSO to the same datasets, comprising a training and a test set. The primary focus is on optimizing the critical parameters: the number of variables (k) for MIQP and the regularization parameter (λ) for LASSO, using 10-fold cross-validation.

The objective of this study is to determine whether LASSO's 'shrinkage' component, which also helps in reducing overfitting, is more beneficial than the direct selection of the most suitable variables using MIQP. This comparative analysis aims to provide insights into the relative advantages of each approach, thereby contributing to both the theoretical understanding and practical application of variable selection in the field of predictive analytics.
2. Methodology
2.1. Data
This study utilizes two distinct datasets containing a set of 50 independent variables (X) and a dependent variable (y), each serving specific purposes in the analysis:


The Training Dataset (training_data.csv):
Composition: Consists of 250 rows.
Purpose: The training dataset is used for building the regression models and optimizing their hyperparameters. It is essential for training both the MIQP and LASSO models, determining the optimal number of variables (k) for MIQP, and the regularization parameter (λ) for LASSO.

Testing Dataset (test_data.csv):
Composition: Comprises 50 rows.
Purpose: The testing dataset is employed to evaluate the performance of the trained models. It is used to test the models' ability to predict y values accurately and to compare these predictions with the true y values, thus assessing the effectiveness of the models on new, unseen data.
2.2. Direct Variable Selection: The MIQP Approach
The core idea behind Mixed Integer Quadratic Programming (MIQP) is to pose the variable selection problem as an optimization task. Given a dataset of m independent variables (X) and a dependent variable (y), we seek to identify the optimal subset of variables to include in our regression model while minimizing the least squares error. 

To incorporate variable selection into this problem, binary variables, 𝑧𝑗, are introduced. These force the corresponding values of  𝛽𝑗 to be zero if 𝑧𝑗 is zero, using the big-M method. If we want to include at most  𝑘 variables from  𝑋, then we can pose this as:


subject to:



Note that 𝑚 and 𝑀 are different entities;  𝑚 represents the number of independent variables while 𝑀 is a large constant used in the big-M method. This Big-M value must be large enough so as not to limit the range of the beta coefficients.

In MIQP, we don't restrict the model from having an intercept term (𝛽0) because forcing 𝛽0 to be zero could lead to biased and inaccurate model predictions, especially when an intercept is theoretically justified or necessary for accurate regression.

The hyperparameter 𝑘, representing the maximum number of variables to be included, can be chosen using cross-validation.
2.3. Indirect Variable Selection: The LASSO Approach
LASSO, or Least Absolute Shrinkage and Selection Operator, is a robust technique for variable selection in regression analysis, adept at identifying the most relevant features while reducing overfitting. 

LASSO's standout advantage lies in its automatic feature selection ability, autonomously identifying and retaining influential variables for predicting the target variable and discarding less relevant ones. It combats overfitting through a shrinkage effect, which pulls coefficients closer to zero, enhancing model accuracy and reliability.

LASSO does all of this by introducing a penalty term to the traditional least squares regression problem. The objective is to minimize the sum of squared errors, like ordinary least squares, while adding a penalty based on the absolute values of the regression coefficients, excluding the intercept term. This penalty term encourages some coefficients to become exactly zero, effectively selecting a subset of the available variables.



Crucially, LASSO retains the intercept term (β0), ensuring that the model includes a baseline value. The hyperparameter λ controls the trade-off between regularization strength and model fit. The selection of an appropriate λ value is often determined through cross-validation.

In our analysis, we implement the LASSO regression model using the scikit-learn library, a widely used Python toolkit for machine learning. The next sections of this report will present a comparative analysis of the results obtained from LASSO and the direct variable selection approach using MIQP. This analysis will shed light on whether LASSO's unique combination of feature selection and shrinkage is more advantageous for our specific regression problem.
2.4. Cross-Validation
2.4.1 Manual Cross Validation for MIQP
Adapting to MIQP's Unique Requirements:
MIQP presents unique challenges in cross-validation due to its nature as an optimization technique, as opposed to standard predictive models. Currently, there isn't a standard package designed for applying K-Fold Cross-Validation (KFold) directly to MIQP problems. This stems from the distinct purposes of MIQP and KFold: MIQP focuses on finding optimal solutions under constraints, while KFold is used for assessing a model's generalizability.

Implementing a Manual Approach:
Given these constraints and the lack of a standard tool, a manual cross-validation method was devised for MIQP. This involved randomly shuffling the training dataset, followed by a division into 10 equal-sized segments, each serving as a validation set in turn.
For each specified k value, the MIQP model was applied to the respective training set segments, and its performance was assessed on the validation set.
This approach allowed for a detailed evaluation of MIQP's predictive ability across different data subsets, aligning with the general principles of cross-validation but tailored to the specificities of MIQP.
2.4.2 Automated K-Fold Cross Validation for LASSO
Standard K-Fold for LASSO:
Unlike MIQP, the LASSO regression model is well-suited for the standard K-Fold cross-validation method. LASSO, being a predictive modeling technique, aligns with the objectives of KFold, which evaluates model performance on unseen data.

Cross Validation Process:
The LassoCV function from scikit-learn was employed to perform 10-fold cross-validation, automatically determining the optimal value of the regularization parameter alpha (λ). Prior to applying KFold, the training data was normalized using StandardScaler, a necessary preprocessing step for LASSO regression. This automated process efficiently optimized the LASSO model, ensuring a thorough assessment of its predictive accuracy.
3. Code and Results
3.1. MIQP
In this coding section of the report, we detail the practical implementation of MIQP in feature selection for predictive modeling, structured in the following key steps:

Define the MIQP Optimization Function: This involves setting up an optimization function specific to MIQP, which selects the most relevant features from the dataset.

Define the Manual Cross-Validation Function: A crucial step in the process, manual cross-validation ensures the robustness and reliability of the feature selection. This part covers the implementation details for conducting cross-validation manually, iterating over a range of k-values and evaluating the performance across different data subsets.

Determine the Optimal Number of Features (k): This step involves analyzing the results from the cross-validation process to identify the optimal k value. The focus is on striking a balance between model complexity and predictive accuracy, ensuring the model is neither overfit nor underfit.

Utilize the Optimal k Value: The final step is applying the optimal k value determined from the previous steps to estimate model coefficients using the entire training dataset and then evaluating the model's performance on a separate test set to gauge its predictive power on unseen data.

Let's now dive in and take a closer look at the code and output.

Step 1: Define the MIQP Optimization Function
We define a function to perform the MIQP optimization, specifying the following parameters:
X: independent variables
y: dependent variable
k: maximum number of variables to select
time_limit: maximum time allowed for the optimization process

As mentioned earlier, the Big-M value must be large enough so as not to limit the range of the beta coefficients. So we picked a value of 10000. 

This function returns the selected variables, their corresponding coefficients, and the intercept term.


















Code:






Step 2: Define the Manual Cross-Validation Function
We implement a function for conducting manual cross-validation for MIQP (for reasons mentioned above).
The parameters include: 
data: This is the dataset that will be used in the process.
target: This refers to the target variable in the dataset.
k_values: A list of k-values that the function will iterate over.
output_csv: The filename for the CSV file where results will be written.
total_time_limit: The overall time constraint imposed on the cross-validation process.
n_folds: The number of folds to be used in the cross-validation.

This function produces a DataFrame named results_df, which encapsulates the performance details of each fold in the cross-validation process. The DataFrame is structured with the following columns:

Fold: Identifies the specific fold in the cross-validation.
K: Represents the value of k used in that fold.
Betas: Contains the beta coefficients determined during the fold.
Intercept: The intercept value computed in the fold.
Squared_Error: The squared error for the fold's predictions.
Mean_Squared_Error: The mean squared error calculated for the fold.
Fold_Size: Indicates the number of data points in the fold.

An important aspect of the Fold_Size column is its variability. This occurs because when the total number of data points is not perfectly divisible by the number of folds, the leftover data points are allocated to the last fold. As a result, the size of the folds may vary.

To ensure data integrity and continuity, the results from each fold are continuously saved to a specified CSV file. This approach guarantees that no data is lost after each fold's computation. 














Step 3: Determine the Best k
Now that we’ve created the relevant functions, we want to use them to find the optimal value of k. We begin by importing the training dataset. Then, using the defined range of potential k values, we execute a thorough cross-validation procedure.
Our objective is to identify the k that yields the lowest average Mean Squared Error (MSE).

Code:


Output:


Note that the output provided is just a part of the complete result, which includes the model running across all 10 folds for each k value. 
The following chart demonstrates the variation in MSE observed across different values of k

Figure 1:

The results indicate that a k of 10 is optimal, achieving an average Mean Squared Error (MSE) of 2.82

Step 4: Utilize the Optimal Value of k
With our optimal k identified, we estimate the model coefficients using the entire training data.
These coefficients are then used to predict the target variable in the test dataset.
The Test MSE provides a measure of the model's predictive power.

Code:


Output:

The output shows that the Mean Squared Error (MSE) for the test set, using MIQP optimization, is 2.33

To further understand and assess the performance of the MIQP model on both the training and test datasets, we turn to the following visualization.

Figure 2:


On the left, the scatter plot compares actual versus predicted values for the training set, demonstrating the model's learning capabilities. On the right, a similar plot for the test set highlights the model's ability to generalize to new data. In both plots, the proximity of points to the diagonal line is indicative of the model's predictive accuracy, a key measure in assessing the success of MIQP models.
3.2. LASSO
In this section, we present the coding implementation of LASSO Regression, focusing on optimizing the alpha (lambda) parameter and assessing the model's performance through Mean Squared Error (MSE) on the test set. 
We use KFold from scikit-learn to perform 10-fold cross-validation and identify the optimal λ for our LASSO model. Once the best λ is determined, we fit the model to the entire training dataset. 
Next, we use the resulting β coefficients to predict y values on the test set, and finally calculate the Test MSE to assess the model's predictive accuracy.
Code:


Output:

The LASSO regression output reveals an optimal alpha (lambda) value of 0.085 and Test MSE of 2.36


The 10-fold cross-validation attempts to balance the trade-off between model complexity and fit. 

To enhance our understanding and provide a clearer picture of this trade-off, we include the following chart that displays the Mean Squared Error (MSE) for various alpha (lambda) values. This visualization is crucial as it helps better grasp how changes in lambda affect the model's performance, clearly illustrating the relationship between regularization strength and prediction accuracy.

Figure 3:


Let’s look at the model output more closely now. 
The intercept of the model, β0, is 1.28, setting the baseline prediction. 
The LASSO coefficients show a mixture of zero and non-zero values, demonstrating LASSO's feature selection capability by reducing the coefficients of less influential variables to zero. This effectively simplifies the model by focusing on the most significant predictors. Only 18 coefficients were non-zero, signifying their more substantial influence on the dependent variable in the model.



Mirroring the approach taken with the MIQP model, we have created a similar chart for the LASSO regression model. 

Figure 4:


We observed that the LASSO approach achieves a Mean Squared Error (MSE) of 2.36 on the test set, demonstrating a high level of accuracy in its predictions. 
Note that this is only slightly higher than the Test set MSE of 2.33 obtained by using MIQP.
In the upcoming section, we will compare the LASSO and MIQP approaches in detail. 
4. MIQP vs LASSO - A Comparison
4.1 Accuracy
A key metric in our analysis for evaluating model performance is the Mean Squared Error (MSE) on the test sets. This measure provides a quantitative foundation for comparing the accuracy of the MIQP and LASSO models.

The results indicate a slight variation in the MSE values between the two approaches. The MIQP model exhibits an MSE of 2.33, marginally outperforming the LASSO model, which has an MSE of 2.36. Although this difference is small, it suggests a slightly higher accuracy in the predictions made by the MIQP model. The below chart demonstrates this.




Figure 5:


In summary, while the MIQP model demonstrates a marginally higher accuracy as reflected in the MSE values, the difference in performance is relatively minimal.
4.2 Variable Selection
In our comparative analysis of variable selection, the MIQP model opted for a more concise set of variables: X9, X15, X16, X23, X24, X26, X34, X45, X47, X48, emphasizing a streamlined approach. Conversely, the LASSO model selected a broader array, including all variables chosen by MIQP plus additional ones like X11, X22, X29, X33, X35, X39, X44, X46. This difference highlights MIQP's focus on a smaller, potentially more impactful subset, whereas LASSO's strategy aims to capture a wider spectrum of influences. 











Figure 6:


The above Venn diagram visually encapsulates this overlap and distinction in variable selection, illustrating the choices made by each model.
4.3 Comparison of Coefficients
Figure 7:


In the comparison of coefficient magnitudes between the MIQP and LASSO models, LASSO demonstrates higher absolute values for 11 variables, while showing lower absolute values for 2 variables. Moreover, for some variables, both models yield coefficients that are either zero or very similar in magnitude, underscoring a level of agreement between the two methods.

The overall trend does not decisively favor one model over the other in terms of consistently producing higher coefficient magnitudes. While Lasso tends to produce non-zero coefficients for more variables, suggesting a broader selection of influential factors, MIQP's approach results in a higher magnitude for certain key variables. The consistency in the sign of coefficients across both models indicates agreement on the direction of influence for these variables. 
4.4 Computational Efficiency
Figure 8:


The comparison between MIQP and LASSO in terms of computational efficiency reveals a clear distinction in their practical applicability. MIQP, known for its nuanced approach in variable selection, incurs significantly longer computation times. This is best visualized in the above chart, which starkly contrasts MIQP's total computation time of 1356.62 seconds against LASSO's almost instantaneous processing of 0.12 seconds.




Figure 9:


On further examination we see that MIQP's computation times across different numbers of variables (k) show considerable fluctuation. As illustrated in the above chart, the time required for MIQP peaks notably at k=20 with about 70.94 seconds before decreasing for larger values of k. This nonlinear pattern in computation time with the increase in the number of variables suggests that MIQP can be more efficient for either a very low or high number of variables, while it becomes markedly less efficient for mid-range values.
In contrast, LASSO demonstrates a consistently low computation time, regardless of the number of variables.






4.5 Summary of Comparison
Before proceeding with our recommendations, let's present a table that summarizes the differences between MIQP and LASSO across relevant dimensions:

Dimension
LASSO
MIQP
Computational Efficiency
Faster, especially with large datasets. Efficient for quick model updates.
Slower, with increased computational time as dataset size grows.
Handling Overfitting
Provides regularization which helps in reducing overfitting.
Does not inherently provide regularization against overfitting.
Scalability
Well-suited for high-dimensional data. Scalable with the number of predictors.
May face scalability issues with very large datasets.
Implementation Ease
Widely supported and easy to implement in various software tools.
Requires sophisticated optimization solvers; more complex to implement.
Flexibility in Variable Selection
Limited control over the exact number of variables selected.
 More explicit control over the number of variables included.
Accuracy
Generally provides robust predictions. Shrinkage factor might introduce bias in large coefficients.
Potentially more accurate but the difference might be marginal.


5. Conclusion and Recommendations
Based on our analysis and the insights gained from comparing the MIQP and LASSO models in terms of accuracy, variable selection, coefficient magnitudes, and computational efficiency, we have formulated the following recommendations:

Prioritize LASSO for Efficiency: In most scenarios, especially where quick processing and computational efficiency are essential, LASSO should be the primary choice. Its ability to handle high-dimensional data swiftly makes it suitable for standard applications.

Use MIQP for Accuracy-Critical Projects: For projects where utmost accuracy is crucial and computational time and resources are less of a concern, the MIQP model becomes more relevant. This approach is particularly beneficial in situations demanding precise control over variable selection.

Balance Model Selection Based on Needs: The choice between MIQP and LASSO should be guided by the specific requirements of each project. Factors like the desired level of accuracy, computational limitations, and project timelines should inform the decision-making process, ensuring the most appropriate model is utilized for each task.

To sum up, the decision to use LASSO or MIQP should be carefully aligned with project-specific needs—LASSO for its rapid processing and efficiency in standard settings, and MIQP for its higher accuracy in more complex, detail-oriented tasks. This tailored approach in model selection is key to achieving the most effective and efficient analytical results.









