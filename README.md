Introduction

In predictive analytics, the selection of variables is crucial for the performance of regression models. This project focuses on comparing the Mixed Integer Quadratic Programming (MIQP) and the Least Absolute Shrinkage and Selection Operator (LASSO) methods. MIQP is a mathematical optimization technique for selecting variables to optimize a quadratic objective under linear constraints. LASSO, on the other hand, was developed to tackle computational challenges associated with direct variable selection methods, combining variable selection and regularization to enhance accuracy and interpretability. The study aims to compare the effectiveness of these methods, considering aspects such as computational efficiency and the benefits of shrinkage in variable selection.
Methodology

The methodology involves applying MIQP and LASSO to two distinct datasets, each containing 50 independent variables (X) and a dependent variable (y). The primary focus is on optimizing key parameters: the number of variables (k) for MIQP and the regularization parameter (λ) for LASSO, using 10-fold cross-validation. This approach aims to assess the relative advantages of each method in different analytical contexts.
Data Description

    Training Dataset: Consists of 250 rows, used for building regression models and optimizing hyperparameters.
    Testing Dataset: Comprises 50 rows, employed to evaluate the performance of the trained models on unseen data.

MIQP Approach

MIQP involves identifying the optimal subset of variables in a regression model while minimizing the least squares error. This includes introducing binary variables to incorporate variable selection into the optimization task.
LASSO Approach

LASSO focuses on automatic feature selection, identifying influential variables and discarding less relevant ones, addressing overfitting through a shrinkage effect. This method involves minimizing the sum of squared errors while adding a penalty based on the absolute values of the regression coefficients.
Results
MIQP

The MIQP implementation involved defining an optimization function for feature selection, conducting manual cross-validation, determining the optimal number of features, and applying this value to estimate model coefficients. The test mean squared error (MSE) for MIQP optimization was found to be 2.33.
LASSO

LASSO implementation used the LassoCV function from scikit-learn for 10-fold cross-validation to determine the optimal λ. The optimal lambda was found to be 0.085, with a test MSE of 2.36.
Comparative Analysis

    Accuracy: MIQP showed a slightly higher accuracy with a test MSE of 2.33 compared to LASSO’s 2.36.
    Variable Selection: MIQP selected a concise set of variables, whereas LASSO chose a broader array.
    Coefficients Comparison: LASSO demonstrated higher absolute values for some variables while MIQP resulted in higher magnitudes for key variables.
    Computational Efficiency: MIQP incurred longer computation times compared to LASSO's almost instantaneous processing.
    Summary: MIQP is more accurate but less efficient, while LASSO is computationally efficient and well-suited for high-dimensional data.

Conclusion

The study recommends using LASSO for scenarios requiring quick processing and computational efficiency, and MIQP for projects where accuracy is crucial. The decision between MIQP and LASSO should be guided by the specific requirements of each project, including desired accuracy, computational limitations, and timelines.

This tailored approach in model selection is key to achieving effective and efficient analytical results in the field of predictive analytics.
