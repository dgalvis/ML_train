## Introduction

Linear regression is a type of supervised learning, where the goal is to identify a best-fit function between a dependent variable and one or more independent variables.

---

### Model Definition

We assume the model takes the form:

\[
y_i = \beta_0 + \beta_1 x_{i,1} + \dots + \beta_N x_{i,N} + \epsilon_i,
\]

where \( i = 1, \dots, M \) indexes the samples in the training set.

The components of the model are:

- \( x_{i,j} \) — Feature \( j \) for sample \( i \); there are \( N \) features.
- \( y_i \) — The output (target) for sample \( i \).
- \( \epsilon_i \) — The error term, representing noise or unexplained variation.
- \( \beta_0 \) — The intercept (bias).
- \( \beta_j \) — The coefficient for feature \( j \).

> *Note:* We typically use \( \beta \) for both the true model parameters and the estimated parameters. It's common to overload the symbol and rely on context, though you may also see \( \hat{\beta} \) used for estimated values.

---

### Fitting the Model: Ordinary Least Squares (OLS)

The most common way to fit a linear regression model is with the **ordinary least squares (OLS)** method. The objective is to find parameters \( \boldsymbol{\beta} \) that minimise the following cost function:

\[
J(\boldsymbol{\beta}) = \frac{1}{2} \sum_{i=1}^{M} \left(h_{\boldsymbol{\beta}}(X_i) - y_i\right)^2,
\]

where:

- \( \boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_N \end{bmatrix} \) — Vector of model parameters.
- \( X_i = \begin{bmatrix} 1 & x_{i,1} & x_{i,2} & \dots & x_{i,N} \end{bmatrix} \) — Row vector of features for sample \( i \), including a 1 to account for the bias term \( \beta_0 \).
- \( h_{\boldsymbol{\beta}}(X_i) \) — The hypothesis (model) function, defined as:

\[
h_{\boldsymbol{\beta}}(X_i) = \beta_0 + \beta_1 x_{i,1} + \dots + \beta_N x_{i,N} = \boldsymbol{\beta}^\top X_i^\top.
\]

Thus, we are minimising the **sum of squared differences** between the predicted outputs \( h_{\boldsymbol{\beta}}(X_i) \) and the actual outputs \( y_i \) across all training samples.

---

### Closed-Form Solution

To simplify notation and enable vectorised computation, we define:

- The **design matrix**:

\[
X = \begin{bmatrix}
X_1 \\
X_2 \\
\vdots \\
X_M
\end{bmatrix}
\quad \text{(an \( M \times (N+1) \) matrix)}
\]

- The output vector:

\[
\mathbf{y} = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_M
\end{bmatrix}
\]

OLS is one of the few machine learning methods with a **closed-form analytic solution** for the best-fit parameters. The solution is:

\[
\boldsymbol{\beta} = (X^\top X)^{-1} X^\top \mathbf{y}
\]

This gives the value of \( \boldsymbol{\beta} \) that minimises the cost function and fits the training data as closely as possible in the least-squares sense.

<hr>
<hr>
