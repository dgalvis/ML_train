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

## Example from `R`

The programming language `R` is widely used in statistics and data visualisation. It also provides a useful library of built-in datasets.

To explore this, navigate to the folder `./codes/` in the repository. Activate the virtual environment we created earlier (see [Requirements](requirements.md)).

Then, launch Jupyter Notebook:

``` bash {linenums="1"}
jupyter notebook
```

Open the notebook named `linear_regression.ipynb`, which contains `R` code. **Make sure you use the `R` kernel.** 

We begin by importing `datasets` package:

```r {linenums="1"}
library(datasets)
```

This package includes several classic datasets, one of which is `USJudgeRatings`. This dataset contains average ratings for 43 judges across several dimensions (e.g., integrity, diligence, writing quality). The 12th column is the rating for “worthy of retention,” which we’ll try to predict using the first 11 features.

```r {linenums="1"}
dataset = USJudgeRatings
head(dataset)
```
This command prints the first six rows of the dataset:

<table class="dataframe">
<caption>US Judge Ratings</caption>
<thead>
	<tr><th></th><th scope=col>CONT</th><th scope=col>INTG</th><th scope=col>DMNR</th><th scope=col>DILG</th><th scope=col>CFMG</th><th scope=col>DECI</th><th scope=col>PREP</th><th scope=col>FAMI</th><th scope=col>ORAL</th><th scope=col>WRIT</th><th scope=col>PHYS</th><th scope=col>RTEN</th></tr>
</thead>
<tbody>
	<tr><th scope=row>AARONSON,L.H.</th><td>5.7</td><td>7.9</td><td>7.7</td><td>7.3</td><td>7.1</td><td>7.4</td><td>7.1</td><td>7.1</td><td>7.1</td><td>7.0</td><td>8.3</td><td>7.8</td></tr>
	<tr><th scope=row>ALEXANDER,J.M.</th><td>6.8</td><td>8.9</td><td>8.8</td><td>8.5</td><td>7.8</td><td>8.1</td><td>8.0</td><td>8.0</td><td>7.8</td><td>7.9</td><td>8.5</td><td>8.7</td></tr>
	<tr><th scope=row>ARMENTANO,A.J.</th><td>7.2</td><td>8.1</td><td>7.8</td><td>7.8</td><td>7.5</td><td>7.6</td><td>7.5</td><td>7.5</td><td>7.3</td><td>7.4</td><td>7.9</td><td>7.8</td></tr>
	<tr><th scope=row>BERDON,R.I.</th><td>6.8</td><td>8.8</td><td>8.5</td><td>8.8</td><td>8.3</td><td>8.5</td><td>8.7</td><td>8.7</td><td>8.4</td><td>8.5</td><td>8.8</td><td>8.7</td></tr>
	<tr><th scope=row>BRACKEN,J.J.</th><td>7.3</td><td>6.4</td><td>4.3</td><td>6.5</td><td>6.0</td><td>6.2</td><td>5.7</td><td>5.7</td><td>5.1</td><td>5.3</td><td>5.5</td><td>4.8</td></tr>
	<tr><th scope=row>BURNS,E.B.</th><td>6.2</td><td>8.8</td><td>8.7</td><td>8.5</td><td>7.9</td><td>8.0</td><td>8.1</td><td>8.0</td><td>8.0</td><td>8.0</td><td>8.6</td><td>8.6</td></tr>
</tbody>
</table>

To prepare the data and run linear regression:

```r {linenums="1"}
x <- as.matrix(dataset[, c(1:11)]) # First 11 dimensions are Features
y <- as.matrix(dataset[, 12]) # last dimeions is the Output (Retention rating)
reg <- lm(y ~ x)
```

This creates matrices `x` and `y` and uses the `lm` function to perform linear regression.


To see the model summary:

```r {linenums="1"}
summary(reg)
```

This produces several useful statistics:

    Call:
    lm(formula = y ~ x)

    Residuals:
         Min       1Q   Median       3Q      Max
    -0.22123 -0.06155 -0.01055  0.05045  0.26079

    Coefficients:
                Estimate Std. Error t value Pr(>|t|)
    (Intercept) -2.11943    0.51904  -4.083 0.000290 ***
    xCONT        0.01280    0.02586   0.495 0.624272
    xINTG        0.36484    0.12936   2.820 0.008291 **
    xDMNR        0.12540    0.08971   1.398 0.172102
    xDILG        0.06669    0.14303   0.466 0.644293
    xCFMG       -0.19453    0.14779  -1.316 0.197735
    xDECI        0.27829    0.13826   2.013 0.052883 .
    xPREP       -0.00196    0.24001  -0.008 0.993536
    xFAMI       -0.13579    0.26725  -0.508 0.614972
    xORAL        0.54782    0.27725   1.976 0.057121 .
    xWRIT       -0.06806    0.31485  -0.216 0.830269
    xPHYS        0.26881    0.06213   4.326 0.000146 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

    Residual standard error: 0.1174 on 31 degrees of freedom
    Multiple R-squared:  0.9916,	Adjusted R-squared:  0.9886
    F-statistic: 332.9 on 11 and 31 DF,  p-value: < 2.2e-16

I don't know about you, but I think this is **mega**. However, we’ll take it a step further and write our own `Python` code — mostly from scratch — to calculate these statistics ourselves and better understand what they mean.

<hr>
<hr>

## Example from Python

