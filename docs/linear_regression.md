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

``` bash {linenums="1", title='console'}
jupyter notebook
```

Open the notebook named `linear_regression_R.ipynb`, which contains `R` code. **Make sure you use the `R` kernel.** 

We begin by importing `datasets` package:

```r {linenums="1", title='R'}
library(datasets)
```

This package includes several classic datasets, one of which is `USJudgeRatings`. This dataset contains average ratings for 43 judges across several dimensions (e.g., integrity, diligence, writing quality). The 12th column is the rating for “worthy of retention,” which we’ll try to predict using the first 11 features.

This command prints the first six rows of the dataset:

```r {linenums="1", title='R'}
dataset = USJudgeRatings
head(dataset)
```

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

```r {linenums="1", title='R'}
x <- as.matrix(dataset[, c(1:11)]) # First 11 dimensions are Features
y <- as.matrix(dataset[, 12]) # last dimeions is the Output (Retention rating)
reg <- lm(y ~ x)
```

This creates matrices `x` and `y` and uses the `lm` function to perform linear regression.


To see the model summary:

```r {linenums="1", title='R'}
summary(reg)
```

This produces several useful statistics:
``` {title='Output'}
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
```

I don't know about you, but I think this is **mega**. However, we’ll take it a step further and write our own `Python` code — mostly from scratch — to calculate these statistics ourselves and better understand what they mean.

<hr>
<hr>

## Example from Python

### Using a class to reproduce the `R` statistics

In this example, we will implement a `LinearRegression` class in Python that performs linear regression and reproduces the same statistics that the `R` `summary` function outputs.

To explore this example, navigate to the folder `./codes/` in the repository.
Activate the virtual environment we created earlier (see [Requirements](requirements.md)).

Launch Jupyter Notebook:

``` bash {linenums="1", title='console'}
jupyter notebook
```

Open the notebook named `linear_regression_py.ipynb`, which contains `Python` code. **Make sure you use the `Python 3` kernel.** 

We start by importing `pandas` for data handling and the `LinearRegression` class from our module:

``` python {linenums="1", title='Python'}
import pandas as pd
from modules.lin_reg import LinearRegression 
```

The `pandas` library makes it easy to load tabular datasets, such as the `USJudgeRatings` data stored in a CSV file:

``` python {linenums="1", title='Python'}
df = pd.read_csv("data/USJudgeRatings.csv", index_col=0)
print(df.head())
```

To prepare the data, we convert the `DataFrame` to NumPy arrays:

``` python {linenums="1", title='Python'}
X = df.to_numpy()[:, :-1]
y = df.to_numpy()[:, -1]
```

Then, we create the `LinearRegression` instance and fit the data:

``` python {linenums="1", title='Python'}
model = LinearRegression()
_ = model.fit(X,y)
```

To display the regression results, call the `summary()` method::

``` python {linenums="1", title='Python'}
model.summary()
```

This produces the same statistics as the `R` summary:

``` {title="Output"}
Residuals:
Min: -0.2212
Q1:  -0.0615
Med: -0.0105
Q3:  0.0505
Max: 0.2608

 Coefficient   Std Error     t-value     p-value
     -2.1194      0.5190     -4.0834   0.0002896
      0.0128      0.0259      0.4947      0.6243
      0.3648      0.1294      2.8204    0.008291
      0.1254      0.0897      1.3978      0.1721
      0.0667      0.1430      0.4663      0.6443
     -0.1945      0.1478     -1.3163      0.1977
      0.2783      0.1383      2.0129     0.05288
     -0.0020      0.2400     -0.0082      0.9935
     -0.1358      0.2672     -0.5081       0.615
      0.5478      0.2772      1.9759     0.05712
     -0.0681      0.3148     -0.2162      0.8303
      0.2688      0.0621      4.3263   0.0001464

Residual standard error: 0.1174
R-squared:             0.9916
Adjusted R-squared:    0.9886
F-statistic:           332.8597
F-statistic p-value:   1.11e-16
```

Next, we will unpack the `LinearRegression` class to see how it works!

<hr>

### The `LinearRegression` class

All Python source code for this example is stored in the folder `./codes/modules/`.

- The file `__init__.py` marks the directory as a **Python package**, allowing you to import its modules elsewhere in your project.
- The file `lin_reg.py` defines the `LinearRegression` **class**, which implements our regression model and associated methods.
- The file `test_lin_reg.py` contains **unit tests** to verify that the implementation in `lin_reg.py` works correctly[^1].

[^1]: Testing is beyond the scope of this tutorial, but it is an essential skill in software development. Automated tests help ensure that your code produces the expected results, prevent regressions when you make changes, and improve confidence in the correctness of your program.

We start by importing the required libraries:

``` python {linenums="1", title='Python'}
import numpy as np
from scipy import stats
from typing import Tuple
from numpy.typing import NDArray
```

Every `Python` class can define an **initialiser method** (often called the *constructor* in other languages) using `__init__`.
This special method is automatically executed when a new instance of the class is created[^2].

[^2]: In general, I will reduce the docstrings for presentation here. You can gain additional context by looking through the docstrings in `lin_reg.py`.

``` python {linenums="1", title='Python'}
class LinearRegression:
    """Ordinary Least Squares (OLS) linear regression using the normal equation."""

    def __init__(self, add_bias: bool = True):
        self._add_bias = add_bias # Whether to include an intercept β₀
        self.beta = None # Will hold fitted coefficients 
        self.X = None # Stores training features (with bias if added) 
        self.y = None # Stores training target values
```

#### Fitting the model

The `fit` method computes the OLS solution:

- Adds a column of ones to `X` if an intercept term is included.
- Computes the regression coefficients using the pseudoinverse of $X^T X$.
- Stores `X`, `y` and the fitted coefficients in the instance for later use.

``` python {linenums="1", title='Python'}
def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    if self._add_bias:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))
    
    XtX_inv = np.linalg.pinv(X.T @ X)
    self.beta = XtX_inv @ X.T @ y

    self.X = X
    self.y = y

    return self.beta
```

#### Making predictions (train vs. test)

The **learned relationship** lives entirely in the fitted coefficients \( \boldsymbol{\beta} \) that are computed during `fit(X_train, y_train)`.  
Once `beta` is set, you can call `predict(X)` on **any** dataset that has the same feature schema:

- **Training predictions:** `predict(X_train)`
- **Test/Validation predictions:** `predict(X_test)`

Mathematically, predictions are always
$$
\hat{\mathbf{y}} = X\,\boldsymbol{\beta}.
$$

If the model was created with `add_bias=True`, `predict` will **automatically prepend** the column of ones to whatever `X` you pass in. It then delegates to the internal `_predict`, which assumes `X` is already in final matrix form (bias included if needed).

> Note: Do not re-fit on test data. Fit once on training data, then reuse the same beta to evaluate on validation/test sets (using `predict`).

```python {linenums="1", title="Python"}
def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")

    if self._add_bias:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

    return self._predict(X)

def _predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
    return X @ self.beta
```

#### Computing residuals (train vs. test)

Residuals are the differences between observed targets and predictions:
$$
\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - X\boldsymbol{\beta}
$$

Use the same `residuals(X, y)` method for either training or test data:

- **Training residuals**: `residuals(X_train, y_train)`
- **Test/validation residuals**:`residuals(X_test, y_test)`

As with `predict`, if `add_bias=True`, the public `residuals` method will **add the bias column for you** before delegating to `_residuals`. The private `_residuals` assumes X is already prepared.

``` python {linenums="1", title='Python'}
def residuals(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")

    if self._add_bias:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

    return self._residuals(X, y)

def _residuals(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    y_pred = self._predict(X)
    return y - y_pred
```

#### Calculating Statistics

##### Residual summary statistics

The `residual_stats` method returns the **five-number summary** of the model residuals:

- **Minimum**: the smallest residual value.
- **Q1 (25th percentile)**: the value below which 25% of residuals lie.
- **Median**: the middle value (50th percentile) of the residuals.
- **Q3 (75th percentile)**: the value below which 75% of residuals lie.
- **Maximum**: the largest residual value.

These statistics are useful for identifying skewness or outliers in the residual distribution.
They mirror the residual summary output seen in statistical software like R.

Mathematically, if $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$, then:

$$
\text{summary}(\mathbf{r}) =
\big[
\min(\mathbf{r}),
Q_1(\mathbf{r}),
\operatorname{median}(\mathbf{r}),
Q_3(\mathbf{r}),
\max(\mathbf{r})
\big]
$$

``` python {linenums="1", title='Python'}
def residual_stats(self) -> np.ndarray:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")
        
    # Compute residuals using stored training data
    residuals = self._residuals(self.X, self.y)

    # Compute and return five-number summary as a NumPy array
    return np.array([
        np.min(residuals),
        np.percentile(residuals, 25),
        np.median(residuals),
        np.percentile(residuals, 75),
        np.max(residuals)
    ])
```

##### Residual Standard Error (RSE)

The **Residual Standard Error** (RSE) measures the typical size of the residuals — in other words, how far the model's predictions are from the observed values on average.

It is defined as:
$$
\mathrm{RSE} = \sqrt{\frac{\mathrm{RSS}}{n - p}}
$$

where:

- $\mathrm{RSS} = \sum_{i=1}^n r_i^2$ is the **Residual Sum of Squares**,
- $n$ is the number of observations,
- $p$ is the number of estimated parameters (including the intercept).

This formula is equivalent to taking the square root of the estimated error variance:
$$
\hat{\sigma}^2 = \frac{\mathrm{RSS}}{n - p}
\quad\Rightarrow\quad
\mathrm{RSE} = \sqrt{\hat{\sigma}^2}
$$

The RSE is expressed in the same units as the dependent variable, making it easy to interpret: a smaller RSE means the model predictions are closer to the observed values.

``` python {linenums="1", title='Python'}
def residuals_SE(self) -> float:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")

    # Compute residuals using stored data
    residuals = self._residuals(self.X, self.y)

    # Number of observations and parameters
    n = len(residuals)
    p = len(self.beta)

    # Compute residual sum of squares and standard error
    RSS = np.sum(residuals ** 2)
    RSE = np.sqrt(RSS / (n - p))

    return RSE
```

##### Coefficient of Determination: $R^2$ and adjusted $R^2$

The **coefficient of determination** $R^2$ measures the proportion of variability in the dependent variable that is explained by the model:

$$
R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}}
$$

where:

- $\mathrm{RSS} = \sum_{i=1}^n r_i^2$ is the **Residual Sum of Squares** (unexplained variance),
- $\mathrm{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2$ is the **Total Sum of Squares** (total variance in the data).

While $R^2$ indicates goodness of fit, it always increases when more predictors are added — even if they are not useful.  
To account for this, we compute the **adjusted $R^2$**:

$$
R^2_{\text{adj}} = 1 - \frac{\mathrm{RSS}/(n - p)}{\mathrm{TSS}/(n - 1)}
$$

where:

- $n$ = number of observations,
- $p$ = number of estimated parameters (including the intercept).

Adjusted $R^2$ penalizes the inclusion of unnecessary predictors, making it a better measure for comparing models with different numbers of features.

``` python {linenums="1", title='Python'}
def R_squared(self) -> Tuple[float, float]:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")
    
    # Compute residuals using stored training data
    residuals = self._residuals(self.X, self.y)

    # Number of observations and estimated parameters
    n = len(residuals)
    p = len(self.beta)

    # Residual Sum of Squares (unexplained variance)
    RSS = np.sum(residuals ** 2)

    # Total Sum of Squares (total variance in y)
    TSS = np.sum((self.y - np.mean(self.y)) ** 2)

    # R² = 1 - RSS/TSS
    R_squared = 1 - RSS / TSS

    # Adjusted R² penalizes for model complexity
    R_squared_adj = 1 - (RSS / (n - p)) / (TSS / (n - 1))

    return R_squared, R_squared_adj
```

##### Overall significance: the F-stat and p-value

The **$F$-statistic** tests the null hypothesis that **all regression coefficients except the intercept are equal to zero**:

$$
H_0: \beta_1 = \beta_2 = \dots = \beta_{p-1} = 0
$$

In other words, it checks whether the model provides a better fit than one with only the intercept.

- **Calculate MSR and MSE**: Let $\mathrm{TSS}$ be the **Total Sum of Squares** and $\mathrm{RSS}$ be the **Residual Sum of Squares**:
    - **Mean Square Regression (MSR)** — average explained variance per parameter:
      $$
      \mathrm{MSR} = \frac{\mathrm{TSS} - \mathrm{RSS}}{df_1}
      $$
      where:
        - $df_1 = p - 1$ (if `self._add_bias=True`) 
        - $df_1 = p$ (if `self._add_bias = False`).
    - **Mean Square Error (MSE)** — average unexplained variance per residual degree of freedom:
      $$
      \mathrm{MSE} = \frac{\mathrm{RSS}}{df_2}
      $$
      where $df_2 = n-p$.

- **Calculate  $F$-statistic and p-value**: 
    The $F$-statistic is the ratio:
    $$
    F = \frac{\mathrm{MSR}}{\mathrm{MSE}}
    $$

    A large $F$-value suggests that the model explains significantly more variance than would be expected by chance.
    The **p-value** is computed from the right tail of the $F$-distribution with $(df_1, df_2)$ degrees of freedom.
    The one-tailed p-value is:
    $$
    p = \left( 1 - \text{CDF}_{df_1, df_2}(F)\right).
    $$


- **Interpretation**:
    - Small p-value (reject $H_0$): at least one predictor is significantly associated with the response.
    - Large p-value (fail to reject $H_0$): no evidence the predictors improve the model.



``` python {linenums="1", title='Python'}
def F_score(self) -> Tuple[float, float]:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")
       
    # Compute residuals using stored training data
    residuals = self._residuals(self.X, self.y)

    # Number of observations and number of estimated parameters
    n = len(residuals)
    p = len(self.beta)

    # Residual Sum of Squares (RSS) — unexplained variation
    RSS = np.sum(residuals ** 2)

    # Total Sum of Squares (TSS) — total variation in y
    TSS = np.sum((self.y - np.mean(self.y)) ** 2)

    # Degrees of freedom
    if self._add_bias:
        df1 = p - 1            # Numerator degrees of freedom (model)
    else:
        df1 = p
    df2 = n - p            # Denominator degrees of freedom (residuals)

    # Mean Square Regression and Mean Square Error
    MSR = (TSS - RSS) / df1  # Explained variance per parameter
    MSE = RSS / df2          # Unexplained variance per residual degree of freedom

    # F-statistic: ratio of explained to unexplained variance
    F_stat = MSR / MSE

    # p-value from the F-distribution (right-tailed test)
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)

    return F_stat, p_value
```

##### Standard Errors of Coefficients

The **standard error** of each regression coefficient measures the variability of its estimate across hypothetical repeated samples.
It comes from the diagonal entries of the **variance–covariance matrix** of \( \boldsymbol{\beta} \).

The method works as follows:

- **Get residuals from stored training data**:
   We call the internal `_residuals(self.X, self.y)` to compute
   $$
   \mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}.
   $$

- **Compute $(X^\top X)^{-1}$**:
   This matrix appears in the closed-form OLS solution and is needed to propagate uncertainty into the coefficient estimates.

- **Estimate the variance of the residuals**:
   Using
   $$
   \hat{\sigma}^2 = \frac{\text{RSS}}{n - p}
   $$
   where:
       - $\text{RSS} = \sum_i r_i^2$ is the residual sum of squares,
       - $n$ is the number of observations,
       - $p$ is the number of estimated parameters (including the intercept).

- **Form the variance–covariance matrix of $\boldsymbol{\beta}$**:
   $$
   \text{Var}(\boldsymbol{\beta}) = \hat{\sigma}^2 \, (X^\top X)^{-1}.
   $$

- **Extract standard errors**:
   Take the square root of each diagonal element to get
   $$
   \text{SE}(\beta_j) = \sqrt{ \text{Var}(\beta_j) }.
   $$

These standard errors are crucial for computing **t-statistics** and **p-values** when testing the significance of each coefficient.


``` python {linenums="1", title='Python'}
def coefficients_SE(self) -> NDArray[np.float64]:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")

    # Compute residuals from stored data
    residuals = self._residuals(self.X, self.y)

    # Compute (X^T X)^(-1)
    XtX_inv = np.linalg.pinv(self.X.T @ self.X)

    # Compute estimated variance of errors
    n = len(residuals)
    p = len(self.beta)
    RSS = np.sum(residuals ** 2)
    sigma_squared = RSS / (n - p)

    # Compute variance-covariance matrix of beta
    var_beta = sigma_squared * XtX_inv

    # Standard errors are the square roots of the diagonal entries
    coeff_RSE = np.sqrt(np.diag(var_beta))

    return coeff_RSE
```

##### t-stats and p-values for Coefficients

Once we have the **standard errors** of the regression coefficients, we can test whether each coefficient is statistically different from zero.

-  **Compute t-statistics**
   For each coefficient \( \beta_j \), the **t-statistic** is computed as:
   $$
   t_j = \frac{\beta_j}{\mathrm{SE}(\beta_j)}
   $$
   This measures how many standard errors the coefficient is away from zero.

-  **Determine degrees of freedom**
   We use:
   $$
   \text{df} = n - p
   $$
   where:
       - $n$ is the number of observations,
       - $p$ is the number of parameters estimated (including the intercept).

- **Compute two-tailed p-values**
   Under the null hypothesis $H_0 : \beta_j = 0$, the t-statistic follows a **Student’s t-distribution** with $n - p$ degrees of freedom.
   The two-tailed p-value is:
   $$
   p_j = 2 \left( 1 - F_t\left( \left| t_j \right| \right) \right)
   $$
   where $F_t$ is the cumulative distribution function (CDF) of the t-distribution.



These p-values indicate the probability of observing such extreme $t$-statistics if the true coefficient were zero.
Small p-values (commonly below 0.05) suggest that the coefficient is statistically significant.

``` python {linenums="1", title='Python'}
def coefficients_p_values(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    if self.beta is None:
        raise ValueError("Model has not been fitted yet.")

    # Compute t-statistics: each beta divided by its standard error
    t_values = self.beta / self.coefficients_SE()

    # Number of observations and number of parameters
    n = len(self.y)
    p = len(self.beta)

    # Compute two-tailed p-values using the t-distribution CDF
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=n - p))

    return t_values, p_values
```

##### Summary method

Together, all of these statistics can be used to generate our own summary function. See `summary(self)` in `lin_reg.py`.

<hr>
<hr>
