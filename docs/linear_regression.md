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

Every `Python` class can define an **initialiser method** (often called the *constructor* in other languages) using `__init__`.
This special method is automatically executed when a new instance of the class is created[^2].

[^2]: In general, I will reduce the docstrings for presentation here. You can gain additional context by looking through the docstrings in `lin_reg.py`.

``` python {linenums="1", title='Python'}
class LinearRegression:
    """
    Ordinary Least Squares (OLS) linear regression using the normal equation.
    For full details, see the complete class docstring in `lin_reg.py`
    """

    def __init__(self, add_bias: bool = True):
        """
        Initializes the model.

        Parameters
        ----------
        add_bias : bool, optional (default=True)
            If True, automatically adds a column of ones to X to estimate
            an intercept term (β₀). If False, the model is fit without an
            intercept (β₀ fixed at 0).
        """
        self.add_bias = add_bias # Instance attribute: model configuration
        self.beta = None  # Instance attribute: will store the estimated coefficients after fitting
        
        self.X = None # Instance attribute: will store the independent variables of the training set
        self.y = None # Instance attribute: will store the dependent variable of the training set
```

<hr>
<hr>
