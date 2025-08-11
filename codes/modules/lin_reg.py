import numpy as np
from scipy import stats
from typing import Tuple

class LinearRegression:
    """
    Ordinary Least Squares (OLS) linear regression using the normal equation.

    This implementation estimates regression coefficients by solving the closed-form
    OLS solution:

        β = (Xᵀ X)⁻¹ Xᵀ y

    It reproduces key summary statistics found in R's `summary.lm()` output,
    including:
    - Residual quartiles summary
    - Estimated coefficients and standard errors
    - t-statistics and p-values for coefficients
    - Residual standard error (RSE)
    - R² and adjusted R²
    - F-statistic and corresponding p-value

    Parameters
    ----------
    add_bias : bool, default=True
        If True, automatically adds a column of ones to X to estimate
        an intercept term (β₀). If False, the model is fit without an
        intercept (β₀ fixed at 0).

    Attributes
    ----------
    add_bias : bool
        Whether the intercept term is included in the model.
    beta : np.ndarray or None
        Estimated regression coefficients after fitting. If `add_bias` is True,
        the first element corresponds to the intercept.
    X : np.ndarray
        Design matrix used to fit the model (including bias column if applicable).
    y : np.ndarray
        Target vector used to fit the model.

    Notes
    -----
    - The model uses the pseudoinverse to handle potentially singular
      `Xᵀ X` matrices.
    - Assumes that the relationship between predictors and target is
      linear, and that residuals are approximately normally distributed
      with constant variance.
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

    def fit(self, X: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Fits the linear regression model using the normal equation.

        This method solves for the regression coefficients (beta) that minimize the residual sum of squares (RSS)
        between the observed targets and those predicted by the linear model.

        Parameters
        ----------
        X : np.ndarray
            The design matrix of shape (n_samples, n_features). Should not include a bias column.
        y : np.ndarray
            The target vector of shape (n_samples,).

        Returns
        -------
        beta : np.ndarray
            The estimated regression coefficients. If `add_bias` is True, the first element is the intercept.
        """
        if self.add_bias:
            # Add a column of ones to account for the intercept term
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
        
        # Compute (X^T X)^(-1)
        XtX_inv = np.linalg.pinv(X.T @ X)

        # Compute the least squares solution beta = (X^T X)^(-1) X^T y
        self.beta = XtX_inv @ X.T @ y

        # Store X and y for later use in residual and error calculations
        self.X = X
        self.y = y

        return self.beta

    def predict(self, X: np.ndarray[float]) -> np.ndarray[float]:
        """
        Generates predictions using the fitted linear model.

        Parameters
        ----------
        X : np.ndarray
            The design matrix of shape (n_samples, n_features). Should not include a bias column.

        Returns
        -------
        y_pred : np.ndarray
            Predicted target values of shape (n_samples,).
        """
        if self.add_bias:
            # Add a column of ones to account for the intercept
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        # Delegate to the internal _predict method
        y_pred = self._predict(X)
        return y_pred

    def _predict(self, X: np.ndarray[float]) -> np.ndarray[float]:
        """
        Internal method to perform matrix multiplication to compute predictions from the design matrix and fitted coefficients.

        Parameters
        ----------
        X : np.ndarray
            The design matrix (already processed to include bias if needed).

        Returns
        -------
        y_pred : np.ndarray
            Predicted target values of shape (n_samples,).
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")

        # Compute predictions as X @ beta
        y_pred = X @ self.beta
        return y_pred

    def residuals(self, X: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Computes residuals between observed and predicted values.

        Parameters
        ----------
        X : np.ndarray
            The design matrix of shape (n_samples, n_features). Should not include a bias column.
        y : np.ndarray
            The observed target values of shape (n_samples,).

        Returns
        -------
        residuals : np.ndarray
            The difference y - y_pred, of shape (n_samples,).
        """
        if self.add_bias:
            # Add bias column if needed
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))       

        # Predict values and return residuals
        y_pred = self._predict(X)
        return y - y_pred

    def _residuals(self, X: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        """
        Internal method to compute residuals from already-prepared data.

        Parameters
        ----------
        X : np.ndarray
            The design matrix (including bias column if applicable).
        y : np.ndarray
            The observed target values.

        Returns
        -------
        residuals : np.ndarray
            The residuals (y - y_pred), of shape (n_samples,).
        """
        y_pred = self._predict(X)
        return y - y_pred

    def residuals_SE(self) -> float:
        """
        Computes the Residual Standard Error (RSE) for the fitted model.

        Uses the formula:
            RSE = sqrt(RSS / (n - p))
        where RSS is the residual sum of squares, n is the number of samples, and p is the number of parameters.

        Returns
        -------
        float
            The residual standard error.
        """
        # Compute residuals using stored data
        residuals = self._residuals(self.X, self.y)

        # Number of observations and parameters
        n = len(residuals)
        p = len(self.beta)

        # Compute residual sum of squares and standard error
        RSS = np.sum(residuals ** 2)
        RSE = np.sqrt(RSS / (n - p))
        
        return RSE

    def coefficients_SE(self) -> np.ndarray[float]:
        """
        Computes the standard error for each regression coefficient.

        This is derived from the diagonal of the variance-covariance matrix of the coefficients.

        Returns
        -------
        np.ndarray
            Standard errors for each estimated coefficient (same shape as beta).
        """
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

    def coefficients_p_values(self) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Computes the t-statistics and two-sided p-values for each coefficient
        in the fitted linear regression model.
    
        The t-statistic for each coefficient is computed as:
            t = beta / SE(beta)
    
        The p-value is then calculated from the cumulative distribution function (CDF)
        of the Student's t-distribution, assuming the null hypothesis that each coefficient is 0.
    
        Returns
        -------
        t_values : np.ndarray
            The t-statistics for each coefficient.
        p_values : np.ndarray
            The two-sided p-values for each coefficient, indicating the probability
            of observing such an extreme t-value under the null hypothesis.
        """
        # Compute t-statistics: each beta divided by its standard error
        t_values = self.beta / self.coefficients_SE()
    
        # Number of observations and number of parameters
        n = len(self.y)
        p = len(self.beta)
    
        # Compute two-tailed p-values using the t-distribution CDF
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=n - p))
    
        return t_values, p_values

    def R_squared(self) -> Tuple[float, float]:
        """
        Computes the R-squared and adjusted R-squared values for the fitted model.
    
        R-squared measures the proportion of variance in the target variable that is 
        explained by the model. Adjusted R-squared corrects for model complexity, 
        penalizing for the number of predictors.
    
        Returns
        -------
        R_squared : float
            The coefficient of determination (R²), a measure of model fit.
        R_squared_adj : float
            The adjusted R², which adjusts for the number of predictors in the model.
        """
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

    def F_score(self) -> Tuple[float, float]:
        """
        Computes the F-statistic and associated p-value for the overall model fit.
    
        The F-statistic tests the null hypothesis that all regression coefficients 
        (except the intercept) are equal to zero — i.e., that the model provides no
        better fit than a model with just the intercept.
    
        Returns
        -------
        F_stat : float
            The F-statistic value.
        p_value : float
            The p-value corresponding to the F-statistic under the null hypothesis.
        """
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
        if self.add_bias:
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
        
    def residual_stats(self) -> np.ndarray:
        """
        Returns summary statistics of the residuals from the fitted model.
    
        The output includes the five-number summary:
        minimum, first quartile (Q1), median, third quartile (Q3), and maximum.
    
        Returns
        -------
        np.ndarray
            A 1D NumPy array containing the summary statistics in the following order:
            [min, 25th percentile (Q1), median, 75th percentile (Q3), max]
        """
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

    def summary(self) -> None:
        """
        Prints a summary of the fitted linear regression model, including:
        - Residual five-number summary
        - Coefficients, standard errors, t-values, and p-values
        - Residual standard error
        - R-squared and adjusted R-squared
        - F-statistic and corresponding p-value
        """
        # Get residual summary statistics
        quartiles = self.residual_stats()
        print('Residuals:')
        print(f'Min: {quartiles[0]:.4f}')
        print(f'Q1:  {quartiles[1]:.4f}')
        print(f'Med: {quartiles[2]:.4f}')
        print(f'Q3:  {quartiles[3]:.4f}')
        print(f'Max: {quartiles[4]:.4f}\n')
    
        # Get coefficient statistics
        t, p = self.coefficients_p_values()
        coefs = np.column_stack((self.beta, self.coefficients_SE(), t, p))
    
        print(f'{"Coefficient":>12}  {"Std Error":>10}  {"t-value":>10}  {"p-value":>10}')
        for row in coefs:
            print(f'{row[0]:12.4f}  {row[1]:10.4f}  {row[2]:10.4f}  {row[3]:10.4g}')
        
        print()
    
        # Residual standard error
        rse = self.residuals_SE()
        print(f'Residual standard error: {rse:.4f}')
    
        # R-squared and Adjusted R-squared
        R, R_adj = self.R_squared()
        print(f'R-squared:             {R:.4f}')
        print(f'Adjusted R-squared:    {R_adj:.4f}')
    
        # F-statistic and p-value
        F, pF = self.F_score()
        print(f'F-statistic:           {F:.4f}')
        print(f'F-statistic p-value:   {pF:.4g}')



if __name__ == "__main__":
    pass
