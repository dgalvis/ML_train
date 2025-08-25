{-|
Module      : LogisticRegression
Description : Minimal logistic regression implementation

This module implements a small toolkit for binary logistic regression:

* 'numFeatures' — infer the number of features (columns) from a design matrix.
* 'addIntercept' — prepend a 1.0 intercept column to a design matrix.
* 'predict' — vectorised probability predictions @p_i := p(yi=1 \| xi)@ for sample @i@.
* 'logLoss' — loss function given @y@ and @p@ (from 'predict').
* 'initBeta' — random @Normal(0,1)@ initial parameters (including intercept).
* 'trainGD' — batch gradient descent with simple early stopping.
* 'to01' — threshold probabilities at @yHati := p_i > 0.5@ for sample @i@ to get class labels.

Notes:

* Shapes: let @m@ be number of rows (samples) and @n@ be number of features.
* After 'addIntercept', each row has length @n+1@; @beta@ must also have length @n+1@.
* Numerical safety: 'logLoss' clamps probabilities to avoid @log 0@.
* Random init: 'initBeta' uses Box–Muller to sample standard normals.
* This implementation uses full-batch gradients (no mini-batching).
-}

module LogisticRegression (numFeatures, addIntercept, predict, logLoss, initBeta, trainGD, to01) where

import Data.List (transpose)
import System.Random (randomRIO)

-- | Infer the number of features (columns) from an input matrix.
--
--   __Input__
--
--   * [[a]] : design matrix, assumption is that all rows have the same number of elements.
--   * Returns @0@ for an empty matrix of @[]@ or @[[]]@.
--
--   __Returns__
--
--   * Int : the number of columns (i.e., length of each row).
--
--   __Example__
--
--   >>> numFeatures [[1,2,3],[4,5,6]]
--   3
numFeatures :: [[a]] -> Int
numFeatures []    = 0
numFeatures [[]]  = 0
numFeatures (r:_) = length r

-- | Dot product of two vectors.
--
--   __Input__
--
--   * [Double], [Double] : two vectors (same length).
--
--   __Returns__
--
--   * Double : @sum_j (u_j * v_j)@.
--
--   __Example__
--
--   >>> dot [1,2,3] [4,5,6]
--   32.0
dot :: [Double] -> [Double] -> Double
dot u v = sum (zipWith (*) u v)

-- | Logistic sigmoid function.
--
--   __Input__
--
--   * Double : a scalar @z@.
--
--   __Returns__
--
--   * Double : @sigmoid(z) = 1 / (1 + e^{-z})@.
--
--   __Example__
--
--   >>> sigmoid 0
--   0.5
sigmoid :: Double -> Double
sigmoid z = 1.0 / (1.0 + exp ( - z))

-- | Add an intercept column to the design matrix by prepending @1.0@ to every feature row.
--
--   __Input__
--
--   * [[Double]] : X, design matrix @X := [xi]@ with each @xi \in R^n@.
--
--   __Returns__
--
--   * [[Double]] : XInt, matrix with intercept column, i.e., each row becomes @xi:= [1.0] ++ xi@.
--
--   __Example__
--
--   >>> addIntercept [[2,3],[4,5]]
--   [[1.0,2.0,3.0],[1.0,4.0,5.0]]
addIntercept :: [[Double]] -> [[Double]]
addIntercept = map (1.0 :)

-- | Probability prediction for a single sample given parameters.
--
--   __Input__
--
--   * [Double] : @beta@, parameters of length @n+1@ (including intercept).
--   * [Double] : @xi@, a single sample with a prepended @1.0@ (i.e., already intercept-augmented).
--
--   __Returns__
--
--   * Double : @p_i := p(yi = 1 | xi) = sigmoid(dot beta xi)@ for this sample.
--
--   __Example__
--
--   >>> predictRow [1,2,3] [1,4,6]   -- 1 is the intercept already in xi
--   sigmoid (1*1 + 2*4 + 3*6)
predictRow :: [Double] -> [Double] -> Double
predictRow beta row = sigmoid (dot beta row)

-- | Vectorised probability predictions for a design matrix.
--
--   __Input__
--
--   * [Double]   : @beta@, parameters (length @n+1@).
--   * [[Double]] : @XInt@, matrix of intercept-augmented samples @xi@ (each starts with @1.0@).
--
--   __Returns__
--
--   * [Double] : @p := [p_i]@ where @p_i := p(yi = 1 | xi)@.
--
--   __Example__
--
--   >>> predict [1,2,3] [[1,2,3],[1,4,5]]
--   [ sigmoid(1*1 + 2*2 + 3*3)
--   , sigmoid(1*1 + 2*4 + 3*5)
--   ]
predict :: [Double] -> [[Double]] -> [Double]
predict beta x = map (predictRow beta) x

-- | Convert a probability @p_i@ to a hard prediction @yHati@ using a 0.5 threshold.
--
--   __Input__
--
--   * Double : @p_i in (0,1)@.
--
--   __Returns__
--
--   * Double : @yHati in {0,1}@.
--
--   __Example__
--
--   >>> to01 0.7
--   1.0
--   >>> to01 0.3
--   0.0
to01 :: Double -> Double
to01 p_i = if p_i > 0.5 then 1.0 else 0.0

-- | Logistic log-loss (negative log-likelihood / cross-entropy).
--
--   __Input__
--
--   * [Double] : @y := [yi]@, true labels with @yi in {0,1}@.
--   * [Double] : @p := [p_i]@, predicted probabilities with @p_i in (0,1)@.
--
--   __Returns__
--
--   * Double : @L = - sum_i (yi log p_i + (1-yi) log(1-p_i))@.
--
--   __Notes__
--
--   * Uses a small @eps@ to clamp probabilities away from 0 and 1 for numerical stability.
--
--   __Example__
--
--   >>> logLoss [1.0, 0.0] [0.9, 0.1]
--   -- equals: -( log 0.9 + log 0.9 )
--   0.210721031...
logLoss :: [Double] -> [Double] -> Double
logLoss y p =
  let eps   = 1e-12
      clamp t = max eps (min (1 - eps) t)
  in  negate $ sum (zipWith (\yi p_i -> yi*log (clamp p_i) + (1-yi)*log (clamp (1-p_i))) y p)

-- | Sample one standard normal @Normal(0,1)@ using the Box–Muller transform.
--
--   __Inputs__
--
--   * N/A
--
--   __Returns__
--
--   * IO Double : @z ~ Normal(0,1)@.
--
--   __Example__
--
--   >>> z <- normalZ
--   >>> print z
--   -- prints a random number
normalZ :: IO Double
normalZ = do
  u1 <- randomRIO (1e-12, 1.0)  -- uniform (0,1], avoid log 0
  u2 <- randomRIO (0.0, 1.0)    -- uniform (0,1)
  let r     = sqrt (-2 * log u1)
      theta = 2 * pi * u2
  return (r * cos theta)

-- | Initialise a parameter vector @beta@ of length @(n+1)@ with @Normal(0,1)@ draws.
--
--   __Input__
--
--   * Int : @n@, number of features (without intercept).
--
--   __Returns__
--
--   * IO [Double] : parameter vector @[beta_j]@ where each @beta_j ~ Normal(0,1)@.
--
--   __Example__
--
--   >>> b <- initBeta 3  -- for n=3 features -> length 4
--   >>> print (length b)
--   3
initBeta :: Int -> IO [Double]
initBeta n = sequence (replicate (n) normalZ)

-- | Gradient of the negative log-likelihood for logistic regression
--   with respect to @beta@ (full batch).
--
--   __Input__
--
--   * [Double]   : @beta@, current parameters (length @n+1@).
--   * [[Double]] : @XInt@, intercept-augmented design matrix (each row is @xi@ with leading @1.0@).
--   * [Double]   : @y := [yi]@, labels.
--
--   __Returns__
--
--   * [Double] : @gradient wrt beta of L = XInt^T (p - y)@ where @p := [pi]@ and @pi := sigmoid(dot beta xi)@.
--
--   __Notes__
--
--   * Implementation computes @err = p - y@ then multiplies by columns of @XInt@.
gradient :: [Double] -> [[Double]] -> [Double] -> [Double]
gradient beta xInt y =
  let p = predict beta xInt
      err  = zipWith (-) p y -- yHat - y
      xT   = transpose xInt     -- columns of X
  in  map (\col -> sum (zipWith (*) err col)) xT

-- | One gradient-descent update: @beta := beta - alpha * gradient(beta)@.
--
--   __Input__
--
--   * Double     : @alpha@, learning rate.
--   * [Double]   : @beta@, current parameters.
--   * [[Double]] : @XInt@, intercept-augmented design matrix.
--   * [Double]   : @y := [yi]@, labels.
--
--   __Returns__
--
--   * [Double] : @beta@, updated parameters after one step.
gdStep :: Double -> [Double] -> [[Double]] -> [Double] -> [Double]
gdStep alpha beta xInt y =
  let g = gradient beta xInt y
  in  zipWith (-) beta (map (alpha *) g)

-- | Train by batch gradient descent up to @maxIters@ with early stopping.
--
--   __Input__
--
--   * Int        : @maxIters@, maximum number of iterations.
--   * Double     : @alpha@, learning rate.
--   * [Double]   : @beta0@, initial parameters.
--   * [[Double]] : @xInt@, intercept-augmented design matrix.
--   * [Double]   : @y := [yi]@, labels.
--
--   __Returns__
--
--   * ([Double], [Double]) : @(finalBeta, lossHistory_oldestFirst)@.
--
--   __Early stopping__
--
--   * If the loss increases at an iteration, training stops and the previous
--     parameters are returned (the loss history excludes the increasing step).
--
--   __Example__
--
--   >>> let (b, hist) = trainGD 100 0.1 b0 xInt y
--   >>> print(length hist > 10)
--   True
trainGD :: Int -> Double -> [Double] -> [[Double]] -> [Double] -> ([Double], [Double])
trainGD maxIters alpha beta0 xInt y =
  let l0 = logLoss y (predict beta0 xInt)
  in  go 0 beta0 l0 [l0]
  where
    go k beta prevLoss acc
      | k >= maxIters = (beta, reverse acc)
      | otherwise =
          let beta' = gdStep alpha beta xInt y
              loss' = logLoss y (predict beta' xInt)
          in  if loss' > prevLoss
                then (beta, reverse acc)        -- early stop; keep previous beta
                else go (k + 1) beta' loss' (loss' : acc)
