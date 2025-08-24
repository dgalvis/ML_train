{-|
Module      : LogisticRegression
Description : Minimal logistic regression implementation

This module implements a small toolkit for binary logistic regression:

* 'numFeatures' — infer the number of features (columns) from a design matrix.
* 'addIntercept' — prepend a 1.0 intercept column to a design matrix.
* 'predict' — vectorised probability predictions @p(y=1 \| x)@.
* 'logLoss' - Loss function given y and yHat (from 'predict')
* 'initBeta' — random @N(0,1)@ initial parameters (including intercept).
* 'trainGD' — batch gradient descent with simple early stopping.
* 'to01' — threshold probabilities at 0.5 to get class labels.

Notes:

* Shapes: let @m@ be number of rows (samples) and @n@ be number of features.
* After 'addIntercept', each row has length @p+1@; @beta@ must also have length @p+1@.
* Numerical safety: 'logLoss' clamps probabilities to avoid @log 0@.
* Random init: 'initBeta' uses Box–Muller to sample standard normals.
* This implementation uses full-batch gradients (no mini-batching).
-}

-- module LogisticRegression (numFeatures, addIntercept, predict, logLoss, initBeta, trainGD, to01, gradient, gdStep) where
module LogisticRegression  where

import Data.List (transpose)
import System.Random (randomRIO)

-- | Infer the number of features (columns) from an input matrix.
--
--   __Input__
--
--   * [[a]] : assumption is that all rows have the same number of elements
--   * Returns @0@ for an empty matrix of [] or [[]]
--
--   __Returns__
--
--   * Int:  the number of columns (i.e., length of each row)
--
--   __Example__
--  
--   >>> numFeatures [[1,2,3],[4,5,6]]
--   3
numFeatures :: [[a]] -> Int
numFeatures []    = 0
numFeatures [[]]    = 0
numFeatures (r:_) = length r

-- | Dot product of two vectors.
--
--  __Input__
--
--   * [Double], [Double] : two vectors, assumption is that both vectors have the same length.
--
--   __Returns__
--
--   * Double: dot product of two vectors
--
--   __Example__
--  
--   >>> dot [1,2,3] [4,5,6]
--   32.0 
dot :: [Double] -> [Double] -> Double
dot u v = sum (zipWith (*) u v)

-- | Logistic sigmoid function 
--
--   __Input__
--
--   * Double : some number z
--
--   __Returns__
--
--   * Double :  1 / (1 + e^{-z})
--
--   __Example__
--
--   >>> sigmoid 0
--   0.5
sigmoid :: Double -> Double
sigmoid z = 1.0 / (1.0 + exp ( - z))

-- | Add an intercept term by prepending @1.0@ to every feature row.
--
--   >>> addIntercept [[2,3],[4,5]]
--   [[1.0,2.0,3.0],[1.0,4.0,5.0]]
addIntercept :: [[Double]] -> [[Double]]
addIntercept = map (1.0 :)

-- | Predict probability for a single row given parameters.
--   Equivalent to @sigmoid (dot beta row)@.
predictRow :: [Double] -> [Double] -> Double
predictRow beta row = sigmoid (dot beta row)

-- | Vectorised probability predictions for a design matrix.
--
--   The input @x@ is expected to already include an intercept column
--   (use 'addIntercept'). The length of @beta@ must match each row.
predict :: [Double] -> [[Double]] -> [Double]
predict beta x = map(predictRow beta) x

-- | Logistic log-loss (negative log-likelihood / cross-entropy).
--
--   Uses a small @eps@ to clamp probabilities away from 0 and 1 for
--   numerical stability.
logLoss :: [Double] -> [Double] -> Double
logLoss y yHat =
  let eps   = 1e-12
      clamp t = max eps (min (1 - eps) t)
  in  negate $ sum (zipWith (\yi yHi -> yi*log (clamp yHi) + (1-yi)*log (clamp (1-yHi))) y yHat)

-- | Sample one standard normal @N(0,1)@ using the Box–Muller transform.
--
--   Draws two uniforms @u1 \\in (0,1], u2 \\in [0,1)@ and returns
--   @r \\cos \\theta@ where @r = \\sqrt{-2 \\log u1}@ and @\\theta = 2\\pi u2@.
normalZ :: IO Double
normalZ = do
  u1 <- randomRIO (1e-12, 1.0)  -- uniform (0,1], avoid log 0
  u2 <- randomRIO (0.0, 1.0)    -- uniform (0,1)
  let r     = sqrt (-2 * log u1)
      theta = 2 * pi * u2
  return (r * cos theta)

-- | Initialise a parameter vector @beta@ of length @(N+1)@ with @Normal(0,1)@ draws.
--
--   The extra 1 accounts for the intercept term.
initBeta :: Int -> IO [Double]
initBeta n = sequence (replicate (n+1) normalZ)

-- | Gradient of the negative log-likelihood for logistic regression
--   with respect to @beta@ (full batch).
--
--   Given @X@ (with intercept), @yHat = sigmoid(X beta)@, and @err = yHat - y@,
--   the @j@-th component is @\\sum_i err_i * x_{i,j}@.
gradient :: [Double] -> [[Double]] -> [Double] -> [Double]
gradient beta xInt y =
  let yHat    = predict beta xInt
      err  = zipWith (-) yHat y -- yHat - y
      xT   = transpose xInt -- columns of X
  in  map (\col -> sum (zipWith (*) err col)) xT

-- | One gradient-descent update:
--
-- @beta := beta - alpha * gradient(beta)@
gdStep :: Double -> [Double] -> [[Double]] -> [Double] -> [Double]
gdStep alpha beta xInt y =
  let g = gradient beta xInt y
  in  zipWith (-) beta (map (alpha *) g)

-- | Train by batch gradient descent up to @maxIters@ with early stopping.
--
--   Returns @(finalBeta, lossHistory_oldestFirst)@.
--
--   Early stopping: if the loss increases at an iteration, training stops
--   and the previous parameters are returned.
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

-- | Convert a probability to a hard prediction using a 0.5 threshold.
--
--   >>> to01 0.7
--   1.0
--   >>> to01 0.3
--   0.0
to01 :: Double -> Double
to01 p = if p > 0.5 then 1.0 else 0.0

