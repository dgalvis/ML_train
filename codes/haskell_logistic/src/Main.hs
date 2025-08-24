{-|
Module      : Main
Description : Command-line runner for the logistic regression example.

This executable:

- Reads a whitespace-delimited dataset from a CSV-like file.
  __Caution__: The loader expects the csv file to contain n+1 columns (where n is the number of regressors).
  The final column is the output with y in {0, 1}.
  There has to be at least one row and every row must have the same length.
- Splits the data into features (@X@) and labels (@y@).
- Adds an intercept column to @X@.
- Initialises random parameters with 'initBeta'.
- Trains logistic regression using 'trainGD'.
- Predicts probabilities and converts them into binary class predictions.
- Prints each row with its true label and predicted label.


Usage:

@
stack run <datafile>
@

If no file is provided, defaults to @\"data.csv\"@ in the working directory.
-}
module Main (main) where

import System.Environment (getArgs)
import DataLoader (loadCSV, splitXY)
import LogisticRegression (numFeatures, initBeta, predict, trainGD, to01, addIntercept)

-- | Entry point for the logistic regression demo.
--
--   Reads the dataset, trains a model, and prints predictions row by row.
main :: IO ()
main = do
  args <- getArgs
  let fp = case args of
             (file:_) -> file          -- first CLI argument = file path
             []       -> "data.csv"    -- default filename if none provided
  nums <- loadCSV fp
  let (x, y) = splitXY nums         -- split dataset into (features, labels)
      n      = numFeatures x        -- number of features

  beta <- initBeta n                -- random initial parameter vector

  let xInt           = addIntercept x             -- add intercept term
      maxIters       = 5000                       -- maximum iterations
      alpha          = 0.01                       -- learning rate
      (betaTrained, _) = trainGD maxIters alpha beta xInt y
      yHat           = predict betaTrained xInt   -- predicted probabilities
      -- append true y and thresholded prediction to each feature row
      rows           = zipWith3 (\xs yi yHi -> xs ++ [yi, to01 yHi]) x y yHat :: [[Double]]

  putStrLn "x y pred"
  mapM_ print rows

