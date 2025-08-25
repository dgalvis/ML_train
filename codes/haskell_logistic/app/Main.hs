{-|
Module      : Main
Description : Command-line runner for the logistic regression example.

This executable:

* Reads a JSON configuration file (e.g. @config.json@) containing fields:

    * @file@     – dataset filename (string)
    * @maxIters@ – number of gradient descent iterations (integer)
    * @alpha@    – learning rate (double)

* Reads a whitespace-delimited dataset from the given file (in JSON file).  
  __Caution__: The loader expects the file to contain @n+1@ columns (where @n@ is
  the number of regressors).  
  The final column is the output with @y ∈ {0, 1}@.  
  There must be at least one row and every row must have the same length.
  The number of rows is the number of samples, @m@

* Splits the data into features (@X@) and labels (@y@).
* Adds an intercept column to @X@.
* Initialises random parameters with 'initBeta'.
* Trains logistic regression using 'trainGD'.
* Predicts probabilities and converts them into binary class predictions.
* Prints each row with its true label and predicted label.

Usage:

@
stack build
stack run config.json
@

If no configuration file is provided, the program exits with a usage message.
-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import System.Environment (getArgs)
import qualified Data.ByteString.Lazy as BL
import GHC.Generics (Generic)
import Data.Aeson (FromJSON, eitherDecode)

import DataLoader (loadCSV, splitXY)
import LogisticRegression (numFeatures, initBeta, predict, trainGD, to01, addIntercept)

-- config type, all fields required
data Config = Config
  { file     :: FilePath
  , maxIters :: Int
  , alpha    :: Double
  } deriving (Show, Generic)

instance FromJSON Config

-- | Entry point for the logistic regression demo.
--
--   Reads the dataset, trains a model, and prints predictions row by row.
main :: IO ()
main = do
  args <- getArgs
  cfg <- case args of
           (fp:_) -> do
             bs <- BL.readFile fp
             case eitherDecode bs of
               Right c -> pure c
               Left err -> fail ("Could not parse config: " ++ err)
           [] -> fail "Usage: run <config.json>"

  nums <- loadCSV (file cfg)
  let (x, y) = splitXY nums         -- split dataset into (features, labels)
      n      = numFeatures x        -- number of features

  beta <- initBeta (n + 1)              -- random initial parameter vector

  let xInt           = addIntercept x             -- add intercept term
      (betaTrained, _) = trainGD (maxIters cfg) (alpha cfg) beta xInt y -- produce optimised parameters
      p           = predict betaTrained xInt   -- predicted probabilities
      rows           = zipWith3 (\xi yi p_i -> xi ++ [yi, to01 p_i]) x y p :: [[Double]] -- regressors, output, predicted output

  putStrLn "x y pred"
  mapM_ print rows
