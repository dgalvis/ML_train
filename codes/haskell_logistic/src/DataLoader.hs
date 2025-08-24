{-|
Module      : DataLoader
Description : Simple CSV-like loader and feature/target splitter.

This module provides:

* 'loadCSV' — reads a whitespace-delimited text file into a list of rows of 'Double's.
* 'splitXY' — splits each row into features (all but last column) and targets (last column).

__Caution__:

* The loader currently splits on a single space character (" "), not on commas.
  Update the delimiter if your data uses a different separator.
* No validation is performed; malformed lines will raise a runtime error via 'read'.
* The loader expects the csv file to contain n+1 columns (where n is the number of regressors).
  The final column is the output with y in {0, 1}.
  There has to be at least one row and every row must have the same length.
-}
module DataLoader (loadCSV, splitXY) where

import Data.List.Split (splitOn)

-- | Read a whitespace-delimited file and parse each field as a 'Double'.
--
--   * Each line becomes one row.
--   * Columns are split using a single space character (" ").
--   * Every field must parse as 'Double' (@read@ is used directly).
loadCSV :: FilePath -> IO [[Double]]
loadCSV fp = do
  contents <- readFile fp
  let rows = lines contents :: [String]         -- split into rows
      cols = map (splitOn " ") rows :: [[String]]  -- split each row by commas
      nums = map (map read) cols :: [[Double]] -- parse fields to Doubles
  return nums

-- | Split a matrix into features @Xs@ and targets @ys@ by row.
--
--   For each row, all but the last element are treated as features, and
--   the last element is treated as the target.
splitXY :: [[Double]] -> ([[Double]], [Double])
splitXY rows = 
  let xs = map init rows -- drop last element of each row
      ys = map last rows -- keep last element of each row
  in (xs, ys)
