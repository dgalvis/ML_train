{-|
Module      : DataLoader
Description : Simple whitespace-delimited CSV loader. Splits features and target.

This module provides:

* 'loadCSV' — read a whitespace-delimited text file into a list of rows of 'Double's.
* 'splitXY' — split each row into features (all but last column) and targets (last column).
-}
module DataLoader (loadCSV, splitXY) where

import Data.List.Split (splitOn)

-- | Read a whitespace-delimited file and parse each field as a 'Double'.
--
--   __Input__
--
--   * FilePath : path to a text file where each line is one sample.
--
--   __Returns__
--
--   * IO [[Double]] : design matrix with @m@ rows and @n+1@ columns.
--
--   __Assumptions__
--
--   * Columns are separated by a single space character @" "@ (not commas).
--   * Every row has the same number of columns.
--   * Each field parses with 'read' as a 'Double'.
--   * The last column is assumed to be the target @yi ∈ {0,1}@.
--   * There must be at least one row.
--
--   __Example__
--
--   File contents:
--
--   > 1.0 2.0 0.0
--   > 3.0 4.0 1.0
--
--   >>> loadCSV "data.txt"
--   [[1.0,2.0,0.0],[3.0,4.0,1.0]]
loadCSV :: FilePath -> IO [[Double]]
loadCSV fp = do
  contents <- readFile fp
  let rows = lines contents :: [String]            -- split into rows
      cols = map (splitOn " ") rows :: [[String]]  -- split each row by spaces
      nums = map (map read) cols :: [[Double]]     -- parse fields to Doubles
  return nums

-- | Split a matrix into features @X@ and targets @y@ by row.
--
--   __Input__
--
--   * [[Double]] : data matrix with @m@ rows and @n+1@ columns.
--
--   __Returns__
--
--   * ([[Double]], [Double]) : pair @(X, y)@
--     - @X@ is @m × n@ (all but last column of each row).
--     - @y := [yi]@ is the last column, with @yi ∈ {0,1}@.
--
--   __Example__
--
--   >>> splitXY [[1.0,2.0,0.0],[3.0,4.0,1.0]]
--   ([[1.0,2.0],[3.0,4.0]],[0.0,1.0])
splitXY :: [[Double]] -> ([[Double]], [Double])
splitXY rows = 
  let xs = map init rows -- drop last element of each row
      ys = map last rows -- keep last element of each row
  in (xs, ys)

