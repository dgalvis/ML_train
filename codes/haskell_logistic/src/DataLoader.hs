module DataLoader (loadCSV, splitXY) where

-- helper functions: split string by chosen delimeter
splitBy :: Char -> String -> [String]
splitBy d s =
  case break (== d) s of
    (chunk, "")     -> [chunk]
    (chunk, _:rest) -> chunk : splitBy d rest

-- main loader: read file, parse to Doubles
loadCSV :: FilePath -> IO [[Double]]
loadCSV fp = do
  contents <- readFile fp
  let rows = lines contents :: [String]         -- split into rows
      cols = map (splitBy ' ') rows :: [[String]]  -- split each row by commas
      nums = map (map read) cols :: [[Double]]
  return nums

splitXY :: [[Double]] -> ([[Double]], [Double])
splitXY rows = 
  let xs = map init rows -- drop last element of each row
      ys = map last rows -- keep last element of each row
  in (xs, ys)
