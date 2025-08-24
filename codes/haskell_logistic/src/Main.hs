module Main where

import System.Random (randomRIO)
import DataLoader (loadCSV, splitXY)

-- compute dot product of two vectors
dot :: [Double] -> [Double] -> Double
dot u v = sum (zipWith (*) u v)

-- sigmoid function
sigmoid :: Double -> Double
sigmoid z = 1.0 / (1.0 + exp ( - z))

-- add a leading 1.0 to every feature
addIntercept :: [[Double]] -> [[Double]]
addIntercept = map (1.0 :)

-- elementwise linear predictor -> probability
predictRow :: [Double] -> [Double] -> Double
predictRow beta row = sigmoid (dot beta row)

-- vectorised predictions for a design matrix
predict :: [Double] -> [[Double]] -> [Double]
predict beta x = map(predictRow beta) x

-- sum of squared errors (useful for sanity checks)
sseLoss :: [Double] -> [Double] -> Double
sseLoss y yHat = sum (map (**2) (zipWith (-) y yHat))

-- the logistic log-loss (cross-entropy)
-- add an epsilon for numerical safety
logLoss :: [Double] -> [Double] -> Double
logLoss y yHat =
  let eps   = 1e-12
      clamp t = max eps (min (1 - eps) t)
  in  negate $ sum (zipWith (\yi yHi -> yi*log (clamp yHi) + (1-yi)*log (clamp (1-yHi))) y yHat)

-- generate one standard normal N(0,1) using Box–Muller
normalZ :: IO Double
normalZ = do
  u1 <- randomRIO (1e-12, 1.0)  -- uniform (0,1], avoid log 0
  u2 <- randomRIO (0.0, 1.0)    -- uniform (0,1)
  let r     = sqrt (-2 * log u1)
      theta = 2 * pi * u2
  return (r * cos theta)

main :: IO ()
main = do
  nums <- loadCSV "data.csv"
  let (x, y) = splitXY nums -- inputs/outputs as ([[Double]], [Double])
      p = case x of
            [] -> 0
            (r:_) -> length(r) -- number of regressors

  beta <- sequence ( replicate (p+1) normalZ) -- define parameter list

  let xInt = addIntercept x -- add intercept to design matrix
      yHat = predict beta xInt -- prediction
      sse = sseLoss y yHat
      lle = logLoss y yHat

  print(sse)
  print(lle)

