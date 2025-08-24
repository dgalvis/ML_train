module Main (main) where

import Data.List (transpose)
import System.Random (randomRIO)
import System.Environment (getArgs)
import DataLoader (loadCSV, splitXY)

-- find the number of features in a design matrix
numFeatures :: [[a]] -> Int
numFeatures []    = 0
numFeatures (r:_) = length r

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
-- sseLoss :: [Double] -> [Double] -> Double
-- sseLoss y yHat = sum (map (**2) (zipWith (-) y yHat))

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

-- initialise a beta vector of length (p+1) with random N(0,1) values
initBeta :: Int -> IO [Double]
initBeta p = sequence (replicate (p+1) normalZ)


-- transpose function (instructive!)
-- transpose :: [[a]] -> [[a]]
-- transpose [] = []
-- transpose ([] :_) = []
-- transpose rows = map head rows : transpose (map tail rows)

-- Gradient (negative log-likelihood) for logistic regression:
-- gradient_j = sum_i (p_i - y_i) * x_{i,j}
gradient :: [Double] -> [[Double]] -> [Double] -> [Double]
gradient beta xInt y =
  let yHat    = predict beta xInt
      err  = zipWith (-) yHat y -- p - y
      xT   = transpose xInt -- columns of X
  in  map (\col -> sum (zipWith (*) err col)) xT

-- one GD step: beta <- beta - alpha * Gradient
gdStep :: Double -> [Double] -> [[Double]] -> [Double] -> [Double]
gdStep alpha beta xInt y =
  let g = gradient beta xInt y
  in  zipWith (-) beta (map (alpha *) g)

-- train with up to maxIters, early-stop if loss increases
-- returns (finalBeta, lossHistory_oldestFirst)
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

-- turn probability into prediction
to01 :: Double -> Double
to01 p = if p > 0.5 then 1.0 else 0.0

main :: IO ()
main = do
  args <- getArgs
  let fp = case args of
             (file:_) -> file -- first argument
             []       -> "data.csv" -- default
  nums <- loadCSV fp
  let (x, y) = splitXY nums -- inputs/outputs as ([[Double]], [Double])
      p = numFeatures x

  beta <- initBeta p -- define parameter list

  let xInt = addIntercept x -- add intercept to design matrix
      maxIters = 5000
      alpha = 0.01
      (betaTrained, _) = trainGD maxIters alpha beta xInt y
      yHat = predict betaTrained xInt
      rows = zipWith3 (\xs yi yHi -> xs ++ [yi, to01 yHi]) x y yHat :: [[Double]]

  putStrLn "x y pred"
  mapM_ print (rows)
