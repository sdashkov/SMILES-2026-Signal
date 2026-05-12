# SMILES-2026 Signal Interference Cancellation

## Environment

Python version:
- Python 3.10+

Required packages:
```bash
pip install numpy scipy gdown

python applicant_solution.py
```
The script downloads challenge.mat, runs the baseline and my solution, and generates results.json.

## Final result
- baseline:
  ~4.02 dB
- my solution:
  ~13.32 dB

## Solution overview

The main idea of the solution is to separate two different types of interference described in the task
 1. TX nonlinear intereference
 2. External spatially coherent intereference

## Main changes compared to baseline

1. Processing only inside the scoring band
The scorer evaluates only a narrow frequency region.
Instead of processing the entire spectrum equally so firstly I applied the same band filter used in scoring and performed most operations inside this band that made the estimation more stable and focused optimization directly on the evaluated region.

2. External interference estimation
After obtaining the initial TX-related interference estimate, I additionally modeled the remaining residual as a spatially coherent rank-1 interference component shared across all RX channels, since direct cancellation using only the baseline TX prediction was still leaving a strong correlated structure inside the scoring band. To estimate this component, I computed the covariance matrix of the residual signal after TX cancellation and reconstructed the dominant coherent interference contribution using the principal eigenvector corresponding to the largest eigenvalue.

3. Iterative refinement
Since the TX-driven interference estimate and the coherent external interference estimate affect each other, performing both estimation steps only once produced noticeably worse separation results, especially in channels with stronger residual correlation. To improve the stability of the decomposition, I used an iterative refinement procedure in which the coherent component was estimated from the residual after TX cancellation, removed from the received signal, and then the TX-related interference was re-estimated again using the updated residual signal.

4. Final fitting
After estimating both interference components, I performed a joint complex least-squares fitting step independently for each RX channel in order to determine stable subtraction coefficients before applying the final cancellation in the time domain.
Additionally, I introduced small regularization terms together with coefficient magnitude clipping to reduce numerical instability and prevent occasional over-cancellation artifacts that appeared during earlier experiments.
