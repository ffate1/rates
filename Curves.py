import scipy.interpolate
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Any

class ParCurves:
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.x_values = df['Time to expiry'].to_numpy()
        self.y_values = df['EOD YTM'].to_numpy()

    def univariate_spline(self, smoothness: float = 0.5, return_data: bool = True):
        smooth_spline = scipy.interpolate.UnivariateSpline(x=self.x_values,
                                                           y=self.y_values,
                                                           s=smoothness)
        if return_data:
            new_y_values = smooth_spline(np.arange(0.25, 30, 0.01))
            new_x_values = np.arange(0.25, 30, 0.01)
            return new_x_values, new_y_values
        
        return smooth_spline
    
    def Bspline_with_knots(self,
                           knots: Optional[np.ndarray],
                           k:int = 3,
                           return_data: bool = True):
        
        tck = scipy.interpolate.splrep(x=self.x_values, y=self.y_values, k=k, t=knots)
        bspline_model = scipy.interpolate.BSpline(*tck)
        if return_data:
            x_new = np.arange(90/365, 30, 1/365)
            y_curve = bspline_model(x_new)
            return x_new, y_curve
         
        return bspline_model
        

class MerrillLynchExponentialSpline(ParCurves):
    """
    Calculates the MLES yield curve for a given set of data and parameters.
    The optimization is handled by an external calibration function.
    """
    def __init__(self, alpha: float, N: int, z: np.ndarray):
        self.z = None
        self.N = N
        self.alpha = 0.1

    def e_basis(self, k: int, t: float) -> float:
        return np.exp(-self.alpha * k * t)

    def discount(self, t: float) -> float:
        return sum(self.z[k] * self.e_basis(k, t) for k in range(self.N))

    def construct_H(self, T: np.ndarray) -> np.ndarray:
        H = np.zeros((len(T), self.N))
        for j in range(len(T)):
            for k in range(self.N):
                H[j, k] = self.e_basis(k, T[j])
        return H

    def gls(self, H: np.ndarray, W: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Generalized Least Squares (GLS) estimate of the MLES basis parameters."""
        return np.linalg.inv(H.T @ W @ H) @ H.T @ W @ p

    def fit(self, maturities: np.ndarray, yields: np.ndarray, W: np.ndarray):
        """Fits the model to observed yields."""
        prices = np.exp(-yields * maturities / 100)
        H = self.construct_H(maturities)
        # Perform GLS estimation
        self.z = self.gls(H, W, prices)

    def theoretical_yields(self, maturities: np.ndarray) -> np.ndarray:
        prices = np.array([self.discount(t) for t in maturities])
        return -np.log(prices) / maturities * 100
    
def calibrate_mles(
        maturities: npt.NDArray[np.float64],
        yields: npt.NDArray[np.float64],
        N: int = 9,
        regularization: float = 1e-4,
        overnight_rate: Optional[float] = None
        ) -> Tuple[MerrillLynchExponentialSpline, Any]:
    
    if maturities[0] != 0:
        short_rate = overnight_rate or yields[0]
        maturities = np.insert(maturities, 0, 1 / 365)
        yields = np.insert(yields, 0, short_rate)

    """Fit the MLES model to the given yields using OLS."""
    initial_guess = [0.1] + [1.0] * N

    def objective(params: np.ndarray) -> float:
        alpha = params[0]
        z = np.array(params[1:])
        curve = MerrillLynchExponentialSpline(alpha, N, z)
        curve.fit(np.array(maturities), np.array(yields), np.eye(len(maturities)))
        theoretical_yields = curve.theoretical_yields(np.array(maturities))
        regularization_term = regularization * np.sum(np.diff(z) ** 2)

        return np.sum((theoretical_yields - np.array(yields)) ** 2) + regularization_term

    result = minimize(fun=objective, x0=initial_guess, method="BFGS")
    optimized_params = result.x
    optimized_alpha = optimized_params[0]
    optimized_z = np.array(optimized_params[1:])
    curve = MerrillLynchExponentialSpline(optimized_alpha, N, optimized_z)
    curve.fit(np.array(maturities), np.array(yields), np.eye(len(maturities)))

    return curve, result