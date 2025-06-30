from scipy.interpolate import CubicSpline
import pandas as pd


class ParCurves:
    def __init__(self, df: pd.DataFrame):
        self.data = df[['Time to expiry', 'EOD YTM']]
    
    def cubicspline_interpolator(self):
        data = self.data.drop_duplicates('Time to expiry')
        f = CubicSpline(x=data['Time to expiry'].to_numpy(),
                        y=data['EOD YTM'].to_numpy(),
                        bc_type='natural')
        return f
    
# Bspline & Merrill Lynch spline to do