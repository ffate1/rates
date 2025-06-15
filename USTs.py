from DataFetcher import DataFetcher
import pandas as pd
from typing import Optional

class USTs:
    def __init__(self,
                 auction_data: pd.DataFrame,
                 price_data: Optional[pd.DataFrame],
                 curve_data: Optional):
        self.auction_data = auction_data
        self.price_data = price_data
        if curve_data:
            self.curve_data = curve_data

    def get_nth_OTRs(self, n: int) -> pd.DataFrame:
        data = self.auction_data[["cusip", "auction_date", "security_term", "avg_med_yield", "maturity_date"]]
        data['run'] = 0

        Tenors = ["2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"]
        otr_df = pd.DataFrame()

        col_loc = data.columns.get_loc('security_term')

        for tenor in Tenors:
            runs = 0
            for i in range(len(data)):
                if runs < n:
                    if data.iloc[i, col_loc] == tenor:
                        otr_df = pd.concat([otr_df, data.iloc[[i], :]], ignore_index=True)
                        otr_df.iloc[-1, -1] = runs + 1
                        runs += 1
        return otr_df