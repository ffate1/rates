from DataFetcher import DataFetcher
import pandas as pd
from typing import Optional

class USTs:
    def __init__(self,
                 auction_data: Optional[pd.DataFrame],
                 price_data: Optional[pd.DataFrame],
                 discount_curve: Optional):

        self.auction_data = auction_data
        self.price_data = price_data
        self.discounts = discount_curve

    def get_current_UST_set(self):
        # Checking all necessary data is provided
        auction_check = (self.auction_data.empty or self.auction_data is None)
        price_check = (self.price_data.empty or self.price_data is None)
        if auction_check or price_check:
            print("Cannot produce UST set due to missing data")
        
        auctions = self.auction_data
        auctions = auctions[auctions['security_term'] == auctions['original_security_term']]
        auction_cols_to_keep = ['cusip', 'security_term', 'auction_date']
        auctions = auctions[auction_cols_to_keep]
        
        prices = self.price_data
        ust_set = pd.merge(prices, auctions, how='inner', left_on='Cusip', right_on='cusip')

        if len(ust_set) == len(prices):
            if bool((ust_set['Cusip'] == ust_set['cusip']).all()):
                print("Merged auction and price data successfully\nNo missing or excess data\nAll CUSIPs are identical between DataFrames")
                ust_set = ust_set.drop(columns='cusip')
                return ust_set
            else:
                print("CUSIPs differ between DataFrames - verify data")
                return None
        else:
            print("Length of DataFrames differs - verify data")
            return None        

    def get_nth_OTRs(self, n: int) -> pd.DataFrame:
        data = self.auction_data[["cusip", "auction_date", "security_term", "avg_med_yield", "maturity_date"]].copy()
        data.loc[:, 'run'] = 0

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
        otr_df['maturity_date'] = pd.to_datetime(otr_df['maturity_date'])
        otr_df['auction_date'] = pd.to_datetime(otr_df['auction_date'])
        otr_df['avg_med_yield'] = pd.to_numeric(otr_df['avg_med_yield'], errors='coerce')              
        return otr_df
    
