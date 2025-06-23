import pandas as pd
import numpy as np
from typing import Optional, List
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import calendar

class USTs:
    def __init__(self,
                 auction_data: Optional[pd.DataFrame],
                 price_data: Optional[pd.DataFrame]):

        self.auction_data = auction_data
        self.price_data = price_data

    def get_current_UST_set(self):
        # Checking all necessary data is provided
        auction_check = (self.auction_data is None or self.auction_data.empty )
        price_check = (self.price_data is None or self.price_data.empty)
        if auction_check or price_check:
            print("Cannot produce UST set due to missing data")
        
        auctions = self.auction_data
        auctions = auctions[auctions['security_term'] == auctions['original_security_term']]
        auction_cols_to_keep = ['cusip', 'security_term', 'issue_date']
        auctions = auctions[auction_cols_to_keep]
        
        prices = self.price_data
        ust_set = pd.merge(prices, auctions, how='inner', left_on='Cusip', right_on='cusip')
        ust_set['issue_date'] = pd.to_datetime(ust_set['issue_date'])

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

    def get_bill_discount_rate(self, price: float, issue_date: datetime.date, maturity_date: datetime.date) -> float:
        """Returns simple discount rate using ACT/360 convention"""
        time = (maturity_date - issue_date).days
        discount_rate = (100 - price)/100 * 360/time
        return round(discount_rate * 100, 3)

    def get_bill_BEYTM(self, price: float, issue_date: datetime.date, maturity_date: datetime.date, print_steps = False) -> float:
        """Following TreasuryDirect methodology to get bond-equivalent yields for maturities
        less than or greater than 6 months"""
        time = (maturity_date - issue_date).days
        if not (time < 366):
            print(f"Days to expiry is: {time}. Ensure correct dates have been entered.")
            return None
        
        if time < 185:
            bond_equivalent_ytm = (100-price)/100 * 365/time
            return round(bond_equivalent_ytm * 100, 3)
        else:
            a = (time/(2*365)) - 0.25
            b = time/365
            c = (price - 100)/price

            bond_equivalent_ytm = ((b*-1) + np.sqrt((b*b)-(4*a*c)))/(2*a)
            if print_steps:
                print(f"A: {a}\nB: {b}\nC: {c}")
            return float(round(bond_equivalent_ytm * 100, 3))
        
    def get_bond_ytm(self,
                     price: float,
                     coupon: float,
                     issue_date: datetime.date,
                     maturity_date: datetime.date,
                     face_value: int = 100,
                     pricing_date: datetime.date = datetime.datetime.now().date()):
        pass

    def _create_date_set(self, issue_date, maturity_date) -> List[datetime.date]: # type: ignore
        tenor = maturity_date.year - issue_date.year
        coupons = tenor * 2
        today = datetime.datetime.now().date()
        payment_set = []
        if tenor in [3, 10, 20, 30]:
            DAY = 15
            first_payment = issue_date + relativedelta(months=6)
            date = datetime.date(first_payment.year, first_payment.month, DAY)
            payment_set.append(date)
            while len(payment_set) < coupons - 1:
                date = date + relativedelta(months=6)
                date = datetime.date(date.year, date.month, DAY)
                while date.weekday() in [5, 6]:
                    date = date + timedelta(days=1)
                payment_set.append(date)
        
        if tenor in [2, 5, 7]:
            month = min(issue_date.month, maturity_date.month)
            _, day = calendar.monthrange(issue_date.year, month)
            date = datetime.date(issue_date.year, month, day) + relativedelta(months=6)
            payment_set.append(date)
            while len(payment_set) < coupons - 1:
                date = date + relativedelta(months=6)
                while date.weekday() in [5, 6]:
                    date = date + timedelta(days=1)
                payment_set.append(date)

        payment_set.append(maturity_date)
        if len(payment_set) == coupons:
            print(f"{coupons} coupons dates in set, matching the expected number.")
            return payment_set
        
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