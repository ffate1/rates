import pandas as pd
import numpy as np
from typing import Optional, List
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import newton
import pandas_market_calendars as mcal

class USTs:
    def __init__(self,
                 auction_data: Optional[pd.DataFrame],
                 price_data: Optional[pd.DataFrame]):

        self.auction_data = auction_data
        self.price_data = price_data
        
        holiday_array = mcal.get_calendar('NYSE').holidays().holidays
        start, end = datetime.date(1990, 1, 1), datetime.date(2060, 1, 1)
        holidays_list = [pd.to_datetime(np_date).date() for np_date in holiday_array]
        self.holidays = [holiday for holiday in holidays_list if start < holiday < end]

    def get_current_UST_set(self,
                            as_of_date: Optional[datetime.date],
                            get_ytms: bool = True, 
                            include_FRNs: bool = False,
                            include_TIPS: bool = False):
        # Checking all necessary data is provided
        auction_check = (self.auction_data is None or self.auction_data.empty )
        price_check = (self.price_data is None or self.price_data.empty)
        if auction_check or price_check:
            print("Cannot produce UST set due to missing data")
            return None
        
        auctions = self.auction_data
        auctions = auctions[auctions['security_term'] == auctions['original_security_term']]
        auction_cols_to_keep = ['cusip', 'security_term', 'issue_date']
        auctions = auctions[auction_cols_to_keep]
        
        prices = self.price_data
        ust_set = pd.merge(prices, auctions, how='inner', left_on='Cusip', right_on='cusip')
        ust_set['issue_date'] = pd.to_datetime(ust_set['issue_date'])

        if len(ust_set) == len(prices):
            if not include_FRNs:
                ust_set = ust_set[ust_set['Security type'] != 'FRN']
            if not include_TIPS:
                ust_set = ust_set[ust_set['Security type'] != 'TIPS']
            if get_ytms:
                for row in range(len(ust_set)):
                    if ust_set.loc[row, 'Security type'] == 'Bill':
                        ust_set.loc[row, 'EOD YTM'] = self.get_bill_BEYTM(price=ust_set.loc[row, 'End of day'],
                                                                      issue_date=ust_set.loc[row, 'issue_date'].date(),
                                                                      maturity_date=ust_set.loc[row, 'Maturity date'].date())
                    elif ust_set.loc[row, 'Security type'] in ['Note', 'Bond']:
                        ust_set.loc[row, 'EOD YTM'] = self.get_coupon_ytm(price=ust_set.loc[row, 'End of day'],
                                                                      issue_date=ust_set.loc[row, 'issue_date'].date(),
                                                                      maturity_date=ust_set.loc[row, 'Maturity date'].date(),
                                                                      as_of_date=as_of_date,
                                                                      coupon=ust_set.loc[row, 'Rate'],
                                                                      dirty=False)
            if bool((ust_set['Cusip'] == ust_set['cusip']).all()):
                print("Merged auction and price data successfully\nNo missing or excess data\nAll CUSIPs are identical between DataFrames")
                ust_set = ust_set.drop(columns='cusip')
                return ust_set
            else:
                print("CUSIPs differ between DataFrames - verify data")
                return None
        else:
            print("Length of DataFrames differs - verify data")
            print(len(ust_set), len(prices))
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
    
    def adjust_for_bad_day(self, date: datetime.date) -> datetime.date: # type: ignore
        while date.weekday() in [5, 6] or date in self.holidays:
            date = date + timedelta(days=1)
        return date

    def get_dates_and_cashflows(self,
                                issue_date: datetime.date,
                                maturity_date: datetime.date,
                                coupon: float,
                                FV: int = 100):
        payment_dates = [maturity_date]
        current_date = maturity_date
        while current_date > issue_date:
            current_date = maturity_date - relativedelta(months=len(payment_dates) * 6)
            if current_date > issue_date:
                payment_dates.append(current_date)
        payment_dates = sorted(payment_dates)
        
        coupon_amt = coupon / 2
        return_list = [(issue_date, 0.0)] # Issue date
        for date in payment_dates[:-1]:
            return_list.append((date, coupon_amt))
        return_list.append((payment_dates[-1], coupon_amt + FV)) # Maturity with principal repayment
        return return_list

    def calculate_accrued_interest(self, dates_and_cashflows, as_of_date) -> float:
        for date, _ in dates_and_cashflows:
            if date > as_of_date:
                next_coupon_date = date
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = dates_and_cashflows[previous_index][0]
                break

        days_in_period = (next_coupon_date - last_date).days
        days_accrued = (as_of_date - last_date).days
        if days_in_period == 0:
            return 0.0

        coupon_amt = dates_and_cashflows[1][1]
        accrued = coupon_amt * days_accrued / days_in_period
        return accrued

    def calculate_bond_price(self,
                             issue_date: datetime.date,
                             maturity_date: datetime.date,
                             as_of_date: datetime.date,
                             coupon: float,
                             discount_rate: float,
                             dirty: bool = False
                             ) -> float:
        
        dates_and_cashflows = self.get_dates_and_cashflows(issue_date=issue_date,
                                                        maturity_date=maturity_date,
                                                        coupon=coupon)
        for date, _ in dates_and_cashflows:
            if date > as_of_date:
                first_pmt_date = self.adjust_for_bad_day(date)
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = self.adjust_for_bad_day(dates_and_cashflows[previous_index][0])
                first_pmt_period = (first_pmt_date - last_date).days
                time_to_pmt = (first_pmt_date - as_of_date).days
                break
        first_period_fraction = time_to_pmt / first_pmt_period

        PV = 0
        exponent = first_period_fraction
        previous_date = None
        for unadjusted_date, cashflow in dates_and_cashflows:
            date = self.adjust_for_bad_day(unadjusted_date)
            if date > as_of_date:
                if previous_date:
                    days_in_period = (date - previous_date).days
                    exponent += (days_in_period / (365.25 / 2))
                pmt_pv = cashflow / ((1 + discount_rate/2) ** exponent)
                PV += pmt_pv
                previous_date = date

        if not dirty: # Clean price
            accrued = self.calculate_accrued_interest(dates_and_cashflows=dates_and_cashflows,
                                                as_of_date=as_of_date)
            PV -= accrued
        return PV

    def error_function(self,
                       discount_rate: float,
                       price: float,
                       issue_date: datetime.date,
                       maturity_date: datetime.date,
                       as_of_date: datetime.date,
                       coupon: float,
                       dirty: bool=False
                       ) -> float:
        calculated_price = self.calculate_bond_price(issue_date=issue_date,
                                                maturity_date=maturity_date,
                                                discount_rate=discount_rate,
                                                coupon=coupon,
                                                as_of_date=as_of_date,
                                                dirty=dirty)
        error = price - calculated_price
        return error


    def get_coupon_ytm(self,
                       price: float, 
                       issue_date: datetime.date,
                       maturity_date: datetime.date,
                       as_of_date: datetime.date,
                       coupon: float,
                       dirty: bool=False
                       ) -> float:
        guess = coupon / 100
        try:
            ytm = newton(
                func=self.error_function,
                x0=guess,
                args=(price, issue_date, maturity_date, as_of_date, coupon, dirty)
            )
            return round(float(ytm * 100), 6)
        except RuntimeError:
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