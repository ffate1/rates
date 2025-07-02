import pandas as pd
from typing import Optional
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
                            settlement_date: datetime.date,
                            get_ytms: bool = True, 
                            include_FRNs: bool = False,
                            include_TIPS: bool = False):
        # Checking all necessary data is provided
        auction_check = (self.auction_data is None or self.auction_data.empty )
        price_check = (self.price_data is None or self.price_data.empty)
        if auction_check or price_check:
            print("Cannot produce UST set due to missing data")
            return None
        
        # Merging auction and price data
        auctions = self.auction_data
        auction_cols_to_keep = ['Cusip', 'Original security term', 'Issue date', 'Currently outstanding']
        auctions = auctions[auction_cols_to_keep]
        prices = self.price_data
        ust_set = pd.merge(prices, auctions, how='inner', on='Cusip')
        
        # Cleaning up data
        outstanding_values = ust_set[~ust_set.duplicated('Cusip', keep='first')][['Cusip', 'Currently outstanding']].set_index('Cusip')
        ust_set = ust_set.drop_duplicates(subset='Cusip', keep='last').set_index('Cusip')
        ust_set.update(outstanding_values)
        ust_set['Time to expiry'] = ((ust_set['Maturity date'] - pd.to_datetime(settlement_date)).dt.days + 1)/365
        ust_set.sort_values(by='Time to expiry', ascending=True, inplace=True)
        ust_set = ust_set.reset_index(drop=False)

        if len(ust_set) == len(prices):
            ust_set = ust_set[~ust_set['Original security term'].isin(['30-Year 3-Month', '29-Year 9-Month'])]
            if not include_FRNs:
                ust_set = ust_set[ust_set['Security type'] != 'FRN']
            if not include_TIPS:
                ust_set = ust_set[ust_set['Security type'] != 'TIPS']
            if get_ytms:
                ust_set['EOD YTM'] = ust_set.apply(lambda row: self._get_df_ytm(row, settlement_date), axis='columns')
            
            ust_set['Duration'] = ust_set.apply(lambda row: self._get_duration(row, settlement_date), axis='columns')
                        
            print("Merged auction and price data successfully\nNo missing or excess data\nAll CUSIPs are identical between DataFrames")
            ust_set = ust_set.drop(columns=['Buy', 'Sell'])
            ust_set = self._get_ranks(data=ust_set)
            return ust_set
            
        else:
            print("Length of DataFrames differs - verify data")
            print(len(ust_set), len(prices))
            return None        
    
    def get_bill_discount_rate(self,
                               price: float,
                               maturity_date: datetime.date,
                               settlement_date: datetime.date
                               ) -> float:
        """Returns  discount rate using ACT/360 convention"""
        time = (maturity_date - settlement_date).days
        discount_rate = (100 - price)/100 * 360/time
        return round(discount_rate * 100, 6)

    def get_bill_BEYTM(self,
                       price: float,
                       maturity_date: datetime.date,
                       settlement_date: datetime.date) -> float:
        """Following TreasuryDirect methodology to get bond-equivalent yields for maturities
        less than or greater than 6 months"""
        tenor = (maturity_date - settlement_date).days
        YTM = (100 - price) / price / (tenor / 365)
        return YTM * 100

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

    def calculate_accrued_interest(self, dates_and_cashflows, settlement_date) -> float:
        for date, _ in dates_and_cashflows:
            if date > settlement_date:
                next_coupon_date = date
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = dates_and_cashflows[previous_index][0]
                break

        days_in_period = (next_coupon_date - last_date).days
        days_accrued = (settlement_date - last_date).days
        if days_in_period == 0:
            return 0.0

        coupon_amt = dates_and_cashflows[1][1]
        accrued = coupon_amt * days_accrued / days_in_period
        return accrued

    def calculate_bond_price(self,
                             issue_date: datetime.date,
                             maturity_date: datetime.date,
                             settlement_date: datetime.date,
                             coupon: float,
                             discount_rate: float,
                             dirty: bool = False
                             ) -> float:
        
        dates_and_cashflows = self.get_dates_and_cashflows(issue_date=issue_date,
                                                        maturity_date=maturity_date,
                                                        coupon=coupon)
        for date, _ in dates_and_cashflows:
            if date > settlement_date:
                first_pmt_date = self.adjust_for_bad_day(date)
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = self.adjust_for_bad_day(dates_and_cashflows[previous_index][0])
                first_pmt_period = (first_pmt_date - last_date).days
                time_to_pmt = (first_pmt_date - settlement_date).days
                break
        first_period_fraction = time_to_pmt / first_pmt_period

        PV = 0
        exponent = first_period_fraction
        previous_date = None
        for unadjusted_date, cashflow in dates_and_cashflows:
            date = self.adjust_for_bad_day(unadjusted_date)
            if date > settlement_date:
                if previous_date:
                    days_in_period = (date - previous_date).days
                    exponent += (days_in_period / (365.25 / 2))
                pmt_pv = cashflow / ((1 + discount_rate/2) ** exponent)
                PV += pmt_pv
                previous_date = date

        if not dirty: # Clean price
            accrued = self.calculate_accrued_interest(dates_and_cashflows=dates_and_cashflows,
                                                settlement_date=settlement_date)
            PV -= accrued
        return PV

    def error_function(self,
                       discount_rate: float,
                       price: float,
                       issue_date: datetime.date,
                       maturity_date: datetime.date,
                       settlement_date: datetime.date,
                       coupon: float,
                       dirty: bool=False
                       ) -> float:
        calculated_price = self.calculate_bond_price(issue_date=issue_date,
                                                maturity_date=maturity_date,
                                                discount_rate=discount_rate,
                                                coupon=coupon,
                                                settlement_date=settlement_date,
                                                dirty=dirty)
        error = price - calculated_price
        return error


    def get_coupon_ytm(self,
                       price: float, 
                       issue_date: datetime.date,
                       maturity_date: datetime.date,
                       settlement_date: datetime.date,
                       coupon: float,
                       dirty: bool=False
                       ) -> float:
        guess = coupon / 100
        try:
            ytm = newton(
                func=self.error_function,
                x0=guess,
                args=(price, issue_date, maturity_date, settlement_date, coupon, dirty)
            )
            return round(float(ytm * 100), 6)
        except RuntimeError:
            return None
    
    def _get_df_ytm(self, row: pd.Series, settlement_date: datetime.date) -> float:
        if row['Security type'] == 'Bill':
            ytm = self.get_bill_BEYTM(row['End of day'],
                                      row['Maturity date'].to_pydatetime(),
                                      settlement_date)
        elif row['Security type'] in ['Note', 'Bond']:
            ytm = self.get_coupon_ytm(price=row['End of day'],
                                      issue_date=row['Issue date'].to_pydatetime(),
                                      maturity_date=row['Maturity date'].to_pydatetime(),
                                      settlement_date=settlement_date,
                                      coupon=row['Rate'],
                                      dirty=False)
        return ytm
    
    def _get_duration(self, row: pd.Series, settlement_date: datetime.date):
        maturity = row['Maturity date'].to_pydatetime()
        issue = row['Issue date'].to_pydatetime()
        ytm = row['EOD YTM']
        rate = row['Rate']
        
        dates_and_cashflows = self.get_dates_and_cashflows(issue_date=issue,
                                                        maturity_date=maturity,
                                                        coupon=rate)
        for date, _ in dates_and_cashflows:
            if date > settlement_date:
                first_pmt_date = self.adjust_for_bad_day(date)
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = self.adjust_for_bad_day(dates_and_cashflows[previous_index][0])
                first_pmt_period = (first_pmt_date - last_date).days
                time_to_pmt = (first_pmt_date - settlement_date).days
                break
        first_period_fraction = time_to_pmt / first_pmt_period

        PV = 0
        TW_PV = 0
        t = 0
        exponent = first_period_fraction
        previous_date = None
        for unadjusted_date, cashflow in dates_and_cashflows:
            date = self.adjust_for_bad_day(unadjusted_date)
            if date > settlement_date:
                if previous_date:
                    days_in_period = (date - previous_date).days
                    exponent += (days_in_period / (365.25 / 2))
                pmt_pv = cashflow / ((1 + ytm/2) ** exponent)
                PV += pmt_pv
                TW_PV += pmt_pv * exponent
                previous_date = date

        mac_duration = TW_PV / PV / 2
        modified_duration = mac_duration / (1 + (ytm / 2)) 

        return modified_duration
        
    def _get_ranks(self, data: pd.DataFrame) -> pd.DataFrame:
        data.sort_values(
            by=['Original security term', 'Time to expiry'],
            ascending=[True, False],
            inplace=True
        )
        data['Rank'] = data.groupby(by='Original security term').cumcount() + 1
        data.sort_values(by='Time to expiry', ascending=True, inplace=True)
        return data