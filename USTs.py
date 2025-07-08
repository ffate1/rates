import scipy.interpolate
from DataFetcher import DataFetcher
from Curves import ParCurves, calibrate_mles
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import scipy
import pandas_market_calendars as mcal
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


class USTs:
    def __init__(self, date: datetime.date):

        self.auction_data = DataFetcher().fetch_auction_data()
        self.price_data = DataFetcher().fetch_historical_UST_data(
            date=datetime.datetime(date.year, date.month, date.day)
        )
        
        holiday_array = mcal.get_calendar('NYSE').holidays().holidays
        start, end = datetime.date(1990, 1, 1), datetime.date(2060, 1, 1)
        holidays_list = [pd.to_datetime(np_date).date() for np_date in holiday_array]
        self.holidays = [holiday for holiday in holidays_list if start < holiday < end]

    def get_current_UST_set(self,
                            settlement_date: datetime.date,
                            auction_data: Optional[pd.DataFrame] = None,
                            price_data: Optional[pd.DataFrame] = None,
                            get_ytms: bool = True, 
                            include_FRNs: bool = False,
                            include_TIPS: bool = False,
                            include_outstanding = False):
        # Checking all necessary data is provided
        if auction_data is None:
            auction_data = self.auction_data
        if price_data is None:
            price_data = self.price_data

        auction_check = (auction_data is None or auction_data.empty)
        price_check = (price_data is None or price_data.empty)
        if auction_check or price_check:
            print("Cannot produce UST set due to missing data")
            return None
        
        # Merging auction and price data
        auction_cols_to_keep = ['Cusip', 'Original security term', 'Issue date', 'Currently outstanding']
        auctions = auction_data[auction_cols_to_keep].copy()
        auctions = auctions.rename(columns={'Original security term': 'Security term'})
        prices = price_data.copy()
        ust_set = pd.merge(prices, auctions, how='inner', on='Cusip')
        
        # Cleaning up data
        outstanding_values = ust_set[~ust_set.duplicated('Cusip', keep='first')][['Cusip', 'Currently outstanding']].set_index('Cusip')
        ust_set = ust_set.drop_duplicates(subset='Cusip', keep='last').set_index('Cusip')
        ust_set.update(outstanding_values)
        ust_set['Years to maturity'] = ((ust_set['Maturity date'] - pd.to_datetime(settlement_date)).dt.days + 1)/365
        ust_set.sort_values(by='Years to maturity', ascending=True, inplace=True)

        if len(ust_set) == len(prices):
            ust_set = ust_set[~ust_set['Security term'].isin(['30-Year 3-Month', '29-Year 9-Month'])]
            ust_set['UST label'] = ust_set.apply(lambda row: self._get_ust_label(row), axis='columns')
            if not include_FRNs:
                ust_set = ust_set[ust_set['Security type'] != 'FRN']
            if not include_TIPS:
                ust_set = ust_set[ust_set['Security type'] != 'TIPS']
            if get_ytms:
                ust_set['EOD YTM'] = ust_set.apply(lambda row: self._get_df_ytm(row, settlement_date), axis='columns')
            if not include_outstanding:
                ust_set.drop(columns='Currently outstanding', inplace=True)
            
            ust_set['Duration'] = ust_set.apply(lambda row: self._get_duration(row, settlement_date), axis='columns')
                        
            print("Merged auction and price data successfully\nNo missing or excess data\nAll CUSIPs are identical between DataFrames")
            ust_set = ust_set.drop(columns=['Buy', 'Sell'])
            ust_set = self._get_ranks(data=ust_set)
            self.ust_set = ust_set.reset_index(drop=False).set_index('Cusip')
            return self.ust_set
            
        else:
            print("Length of DataFrames differs - verify data")
            print(len(ust_set), len(prices))
            return None
    
    def plot_ust_curve(self, bspline_curve: bool = True, univariate_spline: bool = True, mles_spline: bool = False):
        fig = go.Figure()
        hovertemplate = (
            "<b>CUSIP:</b> %{customdata[0]}<br>"+
            "<b>YTM:</b> %{y:.3%}<br>"+
            "<b>Coupon</b> %{customdata[2]}<br>"+
            "<b>Price:</b> %{customdata[4]:.4f}<br>"+
            "<b>Duration:</b> %{customdata[10]:.2f}<br>"+
            "<b>Maturity date:</b> %{customdata[3]|%Y-%m-%d}<br>"+
            "<b>Issue date:</b> %{customdata[6]|%Y-%m-%d}<br>"
        )
        years = ['2', '3', '5', '7', '10', '20', '30']
        terms = [tenor + '-Year' for tenor in years] # Ordered list
        for tenor in terms:
            subset = self.ust_set[(self.ust_set['Security term'] == tenor) &
                                  (self.ust_set['Years to maturity'] > 90/365)].reset_index(drop=False)
            custom_data = subset.to_numpy()
            fig.add_trace(go.Scatter(
                x=subset['Years to maturity'],
                y=subset['EOD YTM']/100,
                mode='markers',
                name=tenor,
                hovertemplate=hovertemplate,
                customdata=custom_data,
                text=subset['EOD YTM'],
                marker=dict(
                    size=8
                )
            ))
        filtered_set = self.ust_set[(self.ust_set['Years to maturity'] > 90/365) &
                                    (self.ust_set['Security type'] != 'Bill') &
                                    (self.ust_set['Rank'] > 2)] # Removing OTR securities
        curve_builder = ParCurves(filtered_set)

        if bspline_curve:
            bspline_x, bspline_y = curve_builder.Bspline_with_knots(knots=[0.5, 1, 2, 3, 5, 7, 8, 9, 10, 15, 20, 25])
            fig.add_trace(go.Scatter(
                x=bspline_x,
                y=bspline_y/100,
                mode='lines',
                name='Cubic B-spline with knots',
                line=dict(width=3)
            ))
        if univariate_spline:
            spline_x, spline_y = curve_builder.univariate_spline(smoothness=0.2, return_data=True)
            fig.add_trace(go.Scatter(
                x=spline_x,
                y=spline_y/100,
                mode='lines',
                name='Smoothed cubic spline'
            ))
        if mles_spline:
            mles_curve, _ = calibrate_mles(maturities=filtered_set['Years to maturity'].to_numpy(), # Merrill Lynch Exponential Spline (parametric model)
                        yields=filtered_set['EOD YTM'].to_numpy(),
                        N=9,
                        overnight_rate=4.33)
            mles_x = np.arange(0.25, 30, 1/365)
            mles_y = mles_curve.theoretical_yields(maturities=mles_x)
            fig.add_trace(go.Scatter(
                x=mles_x,
                y=mles_y/100,
                mode='lines',
                name='Merrill Lynch Exponential Spline',
                line=dict(color='blue')
            ))

        fig.update_layout(
            title='US Treasury Yield Curve',
            width=1400,
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title='Years to maturity',
            yaxis_title='End-of-Day Yield to Maturity (YTM)',
            yaxis_tickformat='.2%',
            legend_title_text='Security Tenors',
            template='plotly_dark'
        )
        fig.show()
        
    def get_cusip_timeseries(self, CUSIPs: list, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        price_data = DataFetcher().fetch_cusip_timeseries(CUSIPs, start_date, end_date)
        self.ytm_timeseries = price_data.copy()

        for cusip in price_data.columns:
            cusip_info = self.ust_set.loc[cusip]
            maturity = cusip_info['Maturity date'].date()
            issue = cusip_info["Issue date"].date()
            coupon = float(cusip_info['Rate'])

            for date in price_data.index:
                settlement = self.adjust_for_bad_day(date.date() + relativedelta(days=1))
                price = self.ytm_timeseries.loc[date, cusip]
                ytm = self.get_coupon_ytm(price, issue, maturity, settlement, coupon, dirty=False)
                self.ytm_timeseries.loc[date, cusip] = ytm
        
        if len(CUSIPs) == 2:
            if self.ytm_timeseries.shape[1] < 2:
                print("wrong shape")
            else:
                self.ytm_timeseries['Spread'] = self.ytm_timeseries.iloc[:, 1] - self.ytm_timeseries.iloc[:, 0]
        return self.ytm_timeseries, price_data

    def get_residuals(self, curve: Tuple[np.ndarray, np.ndarray], return_full_df: bool = True, plot_residuals: bool = True) -> pd.DataFrame:
        
        bond_df = self.ust_set[(self.ust_set['Security type'] != 'Bill') & (self.ust_set['Years to maturity'] > 90/365)].copy()
        curve_df = pd.DataFrame({"Years to maturity": curve[0], "Theoretical YTM": curve[1]})
        self.par_curve = curve_df.copy()

        merged_df = pd.merge_asof(left=bond_df.reset_index(drop=False), right=curve_df, on='Years to maturity').set_index('Cusip')
        merged_df['Residual'] = (merged_df['EOD YTM'] - merged_df['Theoretical YTM']) * 100
        self.residuals_df = merged_df[['Years to maturity', 'Maturity date', 'Security term', 'EOD YTM', 'Theoretical YTM', 'Residual']]
        self.bond_set_with_residuals = merged_df.copy()

        if plot_residuals:
            plt.figure(figsize=(10,6))
            sns.scatterplot(data=self.residuals_df, x='Years to maturity', y='Residual', hue='Security term',
                            hue_order=['2-Year', '3-Year', '5-Year', '7-Year', '10-Year', '20-Year', '30-Year'],
                            s=50, palette='bright')
            sns.despine()
            plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
            plt.xlabel("Years to Maturity", fontdict={"size": 12})
            plt.ylabel("Residual to fitted curve (bps)", fontdict={"size": 12})
            plt.legend(loc="lower right", frameon=False)
            plt.show()
        
        if return_full_df:
            return self.bond_set_with_residuals
        else:
            return self.residuals_df
        
    def get_initial_screening_set(self,
                                  ytm_threshold: float = 0.1,
                                  duration_threshold: float = 1.0,
                                  maturity_threshold: float = 4.0) -> pd.DataFrame:
        filtered_set = self.ust_set[(self.ust_set['Years to maturity'] > 90/365) &
                                    (self.ust_set['Security type'] != 'Bill') &
                                    (self.ust_set['Rank'] > 2)] # Removing OTR securities
        Bspline_model = ParCurves(filtered_set).Bspline_with_knots(knots=[0.5, 1, 2, 3, 5, 7, 8, 9, 10, 15, 20, 25],
                                                                   return_data=False)
        cols_to_drop = ['Security type', 'Rate', 'Maturity date', 'Issue date', 'End of day', 'Rank']
        screening_df = self.ust_set.drop(columns=cols_to_drop)
        
        # Cross merging to filter through data
        screening_df['key'] = 1
        all_pairs_df = pd.merge(
            screening_df, 
            screening_df, 
            on='key', 
            suffixes=(' long', ' short')
        ).drop('key', axis=1)

        filtered_df = all_pairs_df[
            (abs(all_pairs_df['Years to maturity long'] - all_pairs_df['Years to maturity short']) <= maturity_threshold) &
            (abs(all_pairs_df['Duration long'] - all_pairs_df['Duration short']) <= duration_threshold) &
            (abs(all_pairs_df['EOD YTM long'] - all_pairs_df['EOD YTM short']) >= ytm_threshold) &
            (all_pairs_df['Years to maturity long'] > 10)].copy()

        # Interpolating duration to find curves for each security pair
        otr_set = self.ust_set[self.ust_set['Rank'] == 1]
        yearly_index = np.arange(1, 30 + 1, 0.2)
        interpolator = scipy.interpolate.interp1d(
            x=otr_set['Years to maturity'],
            y=otr_set['Duration'],
            kind='cubic',
            fill_value='extrapolate')
        interpolated_duration = interpolator(yearly_index) 
        interpolated_duration_df = pd.DataFrame({
            "Years to maturity": yearly_index,
            "Interpolated duration": interpolated_duration
        })

        first_merge = pd.merge_asof(
            filtered_df.sort_values('Duration long'),
            interpolated_duration_df,
            left_on='Duration long',
            right_on='Interpolated duration',
            direction='nearest'
        ).rename(columns={"Years to maturity": "Implied tenor long"}).drop(columns='Interpolated duration')

        second_merge = pd.merge_asof(
            first_merge.sort_values('Duration short'),
            interpolated_duration_df,
            left_on='Duration short',
            right_on='Interpolated duration',
            direction='nearest'
        ).rename(columns={"Years to maturity": "Implied tenor short"}).drop(columns='Interpolated duration')
        
        final_df = second_merge 
        final_df['Curve exposure'] = np.where(
            final_df['Duration long'] < final_df['Duration short'],
            final_df['Implied tenor long'].astype(int).astype(str) + 's' + final_df['Implied tenor short'].astype(int).astype(str) + 's steepener',
            final_df['Implied tenor short'].astype(int).astype(str) + 's' + final_df['Implied tenor long'].astype(int).astype(str) + 's flattener'
        )
        final_df['Current spread'] = np.where(
            final_df['EOD YTM long'] > final_df['EOD YTM short'],
            final_df['EOD YTM long'] - final_df['EOD YTM short'],
            final_df['EOD YTM short'] - final_df['EOD YTM long']
        )
        final_df['Par curve slope'] = np.where(
            final_df['Implied tenor short'] < final_df['Implied tenor long'],
            Bspline_model(final_df['Implied tenor long']) - Bspline_model(final_df['Implied tenor short']),
            Bspline_model(final_df['Implied tenor short']) - Bspline_model(final_df['Implied tenor long'])
        )
        final_df['Spread to par curve'] = np.where(
            final_df['Implied tenor short'] < final_df['Implied tenor long'],
            final_df['EOD YTM long'] - final_df['EOD YTM short'] - final_df['Par curve slope'], # flattener spread to par calculation - CLEAN UP for filtering steepeners and flatteners
            final_df['Par curve slope'] - final_df['EOD YTM short'] - final_df['EOD YTM long'] # steepener spread to par calculation
        )
        final_df = final_df[final_df['Spread to par curve'] > 0.02]
        final_df = final_df.sort_values(by='Spread to par curve', ascending=False)

        columns=['Duration exposure', 'Current spread', 'Par curve spread',
                 'Long bond', 'Tenor', 'YTM', 'Short bond', 'Tenor', 'YTM', 'Duration long', 'Duration short']
        data = []
        for i in range(5):
            row = final_df.iloc[i, :]
            row_duration = 'Long duration' if row['Duration long'] > row['Duration short'] else 'Short duration'
            row_spread = f"{(row['Current spread'] * 100):.1f} bps"
            row_par_curve = f"{(row['Par curve slope'] * 100):.1f} bps"
            row_long, tenor_long, ytm_long = row['UST label long'], row['Security term long'], row['EOD YTM long']
            row_short, tenor_short, ytm_short = row['UST label short'], row['Security term short'], row['EOD YTM short']
            duration_long, duration_short = row['Duration long'], row['Duration short']
            row_data = [row_duration, row_spread, row_par_curve,
                        row_long, tenor_long, ytm_long,
                        row_short, tenor_short, ytm_short,
                        duration_long, duration_short]
            data.append(row_data)

        self.trade_screening_set = pd.DataFrame(columns=columns, data=data)
        return self.trade_screening_set.drop(columns=['Duration long', 'Duration short'])
    
    def plot_trades(self):
        pass

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
        if tenor == 0:
            return 0
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
        time_to_pmt, first_pmt_period = None, None
        for date, _ in dates_and_cashflows:
            if date > settlement_date:
                first_pmt_date = self.adjust_for_bad_day(date)
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = self.adjust_for_bad_day(dates_and_cashflows[previous_index][0])
                first_pmt_period = (first_pmt_date - last_date).days
                time_to_pmt = (first_pmt_date - settlement_date).days
                break
        if time_to_pmt is None or first_pmt_period is None:
            return 100
        
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
            ytm = scipy.optimize.newton(
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
        ytm = row['EOD YTM']/100
        rate = row['Rate']
        
        dates_and_cashflows = self.get_dates_and_cashflows(issue_date=issue,
                                                        maturity_date=maturity,
                                                        coupon=rate)
        time_to_pmt, first_pmt_period = None, None
        for date, _ in dates_and_cashflows:
            if date > settlement_date:
                first_pmt_date = self.adjust_for_bad_day(date)
                previous_index = dates_and_cashflows.index((date, _)) - 1
                last_date = self.adjust_for_bad_day(dates_and_cashflows[previous_index][0])
                first_pmt_period = (first_pmt_date - last_date).days
                time_to_pmt = (first_pmt_date - settlement_date).days
                break
        if time_to_pmt is None or first_pmt_period is None:
            return 0.0
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
            by=['Security term', 'Years to maturity'],
            ascending=[True, False],
            inplace=True
        )
        data['Rank'] = data.groupby(by='Security term').cumcount() + 1
        data.sort_values(by='Years to maturity', ascending=True, inplace=True)
        return data
    
    def _get_ust_label(self, row: pd.Series) -> str:
        rate_str = f"{row['Rate']:.3f}%"
        date_str = row['Maturity date'].strftime("%b-%y")
        return (rate_str + " " + date_str)
    
    def _get_cusip_from_label(self, label: str) -> str:
        coords = np.where(self.ust_set == label)
        row, col = coords
        return self.ust_set.index[row][0]