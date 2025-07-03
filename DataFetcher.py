import requests
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

class DataFetcher:
    def __init__(self):
        # Treasury auction data API configuration
        self.auction_base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
        self.FedInvest_url = "https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate"
        

    def fetch_auction_data(self,
                           fields: Optional[List[str]] = ["cusip", "auction_date", "issue_date", "security_term", "maturity_date", "currently_outstanding", "original_security_term"]):
        # Fetching all historical Treasury auction data
        
        params = {
            "sort": "-issue_date",
            "format": "json"
        }

        if fields:
            params["fields"] = ",".join(fields)

        try:
            response = requests.get(self.auction_base_url, params=params)
            url = response.url
            response.raise_for_status()  # Raise an error for bad responses
            self.api_data = response.json()
            self.auction_data = pd.DataFrame(self.api_data['data'])

            next_page_url = self.api_data.get('links', {}).get('next').replace("%5B", "[").replace("%5D", "]") if 'links' in self.api_data else None
            while next_page_url:
                new_url = str(url) + next_page_url
                response = requests.get(new_url)
                response.raise_for_status()
                self.api_data = response.json()
                self.auction_data = pd.concat([self.auction_data, pd.DataFrame(self.api_data['data'])])
                next_page_url = self.api_data.get('links', {}).get('next')
                if next_page_url is not None:
                    next_page_url.replace("%5B", "[").replace("%5D", "]")
            
            date_cols = ['auction_date', 'issue_date', 'maturity_date']
            for col in date_cols:
                self.auction_data[col] = pd.to_datetime(self.auction_data[col], format='%Y-%m-%d')
            self.auction_data['currently_outstanding'] = pd.to_numeric(self.auction_data['currently_outstanding'], errors='coerce')
            col_names = []
            for name in self.auction_data.columns:
                col_names.append(name.replace('_', ' ').capitalize())
            self.auction_data.columns = col_names
            return self.auction_data
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching auction data: {e}")
            return None
        
    def fetch_historical_UST_data(self, date: Optional[datetime]):
        """
        Fetches UST data from FedInvest
        
        Args:
            date (str): Date for pricing to be input in "YYYY-MM-DD" format
        
        Returns:
            DataFrame: Values from website
        """
        # Cleaning input
        if date:
            datetime_object = date
        else:
            datetime_object = datetime.now().date() - timedelta(days=1) 

        while datetime_object.weekday() in [5, 6]:
            datetime_object = datetime_object - timedelta(days=1)

        # Posting to FedInvest to retrieve data
        payload = {
            "priceDate.month": datetime_object.month,
            "priceDate.day": datetime_object.day,
            "priceDate.year": datetime_object.year,
            "submit": "Show Prices"
        }

        try:
            response = requests.post(self.FedInvest_url, data=payload, timeout=30)
            response.raise_for_status() # Raise an error for bad responses
        
        except requests.exceptions.RequestException as e:
            print(f"An error occured during request: {e}")
        
        output = BeautifulSoup(response.text, "html.parser")
        table = output.find("table", class_="data1") # Getting table and its data from webpage
        header_row = table.find_all("th")
        data_rows = table.find_all("tr")[1:]

        if not data_rows:
            print("No price information on this date")
            return None

        headers = [th.get_text(strip=True).capitalize() for th in header_row]
        
        all_rows = list()
        for tr in data_rows:
            row_data = [td.get_text(strip=True) for td in tr.find_all("td")]
            all_rows.append(row_data)

        df = pd.DataFrame(data=all_rows, columns=headers)

        column_name = headers[1] 
        df.loc[df[column_name] == "MARKET BASED BILL", column_name] = "Bill" # Cleaning up naming
        df.loc[df[column_name] == "MARKET BASED NOTE", column_name] = "Note"
        df.loc[df[column_name] == "MARKET BASED BOND", column_name] = "Bond"
        df.loc[df[column_name] == "MARKET BASED FRN", column_name] = "FRN"
        df = df.drop(columns=headers[4]) # Dropping "Call date" column which is empty
        df['Rate'] = pd.to_numeric(df['Rate'].str.rstrip('%'))
        df['Maturity date'] = pd.to_datetime(df['Maturity date'], errors='coerce')
        df['Buy'] = pd.to_numeric(df['Buy'])
        df['Sell'] = pd.to_numeric(df['Sell'])
        df['End of day'] = pd.to_numeric(df['End of day'])
        
        return df
    
    def fetch_cusip_timeseries(self, CUSIPs: list, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        date_range = pd.date_range(start=start_date, end=end_date).to_list()
        all_data = dict()
        
        for date in date_range:
            date_price_info = dict()
            payload = {
                        "priceDate.month": date.month,
                        "priceDate.day": date.day,
                        "priceDate.year": date.year,
                        "submit": "Show Prices"
                    }
            response = requests.post(url=self.FedInvest_url, data=payload, timeout=30)
            response.raise_for_status()

            output = BeautifulSoup(response.text, "html.parser")
            table = output.find("table", class_="data1")
            data = table.find_all("tr")[1:]
            if data:
                for tr in data:
                    data_row = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if data_row[0] in CUSIPs: date_price_info[data_row[0]] = data_row[7]
            all_data[date] = date_price_info
        
        prices_df = pd.DataFrame(data=all_data).transpose().rename_axis("Date").dropna(axis=0)
        for col in prices_df.columns:
            prices_df[col] = pd.to_numeric(prices_df[col])
        return prices_df