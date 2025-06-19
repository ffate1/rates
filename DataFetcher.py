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
                           fields: Optional[List[str]] = ["cusip", "auction_date", "issue_date", "security_term", "maturity_date", "avg_med_yield", "currently_outstanding", "original_security_term"]):
        # Fetching all historical Treasury auction data
        
        params = {
            "sort": "-auction_date",
            "format": "json"  # Adjust limit as needed
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
            
            return self.auction_data
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching auction data: {e}")
            return None
        
    def fetch_historical_UST_data(self, date: Optional[str] = (datetime.now().date() - timedelta(days=1)).strftime(format="%Y-%m-%d")):
        """
        Fetches UST data from FedInvest
        
        Args:
            date (str): Date for pricing to be input in "YYYY-MM-DD" format
        
        Returns:
            DataFrame: Values from website
        """
        # Cleaning input
        try:
            datetime_object = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"The format for {date} is incorrect, please use 'YYYY-MM-DD' format.")

        if 
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
        table = output.find("table", class_="data1") # Getting table from webpage

        header_row = table.find_all("th")
        headers = [th.get_text(strip=True).capitalize() for th in header_row]

        data_rows = table.find_all("tr")[1:]
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
        df = df.drop(columns=[headers[4], headers[-1]]) # Dropping "Call date" and "End of day" columns which are empty
        df['Maturity date'] = pd.to_datetime(df['Maturity date'], errors='coerce')
        df['Buy'] = pd.to_numeric(df['Buy'])
        df['Sell'] = pd.to_numeric(df['Sell'])
        return df