import requests
import pandas as pd
import numpy as np
from typing import List, Optional

class DataFetcher:
    def __init__(self):
        # Treasury auction data API configuration
        self.auction_base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
        # self.auction_fields = ""

    def fetch_auction_data(self,
                           fields = ["cusip", "auction_date", "issue_date", "security_term", "avg_med_yield", "currently_outstanding", "original_security_term"]):
        # Fetching all historical Treasury auction data
        params = {
            "fields": ",".join(fields),
            "sort": "auction_date",
            "format": "json"  # Adjust limit as needed
        }

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