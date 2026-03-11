import os
import pandas as pd

class PortfolioDataManager:
    def __init__(self, current_app, finance_data, storage_utils):
        self.cache_dir = current_app.config['DATA_FOLDER']
        self.finance_data = finance_data
        self.storage_utils = storage_utils
        
        self.path_1d = os.path.join(self.cache_dir, 'stocks_price_history_1d.csv')
        self.path_4h = os.path.join(self.cache_dir, 'stocks_price_history_4h.csv')

    def get_data(self, asset_type='stocks'):
        data_4h = self._load_csv(self.path_4h)
        data_1d = self._load_csv(self.path_1d)

        tickers_4h = set(data_4h.columns) if data_4h is not None else set()
        tickers_1d = set(data_1d.columns) if data_1d is not None else set()

        # Integrity check
        if not tickers_4h.issubset(tickers_1d):
            print(f"Syncing {asset_type} 1d columns with 4h columns.")
            download_start = data_4h.index.min()
            self.finance_data.fetch_latest_metrics(  list(tickers_4h),
                                            category_name=asset_type,
                                            interval='1d',
                                            target_start_date=download_start) # Use daily prices here!
            data_1d = self._load_csv(self.path_1d)

        # Recency check
        if data_1d is not None and data_4h is not None:
            last_1d = data_1d.index.max().normalize()
            last_4h = data_4h.index.max().tz_localize(None).normalize()

            if last_4h > last_1d:
                print(f"Updating 1d data from {last_1d}")
                self.finance_data.fetch_latest_metrics(  list(tickers_1d),
                                            category_name='stocks',
                                            interval='1d',
                                            force_update=True,
                                            target_start_date=last_1d) # Use daily prices here!
                data_1d = self._load_csv(self.path_1d)

        return data_1d

    def _load_csv(self, path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path, index_col='Datetime', parse_dates=True).dropna()
            except Exception as e:
                print(f"Error reading {path}: {e}")
        return None