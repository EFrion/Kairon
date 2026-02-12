import json
import os
from flask import current_app

# Base directory for all data files
def get_data_dir():
    """
    Fetches the data directory from the Flask config.
    """
    return current_app.config['DATA_FOLDER']

#def update_json_file(file_path, new_data): #TODO make a general update function?
#    """Merges new_data into an existing JSON file."""
#    data = {}
#    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#        try:
#            with open(file_path, 'r') as f:
#                data = json.load(f)
#        except Exception: pass
#    
#    data.update(new_data)
#    with open(file_path, 'w') as f:
#        json.dump(data, f, indent=4)
#    return data
    
def get_filename(asset_type, data_kind):
    """
    Generates path like 'data/crypto_shares.json' or 'data/stocks_prices.json'
    data_kind should be 'shares' or 'prices'
    """
    #print(f"get_filename called for {asset_type} of {data_kind}")
    data_dir = get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    
    #print("get_filename out")
    return os.path.join(data_dir, f"{asset_type}_{data_kind}.json")

def load_data(asset_type, data_kind):
    """Loads a dictionary from the asset-specific JSON file."""
    #print(f"load_data called for {asset_type} of {data_kind}")
    filename = get_filename(asset_type, data_kind)
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Error reading {filename}. Returning empty data.")
            return {}
    return {}

def save_data(data, asset_type, data_kind):
    """Saves a dictionary to the asset-specific JSON file."""
    print(f"save_data called for {asset_type} of {data_kind}")
    filename = get_filename(asset_type, data_kind)
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving data to file {filename}: {e}")
    print("save_data out")
        
# Convenience functions
def load_shares(asset_type):
    #print(f"load_shares called for {asset_type}")
    return load_data(asset_type, 'shares')

def save_shares(share_counts, asset_type):
    print(f"save_shares called for {asset_type}")
    save_data(share_counts, asset_type, 'shares')
    
def load_prices(asset_type):
    #print(f"load_prices called for {asset_type}")
    return load_data(asset_type, 'prices')

def save_prices(avg_buy_prices, asset_type):
    print(f"save_prices called for {asset_type}")
    save_data(avg_buy_prices, asset_type, 'prices')
    
def load_env(asset_type):
    #print(f"load_env called for {asset_type}")
    return load_data(asset_type, 'env')

def save_env(env_counts, asset_type):
    print(f"save_env called for {asset_type}")
    save_data(env_counts, asset_type, 'env')
    
def load_soc(asset_type):
    #print(f"load_soc called for {asset_type}")
    return load_data(asset_type, 'soc')

def save_soc(soc_counts, asset_type):
    print(f"save_soc called for {asset_type}")
    save_data(soc_counts, asset_type, 'soc')
    
def load_gov(asset_type):
    #print(f"load_gov called for {asset_type}")
    return load_data(asset_type, 'gov')

def save_gov(gov_counts, asset_type):
    print(f"save_gov called for {asset_type}")
    save_data(gov_counts, asset_type, 'gov')
    
def load_cont(asset_type):
    #print(f"load_cont called for {asset_type}")
    return load_data(asset_type, 'cont')

def save_cont(cont_counts, asset_type):
    print(f"save_cont called for {asset_type}")
    save_data(cont_counts, asset_type, 'cont')
    
def save_cash(amount):
    """Saves the free cash amount to a global JSON file."""
    # Wraped it in a dict to keep it compatible with the json.dump logic
    data = {'free_cash': float(amount)}
    save_data(data, 'global', 'cash')

def load_cash():
    """Loads the free cash amount. Returns 0.0 if not found."""
    data = load_data('global', 'cash')
    return data.get('free_cash', 0.0)

