from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, current_app
import json
import os
from app.utils import plotting_utils, storage_utils, finance_data
from app.models import PortfolioManager

bp = Blueprint('portfolio', __name__)

@bp.route('/portfolio', methods=['GET', 'POST'])
@bp.route('/', methods=['GET', 'POST'])   
def portfolio_feature():
    
    data = get_portfolio_data()
    #print("Weight: ", data['portfolio'].stocks.assets[0].weight)
    
    return render_template(
        'portfolio.html',
        title='Portfolio Dashboard',
        portfolio=data['portfolio'],
        income_plot=data['income_plot'].to_html(full_html=False, include_plotlyjs='cdn')
    )
    
def get_portfolio_data(force_update=False):

    # TODO automatic asset classes handling
    portfolio = PortfolioManager.from_storage(
        asset_classes=['stocks', 'crypto'],
        storage_utils=storage_utils,
        finance_data=finance_data,
        interval='4h',
        force_update=force_update
    )

    income_plot = plotting_utils.create_income_plot(portfolio.total_income_data)

    return {
        'portfolio': portfolio,
        'income_plot': income_plot
    }

# TODO remove this function
@bp.route('/update_portfolio_cache', methods=['POST'])
def update_portfolio_cache():
    """
        Loads cached data when app opens.
    """                       
    data = get_portfolio_data(force_update=True)
        
    return jsonify({
        'portfolio': data['portfolio'].to_dict(),
        'income_plot': data['income_plot'].to_json()
    })
     
@bp.route('/update_portfolio_data/<asset_type>', methods=['POST'])
def update_portfolio_data(asset_type):
    """
        Update data when app is running.
    """
    data = get_portfolio_data(force_update=True)

    return jsonify({
        'portfolio': data['portfolio'].to_dict(),
        'income_plot': data['income_plot'].to_json()
    })
          

@bp.route('/save_single_value/<asset_type>', methods=['POST'])
def save_single_value(asset_type):
    """
    Handles saving feature update from an AJAX request.
    """
    print(f"save_single_value called for asset type {asset_type}")
    data = request.get_json()
    
    ticker = data.get('ticker')
    field = data.get('field')
    value = data.get('value')
    
    if not all([ticker, field, isinstance(value, (int, float))]):
        return jsonify({'status': 'error', 'message': 'Invalid data received.'}), 400
        
    loaders = {
        'shares': storage_utils.load_shares, 'price': storage_utils.load_prices,
        'env': storage_utils.load_env, 'soc': storage_utils.load_soc,
        'gov': storage_utils.load_gov, 'cont': storage_utils.load_cont
    }
    savers = {
        'shares': storage_utils.save_shares, 'price': storage_utils.save_prices,
        'env': storage_utils.save_env, 'soc': storage_utils.save_soc,
        'gov': storage_utils.save_gov, 'cont': storage_utils.save_cont
    }
    
    if field not in loaders:
        return jsonify({'status': 'error', 'message': f'Unknown field: {field}'}), 400

    current_data = loaders[field](asset_type)
    current_data[ticker] = max(0.0, value)
    savers[field](current_data, asset_type)
    session[field] = current_data
    
    print("save_single_value out")
    return jsonify({'status': 'success', 'message': f'Updated {field} for {ticker}.'})
  
@bp.route('/save_cash', methods=['POST'])
def save_cash():
    data = request.get_json()
    cash_value = data.get('cash', 0)
    
    storage_utils.save_cash(cash_value)
    
    # Store in session so it persists for the user
    session['free_cash'] = float(cash_value)
    
    return jsonify({'status': 'success', 'saved_value': cash_value})
      

@bp.route('/add/<asset_class>', methods=['POST'])
def add_asset(asset_class):
    print("add_asset called")
    ticker = request.form.get('ticker', '').upper().strip()
    if ticker:
        assets = storage_utils.get_assets(asset_class)
        if ticker not in assets:
            assets.append(ticker)
            save_assets(assets, asset_class)
            print(f"Added {ticker} to portfolio.")
    return redirect(url_for('portfolio.portfolio_feature'))
    
def save_assets(asset_list, asset_class='stocks'):
    print("save_assets called")
    data_dir = current_app.config['DATA_FOLDER']
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir,f"{asset_class}_list.json")
    unique_assets = sorted(list(set(asset_list))) # Use set to avoid duplicates
    with open(path, 'w') as f:
        json.dump(unique_assets, f)
        
@bp.route('/delete/<asset_class>/<ticker>', methods=['POST'])
def delete_asset(asset_class, ticker):
    print(f"delete_asset called for: {asset_class}/{ticker}")
    assets = storage_utils.get_assets(asset_class)
    
    if ticker in assets:
        # Remove from main asset list
        assets.remove(ticker)
        save_assets(assets, asset_class)
        
        # Clean up associated share & average price data
        # TODO reduce the amount of files used?
        shares = storage_utils.load_shares(asset_class)
        prices = storage_utils.load_prices(asset_class)
        env = storage_utils.load_env(asset_class)
        soc = storage_utils.load_soc(asset_class)
        gov = storage_utils.load_gov(asset_class)
        cont = storage_utils.load_cont(asset_class)
        shares.pop(ticker, None)
        prices.pop(ticker, None)
        env.pop(ticker, None)
        soc.pop(ticker, None)
        gov.pop(ticker, None)
        cont.pop(ticker, None)
        storage_utils.save_shares(shares, asset_class)
        storage_utils.save_prices(prices, asset_class)
        storage_utils.save_env(env, asset_class)
        storage_utils.save_soc(soc, asset_class)
        storage_utils.save_gov(gov, asset_class)
        storage_utils.save_cont(cont, asset_class)
        
        # Clean up metrics and price history
        finance_data.remove_from_metrics(ticker, asset_class)
        finance_data.remove_from_price_history(ticker,'4h',category_name=asset_class)
        finance_data.remove_from_price_history(ticker,'1d',category_name=asset_class)
        
        return jsonify({'status': 'success', 'message': f'{ticker} removed.'})
    
    return jsonify({'status': 'error', 'message': 'Ticker not found.'}), 404
    
# Needed for returning correctly formatted numbers at initialisation
@bp.app_template_filter('format_finance')
def format_finance(val):
    try:
        val = float(val)
        if val == 0: return "0.00"
        if 0 < abs(val) < 0.01: return f"{val:.2e}"
        return f"{val:,.2f}".replace(",", " ")
    except:
        return "N/A"
