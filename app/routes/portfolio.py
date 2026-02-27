from flask import Blueprint, render_template, request, session, redirect, url_for, jsonify, current_app
import json
import os
from app.utils import plotting_utils, storage_utils, finance_data
from app.models import Asset, Portfolio, PortfolioManager

bp = Blueprint('portfolio', __name__)

@bp.route('/portfolio', methods=['GET', 'POST'])
@bp.route('/', methods=['GET', 'POST'])   
def portfolio_feature(): 
    #print("portfolio_feature called")
    
    data = get_full_portfolio_data()
    
    #print("portfolio_feature out")

    return render_template(
        'portfolio.html',
        title='Portfolio Dashboard',
        **data
    )
    
def get_full_portfolio_data():
    asset_classes = ['stocks', 'crypto'] # Only two asset classes for now. TODO
    portfolios = {}
    
    free_cash = storage_utils.load_cash() # Load the cash data

    for asset_type in asset_classes:
        tickers = get_assets(asset_type)
        # Note: force_update=False here to load old data first, then update
        raw_metrics = finance_data.fetch_latest_metrics(tickers, asset_type, interval='4h', force_update=False)
        
        # Load shares and prices from storage_utils
        shares  = storage_utils.load_shares(asset_type)
        avg_price  = storage_utils.load_prices(asset_type)
        env = storage_utils.load_env(asset_type)
        soc = storage_utils.load_soc(asset_type)
        gov = storage_utils.load_gov(asset_type)
        cont = storage_utils.load_cont(asset_type)
                
        assets = [
            Asset(
                ticker=t,
                metrics=next(m for m in raw_metrics if m['Ticker']==t),
                shares=shares.get(t, 0),
                avg_price=avg_price.get(t, 0),
                env=env.get(t, 0),
                soc=soc.get(t, 0),
                gov=gov.get(t, 0),
                cont=cont.get(t, 0)
            ) for t in tickers
        ]
        
        portfolios[asset_type] = Portfolio(assets)
        
        
    print("portfolios:" , portfolios)
    
    portfolio = PortfolioManager(portfolios)
    print("portfolio: ", portfolio)
    
    total_market_value = sum(p.total_market_value for p in portfolio.values())
    grand_total_cost_basis = sum(p.total_cost_basis for p in portfolio.values())
    grand_total_with_cash = total_market_value + free_cash # Total portfolio value

    # Generate the dividend plot (stocks only)    
    stock_shares_dict = {assets.ticker: assets.shares for assets in portfolio.stocks.assets}
    stock_metrics_list = [assets.metrics for assets in portfolio.stocks.assets]
    dividend_plot = create_monthly_dividends_plot(stock_metrics_list, stock_shares_dict)

    return {
        'portfolio': portfolio,
        'total_market_value': total_market_value,
        'total_cost_basis': grand_total_cost_basis,
        'grand_total_with_cash': grand_total_with_cash,
        'free_cash_value': free_cash,
        'monthly_div_plot': dividend_plot,
        'monthly_payment_counts': portfolio.stocks.monthly_dividend_data.counts,
        'total_monthly_dividend_payout': portfolio.stocks.monthly_dividend_data.payouts
    }

def get_assets(asset_class='stocks'):
    #print("get_assets called")
    
    # Paths look like 'data/stocks_list.json'
    data_dir = current_app.config['DATA_FOLDER']
    path = os.path.join(data_dir,f"{asset_class}_list.json")
    
    if os.path.exists(path) and os.path.getsize(path) > 0: # Ensure path exists and file isn't empty
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"File error, returning defaults: {e}")
    
    # Defaults based on type
    defaults = {
        'stocks': ["AMAT", "AMT", "AMUN.PA"],
        'crypto': ["BTC-USD", "ETH-USD"]#,
        #'bonds': ["BND"],
        #'commodities': ["GC=F"] # Gold TODO
    }
    
    print("get_assets out")
    return defaults.get(asset_class, [])


def create_monthly_dividends_plot(stock_metrics, current_shares):
    """
    Generates HTML code for the initial dividend plot (should be updated to pio)
    """
    # Call the utility function to get the figure object
    fig = plotting_utils.create_monthly_dividends_figure(stock_metrics, current_shares)
    
    # Return the HTML string that can be inserted directly into the template
    return fig.to_html(full_html=False, include_plotlyjs='cdn')  #TODO change to json
    
@bp.route('/update_portfolio_cache', methods=['POST'])
def update_portfolio_cache():
    """
        Loads cached data when app opens.
    """
    asset_classes = ['stocks', 'crypto'] #TODO
    for asset_type in asset_classes:
        tickers = get_assets(asset_type)
        # Here call the function with force_update=True to trigger the update logic
        metrics = finance_data.fetch_latest_metrics(  tickers, asset_type,
                                            interval='4h', force_update=True)
                                            
    updated_data = get_full_portfolio_data()
        
    # Update dividend plot
    if asset_type == 'stocks':
        data = request.get_json()
        asset_items = data.get('assets', [])
        current_shares  = {item['ticker']: float(item['shares']) for item in asset_items}
        fig = plotting_utils.create_monthly_dividends_figure(metrics, current_shares)
        updated_data['plot_data'] = fig.to_json()
    
    updated_data['portfolio'] = updated_data['portfolio'].to_dict()
        
    return jsonify(updated_data)
     
@bp.route('/update_portfolio_data/<asset_type>', methods=['POST'])
def update_portfolio_data(asset_type):
    """
        Update data when app is running.
    """
    print(f"update_portfolio_data called for {asset_type}")
    
    # Get data from the AJAX request body
    data = request.get_json()
    
    ticker_list = get_assets(asset_type)
    metrics = finance_data.fetch_latest_metrics(ticker_list, asset_type,
                                                interval='4h', force_update=True)
    asset_items = data.get('assets', [])
    
    # Reconstruct dynamic data from AJAX
    current_shares  = {item['ticker']: float(item['shares']) for item in asset_items}
    current_prices  = {item['ticker']: float(item['price']) for item in asset_items}
#    current_env     = {item['ticker']: float(item['env']) for item in asset_items}
#    current_soc     = {item['ticker']: float(item['soc']) for item in asset_items}
#    current_gov     = {item['ticker']: float(item['gov']) for item in asset_items}
#    current_cont    = {item['ticker']: float(item['cont']) for item in asset_items}
    
    #print("results['total_market_value']: ", results['total_market_value'])
    
    results = calculate_portfolio_metrics(metrics, current_shares, current_prices)
    
    #print("results['total_market_value']: ", results['total_market_value'])
    
    response = {
        'total_market_value': results['total_market_value'],
        'total_cost_basis': results['total_cost_basis'],
        'sector_labels': results['sector_labels'],
        'sector_values': results['sector_values'],
        'portfolio_yield': results['portfolio_yield'],
        'portfolio_div_growth': results['portfolio_div_growth'],
        'annual_dividends' : results['annual_dividends']
    }
    
    if asset_type == 'stocks':
        fig = plotting_utils.create_monthly_dividends_figure(metrics, current_shares)
        response['plot_data'] = fig.to_json()
        response['monthly_payment_counts'] = results['monthly_payment_counts']
    
    print("update_portfolio_data out")
    
    return jsonify(response)
          

@bp.route('/save_single_value/<asset_type>', methods=['POST'])
def save_single_value(asset_type):
    """
    Handles saving a single updated share count or price from an AJAX request.
    """
    print(f"save_single_value called for asset type {asset_type}")
    data = request.get_json()
    
    ticker = data.get('ticker')
    field = data.get('field')
    value = data.get('value')
    
    if not all([ticker, field, isinstance(value, (int, float))]):
        return jsonify({'status': 'error', 'message': 'Invalid data received.'}), 400

    # Update shares count
    if field == 'shares':
        # Load all shares, update only the one that changed
        current_shares = storage_utils.load_shares(asset_type)
        current_shares[ticker] = max(0.0, value)
        storage_utils.save_shares(current_shares, asset_type)
        session['share_counts'] = current_shares
        
    # Update average buy price
    elif field == 'price':
        # Load all prices, update only the one that changed
        current_prices = storage_utils.load_prices(asset_type)
        current_prices[ticker] = max(0.0, value)
        storage_utils.save_prices(current_prices, asset_type)
        session['avg_buy_prices'] = current_prices
        
    # Update environment score
    elif field == 'env':
        current_env = storage_utils.load_env(asset_type)
        current_env[ticker] = max(0.0, value)
        storage_utils.save_env(current_env, asset_type)
        session['env'] = current_env
        
    # Update social score
    elif field == 'soc':
        current_soc = storage_utils.load_soc(asset_type)
        current_soc[ticker] = max(0.0, value)
        storage_utils.save_soc(current_soc, asset_type)
        session['soc'] = current_soc
        
    # Update governance score
    elif field == 'gov':
        current_gov = storage_utils.load_gov(asset_type)
        current_gov[ticker] = max(0.0, value)
        storage_utils.save_gov(current_gov, asset_type)
        session['gov'] = current_gov
        
    # Update controversy score
    elif field == 'cont':
        current_cont = storage_utils.load_cont(asset_type)
        current_cont[ticker] = max(0.0, value)
        storage_utils.save_cont(current_cont, asset_type)
        session['cont'] = current_cont

    else:
        print("save_single_value out")
        return jsonify({'status': 'error', 'message': f'Unknown field: {field}'}), 400

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
        assets = get_assets(asset_class)
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
    assets = get_assets(asset_class)
    
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

def format_price(value):
    #print("format_price called")
    try:
        val = float(value)
    except (ValueError, TypeError):
        return "N/A"

    if val == 0:
        return "0.00"
    
    # If the price is less than 0.01, use scientific notation
    # TODO check that, there seems to be a few bugs in the asset chart pie and crypto table
    if 0 < abs(val) < 0.01:
        # Returns format like 1.20e-7
        return f"{val:.2e}"
    
    #print("format_price out")
    # Otherwise, standard 2 decimal places
    return f"{val:,.2f}".replace(",", " ")

def format_weight(value):
    try:
        val = float(value)
    except (ValueError, TypeError):
        return "0.00"

    if val == 0:
        return "0.00"
    
    # If the weight is less than 0.01%, use scientific notation
    if 0 < abs(val) < 0.01:
        return "{:.2e}".format(val)
    
    # Standard 2 decimal places for significant weights
    return "{:.2f}".format(val)



