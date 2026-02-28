from types import SimpleNamespace
from app.utils import storage_utils

# Define a single asset (one per ticker)
class Asset:
    def __init__(self, ticker, metrics, shares=0.0, avg_price=0.0, env=0, soc=0, gov=0, cont=0):
        self.ticker = ticker
        self.metrics = metrics
        self.shares = float(shares)
        self.avg_price = float(avg_price)
        self.sector = metrics.get("Sector", "Null")
        self.quote_eur = self._safe_float(metrics.get("Quote_EUR"))
        self.latest_div = self._safe_float(metrics.get("Latest_Div_EUR"))
        self.months_paid = metrics.get("Months_Paid", [0]*12)
        self.div_yield = self._safe_float(metrics.get("Div_Yield"))
        self.div_growth = self._safe_float(metrics.get("Div_CAGR"))
        self.env = int(env)
        self.soc = int(soc)
        self.gov = int(gov)
        self.cont = int(cont)
        
    # Ensure floats are retrieved from data
    def _safe_float(self, val):
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0
            
    @property
    def market_value(self):
        return self.shares * self.quote_eur
        
    @property
    def cost_basis(self):
        return self.shares * self.avg_price
        
    @property
    def asset_income(self):
        return self.market_value * self.div_yield
        
    def get_monthly_income(self):
        if not hasattr(self, 'months_paid'): return [0.0] * 12
            
        return [self.latest_div * self.shares if m == 1 else 0.0 for m in self.months_paid]
        
    # Convert to a dictionary for the frontend
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'metrics': self.metrics,
            'shares': self.shares,
            'avg_price': self.avg_price,
            'sector': self.sector,
            'quote_eur': self.quote_eur,
            'latest_div': self.latest_div,
            'months_paid': self.months_paid,
            'div_yield': self.div_yield,
            'div_growth': self.div_growth,
            'env': self.env,
            'soc': self.soc,
            'gov': self.gov,
            'cont': self.cont,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'asset_income': self.asset_income
        }
        
        
    def __repr__(self):
        return f"Asset({self.ticker}, MarketValue={self.market_value:.2f})"
        
# Define a portfolio of a given asset type (stocks, crypto, etc...)        
class Portfolio:
    def __init__(self, assets):
        self.assets = assets
        
    @property
    def total_market_value(self):
        return sum(asset.market_value for asset in self.assets)
        
    @property
    def total_cost_basis(self):
        return sum(asset.cost_basis for asset in self.assets)
        
    @property
    def monthly_income_data(self):
        payouts = [0.0]*12
        counts = [0] * 12
        details = [[] for _ in range(12)] # List of lists to store stock details per month
        
        for asset in self.assets:
            asset_payouts = asset.get_monthly_income()
            for i in range(12):
                if asset_payouts[i] > 0:
                    payouts[i] += asset_payouts[i]
                    counts[i] += 1
                    details[i].append(f"{asset.ticker}: €{asset_payouts[i]:.2f}") # Details for hovering in plot

        # Process details to include the total income in plot
        final_details = []
        for i in range(12):
            if payouts[i] > 0:
                # Combine the ticker list and add the total header
                ticker_list = "<br>".join(details[i])
                final_details.append(f"<b>Total: €{payouts[i]:.2f}</b><br><br>{ticker_list}")
            else:
                final_details.append("No Income")
                
        return SimpleNamespace(
            payouts = payouts,
            counts = counts,
            details = final_details
        )
        
    @property
    def annual_dividends(self):
        return sum(self.monthly_income_data.payouts)
    
    @property
    def portfolio_yield_data(self):
        total_mv = self.total_market_value
        total_income = sum(asset.market_value * asset.div_yield for asset in self.assets)
        total_income_growth = sum(asset.market_value * asset.div_yield * asset.div_growth for asset in self.assets)
        
        if total_mv < 0 or total_income <= 0:
            return SimpleNamespace(
            div_yield = 0,
            div_growth= 0
        )

        return SimpleNamespace(
            div_yield = total_income / total_mv,
            div_growth= total_income_growth / total_income
        )
        
    @property
    def sectors(self):
        sectors = {}
        for asset in self.assets:
            sector = asset.sector
            sectors[sector] = sectors.get(sector,0) + asset.market_value
            
        return SimpleNamespace(
            labels = list(sectors.keys()),
            values = list(sectors.values())
        )

    def to_dict(self):
        return {
            'assets': [a.to_dict() for a in self.assets],
            'total_market_value': self.total_market_value,
            'total_cost_basis': self.total_cost_basis,
            'monthly_income_data': vars(self.monthly_income_data), # vars is used to extract dicts from object instance
            'annual_dividends': self.annual_dividends,
            'portfolio_yield_data': vars(self.portfolio_yield_data),
            'sectors': vars(self.sectors)
        }

    def __repr__(self):
        return f"Portfolio(Assets={len(self.assets)}, TotalValue=€{self.total_market_value:.2f})"

# Load external data into a portfolio
class PortfolioLoader:
    @staticmethod
    def load_asset_data(asset_type):
        return {
            'shares': storage_utils.load_shares(asset_type),
            'avg_price': storage_utils.load_prices(asset_type),
            'env': storage_utils.load_env(asset_type),
            'soc': storage_utils.load_soc(asset_type),
            'gov': storage_utils.load_gov(asset_type),
            'cont': storage_utils.load_cont(asset_type)
        }


# Define the complete portfolio
class PortfolioManager:
    def __init__(self, portfolios_dict, free_cash=0.0, silent=False):
        self._portfolios = portfolios_dict
        self.free_cash = free_cash
        # Set attributes based on asset classes (stocks, crypto, etc.)
        for name, portfolio_obj in portfolios_dict.items():
            setattr(self, name, portfolio_obj)
            
        # Show summary
        if not silent:
            print(self)
            
    @property
    def total_market_value(self):
        return sum(p.total_market_value for p in self._portfolios.values())
            
    @property
    def grand_total_cost_basis(self):
        return sum(p.total_cost_basis for p in self._portfolios.values())
        
    @property
    def grand_total_with_cash(self):
        return self.total_market_value + self.free_cash
        
    @property
    def total_income_data(self):
        grand_total = [0.0] * 12
        grand_details = [[] for _ in range(12)]
        
        for portfolio in self.values():
            report = portfolio.monthly_income_data
            for i in range(12):
                grand_total[i] += report.payouts[i]
                if report.payouts[i] > 0:
                    # Add a header for the asset class in the hover text
                    grand_details[i].append(report.details[i])
                    
        return {
            "payouts": grand_total,
            "details": ["<br>".join(d) for d in grand_details]
        }
    
    def __iter__(self):
        return iter(self._portfolios)
        
    def keys(self):
        return self._portfolios.keys()
        
    def values(self):
        return self._portfolios.values()
        
    def items(self):
        return self._portfolios.items()
        
    def to_dict(self):
        data = {name: p.to_dict() for name, p in self.items()}
        data['summary'] = {
            'total_market_value': self.total_market_value,
            'grand_total_cost_basis': self.grand_total_cost_basis,
            'grand_total_with_cash': self.grand_total_with_cash,
            'free_cash': self.free_cash,
            'total_income_data': self.total_income_data
        }
        return data
            
    def __repr__(self):
        count = 60
        # Header
        lines = [
            "\n" + "="*count,
            "PORTFOLIO SUMMARY",
            "="*count
        ]
        
        # Add each sub-portfolio
        for name, p in self._portfolios.items():
            lines.append(f" • {name.upper():<8}: {p.__repr__()}")
            
        # Add Global Totals
        lines.append("-" * count)
        lines.append(f" CASH       : €{self.free_cash:,.2f}")
        lines.append(f" TOTAL MV   : €{self.total_market_value:,.2f}")
        lines.append(f" GRAND TOTAL: €{self.grand_total_with_cash:,.2f}")
        lines.append("="*count + "\n")
        
        return "\n".join(lines)


























        
    
