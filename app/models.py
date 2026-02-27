from types import SimpleNamespace

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
        
    def get_monthly_dividend(self):
        if self.shares <= 0 or self.latest_div <= 0:
            return [0.0] * 12
            
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
    def monthly_dividend_data(self):
        payouts = [0.0]*12
        counts = [0] * 12
        
        for asset in self.assets:
            asset_payouts = asset.get_monthly_dividend()
            for i in range(12):
                if asset.months_paid[i]==1:
                    counts[i] += 1
                payouts[i] += asset_payouts[i]

        return SimpleNamespace(
            payouts = payouts,
            counts = counts
        )
        
    @property
    def annual_dividends(self):
        return sum(self.monthly_dividend_data.payouts)
    
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
            'monthly_dividend_data': vars(self.monthly_dividend_data), # vars is used to extract dicts from object instance
            'annual_dividends': self.annual_dividends,
            'portfolio_yield_data': vars(self.portfolio_yield_data),
            'sectors': vars(self.sectors)
        }

    def __repr__(self):
        return f"Portfolio(Assets={len(self.assets)}, TotalValue={self.total_market_value:.2f})"



class PortfolioManager:
    def __init__(self, portfolios_dict):
        self._portfolios = portfolios_dict
        # Set attributes based on asset classes (stocks, crypto, etc.)
        for name, portfolio_obj in portfolios_dict.items():
            setattr(self, name, portfolio_obj)
            
    def __iter__(self):
        return iter(self._portfolios)
        
    def keys(self):
        return self._portfolios.keys()
        
    def values(self):
        return self._portfolios.values()
        
    def items(self):
        return self._portfolios.items()
        
    def to_dict(self):
        return {name: p.to_dict() for name, p in self.items()}
            
    def __repr__(self):
        return f"PortfolioManager(Portfolios={len(self._portfolios)})"


























        
    
