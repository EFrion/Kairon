from flask import Flask
from app.utils import database
from config import Config
    
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config) 
    
    from .routes import cashflow, portfolio, test
    app.register_blueprint(cashflow.bp)
    app.register_blueprint(portfolio.bp)
    app.register_blueprint(test.bp)
    
    # Initialise a database    
    with app.app_context():
        database.init_db()

    app.config['TEMPLATES_AUTO_RELOAD'] = True  # CHANGE THIS IN PRODUCTION!
    
    # Register filters with Jinja2
    from .routes.cashflow import datetime_format
    app.jinja_env.filters['strftime'] = datetime_format
    
    from .routes.portfolio import format_weight
    app.jinja_env.filters['smart_weight'] = format_weight
    
    from .routes.portfolio import format_price
    app.jinja_env.filters['smart_price'] = format_price
    
    return app
