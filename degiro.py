
import json
import pandas as pd

from degiroapi.degiro_connector.trading.api import API as Degiro
from degiroapi.degiro_connector.trading.models.trading_pb2 import Credentials, ProductSearch
from degiroapi.degiro_connector.quotecast.api import API as QuotecastAPI
from degiroapi.degiro_connector.quotecast.models.quotecast_parser import QuotecastParser
from degiroapi.degiro_connector.quotecast.models.quotecast_pb2 import Chart, Quotecast, Ticker
from degiroapi.degiro_connector.quotecast.actions.action_get_chart import ChartHelper

class DeGiro:

    def __init__(self, user:str, pwd:str):
        
        # Setup Credentials
        self.credentials = Credentials(
            username = user,
            password = pwd,
            #int_account = YOUR_INT_ACCOUNT,  # OPTIONAL FOR LOGIN
        )

        # Setup Trading API
        self.degiro = Degiro(credentials=self.credentials)
        self.degiro.connect()
        self.session_id = self.degiro.connection_storage.session_id

        # Get Config
        config_table = self.degiro.get_config()
        self.user_token = config_table['clientId']
        self.session_id = config_table['sessionId']

        # Connecting to data
        self.quotecast_api = QuotecastAPI(user_token=self.user_token)

    def _availableProducts(self):
         
        # FETCH DATA - DICT
        products_config_dict = self.degiro.get_products_config(raw=True)
        self.products_config = json.dumps(
            products_config_dict,
            sort_keys=True,
            indent=4,
        )

        return self.products_config

    def getExchanges(self):

        return self._availableProducts()['exchanges']

    def getProductsTypes(self):

        return self._availableProducts()['productTypes']

    def getIndeces(self):

        return self._availableProducts()['indices']
    
    def getProducts(self, exchange_id:int=663, index_id:int=None, country:int=846,
                    limit:int=10000):

        # SETUP REQUEST
        c = 0
        total = limit
        asset_list = []
        while c < total:
            request = ProductSearch.RequestStocks(
                exchange_id=exchange_id,  # NASDAQ
                #index_id=122001,  # NASDAQ 100
                # You can either use `index_id` or `exchange id`
                # See which one to use in the `ProductsConfig` table
                is_in_us_green_list=True,
                stock_country_id=846,  # US
                offset=c,
                limit=limit,
                require_total=True,
                sort_columns="name",
                sort_types="asc",
            )

            # FETCH DATA
            self.search = self.degiro.product_search(request=request, raw=True)
            total = self.search['total']
            asset_list = asset_list + self.search['products']
            c += 1000

        assets = pd.DataFrame(asset_list)
        assets = assets[['symbol', 'name', 'vwdId', 'isin', 'currency', 'productType', 'active', 'onlyEodPrices', 
                        'buyOrderTypes', 'contractSize', 'isShortable', 'id', 'sellOrderTypes', 
                        'closePrice', 'category', 'exchangeId', 'orderBookDepth']]
        assets['country'] = assets['isin'].apply(lambda x: x[:2])

        return assets

    def getPriceData(self, vwdid:int, resolution:str='PT1H', period:str='MAX', tz:str='UTC'):

        request = Chart.Request()
        request.culture = 'es-ES'
        request.period = period
        request.requestid = '1'
        request.resolution = resolution
        request.series.append(f"issueid:{vwdid}")
        request.series.append(f"ohlc:issueid:{vwdid}")
        request.series.append(f"volume:issueid:{vwdid}")
        request.tz = tz # It can be Europe/Madrid

        self.chart = self.quotecast_api.get_chart(
            request=request,
            raw=True,
        )


        ChartHelper.format_chart(chart=self.chart, copy=False)
        df = ChartHelper.serie_to_df(serie=self.chart['series'][1])
        df['volume'] = ChartHelper.serie_to_df(serie=self.chart['series'][2])['volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']

        return df
    
    def getCompanyProfile(self, isin:str):

        return self.degiro.get_company_profile(
            product_isin=isin,
            raw=True,
        )




if __name__ == '__main__':

    degiro = DeGiro('OneMade','Onemade3680')
    products = degiro.getProducts(exchange_id=663,country=846) # Nasdaq exchange
    asset = products.iloc[213] # AAPL -> vwdid = 350015372
    data = degiro.getPriceData(asset['vwdId'], 'PT1H', 'P5Y', tz='UTC')
    profile = degiro.getCompanyProfile('FR0000131906')
