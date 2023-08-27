
import datetime as dt
import json

import pandas as pd
import requests



def pretty_json(data):

    return json.dumps(data, indent=4, sort_keys=True)

class AssetType:

    STOCKS = 'stocks'
    BONDS = 'bonds'
    CURRENCIES = 'currencies'
    FUTURES = 'futures'
    OPTIONS = 'options'
    CFD = 'cfd'
    ETFS = 'etfs'

class Order:

    class Type:

        LIMIT = 0
        STOPLIMIT = 1
        MARKET = 2
        STOPLOSS = 3

        def options(self):

            return [self.LIMIT, self.STOPLIMIT, self.MARKET, 
                    self.STOPLOSS]

    class Time:
        
        DAY = 1
        GTC = 3

        def options(self):

            return [self.DAY, self.GTC]
    
    class Side:

        BUY = 'BUY'
        SELL = 'SELL'

        def options(self):

            return [self.BUY, self.SELL]

class DataType:

    PORTFOLIO = 'portfolio'
    CASHFUNDS = 'cashFunds'
    
class IntervalType:

    D1 = 'P1D',
    W1 = 'P1W',
    M1 = 'P1M',
    M3 = 'P3M',
    M6 = 'P6M',
    Y1 = 'P1Y',
    Y3 = 'P3Y',
    Y5 = 'P5Y',
    Max = 'P50Y'
    
class ResolutionType:

    # PT1M, PT5M, PT10M, PT15M, PT1H, P1D, P7D, P1M, P3M, P1Y

    S1 = 'PT1S'
    M1 = 'PT1M',
    M5 = 'PT5M',
    M10 = 'PT10M',
    M15 = 'PT15M',
    H1 = 'PT1H',
    D1 = 'P1D',
    D7 = 'P7D',

    def resToSeconds(self, resolution:str):

        resolution = resolution.replace('PT', '').replace('P','')
        if 'S' in resolution:
            return int(resolution[:-1])
        elif 'M' in resolution:
            return int(resolution[:-1]) * 60
        elif 'H' in resolution:
            return int(resolution[:-1]) * 60 * 60
        elif 'D' in resolution:
            return int(resolution[:-1]) * 60 * 60 * 24
        else:
            raise ValueError('Not a valid resolution.')

class Product:

    def __init__(self, product):

        self.props = product  # for later access to any property which is not included below
        self.__id = product['id']
        self.__name = product['name']
        self.__isin = product['isin']
        self.__symbol = product['symbol']
        self.__min_contract = product['contractSize']
        self.__currency = product['currency']
        self.__product_type = product['productTypeId']
        self.__tradable = product['tradable']
        self.__close_price = product.get('closePrice')
        close_price_date = product.get('closePriceDate')
        self.__close_price_date = dt.datetime.strptime(close_price_date, '%Y-%m-%d').date() if close_price_date else None
        expiration_date = product.get('expirationDate')
        self.__expiration_date = dt.datetime.strptime(expiration_date, '%d-%m-%Y').date() if expiration_date else None
        self.__strike_price = product.get('strikePrice')

    def __getitem__(self, item):

        """e.g. product["exchangeId"] """
        return self.props[item]

    @property
    def id(self):

        return self.__id

    @property
    def name(self):

        return self.__name

    @property
    def isin(self):

        return self.__isin

    @property
    def symbol(self):

        return self.__symbol

    @property
    def min_contract(self):

        return self.__min_contract

    @property
    def currency(self):

        return self.__currency

    @property
    def product_type(self):

        return self.__product_type

    @property
    def tradable(self):

        return self.__tradable

    @property
    def close_price(self):

        return self.__close_price

    @property
    def close_price_date(self):

        return self.__close_price_date

    @property
    def expiration_date(self):

        return self.__expiration_date

    @property
    def strike_price(self):

        return self.__strike_price

    @property
    def is_option(self):  # stock option?

        return self.product_type == 8
    
class ClientInfo:

    def __init__(self, client_info):

        self.__token = client_info['id']
        self.__account_id = client_info['intAccount']
        self.__username = client_info['username']
        self.__first_name = client_info['firstContact']['firstName']
        self.__last_name = client_info['firstContact']['lastName']
        self.__email = client_info['email']

    @property
    def token(self):
        return self.__token

    @property
    def account_id(self):
        return self.__account_id

    @property
    def username(self):
        return self.__username

    @property
    def first_name(self):
        return self.__first_name

    @property
    def last_name(self):
        return self.__last_name

    @property
    def email(self):
        return self.__email
    
urls = {
    'allocationsUrl': "https://trader.degiro.nl/allocations/",
    'betaLandingPath': "/beta-trader/",
    'clientId': 3442517,
    'companiesServiceUrl': "https://trader.degiro.nl/dgtbxdsservice/",
    'dictionaryUrl': "https://trader.degiro.nl/product_search/config/dictionary/",
    'exanteReportingUrl': "https://trader.degiro.nl/exante-reporting",
    'favoritesUrl': "https://trader.degiro.nl/favorites/",
    'feedbackUrl': "https://trader.degiro.nl/feedback/",
    'i18nUrl': "https://trader.degiro.nl/i18n/",
    'landingPath': "/trader/",
    'latestSearchedProductsUrl': "https://trader.degiro.nl/latest-searched-products/secure/",
    'loginUrl': "https://trader.degiro.nl/login/es",
    'mobileLandingPath': "/trader/",
    'paUrl': "https://trader.degiro.nl/pa/secure/",
    'paymentServiceUrl': "https://trader.degiro.nl/payments/",
    'productNotesUrl': "https://trader.degiro.nl/product-notes-service/secure/",
    'productSearchUrl': "https://trader.degiro.nl/product_search/secure/",
    'productSearchV2Url': "https://internal.degiro.eu/dgproductsearch/secure/",
    'productTypesUrl': "https://trader.degiro.nl/product_search/config/productTypes/",
    'refinitivAgendaUrl': "https://trader.degiro.nl/dgtbxdsservice/agenda/v2",
    'refinitivClipsUrl': "https://trader.degiro.nl/refinitiv-insider-proxy/secure/",
    'refinitivCompanyProfileUrl': "https://trader.degiro.nl/dgtbxdsservice/company-profile/v2",
    'refinitivCompanyRatiosUrl': "https://trader.degiro.nl/dgtbxdsservice/company-ratios",
    'refinitivEsgsUrl': "https://trader.degiro.nl/dgtbxdsservice/esgs",
    'refinitivEstimatesUrl': "https://trader.degiro.nl/dgtbxdsservice/estimates-summaries",
    'refinitivFinancialStatementsUrl': "https://trader.degiro.nl/dgtbxdsservice/financial-statements",
    'refinitivInsiderTransactionsUrl': "https://trader.degiro.nl/dgtbxdsservice/insider-transactions",
    'refinitivInsidersReportUrl': "https://trader.degiro.nl/dgtbxdsservice/insiders-report",
    'refinitivInvestorUrl': "https://trader.degiro.nl/dgtbxdsservice/investor",
    'refinitivNewsUrl': "https://trader.degiro.nl/dgtbxdsservice/newsfeed/v2",
    'refinitivShareholdersUrl': "https://trader.degiro.nl/dgtbxdsservice/shareholders",
    'refinitivTopNewsCategoriesUrl': "https://trader.degiro.nl/dgtbxdsservice/newsfeed/v2/top-news-categories",
    'reportingUrl': "https://trader.degiro.nl/reporting/secure/",
    'sessionId': "CC36B531899BB8A7E010ED1CC9E7E753.prod_a_168_3",
    'settingsUrl': "https://trader.degiro.nl/settings/",
    'taskManagerUrl': "https://trader.degiro.nl/taskmanager/",
    'tradingUrl': "https://trader.degiro.nl/trading/secure/",
    'translationsUrl': "https://trader.degiro.nl/translations/",
    'vwdChartApiUrl': "https://charting.vwdservices.com/hchart/v1/deGiro/api.js",
    'vwdGossipsUrl': "https://solutions.vwdservices.com/customers/degiro.nl/news-feed/api/",
    'vwdNewsUrl': "https://solutions.vwdservices.com/customers/degiro.nl/news-feed/api/",
    'vwdQuotecastServiceUrl' : "https://trader.degiro.nl/vwd-quotecast-service/",
}

class DeGiro(object):

    __LOGIN_URL = 'https://trader.degiro.nl/login/secure/login'
    __LOGIN_TOTP_URL = 'https://trader.degiro.nl/login/secure/login/totp'
    __CONFIG_URL = 'https://trader.degiro.nl/login/secure/config'

    __LOGOUT_URL = 'https://trader.degiro.nl/trading/secure/logout'

    __CLIENT_INFO_URL = 'https://trader.degiro.nl/pa/secure/client'

    __GET_STOCKS_URL = 'https://trader.degiro.nl/products_s/secure/v5/stocks'
    __GET_URL = 'https://trader.degiro.nl/products_s/secure/v5/'
    ##__GET_STOCKS_URL = 'https://trader.degiro.nl/products_s/secure/v5/etfs'
    ##__GET_OPTIONS_URL = 'https://trader.degiro.nl/products_s/secure/v5/options'
    __PRODUCT_SEARCH_URL = 'https://trader.degiro.nl/product_search/secure/v5/products/lookup'
    __PRODUCT_INFO_URL = 'https://trader.degiro.nl/product_search/secure/v5/products/info'
    __ID_DICTIONARY_URL = 'https://trader.degiro.nl/product_search/config/dictionary'
    __TRANSACTIONS_URL = 'https://trader.degiro.nl/reporting/secure/v4/transactions'
    __ORDERS_URL = 'https://trader.degiro.nl/reporting/secure/v4/order-history'
    __ACCOUNT_URL = 'https://trader.degiro.nl/reporting/secure/v6/accountoverview'
    __DIVIDENDS_URL = 'https://trader.degiro.nl/reporting/secure/v3/ca/'

    __PLACE_ORDER_URL = 'https://trader.degiro.nl/trading/secure/v5/checkOrder'
    __ORDER_URL = 'https://trader.degiro.nl/trading/secure/v5/order/'

    __DATA_URL = 'https://trader.degiro.nl/trading/secure/v5/update/'
    __PRICE_DATA_URL = 'https://charting.vwdservices.com/hchart/v1/deGiro/data.js'
    __NEWS_DATA_URL = 'https://solutions.vwdservices.com/customers/degiro.nl/news-feed/api/'
    __CALENDAR_DATA_URL = 'https://trader.degiro.nl/dgtbxdsservice/agenda/v2'
    __TOPNEWS_DATA_URL = 'https://trader.degiro.nl/dgtbxdsservice/newsfeed/v2/top-news-preview?intAccount=51060786&sessionId=C40B0EC272B45977BB4F2FFA75CFC051.prod_a_165_4'
    __LATESTNEWS_DATA_URL = 'https://trader.degiro.nl/dgtbxdsservice/newsfeed/v2/latest-news?offset=0&languages=es&limit=10&intAccount=51060786&sessionId=C40B0EC272B45977BB4F2FFA75CFC051.prod_a_165_4'
    __FINANCIAL_STATEMENTS_DATA_URL = 'https://trader.degiro.nl/dgtbxdsservice/financial-statements/LU1681048804?intAccount=51060786&sessionId=C40B0EC272B45977BB4F2FFA75CFC051.prod_a_165_4'
    __COMPANY_RATIOS = 'https://trader.degiro.nl/dgtbxdsservice/company-ratios/LU1681048804?intAccount=51060786&sessionId=C40B0EC272B45977BB4F2FFA75CFC051.prod_a_165_4'
    __COMPANY_PROFILE = 'https://trader.degiro.nl/dgtbxdsservice/company-profile/v2/LU1681048804?intAccount=51060786&sessionId=C40B0EC272B45977BB4F2FFA75CFC051.prod_a_165_4'

    __GET_REQUEST = 0
    __POST_REQUEST = 1
    __DELETE_REQUEST = 2

    client_token = None
    session_id = None
    client_info = None
    confirmation_id = None

    def __init__(self, username:str, password:str, totp:str=None):

        '''
        Class used to connect DeGiro.
        '''

        self._id_dictionary = None
        self.login(username=username, password=password, totp=totp)
        info = self.clientInfo()
        self.clientToken()

    @staticmethod
    def __request(url, cookie:dict=None, payload:dict=None, headers:dict=None, 
                  data:dict=None, post_params:dict=None, request_type:int=__GET_REQUEST,
                  error_message:str='An error occurred.') -> dict:

        '''
        Carries out the login to DeGiro.

        Parameters
        ----------
        username: str
            Username of the account.
        password: str
            Password of the account.
        totp: str
            One time password, is optional.

        Returns
        -------
        client_info_response: dict
            Contains the information of the client.
        '''

        if not headers:
            headers = {}
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) " \
                                "Chrome/108.0.0.0 Safari/537.36"

        if request_type == DeGiro.__DELETE_REQUEST:
            response = requests.delete(url, headers=headers, json=payload)
        elif request_type == DeGiro.__GET_REQUEST and cookie:
            response = requests.get(url, headers=headers, cookies=cookie)
        elif request_type == DeGiro.__GET_REQUEST:
            response = requests.get(url, headers=headers, params=payload)
        elif request_type == DeGiro.__POST_REQUEST and headers and data:
            response = requests.post(url, headers=headers, params=payload, data=data)
        elif request_type == DeGiro.__POST_REQUEST and post_params:
            response = requests.post(url, headers=headers, params=post_params, json=payload)
        elif request_type == DeGiro.__POST_REQUEST:
            response = requests.post(url, headers=headers, json=payload)
        else:
            raise Exception(f'Unknown request type: {request_type}')

        if response.status_code == 200 or response.status_code == 201:
            try:
                return response.json()
            except:
                return "No data"
        else:
            raise Exception(f'{error_message} Response: {response.text}')

    def login(self, username:str, password:str, totp:str=None) -> dict:

        '''
        Carries out the login to DeGiro.

        Parameters
        ----------
        username: str
            Username of the account.
        password: str
            Password of the account.
        totp: str
            One time password, is optional.

        Returns
        -------
        client_info_response: dict
            Contains the information of the client.
        '''

        # Request login
        login_payload = {
            'username': username,
            'password': password,
            'isPassCodeReset': False,
            'isRedirectToMobile': False
        }
        if totp is not None:
            login_payload["oneTimePassword"] = totp
            url = DeGiro.__LOGIN_TOTP_URL
        else:
            url = DeGiro.__LOGIN_URL
            
        login_response = self.__request(url, payload=login_payload, 
                                        request_type=DeGiro.__POST_REQUEST,
                                        error_message='Could not login.')
        
        self.session_id = login_response['sessionId']

    def clientToken(self) -> dict:

        '''
        Gets the client Token.

        Returns
        -------
        client_token_response: dict
            Contains the information of the client's session.
        '''
        
        client_token_response = self.__request(DeGiro.__CONFIG_URL, cookie={'JSESSIONID': self.session_id}, 
                                               request_type=DeGiro.__GET_REQUEST,
                                               error_message='Could not get client config.')
        self.token_response = client_token_response
        self.client_token = client_token_response['data']['clientId']

        return client_token_response
    
    def clientInfo(self) -> dict:

        '''
        Gets information about the client.

        Returns
        -------
        client_info_response: dict
            Contains the information of the client.
        '''
        
        client_info_payload = {'sessionId': self.session_id}
        client_info_response = self.__request(DeGiro.__CLIENT_INFO_URL, payload=client_info_payload,
                                              error_message='Could not get client info.')
        self.client_info = ClientInfo(client_info_response['data'])

        return client_info_response

    def logout(self) -> None:

        '''
        Carries out the logout of DeGiro.
        '''

        logout_payload = {
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id,
        }

        self.__request(DeGiro.__LOGOUT_URL + ';jsessionid=' + self.session_id, 
                       payload=logout_payload,
                       error_message='Could not log out')

    def searchProducts(self, search_text:str, limit:int=1):

        product_search_payload = {
            'searchText': search_text,
            'limit': limit,
            'offset': 0,
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        return self.__request(DeGiro.__PRODUCT_SEARCH_URL, payload=product_search_payload,
                              error_message='Could not get products.')['products']

    def productInfo(self, product_id:str):

        product_info_payload = {
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        return self.__request(DeGiro.__PRODUCT_INFO_URL, payload=product_info_payload,
                              headers={'content-type': 'application/json'},
                              data=json.dumps([str(product_id)]),
                              request_type=DeGiro.__POST_REQUEST,
                              error_message='Could not get product info.')['data'][str(product_id)]

    @property
    def id_dictionary(self):

        if self._id_dictionary:  # already cached
            return self._id_dictionary

        raw_dict = self.__request(DeGiro.__ID_DICTIONARY_URL, error_message='Could not get Degiro ID dictionary.')
        self._id_dictionary = {k: {str(i["id"]): i for i in ids} for k, ids in raw_dict.items()}

        return self._id_dictionary

    def transactions(self, from_date:dt.datetime=None, to_date:dt.datetime=None, 
                     group_transactions:bool=False) -> dict:

        if to_date == None:
            to_date = dt.datetime.today()
        if from_date == None:
            from_date = to_date - dt.timedelta(days=90)

        transactions_payload = {
            'fromDate': from_date.strftime('%d/%m/%Y'),
            'toDate': to_date.strftime('%d/%m/%Y'),
            'group_transactions_by_order': group_transactions,
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        return self.__request(DeGiro.__TRANSACTIONS_URL, payload=transactions_payload,
                              error_message='Could not get transactions.')['data']

    def accountHistory(self, from_date:dt.datetime=None, to_date:dt.datetime=None) -> dict:

        if to_date == None:
            to_date = dt.datetime.today()
        if from_date == None:
            from_date = to_date - dt.timedelta(days=90)

        account_payload = {
            'fromDate': from_date.strftime('%d/%m/%Y'),
            'toDate': to_date.strftime('%d/%m/%Y'),
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        return self.__request(DeGiro.__ACCOUNT_URL, payload=account_payload,
                              error_message='Could not get account overview.')['data']

    def futureDividends(self) -> dict:

        dividends_payload = {
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        return self.__request(DeGiro.__DIVIDENDS_URL + str(self.client_info.account_id), 
                              payload=dividends_payload,
                              error_message='Could not get future dividends.')['data']

    def getOrders(self, from_date:dt.datetime=None, to_date:dt.datetime=None, 
                  not_executed:bool=False) -> dict:

        if to_date == None:
            to_date = dt.datetime.today()
        if from_date == None:
            from_date = to_date - dt.timedelta(days=90)

        orders_payload = {
            'fromDate': from_date.strftime('%d/%m/%Y'),
            'toDate': to_date.strftime('%d/%m/%Y'),
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        # max 90 days
        if (to_date - from_date).days > 90:
            raise Exception('The maximum timespan is 90 days')
        
        data = self.__request(DeGiro.__ORDERS_URL, payload=orders_payload, 
                              error_message='Could not get orders.')['data']
        data_not_executed = []
        if not_executed:
            for d in data:
                if d['isActive']:
                    data_not_executed.append(d)
            return data_not_executed
        else:
            return data

    def deleteOrder(self, orderId:str):

        delete_order_params = {
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id,
        }

        return self.__request(DeGiro.__ORDER_URL + orderId + ';jsessionid=' + self.session_id,
                              payload=delete_order_params,
                              request_type=DeGiro.__DELETE_REQUEST,
                              error_message='Could not delete order' + " " + orderId)

    @staticmethod
    def _filtercashfunds(cashfunds:dict) -> list:

        data = []
        for item in cashfunds['cashFunds']['value']:
            if item['value'][2]['value'] != 0:
                data.append(item['value'][1]['value'] + " " + str(item['value'][2]['value']))

        return data

    @staticmethod
    def _filterportfolio(portfolio:dict, filter_zero:bool=False) -> list:

        data = []
        data_non_zero = []
        for item in portfolio['portfolio']['value']:
            positionType = size = price = value = breakEvenPrice = None
            for i in item['value']:
                size = i['value'] if i['name'] == 'size' else size
                positionType = i['value'] if i['name'] == 'positionType' else positionType
                price = i['value'] if i['name'] == 'price' else price
                value = i['value'] if i['name'] == 'value' else value
                breakEvenPrice = i['value'] if i['name'] == 'breakEvenPrice' else breakEvenPrice
            data.append({
                "id": item['id'],
                "positionType": positionType,
                "size": size,
                "price": price,
                "value": value,
                "breakEvenPrice": breakEvenPrice
            })
        if filter_zero:
            for d in data:
                if d['size'] != 0.0:
                    data_non_zero.append(d)
            return data_non_zero
        else:
            return data

    def getData(self, datatype:str='portfolio', filter_zero:bool=False):

        data_payload = {
            datatype: 0
        }

        if datatype == DataType.CASHFUNDS:
            return self._filtercashfunds(
                self.__request(DeGiro.__DATA_URL + str(self.client_info.account_id) + ';jsessionid=' + self.session_id,
                               payload=data_payload,
                               error_message='Could not get data'))
        elif datatype == DataType.PORTFOLIO:
            return self._filterportfolio(
                self.__request(DeGiro.__DATA_URL + str(self.client_info.account_id) + ';jsessionid=' + self.session_id,
                               payload=data_payload,
                               error_message='Could not get data'), filter_zero)
        else:
            return self.__request(
                DeGiro.__DATA_URL + str(self.client_info.account_id) + ';jsessionid=' + self.session_id,
                payload=data_payload,
                error_message='Could not get data')

    def getQuote(self, product:Product, interval:str='P50Y') -> dict:
        
        # Get instrument info
        if isinstance(product, str):
            temp = self.productInfo(product)
            vw_id = temp['vwdId']
            vw_type = temp['vwdIdentifierType']
        else:
            vw_id = product.props['vwdId']
            vw_type = product.props['vwdIdentifierType']

        price_payload = {
            'requestid': 1,
            'period': interval,
            'series': [f'{vw_type}:{vw_id}'], # 
            'userToken': self.client_token
        }

        return self.__request(DeGiro.__PRICE_DATA_URL, payload=price_payload,
                             error_message='Could not get real time price')['series'][0]

    @staticmethod
    def _parseStart(start:str, resolution:str=None) -> float:

        """Extract the start timestamp of a timeserie.
        Args:
            times (str):
                Combination of `start date` and `resolution` of the serie.
                Example :
                    times = "2021-10-28/P6M"
                    times = "2021-11-03T00:00:00/PT1H"
        Returns:
            float:
                Timestamp of the start date of the serie.
        """

        if '/' in start:
            (start, resolution) = start.rsplit(sep="/", maxsplit=1)
        resolution = resolution.replace('Z', '')

        date_format = ""
        if resolution.startswith("PT"):
            date_format = "%Y-%m-%dT%H:%M:%S"
        else:
            date_format = "%Y-%m-%d"

        start_datetime = dt.datetime.strptime(start, date_format)
        start_timestamp = start_datetime.timestamp()

        return start_timestamp

    @staticmethod
    def _resToSeconds(resolution: str) -> int:
        """Extract the interval of a timeserie.
        Args:
            times (str):
                Combination of `start date` and `resolution` of the serie.
                Example :
                    times = "2021-10-28/P6M"
                    times = "2021-11-03T00:00:00/PT1H"
        Raises:
            AttributeError:
                if the resolution is unknown.
        Returns:
            int:
                Number of seconds in the interval.
        """

        if '/' in resolution:
            (_start, resolution) = resolution.rsplit(sep="/", maxsplit=1)

        resolution = resolution.replace('PT', '').replace('P','')
        if 'S' in resolution:
            return int(resolution[:-1])
        elif 'M' in resolution:
            return int(resolution[:-1]) * 60
        elif 'H' in resolution:
            return int(resolution[:-1]) * 60 * 60
        elif 'D' in resolution:
            return int(resolution[:-1]) * 60 * 60 * 24
        else:
            raise ValueError('Not a valid resolution.')
        
    @staticmethod
    def _serieToDF(serie:dict) -> pd.DataFrame:

        """Converts a timeserie into a DataFrame.
        Only series with the following types can be converted into DataFrame :
        - serie.type == "time"
        - serie.type == "ohlc"
        Beware of series with the following type :
         - serie.type == "object"
        These are not actual timeseries and can't converted into DataFrame.
        Args:
            serie (Chart.Serie):
                The serie to convert.
        Raises:
            AttributeError:
                If the serie.type is incorrect.
        Returns:
            pd.DataFrame: [description]
        """

        columns = []
        if serie['type'] == 'ohlc' and serie['id'].startswith('ohlc:'):
            columns = [
                'timestamp',
                'open',
                'high',
                'low',
                'close',
            ]
        elif serie['type'] == 'time' and serie['id'].startswith('price:'):
            columns = [
                'timestamp',
                'price',
            ]
        elif serie['type'] == 'time' and serie['id'].startswith('volume:'):
            columns = [
                'timestamp',
                'volume',
            ]
        elif serie['type'] == 'object':
            raise AttributeError(f"Not a timeserie, serie['type'] = {serie['type']}")
        else:
            raise AttributeError(f"Unknown serie, serie['type'] = {serie['type']}")
        
        return pd.DataFrame.from_records(serie['data'], columns=columns)
    
    def getCandles(self, product:Product, resolution:str=ResolutionType.D1, 
                   interval:str=IntervalType.Max, df:bool=True) -> dict:
        
        # Get instrument info
        if isinstance(product, str):
            temp = self.productInfo(product)
            vw_id = temp['vwdId']
            vw_type = temp['vwdIdentifierType']
        else:
            vw_id = product.props['vwdId']
            vw_type = product.props['vwdIdentifierType']

        price_payload = {
            'requestid': 1,
            'period': interval,
            'resolution': resolution,
            'series': [f'{vw_type}:{vw_id}', f'ohlc:{vw_type}:{vw_id}', f'volume:{vw_type}:{vw_id}'], # 
            'userToken': self.client_token,
            'tz': 'UTC',
        }
        data = self.__request(DeGiro.__PRICE_DATA_URL, payload=price_payload,
                             error_message='Could not get real time price')['series']
        
        if df:
            quote = data[0]['data']
            start = quote['windowFirst']
            dfs = []
            for serie in data:
                if serie['type'] in ["time", "ohlc"]:
                    times = serie['times']
                    start = self._parseStart(start=times)
                    interval = self._resToSeconds(resolution=times)
            
                    for datapoint in serie['data']:
                        datapoint[0] = start + datapoint[0] * interval

                    dfs.append(self._serieToDF(serie))
                    
            data = pd.merge(dfs[0], dfs[1], left_index=True, right_index=True)
            data = data[[c for c in data.columns if '_y' not in c]]
            data.rename(columns={k: k.replace('_x','') for k in data.columns}, inplace=True)
            # data = pd.merge(dfs[0], dfs[1], on='timestamp')
            data.columns = [c.capitalize() for c in data.columns]
            if 'Timestamp' in data.columns:
                data['DateTime'] = pd.to_datetime(data['Timestamp'], utc=True, unit='s')
        else:
            data = {d['id'].split(':')[0]: d['data'] for d in data}

        return data

    def tradeOrder(self, productId:str, size:int, side:str, orderType:str=Order.Type.MARKET, 
                 timeType:int=Order.Time.GTC, limit=None, stop_loss=None):

        place_order_params = {
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id,
        }

        if orderType not in Order.Type.options():
            raise Exception('Invalid order type')

        if timeType not in Order.Time.options():
            raise Exception('Invalid time type')
    
        if side not in Order.Side.options():
            raise Exception('Invalid side for the order.')

        place_order_payload = {
            'buySell': side,
            'orderType': orderType,
            'productId': productId,
            'timeType': timeType,
            'size': size,
            'price': limit,
            'stopPrice': stop_loss,
        }

        place_check_order_response = self.__request(DeGiro.__PLACE_ORDER_URL + ';jsessionid=' + self.session_id,
                                                    payload=place_order_payload, post_params=place_order_params,
                                                    request_type=DeGiro.__POST_REQUEST,
                                                    error_message='Could not place order')

        self.confirmation_id = place_check_order_response['data']['confirmationId']

        self.__request(DeGiro.__ORDER_URL + self.confirmation_id + ';jsessionid=' + self.session_id,
                       payload=place_order_payload, post_params=place_order_params,
                       request_type=DeGiro.__POST_REQUEST,
                       error_message='Could not confirm order')

    def getStockList(self, indexId:str=None, stockCountryId:str=None):

        stock_list_params = {
            'indexId': indexId,
            'stockCountryId': stockCountryId,
            'offset': 0,
            'limit': None,
            'requireTotal': "true",
            'sortColumns': "name",
            'sortTypes': "asc",
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        return self.__request(DeGiro.__GET_STOCKS_URL, payload=stock_list_params, 
                           error_message='Could not get stock list')['products']

    def getAsset(self, assetType:str='stocks', args:dict={}):

        '''
        assetType: str
            stocks, etfs
        '''

        params = {
            'offset': 0,
            'limit': None,
            'requireTotal': "true",
            'sortColumns': "name",
            'sortTypes': "asc",
            'intAccount': self.client_info.account_id,
            'sessionId': self.session_id
        }

        if len(list(args.keys())) > 0:
            params.update(args)

        return self.__request(DeGiro.__GET_URL+assetType, payload=params, 
                           error_message='Could not get stock list')['products']
    
if __name__ == '__main__':

    dg = DeGiro('OneMade','Onemade3680')
    products = dg.searchProducts('Amundi S&P 500') # dg.searchProducts('LU1681048804')
    data = dg.getCandles(Product(products[0]), resolution=ResolutionType.H1, 
                                             interval=IntervalType.Y3)
    #products = dg.searchProducts('Amundi Nasdaq-100') # dg.searchProducts('LU1681038243')
    #data = dg.getCandles(Product(products[0]), interval=IntervalType.Max)
    #products = dg.searchProducts('SPY') # dg.searchProducts('IE00B6YX5C33')
