import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, date, timedelta
import requests
import numpy as np
import yfinance as yf

def lowprice_sotck_codes():
    # get all codes
    url = 'https://isin.twse.com.tw/isin/C_public.jsp?strMode=2'
    response = requests.get(url)
    response.encoding = 'big5' 
    data = pd.read_html(response.text)[0]

    stock_codes = data.iloc[2:]
    stock_codes.columns = data.iloc[0]
    stock_codes = stock_codes.iloc[1:]['有價證券代號及名稱']
    stock_codes = stock_codes.str.extract(r'(\d+)').dropna()
    stock_codes = stock_codes[0].tolist()
    stock_codes = np.array(stock_codes)
    stock_codes = stock_codes[np.char.str_len(stock_codes.astype(str)) <=4]
    stock_codes = np.char.add(stock_codes, ".TW")


    # time_range
    today = date(2023, 6, 13)
    end_date = today.strftime("%Y-%m-%d")
    d1 = date.today() - timedelta(days=360*10) #for last 10 years
    start_date = d1.strftime("%Y-%m-%d")

    stock_list = []


    for i , code in enumerate(stock_codes):
        df = yf.download(tickers = code,
                        start = start_date,
                        end = end_date)
        
        if df['Close'].iloc[-1] >= df['Close'].mean()*1.1:
            continue
        stock_list.append(code)
        lowprice_stock_list = np.char.rstrip(stock_list, '.TW')
        
    return lowprice_stock_list


def good_revenue_codes():

    good_revenue = []
    month = 3
    for i in range(3):
        date = f'112_{month}'  
        url = f'https://mops.twse.com.tw/nas/t21/sii/t21sc03_{date}_0.html'
        res = requests.get(url)
        res.encoding = 'big5'
        html_df = pd.read_html(res.text)           
    
        df = pd.concat([df for df in html_df if df.shape[1] == 11]) 
        df.columns = df.columns.get_level_values(1)
        df = df[df['公司名稱'] != '合計']
        df = df.reset_index(drop=True)
        good_revenue_df = df[df['上月比較增減(%)'] >=0]
        good_revenue.append(good_revenue_df['公司代號'].values)  
        month +=1
    sets = [set(x) for x in good_revenue]
    common_values = list(set.intersection(*sets))
    
    return common_values

def main():
    good_revenue_stock = good_revenue_codes()
    lowprice_sotck = lowprice_sotck_codes()

    valuable_stock_code = [stock for stock in lowprice_sotck if stock in good_revenue_stock]
    print('十年線附近:', lowprice_sotck)
    print(valuable_stock_code)








if __name__ == '__main__':
    main()
