import os
import requests
import argparse
import logging
import pandas as pd
import shutil

from colorama import Fore, Style
from datetime import datetime, timedelta, timezone


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def Parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('crypto', nargs='+', help='crypto list')
    parser.add_argument('-timeframes', nargs='+', default=['1m', '30m', '1d'], help='timeframes to download')
    parser.add_argument('-update', action='store_true', help='update databases')
    parser.add_argument('-build', action='store_true', help='build databases')
    parser.add_argument('-split', action='store_true', help='split db in years in raw/')
    parser.add_argument('-filter', type=str, default=None, help='filtered db name')
    parser.add_argument('-delete', action='store_true', help='delete db')
    return parser.parse_args()


def AppendToCsv(data, file_path):
    columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", 
               "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
    df = pd.DataFrame(data, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if not pd.io.common.file_exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)


def FetchHistoricalData(symbol, interval, start_time):
    url = "https://api.binance.com/api/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": 1000
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Erreur API: {response.status_code} - {response.text}")
        return None


def BuildDb(crypto, interval, start_date):
    logging.info(f'{Style.BRIGHT}Building {crypto} {interval} historical database.{Style.RESET_ALL}')
    
    yesterday = datetime.now() - timedelta(1)
    file_path = f'crypto/{crypto}/{interval}/{crypto}_{interval}.csv'
    start_timestamp = int(pd.to_datetime(start_date, dayfirst=True).timestamp() * 1000)
    end_timestamp = int(yesterday.timestamp() * 1000)
    
    while start_timestamp <= end_timestamp:
        historical_data = FetchHistoricalData(crypto, interval, start_timestamp)
        if historical_data:
            AppendToCsv(historical_data, file_path)
            start_timestamp = historical_data[-1][0] + 1
            logging.info(f'{crypto} {interval} data downloaded until {datetime.fromtimestamp(historical_data[-1][0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}')
        else:
            break

    logging.info(f'{Style.BRIGHT}{crypto} {interval} historical database built successfully.{Style.RESET_ALL}')



def UpdateDb(crypto, interval):
    logging.info(f'{Style.BRIGHT}{Fore.GREEN}Updating {crypto} {interval} historical database.{Style.RESET_ALL}')
    
    file_path = f'crypto/{crypto}/{interval}/{crypto}_{interval}.csv'
    dataframe = pd.read_csv(file_path)
    
    if not pd.api.types.is_datetime64_any_dtype(dataframe['timestamp']):
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    
    last_timestamp = int(dataframe.iloc[-2]['timestamp'].timestamp() * 1000)
    now = datetime.now()
    end_timestamp = int(now.timestamp() * 1000)
    
    logging.info(f'Fetching data from {datetime.utcfromtimestamp(last_timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")} to now.')
    new_data = []
    while last_timestamp <= end_timestamp:
        historical_data = FetchHistoricalData(crypto, interval, last_timestamp)
        if historical_data:
            new_data.extend(historical_data)
            last_timestamp = historical_data[-1][0] + 1
        else:
            break
    
    new_data_df = pd.DataFrame(new_data, columns=dataframe.columns)
    new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'], unit='ms')
    
    updated_dataframe = pd.concat([dataframe[:-2], new_data_df]).drop_duplicates(subset='timestamp').reset_index(drop=True)
    updated_dataframe.to_csv(file_path, index=False)
    logging.info(f'{crypto} {interval} database updated successfully.')



def SplitDb(crypto, interval):
    logging.info(f'{Style.BRIGHT}{Fore.GREEN}Splitting {crypto} {interval} database into yearly files.{Style.RESET_ALL}')    

    dataframe = pd.read_csv(f'crypto/{crypto}/{interval}/{crypto}_{interval}.csv')    
    if not pd.api.types.is_datetime64_any_dtype(dataframe['timestamp']):
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    
    dataframe['year'] = dataframe['timestamp'].dt.year
    
    output_dir = f'crypto/{crypto}/{interval}/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    for year, group in dataframe.groupby('year'):
        year_file_path = f'crypto/{crypto}/{interval}/raw/{year}.csv'
        group.drop(columns=['year'], inplace=True)
        group.to_csv(year_file_path, index=False)
        logging.info(f'{Style.BRIGHT}Year {year} data saved to {year_file_path}.{Style.RESET_ALL}')
    
    logging.info(f'{Style.BRIGHT}{crypto} {interval} database split completed.{Style.RESET_ALL}')


def FilterDb(crypto, interval, file_name):
    df = pd.read_csv(f'crypto/{crypto}/{interval}/{crypto}_{interval}.csv', usecols=["timestamp", "open", "close", "high", "low", "volume"])
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    years = list(df['timestamp'].dt.year.unique())
    done = 0
    chosen_years_list = []
    while done != 1:
        for y, year in enumerate(years):
            print(f'\n{str(y)} - {str(year)}')
        answer = input('\nChoose index of the year to include or \'done\': ')
        
        if answer == 'done':
            done = 1

        else:
            try:
                answer = int(answer)
                if answer in range(0, len(years)): 
                    chosen_years_list.append(years[int(answer)])
                    years.remove(years[int(answer)])
                else:
                    print('Answer out of range')

            except Exception as error:
                print('Wrong input\n')

    df_filtered = df[df['timestamp'].dt.year.isin(chosen_years_list)]
    
    if os.path.exists(f'crypto/{crypto}/{interval}/{file_name}'):
        os.remove(f'crypto/{crypto}/{interval}/{file_name}')
    
    df_filtered.to_csv(f'crypto/{crypto}/{interval}/{file_name}', index=False)
    logging.info(f'{Style.BRIGHT}{Fore.GREEN}File {file_name} created for {crypto} {interval}.{Style.RESET_ALL}')



def CreateArch(timeframes, crypto_list):
    if not os.path.exists('crypto'):
        os.makedirs('crypto')

    for crypto in crypto_list:
        if not os.path.exists(f'crypto/{crypto}'):
            os.makedirs(f'crypto/{crypto}')

        for interval in timeframes:
            if not os.path.exists(f'crypto/{crypto}/{interval}'):
                os.makedirs(f'crypto/{crypto}/{interval}')
            if not os.path.exists(f'crypto/{crypto}/{interval}/raw'):
                os.makedirs(f'crypto/{crypto}/{interval}/raw')



def CheckErrors(args):
    if (args.update is True or args.split is True or args.filter is True) and not os.path.exists('crypto'):
        raise Exception('Database is not built, run \'-build\' first')



if __name__ == '__main__':
    try:
        args = Parsing()
        cryptos_starting_dates = {
            'BTCUSDT': '17/08/2017',
            'ETHUSDT': '07/08/2015',
            'SOLUSDT': '11/04/2020',
            'BNBUSDT': '25/07/2017',
            'ADAUSDT': '01/10/2017',
            'AVAXUSDT': '21/09/2020',
            'DOGEUSDT': '19/12/2013',
            'DOTUSDT': '18/08/2020',
            'XRPUSDT': '02/08/2013',
            'LTCUSDT': '07/12/2013',
            'BCHUSDT': '01/08/2017',
            'NEARUSDT': '24/08/2020',
            'AAVEUSDT': '16/10/2020'
        }
        CheckErrors(args)
        CreateArch(args.timeframes, args.crypto)
        for crypto in args.crypto:
            start_date = cryptos_starting_dates[crypto]
            for interval in args.timeframes:
                if args.build is True:
                    BuildDb(crypto, interval, start_date)

                if args.update is True:
                    UpdateDb(crypto, interval)

                if args.split is True or args.update is True:
                    SplitDb(crypto, interval)

                if args.filter != None:
                    FilterDb(crypto, interval, args.filter)

                if args.delete is True:
                    if os.path.exists('crypto'):
                        shutil.rmtree('crypto')
                    logging.info(f'Databases deleted.')
                    exit(0)
                    

    except Exception as error:
        print(f'Error:  + {error} +  moverfuker')
