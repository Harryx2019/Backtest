import pandas as pd
import numpy as np
import os


def pre_process_treasury_futures_data(code,data,save_path):
    """
    国债期货加权合成指数

    Args:
        code(str): 国债期货合约编号
        data(pd.DataFrame): akshare获取的每日国债期货数据
        save_path(str): 数据保存路径

    """
    # 根据持仓量加权合成指数合约
    print('-------------------------------')
    print('正在根据持仓量加权合成指数合约')
    code = code.split('.')
    code = code[0]

    data = data[data['variety'] == code]
    data['datetime'] = pd.to_datetime(data['date'], format="%Y%m%d")
    data = data.dropna()
    
    result = []
    for index, df in data.groupby("date"):
        total_open_interest = df['open_interest'].sum()
        open = (df['open']*df['open_interest']).sum()/total_open_interest
        high = (df['high'] * df['open_interest']).sum() / total_open_interest
        low = (df['low'] * df['open_interest']).sum() / total_open_interest
        close = (df['close'] * df['open_interest']).sum() / total_open_interest
        volume = (df['volume'] * df['open_interest']).sum() / total_open_interest
        open_interest = df['open_interest'].mean()
        result.append([index, open, high, low, close, volume, open_interest])
    index_df = pd.DataFrame(result, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest'])

    index_df.to_csv(os.path.join(save_path,code+'.csv'),index=False)

