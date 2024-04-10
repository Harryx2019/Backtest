import pandas as pd
import numpy as np
import tushare as ts
import akshare as ak

import os
import torch

from sklearn.preprocessing import MinMaxScaler


def get_stock_data_tushare(code, start_date, end_date, token):
    """
    通过tushare获取股票数据
    """
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    df = df.sort_values(by="trade_date", ascending=True)  # 对数据进行排序，以便滑动窗口操作
    df.set_index("trade_date", inplace=True)
    return df


def get_future_data_akshare(code, start_date, end_date, market, save_path):
    """
    通过akshare获取期货数据

    Args:
        code(str): "IC" 中证500 "IH" 上证50 "IF" 沪深300 
                    "T" 10年国债期货  "TF" 5年国债期货  "TS" 2年国债期货  
        start_date(str): 开始日期
        end_date(str): 结束日期
        market(str): "DCE" 大商所  "INE" 能源所  "SHFE" 上期所  
                    "CZCE" 郑商所  "CFFEX" 中金所  "GFEX" 广期所
        save_path(str): 数据保存路径

    Returns:
        df(pd.DataFrame): 获取到的数据
    """
    print('-------------------------------')
    print('正在从Akshare获取数据')
    print(code,start_date,end_date,market)
    get_futures_daily_df = ak.get_futures_daily(start_date=start_date, end_date=end_date, market=market)

    get_futures_daily_df.to_csv(os.path.join(save_path,market+'.csv'),index=False)



def get_future_data_csv(path,columns):
    """
    通过csv获取期货数据

    根据日期范围构建数据集

    Args:
        path(str): 数据路径
        columns(list): 数据列名

    Returns:
        df(pd.DataFrame): 获取到的数据
    """
    print('-------------------------------')
    print('正在从'+path+'获取数据')
    df = pd.read_csv(path, parse_dates=['date'], index_col='date')
    df = df[columns[1:]]
    return df
    

def create_dataset_classify(args, data, mode, begin_date, end_date, scaler = None):
    """
    构建分类单一分类数据集

    根据日期范围构建数据集

    Args:
        data (pd.DataFrame): 原始数据.
        mode(str): 数据集类型(train/valid/test)
        args.window_size(int): 模型滑动窗口.
        args.look_ahead(int): 预测未来第几天的收益
        begin_train_date(str): 数据集开始时间
        end_train_date(str): 数据集结束时间

    Returns:
        X(torch.tensor): 特征[num_winodws, window_size, feature_dim]
        y(torch.tensor): 标签[num_winodws]
        scaler: 训练集标准化器
    """
    data_mask = (data.index >= begin_date) & (data.index <= end_date)
    data_df = data.loc[data_mask]
    print('mode: ',mode)
    print(data_df)

    if mode == 'train':
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_df)
    else:
        data_normalized = scaler.transform(data_df)

    X = []
    y = []
    close_prices = data_df['close']
    for i in range(len(data_df) - args.window_size):
        X.append(data_normalized[i:i + args.window_size])
        if close_prices[i + args.window_size - 1 + args.look_ahead] > close_prices[i + args.window_size - 1]:
            y.append(1)
        else:
            y.append(0)

    X, y = np.array(X), np.array(y)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    print('mode:',mode,' X shape:',X.shape,' y shape:',y.shape)
    print('-------------------------------')

    return X,y,scaler


def create_datasets_classify(args, data):
    """
    构建分类划分数据集

    将输入数据划分为训练集、验证集、测试集

    Args:
        data (pd.DataFrame): 原始数据.
        args.window_size(int): 模型滑动窗口.
        args.look_ahead(int): 预测未来第几天的收益
        args.begin_train_date(str): 训练集开始时间
        args.end_train_date(str): 训练集结束时间
        args.begin_valid_date(str): 验证集开始时间
        args.end_valid_date(str): 验证集结束时间
        args.begin_test_date(str): 测试集开始时间
        args.end_test_date(str): 测试集结束时间

    Returns:
        X_train, y_train(torch.tensor): 训练集特征与标签
        X_valid, y_valid(torch.tensor): 验证集特征与标签
        X_test, y_test(torch.tensor): 测试集特征与标签
        scaler: 训练集标准化器
    """
    
    # 构建训练集
    X_train, y_train, scaler = create_dataset_classify(args, data,'train',
                                                       args.begin_train_date,args.end_train_date)

    # 构建验证集
    X_valid, y_valid, _ = create_dataset_classify(args, data,'valid',
                                                  args.begin_valid_date,args.end_valid_date, scaler)

    # 构建测试集
    X_test, y_test, _ = create_dataset_classify(args, data,'test',
                                                  args.begin_test_date,args.end_test_date, scaler)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler
