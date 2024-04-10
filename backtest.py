import os
import sys
# 获取项目文件夹的绝对路径
strategies_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "strategies"))
# 将项目文件夹路径添加到模块搜索路径
sys.path.append(strategies_folder)

import argparse
import pandas as pd
import backtrader as bt
import pyfolio as pf
# import matplotlib
# matplotlib.rcParams['backend'] = 'Agg'

import warnings
warnings.filterwarnings("ignore")

#正常显示画图时出现的中文和负号
# from pylab import mpl
# mpl.rcParams['font.sans-serif']=['SimHei']
# mpl.rcParams['axes.unicode_minus']=False

from strategies.treasury_futures.SimpleMovingAverageStrategy import SimpleMovingAverageStrategy
from strategies.treasury_futures.RNNStrategy import RNNStrategy
from strategies.treasury_futures.MACDStrategy import MACDStrategy
from strategies.treasury_futures.PatchTSTStrategy import PatchTSTStrategy
from models.PatchTST.PatchTST_self_supervised.datautils import get_dls


from backtrader.comminfo import ComminfoFuturesPercent, ComminfoFuturesFixed  # 期货交易的手续费用，按照比例或者按照金额



class PandasDataPlus(bt.feeds.PandasData):
    lines = ('vwap',)  # 要添加的列名
    # 设置 line 在数据源上新增的位置
    params = (
        ('vwap', -1),  # vwap对应传入数据的列名，这个-1会自动匹配backtrader的数据类与原有pandas文件的列名
        # 如果是个大于等于0的数，比如8，那么backtrader会将原始数据下标8(第9列，下标从0开始)的列认为是turnover这一列
    )


def backtest_treasury_futures(args, code, path, columns, begin_date, end_date, scaler = None):
    """
    国债期货回测模块

    根据设置的策略执行完整回测，流程为：
        step1. Create a cerebro entity
        step2.1 Load index data
            step2.1.1 Add index data to cerebro
            step2.1.2 Add index config to cerebro
        step2.2 Load contract data
            step2.2.1 Add symbol data to cerebro
            step2.2.2 Add symbol config to cerebro
            step2.2.3 For each symbol
        step3. Add strategy to cerebro
        step4. Add analyzer to cerebro (use pyfolio)
        step5. Add startcash to cerebro
        step6. Run over everything
        step7. Plot the result

    Args:
        ---------数据相关参数
        code (str): 投资标的.
        path (str): 数据路径
        columns (list): 数据列名.（不同数据来源设置不同）
        scaler: 数据标准化器

        ----------回测相关参数(后续需要增加仓位管理回测)
        args.strategy_name(str): 策略名称
        begin_date (str): 回测开始时间
        end_date (str): 回测结束时间
        args.startcash(long): 回测初始资产
        args.stake(int): 每笔交易数量
        args.commission(float): 交易手续费率
        args.slip_perc(float): 滑点

    Returns:
        result: 回测结果
    """
    deep_learning_strategies = ['rnn','PatchTST']

    # step1. Create a cerebro entity
    cerebro = bt.Cerebro()

    # step2. Load back test data
    
    # step2.1 加载指数合约数据
    if args.data_source == 'akshare':
        # akshare 加权平均指数
        data = pd.read_csv(path, parse_dates=['datetime'],index_col='datetime')
    else:
        # cgs 主连数据
        data = pd.read_csv(os.path.join(path,code+'_dataset_day.csv'),parse_dates=['date'],index_col='date')
        data = data[columns[1:]]
    backtest_mask = (data.index >= begin_date) & (data.index <= end_date)
    index_df = data.loc[backtest_mask]
    if args.strategy_name in  deep_learning_strategies: 
        # 深度学习模型数据
        concat_df = data.loc[:index_df.index[0]].tail(args.window_size+1)
        index_df = pd.concat([index_df, concat_df], axis=0).sort_index().drop_duplicates()

        # 加入Alpha158
        alpha_path = os.path.join(path,'Alpha158',code.split('.')[0])
        alpha_df = pd.read_csv(os.path.join(alpha_path,begin_date+'.csv'),parse_dates=['datetime'],index_col='datetime')
        alpha_cols = list(alpha_df.columns)
        alpha_cols = alpha_cols[1:-1]
        alpha_df = alpha_df[alpha_cols]
        alpha_df = alpha_df.fillna(0)

        index_df = pd.merge(index_df, alpha_df, left_index=True, right_index=True, how='left')
    print('-------------------------------')
    print('begin backtesing')
    print(index_df)
    print('invest:', args.invest)
    print('code:', code,
        '\nstrategy:', args.strategy_name,
        '\nbegin backtest date:',begin_date, 
        '\nend backtest date:', end_date,
        '\nindex_df length:',len(index_df))
    begin_date = index_df.index[0]
    end_date = index_df.index[-1]
    
    # step2.1.1 Add index data to cerebro
    data = PandasDataPlus(
        dataname=index_df,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        openinterest=5,
        volume=6)
    cerebro.adddata(data, name='index')

    # akshare数据
    # feed = bt.feeds.PandasDirectData(dataname=index_df)
    # cerebro.adddata(feed, name='index')

    # step2.1.2 Add index config to cerebro
    # 设置合约的交易信息，佣金、保证金率，杠杆按照真实的杠杆来
    comm = ComminfoFuturesPercent(commission=args.commission, margin=args.margin, mult=args.multiplier)
    cerebro.broker.addcommissioninfo(comm, name="index")


    # step2.2 加载具体合约数据
    # 加载具体合约数据(akshare加载)
    # TODO：后续更改为从wind数据源加载
    data = pd.read_csv(os.path.join('.','datasets','akshare','CFFEX.csv'), index_col=0) 
    code_ = code.split('.')
    data = data[data['variety'] == code_[0]]
    data['datetime'] = pd.to_datetime(data['date'], format="%Y%m%d")
    data = data.dropna()
    for symbol, df in data.groupby("symbol"):
        df.index = pd.to_datetime(df['datetime'])
        df = df[['open', 'high', 'low', 'close', 'volume', 'open_interest']]
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'openinterest']

        # 单个合约需要截取在回测时间范围内的数据
        df_mask = (df.index >= begin_date) & (df.index <= end_date)
        symbol_df = df.loc[df_mask]
        if(len(symbol_df) == 0):
            continue

        # step2.2.1 Add symbol data to cerebro
        feed = bt.feeds.PandasDirectData(dataname=symbol_df)
        cerebro.adddata(feed, name=symbol)

        # step2.2.2 Add symbol config to cerebro
        # 设置合约的交易信息，佣金、保证金率，杠杆按照真实的杠杆来
        comm = ComminfoFuturesPercent(commission=args.commission, margin=args.margin, mult=args.multiplier)
        cerebro.broker.addcommissioninfo(comm, name=symbol)

    # step3. Add strategy to cerebro

    if args.strategy_name == 'SimpleMovingAverage':
        # 均值策略
        # cerebro.optstrategy(SimpleMovingAverageStrategy,maperiod=range(3, 31))  #导入策略参数寻优
        cerebro.addstrategy(SimpleMovingAverageStrategy,printlog=args.printlog)
    elif args.strategy_name == 'rnn':
        # 简单RNN策略
        cerebro.addstrategy(RNNStrategy,code = code, columns=columns[1:], #columns[1:] 去掉日期列
                            args = args, scaler=scaler, # scaler: 深度学习模型
                            printlog = args.printlog) 
    elif args.strategy_name == 'MACD':
        cerebro.addstrategy(MACDStrategy,period_me1=args.period_me1,  # 短期指数移动平均线
                            period_me2=args.period_me2, # 长期指数移动平均线
                            period_dif=args.period_dif, # 信号线
                            printlog = args.printlog) 
        # cerebro.addstrategy(MACDStrategy)
    elif args.strategy_name == 'PatchTST':
        # 获取scaler
        print('===========================get scaler===============================')
        dls = get_dls(args,columns)
        scaler = dls.train.dataset.scaler
        cerebro.addstrategy(PatchTSTStrategy,code = code, columns=columns[1:], #columns[1:] 去掉日期列
                            args = args,  scaler=scaler, # scaler: 深度学习模型
                            printlog = args.printlog) 

    # step4. Add startcash to cerebro
    # 本金
    cerebro.broker.setcash(args.startcash)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # step5. Add analyzer to cerebro (use pyfolio)
    cerebro.addanalyzer(bt.analyzers.TotalValue, _name='_TotalValue')
    cerebro.addanalyzer(bt.analyzers.PyFolio)

    # step6. Run over everything
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # step7. Plot the result
    # 7/15 由于需要嵌入pyfolio进行可视化 两边画图终端有许多问题 后续需要解决
    # cerebro.plot()

    # 返回到jupyer notebokk中进行可视化
    return results[0], scaler

def backtest_treasury_futures_old(args, code, path, columns, begin_date, end_date, scaler = None):
    """
    国债期货回测模块

    根据设置的策略执行完整回测，流程为：
        step1. Load back test data
        step2. Create a cerebro entity
        step3. Add data to cerebro
        step4. Add strategy to cerebro
        setp5. Add analyzer to cerebro (use pyfolio)
        step6. Add config to cerebro
        step7. Run over everything
        step8. Plot the result

    Args:
        ---------数据相关参数
        code (str): 投资标的.
        path (str): 数据路径
        columns (list): 数据列名.
        scaler: 数据标准化器

        ----------回测相关参数(后续需要增加仓位管理回测)
        args.strategy_name(str): 策略名称
        begin_date (str): 回测开始时间
        end_date (str): 回测结束时间
        args.startcash(long): 回测初始资产
        args.stake(int): 每笔交易数量
        args.commission(float): 交易手续费率
        args.slip_perc(float): 滑点

    Returns:
        result: 回测结果
    """
    deep_learning_strategies = ['rnn']

    # step1. Load back test data
    data = pd.read_csv(path, skiprows=1,names=columns,parse_dates=['date'],index_col='date')
    backtest_mask = (data.index >= begin_date) & (data.index <= end_date)
    backtest_df = data.loc[backtest_mask]
    if args.strategy_name in  deep_learning_strategies: 
        # 深度学习模型数据
        concat_df = data.loc[:backtest_df.index[0]].tail(args.window_size+1)
        backtest_df = pd.concat([backtest_df, concat_df], axis=0).sort_index().drop_duplicates()
    print(backtest_df)
    print('-------------------------------')
    print('begin backtesing')
    print('invest:', args.invest)
    print('code:', code,
        '\nstrategy:', args.strategy_name,
        '\nbegin backtest date:',begin_date, 
        '\nend backtest date:', end_date,
        '\nbacktest length:',len(backtest_df))

    # step2. Create a cerebro entity
    cerebro = bt.Cerebro(runonce=False)

    # step3. Add data to cerebro
    data = PandasDataPlus(
        dataname=backtest_df,
        datetime=None,
        open=0,
        high=1,
        low=2,
        close=3,
        openinterest=5,
        volume=6)
    cerebro.adddata(data)

    # step4. Add strategy to cerebro

    if args.strategy_name == 'SimpleMovingAverage':
        # 均值策略
        # cerebro.optstrategy(SimpleMovingAverageStrategy,maperiod=range(3, 31))  #导入策略参数寻优
        cerebro.addstrategy(SimpleMovingAverageStrategy,printlog=args.printlog)
    elif args.strategy_name == 'rnn':
        # 简单RNN策略
        cerebro.addstrategy(RNNStrategy,columns=columns[1:], #columns[1:] 去掉日期列
                            args = args, scaler=scaler, # scaler: 深度学习模型必选
                            printlog = args.printlog) 

    # step5. Add analyzer to cerebro (use pyfolio)
    cerebro.addanalyzer(bt.analyzers.PyFolio)

    # step6. Add config to cerebro
    # 本金
    cerebro.broker.setcash(args.startcash)
    # 交易数量
    cerebro.addsizer(bt.sizers.FixedSize, stake=args.stake)
    # 手续费
    cerebro.broker.setcommission(commission=args.commission)
    # 滑点
    cerebro.broker.set_slippage_perc(perc=args.slippage_perc)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # step7. Run over everything
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # step8. Plot the result
    # 7/15 由于需要嵌入pyfolio进行可视化 两边画图终端有许多问题 后续需要解决
    # cerebro.plot()

    # 返回到jupyer notebokk中进行可视化
    return results[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # ---------回测相关参数(后续需要增加仓位管理回测)
    parser.add_argument('--strategy_name', type=str, default='SimpleMovingAverage') # 回测策略
    parser.add_argument('--startcash', type=int, default=100000)                    # 回测初始资产
    parser.add_argument('--stake', type=int, default=100)                           # 每笔交易数量
    parser.add_argument('--commission', type=float, default=0.001)                  # 交易手续费率
    parser.add_argument('--slippage_perc', type=float, default=0.001)               # 滑点
    parser.add_argument('--printlog', type=bool, default=False)                     # 是否打印回测记录


    args = parser.parse_args()
    print(args)

    # 设置投资标的
    code_list=['T.CFE','TF.CFE','TS.CFE']
    begin_date_list=["2015-03-20","2013-09-06","2018-08-17"]
    
    columns=['date','open','high','low','close','vwap','oi','volume']
    data_sets = {}
    for i in range(len(code_list)):
        code = code_list[i]
        # step1. 获取数据
        print('-------------------------------')
        path = os.path.join('.','datasets',code+'_dataset_day.csv')
        
        # step2. 配置模型
        model_name = 'SimpleMovingAverage'

        # step3. 执行回测
        begin_backtest_date = begin_date_list[i]
        end_backtest_date = '2023-07-10'
        backtest_treasury_futures(args, code, path, columns, begin_backtest_date, end_backtest_date)


