import pandas as pd
from datetime import datetime
import backtrader as bt
import matplotlib.pyplot as plt
import tushare as ts

plt.rcParams["font.sans-serif"] = ["SimHei"] # 设置画图时中文显示
plt.rcParams["axes.unicode_minus"] = False # 设置画图时的负号显示

ts.set_token('b9ce9683063b45ec99096dad651eb998f34b85e5c5b46d22337927e2')
pro = ts.pro_api()


# 1. 数据加载
def get_data_pro(code='600519.SH',startime='20170101',endtime='20200101'):
    df = pro.daily(ts_code=code,start_date=startime,end_date=endtime)
    df.index = pd.to_datetime(df.trade_date)
    df.rename_axis('date', axis='index', inplace=True)
    df['openinterest'] = 0
    df = df[['open','high','low','close','vol','openinterest']]
    df.columns = ['open','high','low','close','volume','openInterest']
    return df

def get_data(code='600519',startime='2017-01-01',endtime='2020-01-01'):
    df = ts.get_k_data(code,start=startime,end=endtime)
    df.index = pd.to_datetime(df.date)
    df['openinterest'] = 0
    df = df[['open','high','low','close','volume','openinterest']]
    
    return df

stock_df = get_data()
print(stock_df)
stock_df1 = get_data(code='600419')
# 加载并读取数据源 dataname （数据来源） fromdate(date格式) todate
fromdate = datetime(2017,1,1)
todate = datetime(2020,1,1)

# 创建第一个数据集
data = bt.feeds.PandasData(dataname=stock_df,fromdate=fromdate,todate=todate)
# 创建第二个数据集
data1 = bt.feeds.PandasData(dataname=stock_df1,fromdate=fromdate,todate=todate)


# 2. 构建策略
# 上穿20日线买入，跌穿20日均线就卖出
class MyStrategy(bt.Strategy):
    params = dict(period=20)

    def __init__(self):
        self.order = None
        self.ma = bt.indicators.SimpleMovingAverage(self.datas[0],period=self.params.period)

    # 每个bar都会执行一次,回测的每个日期都会执行一次
    # Backtrader默认是“当日收盘后下单，次日以开盘价成交” 这种模式在回测过程中能有效避免使用未来数据
    # cheat-on-open "当日下单，当日以开盘价成交"
    # cheat-on-close "当日下单，当日以以收盘价成交"
    def next(self):
        # # 拿到lines中各个line的name
        # print(self.datas[0].lines.getlinealiases())
        # # 拿到当天的close
        # print(self.datas[0].lines[0][0],self.datas[0].lines[0][-1])
        # print(self.datas[0].close[0],self.datas[0].close[-1])
        # print(bt.num2date(self.datas[0].lines[6][0]))
        # 判断是否有交易指令正在进行
        if(self.order):
            return
        # 空仓
        if(not self.position):
            # datas[0] 为第一只股票的数据
            # close[0] 为当天的数据
            if self.datas[0].close[0] > self.ma[0]:
                self.order = self.buy(size = 200)
                # data 指定证券 size 订单委托数量 price 订单委托价
                # 按目标数量下单
                # self.order = self.order_traget_size(target=size)
                # 按目标金额下单
                # self.order = self.order_traget_value(target=value)
                # 按目标百分比下单
                # self.order = self.order_traget_percent(target=percent)
        else:
            if self.datas[0].close[0] < self.ma[0]:
                self.order = self.sell(size = 200)


# 3. 策略设置
cerebro = bt.Cerebro() # 创建大脑
# 将数据加入回测系统
# 第一个数据集
cerebro.adddata(data,name='maotai')
# 第二个数据集
cerebro.adddata(data1,name='tianrun')
# 加入自己的策略
cerebro.addstrategy(MyStrategy)
# 经纪人
startcash = 100000
cerebro.broker.setcash(startcash)
# 设置手续费
cerebro.broker.setcommission(0.0002)
# 设置百分比滑点
cerebro.broker.set_slippage_perc(perc=0.1)
# 设置交易时机
cerebro.broker.set_coo(True)


# 4. 执行回测
begin = fromdate.strftime("%Y-%m-%d")
end = todate.strftime("%Y-%m-%d")
print(f"初始资金:{startcash}\n回测时间:{begin}-{end}")

cerebro.run()
portval = cerebro.broker.getvalue() # getcash
print(f"剩余总资金:{portval}\n回测时间:{begin}-{end}")

