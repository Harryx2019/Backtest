import backtrader as bt

from BaseStrategy import BaseStrategy


class SimpleMovingAverageStrategy(BaseStrategy):
    """
    国债期货简单均线策略.

    基于参数maperiod实现简单均线策略,收盘价上穿均线买入,反之卖出.

    Attributes:
        maperiod (int): 均线参数.

    Methods:
        next: 策略核心，根据条件执行买卖交易指令
        stop: 回测结束后输出结果
    """
        
    params=(('maperiod',20),)

    def __init__(self):
        #指定价格序列
        self.dataclose=self.datas[0].close

        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buyprice = None
        self.buycomm = None

        #添加移动均线指标
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)

    #策略核心，根据条件执行买卖交易指令（必选）
    def next(self):
        # 检查是否有指令等待执行
        if self.order: 
            return
        # 检查是否持仓   
        if not self.position: # 没有持仓
            #执行买入条件判断：收盘价格上涨突破maperiod日均线
            if self.dataclose[0] > self.sma[0]:
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                #执行买入
                self.order = self.buy()         
        else:
            #执行卖出条件判断：收盘价格跌破maperiod日均线
            if self.dataclose[0] < self.sma[0]:
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                #执行卖出
                self.order = self.sell()
    

    #回测结束后输出结果（可省略，默认输出结果）
    def stop(self):
        self.log('(MA均线: %2d日) 期末总资金 %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)