import backtrader as bt

from BaseStrategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    """
    国债期货简单的双移动平均线和MACD指标的组合策略.

    1. 平多仓位:如果当前有多头仓位且收盘价低于短期指数移动平均线,则平掉多头仓位。
    2. 平空仓位:如果当前有空头仓位且收盘价高于短期指数移动平均线,则平掉空头仓位。
    3. 开多仓位:如果当前没有持仓且短期指数移动平均线上穿长期指数移动平均线,并且MACD指标为正值,则进行多头开仓。
    4. 开空仓位:如果当前没有持仓且短期指数移动平均线下穿长期指数移动平均线,并且MACD指标为负值,则进行空头开仓。
    5. 移仓换月:以持仓量最大的合约作为主力合约,如果当前持仓合约不为主力合约则将当前合约仓位平掉,并在新合约上开仓。

    Attributes:
        period_me1 (int): # 短期指数移动平均线
        period_me2 (int): # 长期指数移动平均线
        period_dif(int):  # 信号线

    Methods:
        next: 策略核心，根据条件执行买卖交易指令
    """
    
    params = (("period_me1", 10), # 短期指数移动平均线
              ("period_me2", 20), # 长期指数移动平均线
              ("period_dif", 9),) # 信号线

    # 初始化策略的数据
    def __init__(self):
        # 基本上常用的部分属性变量
        self.bar_num = 0  # next运行了多少个bar
        self.current_date = None  # 当前交易日
        # 计算macd指标
        self.ema_1 = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.p.period_me1)
        self.ema_2 = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=self.p.period_me2)
        self.dif = self.ema_1 - self.ema_2
        self.dea = bt.indicators.ExponentialMovingAverage(self.dif, period=self.p.period_dif)
        self.macd = (self.dif - self.dea) * 2
        # 保存现在持仓的合约是哪一个
        self.holding_contract_name = None

    def prenext(self):
        # 由于期货数据有几千个，每个期货交易日期不同，并不会自然进入next
        # 需要在每个prenext中调用next函数进行运行
        self.next()
        # pass

    # 在next中添加相应的策略逻辑
    def next(self):
        # 每次运行一次，bar_num自然加1,并更新交易日
        self.current_date = bt.num2date(self.datas[0].datetime[0])
        self.bar_num += 1
        # self.log(f"{self.bar_num},{self.datas[0]._name},{self.broker.getvalue()}")
        # self.log(f"{self.ema_1[0]},{self.ema_2[0]},{self.dif[0]},{self.dea[0]},{self.macd[0]}")
        data = self.datas[0]
        # 开仓，先平后开
        # 平多
        if self.holding_contract_name is not None and self.getpositionbyname(self.holding_contract_name).size > 0 and \
                data.close[0] < self.ema_1[0]:
            print('==========平多===========')
            data = self.getdatabyname(self.holding_contract_name)
            self.close(data)
            self.holding_contract_name = None
        # 平空
        if self.holding_contract_name is not None and self.getpositionbyname(self.holding_contract_name).size < 0 and \
                data.close[0] > self.ema_1[0]:
            print('==========平空===========')
            data = self.getdatabyname(self.holding_contract_name)
            self.close(data)
            self.holding_contract_name = None


        # 判断主力合约
        dominant_contract = self.get_dominant_contract()
        if dominant_contract == None:
            return
        # 开多
        if self.holding_contract_name is None and self.ema_1[-1] < self.ema_2[-1] and self.ema_1[0] > self.ema_2[0] and \
                self.macd[0] > 0:
            print('==========开多===========')
            next_data = self.getdatabyname(dominant_contract)
            self.buy(next_data, size=1) # 每手size为1
            self.holding_contract_name = dominant_contract

        # 开空
        if self.holding_contract_name is None and self.ema_1[-1] > self.ema_2[-1] and self.ema_1[0] < self.ema_2[0] and \
                self.macd[0] < 0:
            print('==========开空===========')
            next_data = self.getdatabyname(dominant_contract)
            self.sell(next_data, size=1)  # 每手size为1
            self.holding_contract_name = dominant_contract

        # 移仓换月
        if self.holding_contract_name is not None:
            # 如果出现了新的主力合约，那么就开始换月
            if dominant_contract != self.holding_contract_name:
                print('==========移仓换月===========')
                # 下个主力合约
                next_data = self.getdatabyname(dominant_contract)
                # 当前合约持仓大小及数据
                size = self.getpositionbyname(self.holding_contract_name).size  # 持仓大小
                data = self.getdatabyname(self.holding_contract_name)
                # 平掉旧的
                self.close(data)
                # 开新的
                if size > 0:
                    self.buy(next_data, size=abs(size))
                if size < 0:
                    self.sell(next_data, size=abs(size))
                self.holding_contract_name = dominant_contract

    def get_dominant_contract(self):

        # 以持仓量最大的合约作为主力合约,返回数据的名称
        # 可以根据需要，自己定义主力合约怎么计算

        # 获取当前在交易的品种
        target_datas = []
        for data in self.datas[1:]:
            # self.log(self.current_date)
            # self.log(bt.num2date(data.datetime[0]))
            try:
                data_date = bt.num2date(data.datetime[0])
                # self.log(f"{data._name},{data_date}")
                if self.current_date == data_date:
                    target_datas.append([data._name, data.openinterest[0]])
            except:
                self.log(f"{data._name}还未上市交易")

        target_datas = sorted(target_datas, key=lambda x: x[1])
        # print(target_datas)
        if len(target_datas) == 0:
            return None
        else:
            return target_datas[-1][0]
    
