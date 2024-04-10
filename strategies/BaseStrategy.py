import backtrader as bt


class BaseStrategy(bt.Strategy):
    """
    策略模版.

    记录交易细节，后续策略均基于此类完成具体策略实现.

    Attributes:
        printlog (bool): 是否打印详细交易细节.

    Methods:
        log: 交易记录日志
        notify_order: 记录交易执行情况
        notify_trade: 记录交易收益情况
    """

    params=(('printlog',False),)

    def __init__(self):
        pass
    
    #策略核心，根据条件执行买卖交易指令（必选）
    def next(self):
        pass
    
    # log相应的信息
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        if self.p.printlog:
            dt = dt or bt.num2date(self.datas[0].datetime[0])
            print('{}, {}'.format(dt.isoformat(), txt))

    def notify_order(self, order):
        # 订单被提交或接受
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 订单被拒绝时
        if order.status == order.Rejected:
            self.log(f"Rejected : order_ref:{order.ref}  data_name:{order.p.data._name}")
        # 订单保证金不足
        if order.status == order.Margin:
            self.log(f"Margin : order_ref:{order.ref}  data_name:{order.p.data._name}")
        # 订单被取消
        if order.status == order.Cancelled:
            self.log(f"Concelled : order_ref:{order.ref}  data_name:{order.p.data._name}")
        # 订单部分成交
        if order.status == order.Partial:
            self.log(f"Partial : order_ref:{order.ref}  data_name:{order.p.data._name}")
        # 订单完全成交
        if order.status == order.Completed:
            # 多头订单
            if order.isbuy():
                self.log(
                    f" BUY : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")
            # 空头订单
            else:  # Sell
                self.log(
                    f" SELL : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")

    def notify_trade(self, trade):
        # 一个trade结束的时候输出信息
        if trade.isclosed:
            self.log('closed symbol is : {} , total_profit : {} , net_profit : {}'.format(
                trade.getdataname(), trade.pnl, trade.pnlcomm))
            # self.trade_list.append([self.datas[0].datetime.date(0),trade.getdataname(),trade.pnl,trade.pnlcomm])

        if trade.isopen:
            self.log('open symbol is : {} , price : {} '.format(
                trade.getdataname(), trade.price))

    def stop(self):
        # 策略停止的时候输出信息
        pass