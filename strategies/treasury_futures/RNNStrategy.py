import os
import pandas as pd
import torch
import torch.nn as nn
import backtrader as bt

from BaseStrategy import BaseStrategy
from models.rnn.rnn import SimpleRNN

class RNNStrategy(BaseStrategy):
    """
    国债期货简单RNN策略.

    在每天收盘时调用训练好的模型，预测后面的行情是否会上涨。
    如果预测上涨，那么就在明天以开盘价买入，如果下跌，就在明天以开盘价卖出。

    Attributes:
        code (str): 投资标的.
        columns (list): 数据列名.
        args: 配置参数.
        scaler: 数据标准化器 深度学习模型必选

    Methods:
        next: 策略核心，根据条件执行买卖交易指令
    """

    params = dict(code = 'T.CFE',
                columns = [],
                args = None,
                scaler = None)

    def __init__(self):
        self.data_columns = self.datas[0].lines.getlinealiases()  # 获取所有自定义数据列的别名
        # print(self.data_columns)
        self.columns = self.params.columns
        # print(self.columns)
        # 对齐backtrader列与自定义数据列
        self.data_columns_index = []
        for col in self.columns:
            for j in range(len(self.data_columns)):
                if(self.data_columns[j] == col):
                    self.data_columns_index.append(j)
                if(col == 'oi' and self.data_columns[j] == 'openinterest'):
                    self.data_columns_index.append(j)
        # print(self.data_columns_index)

        self.data = self.datas[0]
         # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.window_size = self.params.args.window_size
        self.input_size = self.params.args.input_size
        # 加载选择的模型
        if(self.params.args.model_name == 'rnn'):
            self.model = SimpleRNN(self.params.args.input_size, 
                                self.params.args.hidden_size, 
                                self.params.args.num_layers, 
                                self.params.args.num_classes)
            model_path = os.path.join('.','save_models','rnn', 'rnn_'+self.params.code+'.pth')
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.scaler = self.params.scaler

        self.counter = 1

    def next(self):
        if self.counter < self.window_size:
            self.counter += 1
            return
        # 检查是否有指令等待执行
        if self.order:
            return
        # 构建模型预测所需要的数据 （window_size, input_size)
        previous_data = [] 
        for i in range(0,self.window_size):
            previous_data_i = [self.datas[0].lines[j][-i] for j in self.data_columns_index]
            previous_data.append(previous_data_i)
        
        previous_data_df = pd.DataFrame(previous_data,columns=self.columns)
        previous_data_df = self.scaler.transform(previous_data_df)

        X = torch.tensor(previous_data_df).view(1, self.window_size, -1).float()
        prediction = self.model(X)

        max_vals, max_idxs = torch.max(prediction, dim=1)
        predicted_prob, predicted_trend = max_vals.item(), max_idxs.item()

        # 检查是否持仓   
        if not self.position: # 没有持仓
            if predicted_trend == 1:  # 预测上涨趋势买入
                self.order = self.buy() # 买入       
        else:
            if predicted_trend == 0: # 预测下跌趋势卖出
                self.order = self.sell() # 卖出