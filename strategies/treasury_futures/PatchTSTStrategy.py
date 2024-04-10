import os
import numpy as np
import pandas as pd
import torch
import backtrader as bt

from BaseStrategy import BaseStrategy

from models.PatchTST.PatchTST_self_supervised import patchtst_finetune


class PatchTSTStrategy(BaseStrategy):
    """
    国债期货PatchTST策略.

    1. 平多仓位:如果当前有多头仓位且模型预测上涨,则平掉多头仓位。
    2. 平空仓位:如果当前有空头仓位且模型预测下跌,则平掉空头仓位。
    3. 开多仓位:如果当前没有持仓且模型预测上涨,则进行多头开仓。
    4. 开空仓位:如果当前没有持仓且模型预测下跌,则进行空头开仓。
    5. 移仓换月:以持仓量最大的合约作为主力合约,如果当前持仓合约不为主力合约则将当前合约仓位平掉,并在新合约上开仓。

    Attributes:

    Methods:
        next: 策略核心，根据条件执行买卖交易指令
    """
    
    params = dict(code = 'T.CFE',
                    columns = [],
                    args = None,
                    scaler = None)

    # 初始化策略的数据
    def __init__(self):
        # 基本上常用的部分属性变量
        self.args = self.p.args
        self.columns = self.p.columns                                   # 所有数据列名
        self.input_size = self.args.input_size                          # 输入数据维度
        self.target = self.args.target                                  # 预测目标列名
        self.context_points = self.args.context_points                  # 模型lookback window
        self.scaler = self.p.scaler                                     # 训练集上的数据标准化器
        self.weight_path = os.path.join(self.args.save_path,'model.pth') # 训练好的模型

        self.bar_num = 0  # next运行了多少个bar
        self.current_date = None  # 当前交易日
        # 对齐数据集和backtrade Line
        self.build_map()
        self.learner = patchtst_finetune.pred_func(c_in=self.c_in,args=self.args)
        
        # 保存现在持仓的合约是哪一个
        self.holding_contract_name = None

        # context_points长度的数据
        self.previous_data = [] 

    def build_map(self):
        '''
            这里做一个自定义数据列columns和backtrader数据列的映射
            # TODO:此处后续需要优化时间
        '''
        self.data_columns = self.datas[0].lines.getlinealiases()  # 获取所有数据列名
        self.c_in = len(self.data_columns) # 获取输入数据维度（框架自适应）
        # print("-------strategy--------")
        # print('self.data_columns: ',self.data_columns)
        # print('self.columns: ',self.columns)
        self.data_columns_index = []
        for col in self.columns:
            for j in range(len(self.data_columns)):
                if(self.data_columns[j] == col):
                    self.data_columns_index.append(j)
                if(col == 'oi' and self.data_columns[j] == 'openinterest'):
                    self.data_columns_index.append(j)
        # print('self.data_columns_index: ',self.data_columns_index)

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

        previous_data_i = [self.datas[0].lines[j][0] for j in self.data_columns_index]
        self.previous_data.append(previous_data_i)
        
        if self.bar_num < self.context_points:
            return

        # 构建模型预测所需要的数据 [context_points, input_size]
        groundtruth = self.datas[0].close[0] # 今日收盘价
        
        # 将数据列排序由 self.columns -> [cols + [self.target]]
        previous_data_df = pd.DataFrame(self.previous_data,columns=self.columns)
        cols = list(previous_data_df.columns)
        cols.remove(self.target) 
        previous_data_df = previous_data_df[cols + [self.target]]
        previous_data_df = self.scaler.transform(previous_data_df.values)  # [context_points,input_size]


        # 模型预测
        X = torch.tensor(previous_data_df).view(1, self.context_points, -1).float()
        prediction = self.learner.predict(pred_data=X, weight_path=self.weight_path) # [1,1,input_size]
        prediction = prediction.reshape(1,self.input_size)                           # [1,input_size]
        prediction = self.scaler.inverse_transform(prediction)                       # [1,input_size]
        prediction = prediction[-1][-1]                                              # [close]
        # 将模型预测下一日收盘价与今日收盘价做差
        prediction = prediction / groundtruth - 1
        # self.log(f"prediction :{prediction}")

        # 把数据previous_data第一行删除
        self.previous_data = self.previous_data[1:]

        # 开仓，先平后开
        # 平多
        if self.holding_contract_name is not None and self.getpositionbyname(self.holding_contract_name).size > 0 and \
                prediction < 0:
            print('==========平多===========')
            data = self.getdatabyname(self.holding_contract_name)
            self.close(data)
            self.holding_contract_name = None
        # 平空
        if self.holding_contract_name is not None and self.getpositionbyname(self.holding_contract_name).size < 0 and \
                prediction > 0:
            print('==========平空===========')
            data = self.getdatabyname(self.holding_contract_name)
            self.close(data)
            self.holding_contract_name = None


        # 判断主力合约
        dominant_contract = self.get_dominant_contract()
        if dominant_contract == None:
            return
        # 开多
        if self.holding_contract_name is None and prediction > 0:
            print('==========开多===========')
            next_data = self.getdatabyname(dominant_contract)
            self.buy(next_data, size=100) # 每手size为1
            self.holding_contract_name = dominant_contract

        # 开空
        if self.holding_contract_name is None and prediction < 0:
            print('==========开空===========')
            next_data = self.getdatabyname(dominant_contract)
            self.sell(next_data, size=100)  # 每手size为1
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
            try:
                data_date = bt.num2date(data.datetime[0])
                # self.log(f"{data._name},{data_date}")
                if self.current_date == data_date:
                    target_datas.append([data._name, data.openinterest[0]])
            except:
                self.log(f"{data._name}还未上市交易")
        # self.log('-----------------')

        target_datas = sorted(target_datas, key=lambda x: x[1])
        
        if len(target_datas) == 0:
            return None
        else:
            return target_datas[-1][0]
    