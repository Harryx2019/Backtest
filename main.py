import os
import sys
current_path = os.getcwd()
PatchTSTS_path = os.path.join(current_path,'models','PatchTST','PatchTST_self_supervised')
# PatchTST_self_supervised
sys.path.insert(0, PatchTSTS_path)

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from utils.date import *

from backtest import backtest_treasury_futures,backtest_treasury_futures_old

from data_module.data_loader import get_future_data_csv, get_future_data_akshare
from data_module.data_pre_process import pre_process_treasury_futures_data

from run.treasury_futures import run_treasury_futures_rnn,run_treasury_futures_PatchTST


import warnings
warnings.filterwarnings("ignore")



def config():
    """
    系统配置

    Returns:
        args(argparse): 系统配置
    """
    parser = argparse.ArgumentParser()
    # ---------投资品种
    parser.add_argument('--invest', type=str, default="treasury_futures")   # 资产名称（后续增加各种类别资产）

    # ---------数据来源
    parser.add_argument('--data_source', type=str, default="custom")        # 资产名称（后续增加各种类别资产）

    # ---------深度学习模型共享参数（后续随着模型的增多需要不断调整参数）
    parser.add_argument('--model_name', type=str, default="rnn")            # 模型名称（后续根据模型名更换不同模型）
    parser.add_argument('--model_type', type=str, default="classify")       # 模型类别（分类/回归）
    parser.add_argument('--window_size', type=int, default=16)              # 模型滑动窗口
    parser.add_argument('--look_ahead', type=int, default=1)                # 预测未来第几天
    parser.add_argument('--input_size', type=int, default=1)                # 输入数据维度(在框架中自动适配)
    parser.add_argument('--hidden_size', type=int, default=32)              # 隐藏层维度
    parser.add_argument('--batch_size', type=int, default=32)               # 模型训练批量大小
    parser.add_argument('--num_epochs', type=int, default=20)               # 模型训练轮次
    parser.add_argument('--dropout_rate', type=float, default=0.2)          # dropout率
    parser.add_argument('--learning_rate', type=float, default=1e-3)        # 模型训练学习率
    parser.add_argument('--loss_function', type=str, default='mse')         # 模型损失函数

    # ---------RNN模型参数
    parser.add_argument('--num_layers', type=int, default=1)                # 模型层数

    # ---------分类模型参数
    parser.add_argument('--num_classes', type=int, default=2)               # 分类模型类别数

    # ---------PatchTST模型参数
    # Dataset and dataloader
    parser.add_argument('--dset_pretrain', type=str, default='treasury_futures', help='dataset name')
    parser.add_argument('--dset_finetune', type=str, default='treasury_futures', help='dataset name')
    parser.add_argument('--context_points', type=int, default=512, help='sequence length')
    parser.add_argument('--target_points', type=int, default=1, help='forecast horizon')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
    parser.add_argument('--features', type=str, default='MS', help='for multivariate model or univariate model')
    # Patch
    parser.add_argument('--patch_len', type=int, default=12, help='patch length')
    parser.add_argument('--stride', type=int, default=12, help='stride between patch')
    # RevIN
    parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
    # Model args
    parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
    parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
    # Pretrain mask
    parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
    # Pretrained model name
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
    # Optimization args
    parser.add_argument('--n_epochs_finetune', type=int, default=20, help='number of finetuning epochs')
    parser.add_argument('--n_epochs_pretrain', type=int, default=100, help='number of pre-training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    # model id to keep track of the number of models saved
    parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
    parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
    # parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
    parser.add_argument('--do_pretrain', type=bool, default=True)              # 是否进行PatchTST预训练
    parser.add_argument('--do_finetune', type=bool, default=True)              # 是否进行PatchTST微调

    # ---------回测相关参数(后续需要增加仓位管理回测)
    parser.add_argument('--strategy_name', type=str, default='rnn')         # 回测策略
    parser.add_argument('--startcash', type=int, default=10000000)            # 回测初始资产
    parser.add_argument('--stake', type=int, default=100)                   # 每笔交易数量
    parser.add_argument('--commission', type=float, default=0.0002)         # 交易手续费率
    parser.add_argument('--slippage_perc', type=float, default=0.001)       # 滑点
    parser.add_argument('--margin', type=float, default=0.02)                # 保证金
    parser.add_argument('--multiplier', type=float, default=10000)             # 合约乘数
    parser.add_argument('--printlog', type=bool, default=False)             # 是否打印回测记录

    # ---------MACD策略相关参数
    parser.add_argument('--period_me1', type=int, default=10)                # 短期指数移动平均线
    parser.add_argument('--period_me2', type=int, default=20)                # 长期指数移动平均线
    parser.add_argument('--period_dif', type=int, default=9)                 # 信号线

    # ---------运行设置
    parser.add_argument('--do_train', type=bool, default=True)              # 是否进行模型训练
    parser.add_argument('--do_test', type=bool, default=True)               # 是否进行模型预测
    parser.add_argument('--do_backtest', type=bool, default=True)           # 是否进行策略回测

    args = parser.parse_args([])

    return args


def run_treasury_futures(args, code_list, margin_list, multiplier_list, 
                         columns, begin_date_list, end_date):
    """
    国债期货建模与回测

    对每一个投资标的进行建模，并进行回测，流程为：
        step1. 读取每个投资标的数据
        step2. 构建模型训练数据集
        step3. 模型训练配置
        step4. 模型训练
        setp5. 模型预测
        step6. 执行回测

    Args:
        ---------数据相关参数
        code_list (list): 投资标的列表.
        columns (list): 数据列名.
        begin_date_list (list): 投资标的上市日期.
        begin_backtest_date_list (list): 投资标的执行回测开始日期.

        ---------模型相关参数
        args.model_name: 模型名称
        args.window_size(int): 模型滑动窗口
        args.look_ahead(int): 预测未来第几天
        args.batch_size(int): 模型训练批量大小
        args.hidden_size(int): 模型隐藏层维度
        args.num_layers(int): rnn层数
        args.num_classes(int): 分类类别数
        args.dropout_rate(float): dropout比率
        args.num_epochs(int): 模型训练轮次
        args.learning_rate(float): 模型训练学习率
        args.do_train(bool): 是否进行模型训练, False: 直接进行回测 True: 训练模型并回测
        args.do_test(bool): 是否进行模型训练, False: 直接进行回测 True: 训练模型并回测

        ----------回测相关参数
        args.startcash(long): 回测初始资产
        args.stake(int): 每笔交易数量
        args.commission(float): 交易手续费率
        args.do_backtest(bool): 是否进行回测, False: 不回测 True: 回测

    Returns:
        results(list): 回测结果列表
    """
    
    deep_learning_models = ['rnn','PatchTST']

    data_sets = {}
    results = []
    scalers = []
    # 对每一个国债期货品种分别进行分析
    for i in range(len(code_list)):
        results_i = []
        scalers_i = []

        code = code_list[i]
        args.code = code
        # 不同期限国债期货保证金和合约系数
        args.margin = margin_list[i]
        args.multiplier = multiplier_list[i]

        # 将输出重定向到文件
        output_path = os.path.join('.','logs',args.model_name,args.invest)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # sys.stdout = open(os.path.join(output_path,'output_'+code+'.txt'), 'w')

        # step1. 获取数据
        print('-------------------------------')
        if args.data_source == 'custom':
            # 自定义数据
            # CGS国债期货主连拼接日频数据
            # TODO：后续把国债期货主连程序放进来
            path = os.path.join('.','datasets','treasury_futures')
            data_sets[code] = get_future_data_csv(os.path.join(path,code+'_dataset_day.csv'),columns)
        elif args.data_source == 'akshare':
            # akshare获取期货数据
            path = './datasets/akshare' # 中金所所有数据
            if not os.path.exists(path):
                os.makedirs(path)
                current_date = datetime.now()
                # 将日期格式转化为akshare格式
                begin_date = datetime.strptime(begin_date_list[i], '%Y-%m-%d')
                begin_date = begin_date.strftime('%Y%m%d')
                end_date = current_date.strftime('%Y%m%d')
                get_future_data_akshare(code,begin_date,end_date,'CFFEX',path)
            # 国债期货持仓量加权平均合成指数
            code_ = code.split('.')
            index_path = './datasets/akshare'
            if not os.path.exists(index_path):
                os.makedirs(index_path)
                df = pd.read_csv(path)
                pre_process_treasury_futures_data(code, df, index_path)
            path = os.path.join(index_path,code_[0]+'.csv') # 国债期货指数合约路径


        # 模型训练与预测
        flag_break = False # 是否继续滚动训练

        args.begin_train_date = begin_date_list[i]  # 训练集开始时间（以投资标的上市时间开始训练简易模型容易过拟合）


        while True:
            """
                时序交叉验证

                验证集和测试集均为1年 训练集为4年
            """

            scaler = None

            args.end_train_date = get_end_train_date(args.begin_train_date)     # 训练集结束时间

            args.begin_valid_date = get_begin_valid_date(args.end_train_date)     # 验证集开始时间
            args.end_valid_date = get_end_valid_date(args.begin_valid_date)        # 验证集结束时间

            args.begin_test_date = get_begin_test_date(args.end_valid_date)     # 测试集/回测开始时间
            args.end_test_date = get_end_test_date(args.begin_test_date)        # 测试集/回测结束时间

            end_test_date = pd.to_datetime(args.end_test_date)
            end_date = pd.to_datetime(end_date)
            if end_test_date >= end_date:
                end_test_date = end_date
                args.end_test_date = end_test_date.strftime('%Y-%m-%d')
                flag_break = True

            if args.model_name == 'rnn':
                scaler = run_treasury_futures_rnn(args,code,data_sets)
            elif args.model_name == 'PatchTST':
                out,scaler = run_treasury_futures_PatchTST(args,code,columns)
            
            # step6. 执行回测
            result = None
            if args.do_backtest:
                print('=========================================================================')
                print('===========================begin backtesting=============================')

                result,scaler = backtest_treasury_futures(args, code, path, columns, 
                                                        args.begin_test_date, args.end_test_date, scaler)

                print('===========================finish backtesting============================')
                print('=========================================================================')
            results_i.append(result)
            scalers_i.append(scaler)

            if flag_break:
                break
            
            # 更新数据集时间
            args.begin_train_date = get_next_begin_train_date(args.begin_train_date) # 训练集开始时间

        results.append(results_i)
        scalers.append(scalers_i)

        # 恢复输出到标准输出
        # sys.stdout = sys.__stdout__
        break


    return results,scalers




if __name__ == '__main__':
    
    args = config()

    args.invest = 'treasury_futures'            # 设置资产
    args.data_source = 'custom'                 # 数据来源
    args.strategy_name = 'PatchTST'             # 设置策略
    args.model_name = args.strategy_name        # 设置模型(与策略同名)
    args.model_type = 'regression'              # 设置模型类别

    args.do_train = True                        # 是否训练
    args.do_test = True                         # 是否测试
    args.do_backtest = False                     # 是否回测
    args.printlog = True                        # 是否打印日志

    args.do_pretrain = False                    # 是否进行PatchTST预训练
    args.do_finetune = False                    # 是否进行PatchTST微调


    if args.model_type == 'classify':
        args.loss_function = 'cross_entropy'    # 设置损失函数
    elif args.model_type == 'regression':
        args.loss_function = 'mse'
    
    if args.invest == 'treasury_futures':
        # ---------国债期货数据相关配置
        code_list=['T.CFE','TF.CFE','TS.CFE']                                       # 设置投资标的
        margin_list = [0.02,0.01,0.005]                                             # 保证金
        multiplier_list = [10000, 10000, 20000]                                     # 合约系数
        begin_date_list=["2015-03-20","2013-09-06","2018-08-17"]                    # 投资标的上市日期.
        end_date = "2023-08-18"                                                     # 投资标的数据截止日期.
        
        if args.data_source == 'akshare':
            columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest']
        else:
            columns=['date','open','high','low','close','vwap','oi','volume']   # 数据列名（后续增加筛选后因子指标）
        args.input_size = len(columns)-1 + 158                                  # 输入数据维度(在框架中自动适配) -1: 除去日期列 +158: 加入Alpha158因子

        results,scalers = run_treasury_futures(args, code_list, margin_list, multiplier_list,
                                        columns, begin_date_list, end_date)
    