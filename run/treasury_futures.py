import os
import numpy as np
from data_module.data_loader import create_datasets_classify

from models.rnn.rnn_learner import RnnLearner
from models.rnn.rnn import SimpleRNN

from torch.utils.data import TensorDataset, DataLoader
from models.PatchTST.PatchTST_self_supervised import patchtst_pretrain,patchtst_finetune

def choose_model(args):
    """
    选择模型

    后续随着模型的增多，该部分传入的参数也将根据不同模型的参数而增加

    Args:
        model_name(str): 模型名称.
        input_size(int): 模型输入特征维度
        hidden_size(int): 模型中间层维度
        dropout_rate(float): dropout比率
        num_layers(int): rnn层数
        num_classes(int): 分类类别数

    Returns:
        model(nn.Module): 模型
    """
    if args.model_name == 'rnn':
            model = SimpleRNN(args.input_size, args.hidden_size, 
                              args.num_layers, args.num_classes, args.dropout_rate)
    return model

def run_treasury_futures_rnn(args,code,data_sets):
    """
    运行国债期货rnn模型

    Args:
        args(argparse): 配置参数.
        code(str): 国债期货编码
        data_sets(dict): 数据

    Returns:
        scaler: 训练集标准化器
    """
    # step2. 构建模型训练数据集
    if args.model_type == 'classify':
        # （分类模型）
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = create_datasets_classify(args,data_sets[code]) 

    train_data = TensorDataset(X_train, y_train) # sample = train_data[0] 输出: (tensor([1, 2, 3]), tensor(0))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    valid_data = TensorDataset(X_valid, y_valid) 
    valid_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)


    print('future code:', code,
        '\nmodel name:', args.model_name,
        '\nbegin train date:',args.begin_train_date, 
        '\nend train date:', args.end_train_date,
        '\ntrain length:',len(train_data),
        '\nbegin valid date:',args.begin_valid_date, 
        '\nend valid date:', args.end_valid_date,
        '\nvalid length:',len(valid_data))
    print('-------------------------------')
    print(args)
    print('-------------------------------')
    
    if args.do_train:
        # step3. 模型训练配置
        model = choose_model(args)

        # 构建学习器
        model_save_path = os.path.join('.','save_models',args.model_name,args.model_name+'_'+code+'.pth')
        learner = RnnLearner(model, model_save_path, args)

        # step4. 模型训练并保存
        learner.train(train_loader,valid_loader)


    if args.do_test:
        # setp5. 模型预测
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        print('future code:', code,
            '\nmodel name:', args.model_name,
            '\nbegin test date:',args.begin_test_date, 
            '\nend test date:', args.end_test_date,
            '\ntest length:',len(test_data))
        print('-------------------------------')
        
        learner.test(test_loader)
    
    return scaler


def run_treasury_futures_PatchTST(args,code,columns):
    """
    运行国债期货PatchTST模型

    Args:
        args(argparse): 配置参数.
        code(str): 国债期货编

    Returns:
        out(list): a list of [pred, targ, score]
    """
    out = []
    # 保持模型和策略参数的一致
    args.window_size = args.context_points
    args.look_ahead = args.target_points
    
    args.dset = args.dset_pretrain
    # 预训练模型
    args.save_pretrained_model = 'patchtst_pretrained_cw'+str(args.context_points)+ \
                                '_patch'+str(args.patch_len) + \
                                '_stride'+str(args.stride) + \
                                '_epochs-pretrain' + str(args.n_epochs_pretrain) + \
                                '_mask' + str(args.mask_ratio)  + \
                                '_model' + str(args.pretrained_model_id)
    save_path = 'save_models/PatchTST/' + args.dset_pretrain + '/' + code + '/'+ args.begin_train_date + '/pretrain/'
    args.save_path = os.path.join(save_path, args.save_pretrained_model)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)


    # get available GPU devide
    patchtst_pretrain.set_my_device()

    print('PatchTST pretrain args: ',args)
    print('-------------------------------')

    if args.do_pretrain:
        print('=========================================================================')
        print('===========================begin preratin================================')
        suggested_lr = patchtst_pretrain.find_lr(args,columns)
        patchtst_pretrain.pretrain_func(suggested_lr,args,columns)
        print('===========================finish preratin===============================')
        print('=========================================================================')

    # 微调模型

    suffix_name = '_cw'+str(args.context_points)+\
                    '_tw'+str(args.target_points) + \
                    '_patch'+str(args.patch_len) + \
                    '_stride'+str(args.stride) + \
                    '_epochs-finetune' + str(args.n_epochs_finetune) + \
                    '_model' + str(args.finetuned_model_id)
    args.save_finetuned_model = args.dset_finetune+'_patchtst_finetuned'+suffix_name
    save_path = 'save_models/PatchTST/' + args.dset_pretrain + '/' + code + '/' + args.begin_train_date + '/finetune/'
    args.save_path = os.path.join(save_path, args.save_finetuned_model)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)

    args.pretrained_model = os.path.join(save_path, '..', 'pretrain', args.save_pretrained_model, 'model.pth')

    print('PatchTST finetune args: ',args)
    print('-------------------------------')


    # Finetune
    if args.do_finetune:
        print('=========================================================================')
        print('===========================begin finetune================================')
        head_type = 'prediction' # 2023/7/21 在这里做过实验，原模型regression 有很大问题
        suggested_lr = patchtst_finetune.find_lr(args, columns, head_type=head_type)
        patchtst_finetune.finetune_func(suggested_lr,args,columns,head_type=head_type)
        print('===========================finish finetune===============================')
        print('=========================================================================')
   
    # Test
    if args.do_test:
        print('=========================================================================')
        print('===========================begin testing=================================')
        # out: a list of [pred, targ, score]
        out,scaler = patchtst_finetune.test_func(args.save_path,args,columns)
        np.save(os.path.join(args.save_path,'out.npy'),out)           
        print('===========================finish testing================================')
        print('=========================================================================')
    
    return out,scaler

