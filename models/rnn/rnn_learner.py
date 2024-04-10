import torch
import torch.nn as nn

class RnnLearner():
    """
    RNN学习器.

    根据设定的模型学习参数训练模型并可以调用进行预测

    Attributes:
        model(nn.Module): 模型.
        save_model_path (str): 模型保存路径.
        args: 配置参数.

    Methods:
        next: 策略核心，根据条件执行买卖交易指令
    """

    def __init__(self, model, save_model_path, args):
        self.model = model
        self.save_model_path = save_model_path
        self.args = args


    # 方法定义
    def train(self,train_loader,valid_loader):
        """
        训练模型

        Args:
            model(nn.Module): 模型
            train_loader(DataLoader): 训练数据集
            valid_loader(DataLoader): 验证数据集
            self.args.loss_function(str): 模型训练损失函数
            self.args.num_epochs(int): 模型训练轮次
            self.args.learning_rate(float): 模型训练学习率

        Returns:
            model(nn.Module): 训练好的模型
        """
        args = self.args

        if args.loss_function == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)

        for epoch in range(args.num_epochs):
            self.model.train()
            correct = 0
            total = 0
            train_loss = 0
            for i, (batch_X, batch_y) in enumerate(train_loader):
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, batch_y) #分类模型训练：交叉熵损失函数
                train_loss += loss.item()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss = train_loss/total
            train_accuracy = 100 * correct / total

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                valid_loss = 0
                for batch_X, batch_y in valid_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                valid_loss = valid_loss/total
                valid_accuracy = 100 * correct / total

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{args.num_epochs}], Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}')
                print(f'Accuracy of the model on the train data: {train_accuracy}%, valid data: {valid_accuracy}%')

        print('Finished Training')
        print('-------------------------------')
        self.save_model()


    def test(self,test_loader):
        """
        模型预测

        Args:
            test_loader(DataLoader): 测试数据集
        """
        args = self.args
        self.model.eval()
        
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        print('Finished Testing')
        print(f'Accuracy of the model on the test data: {100 * correct / total}%')
        print('-------------------------------')

    def save_model(self):
        # 保存模型
        torch.save(self.model.state_dict(),  self.save_model_path)

        print('Saving Model in: ', self.save_model_path)
        print('-------------------------------')