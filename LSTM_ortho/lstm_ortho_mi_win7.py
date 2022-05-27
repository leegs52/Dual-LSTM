import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import MinMaxScaler
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
#from RNN_based import Co_LSTM


def orthogonal(X_current, W):
    #calculate othogonal term using trace of multiplied matrixs
    batch_size=X_current.size()[0]
    weight_cur = torch.matmul(X_current, W)
    matrix = torch.matmul(X_current, torch.transpose(weight_cur,1,2))
    trace = 0
    for i in range(batch_size):
        trace+=torch.trace(matrix[i])
    return trace


def correlated_past(X_current, X_past, W):
    #calculate correlated(positive correlated with MI) term using trace of multiplied matrixs
    batch_size=X_current.size()[0]
    weight_past = torch.matmul(X_past, W)
    matrix = torch.matmul(torch.transpose(X_current,1,2),weight_past)
    trace = 0
    for i in range(batch_size):
        trace+=torch.trace(matrix[i])
    return trace


def orthonormal(weight):
    n_feature = weight.size()[0]
    othonorm = torch.norm(torch.matmul(weight,weight.T).float().to(device)-torch.eye(n_feature).float().to(device), p='fro')
    
    return othonorm



class Dataset(Dataset):
    def __init__(self, data_dir, mode, mean=None, std=None, seq_len=480, target_delay=24, stride=5, normalize=True, test_col=0):
        self.mode = mode
        self.seq_len = seq_len
        self.target_delay = target_delay
        self.stride = stride
        self.data = pd.read_csv('{}/{}.txt'.format(data_dir, mode)).values[:, 1:6].astype(np.float32)
        self.data = torch.from_numpy(self.data)
        self.test_col=test_col
        
        if mode == 'train':
            assert (mean is None) and (std is None), \
                 "평균과 분산 계산"
            self.mean = torch.mean(self.data, dim=0)
            self.std = torch.std(self.data, dim=0)
            ## 코드 종료 ##
        else:
            assert (mean is not None) and (std is not None), \
                "평균과 분산은 `train_data`변수에 내장한 self.mean 과 self.std 를 사용합니다."
            ## 코드 시작 ##
            self.mean = mean
            self.std = std
            ## 코드 종료 ##
        if normalize:
            self.data = (self.data - self.mean) / self.std
            print(self.data.shape)
        
    def __getitem__(self, index):
        ## 코드 시작 ##
        index = self.stride * index   
        sequence = self.data[index:index + self.seq_len, :]
        target = self.data[index + self.seq_len + self.target_delay-1, self.test_col] #첫번째 컬럼을 예측 대상으로 삼음
   
        ## 코드 종료 ##
        return sequence, target
    
    def __len__(self):
        max_idx = len(self.data) - self.seq_len - self.target_delay
        num_of_idx = max_idx // self.stride
        return num_of_idx

    

class Co_LSTM(nn.Module):
    '''LSTM for univaritate time series data, many-to-one version
    input_dim : dimension of input data
    hidden_layer : dimension of hidden layer
    output_dim : dimension of output/prediction values
    '''
    
    def __init__(self, input_dim=100, hidden_dim=3, sequence_length=7, output_dim=1, batch_size=2):
        seed_val = 0
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        self.device = torch.device('cuda:0')
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.seq_len = sequence_length
        self.batch_size = batch_size
        
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        # (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
        self.hidden_cell1 = (torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device),  
                            torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device))  
        self.hidden_cell2 = (torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device), 
                            torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device))  
        self.W = nn.parameter.Parameter(torch.randn(input_dim, input_dim))


    def forward(self, input_seq):
        
        batch_size = input_seq.size(0)
        self.init_hidden()
        lstm_out1, self.hidden_cell1 = self.lstm1(input_seq, self.hidden_cell1)
        
        weighted_input = torch.matmul(input_seq, self.W)
        lstm_out2, self.hidden_cell2 = self.lstm2(weighted_input, self.hidden_cell2)
        lstm_out1 = lstm_out1[:,-1,:]
        lstm_out2 = lstm_out2[:,-1,:]

        predictions = torch.cat((lstm_out1, lstm_out2), dim=1)
        predictions = self.fc(predictions)

        return predictions

    
    def init_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        self.hidden_cell1 = (torch.rand(1, self.batch_size, self.hidden_dim).to(self.device),  # .to(self.device),when you use gpu
                            torch.rand(1, self.batch_size, self.hidden_dim).to(self.device))  # .to(self.device)) when you use gpu
        self.hidden_cell2 = (torch.rand(1, self.batch_size, self.hidden_dim).to(self.device),  # .to(self.device),when you use gpu
                            torch.rand(1, self.batch_size, self.hidden_dim).to(self.device))  # .to(self.device)) when you use gpu
        return
    
    
    
if __name__=='__main__':
    
    seed_val = 0
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda:0')

    data_dir = '../dataset/occupancy2'
    seq_len=7
    target_delay=1
    stride=1
    normalize=True
    test_col=1
    
    #load dataset
    train_data = Dataset(data_dir, 
                         'train', 
                         mean=None, 
                         std=None, 
                         seq_len=seq_len, 
                         target_delay=target_delay, 
                         stride=stride, 
                         normalize=normalize, 
                         test_col=test_col)
    val_data = Dataset(data_dir, 'val', 
                       mean=train_data.mean, 
                       std=train_data.std, 
                       seq_len=seq_len, 
                       target_delay=target_delay, 
                       stride=stride, 
                       normalize=normalize, 
                       test_col=test_col)

    
    #generate dataset
    batch_size=3
    train_loader = DataLoader(train_data, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              drop_last=True)
    val_loader = DataLoader(val_data, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            drop_last=True)

    
    #train parameters
    input_dim = 5
    #hidden_dim = 3
    seq_len = 7 # we suppose last 7 days data affect to CO2 level of next day
    output_dim = 1
    batch_size=3
    epochs = 50
    
    hidden_dims=[3,5,7]
    lrs=[0.0001, 0.0005, 0.001, 0.005, 0.01]
    lam1s=[0.1, 0.01, 0.001, 0.0001]
    lam2s=[0.1,0.3,0.5,0.7,0.9]
    lam3s=[0.1,0.3,0.5,0.7,0.9]
    
    
    with open('./result_ortho_mi_win7.csv','w',encoding='utf_8_sig') as f:
        f.write('window,hidden,lr,lambda1,lambda2,lambda3,val_MSE,train_MSE,epoch\n')
        
    for hidden_dim in hidden_dims:
        for lr in lrs:
            for lam1 in lam1s:
                for lam2 in lam2s:
                    for lam3 in lam3s:
                        model = Co_LSTM(input_dim=input_dim, 
                                        hidden_dim=hidden_dim, 
                                        sequence_length=seq_len, 
                                        output_dim=output_dim, 
                                        batch_size=batch_size)
                        model.cuda()
                        loss_function = nn.MSELoss(reduction='sum')
                        #optimizer = optim.RMSprop(model.parameters(), lr)
                        optimizer = optim.Adam(model.parameters(), lr)
                        print(model)

                        dtype = torch.FloatTensor
                        colnames = {0:"Temperature", 1:"Humidity", 2:"Light", 3:"CO2", 4:"HumidityRatio"}

                        train_preds = torch.FloatTensor().to(device).detach()
                        train_acts = torch.FloatTensor().to(device).detach()

                        val_loss = []
                        result = []

                        # Train model using Multivariate LSTM, many to one version
                        for i in range(epochs):
                            j = 0
                            #print(model.W)
                            print(orthonormal(model.W))
                            total_loss = 0
                            total_act_loss = 0
                            for t,(seq, label) in enumerate(tqdm(train_loader)):
                                # Input: (batch, seq_len, input_size) when batch_first=True
                                if t==0:
                                    seq_past = seq.clone().detach()
                                    continue
                                seq_past = Variable(seq_past, requires_grad=True).to(device)
                                seq = Variable(seq, requires_grad=True).to(device) #add `.to(device)` when you use gpu
                                label = Variable(label).to(device) # add `.to(device)` when you use gpu
                                output = model(seq)

                                loss = loss_function(output, label)
                                loss += lam1*orthogonal(seq, model.W)
                                loss -= lam2*correlated_past(seq, seq_past, model.W)
                                loss += lam3*orthonormal(model.W)

                                optimizer.zero_grad()
                                loss.backward(retain_graph=True)

                                total_loss += loss.detach()

                                err_pred = loss_function(output, label)
                                total_act_loss += err_pred.detach()

                                optimizer.step()
                                train_preds = torch.cat((train_preds, output.detach())).to(device).detach()
                                train_acts = torch.cat((train_acts, label.detach())).to(device).detach()
                                seq_past = seq.clone().detach()

                            print('epoch: {}, loss: {}'.format(i+1, total_loss.item()/len(train_loader)))
                            print('epoch: {}, train loss: {}'.format(i+1, total_act_loss.item()/len(train_loader)))

                            # Test the model
                            preds = torch.FloatTensor().to(device).detach()
                            actuals = torch.FloatTensor().to(device).detach()
                            with torch.no_grad():
                                total_err = 0
                                total_val_err = 0
                                for seq, label in val_loader:
                                    seq = Variable(seq).to(device)  # .to(device)
                                    label = Variable(label).to(device)  # .to(device)
                                    actuals = torch.cat((actuals, label.detach())).to(device).detach()

                                    output = model(seq)
                                    preds = torch.cat((preds, output.detach())).to(device).detach()
                                    error = loss_function(output, label)  # .view(1, 1)
                                    total_err += error

                                print('epoch: {}, Val Error of the model : {} '.format(i+1, total_err/len(val_loader)))
                                if i==0:
                                    min_mse=float(total_err/len(val_loader))
                                    min_mse_train=float(total_act_loss.item()/len(train_loader))
                                    best_params = {'window':seq_len,'hidden':hidden_dim,'lr':lr,'lambda1':lam1,'lambda2':lam2,'lambda3':lam3,'val_MSE':min_mse,'train_MSE':min_mse_train, 'epoch':i+1}
                                elif min_mse>total_err/len(val_loader):
                                    min_mse=float(total_err/len(val_loader))
                                    min_mse_train=float(total_act_loss.item()/len(train_loader))
                                    best_params = {'window':seq_len,'hidden':hidden_dim,'lr':lr,'lambda1':lam1,'lambda2':lam2,'lambda3':lam3,'val_MSE':min_mse,'train_MSE':min_mse_train, 'epoch':i+1}

                            val_loss.append(total_err/len(val_loader))
                        print(best_params)
                        with open('./result_ortho_mi_win7.csv','a',encoding='utf_8_sig') as f:
                            f.write(str([x for x in best_params.values()])[1:-1].replace("'",'').replace(" ",'')+'\n')
