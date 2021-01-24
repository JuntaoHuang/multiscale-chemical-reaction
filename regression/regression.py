import matplotlib.pyplot as plt
import numpy as np
import torch
import time

class Regression(torch.nn.Module):
    
    def __init__(self, c_np_array):
        super(Regression, self).__init__()
        
        self.num = c_np_array.size
        
        self.c_array = torch.from_numpy(c_np_array).float()        
        self.p_array = torch.nn.Parameter(torch.zeros(self.num))
        
        for p in self.parameters():
            torch.nn.init.uniform_(p, -1.0, 1.0)
                
    def forward(self, x):
        
        y = self.c_array[0] * torch.pow(x, self.p_array[0])
        
        for i in range(self.num-1):
            y = y + self.c_array[i+1] * torch.pow(x, self.p_array[i+1])
        
        return y

    def get_params(self):

        for name, param in self.named_parameters():
            if name == 'p_array':
                p_np_array = param.data.detach().numpy()
        
        return p_np_array

    def print_params(self):

        p_np_array = self.get_params()

        print('p: ', p_np_array)

    def set_params(self, p_np_array):
        
        with torch.no_grad():            
            sd = self.state_dict()            
            sd['p_array'] = torch.from_numpy(p_np_array).float()
            self.load_state_dict(sd)        

    def enforce_weight_integer(self, integer_index):
        
        with torch.no_grad():            
            sd = self.state_dict()
            p_array = sd['p_array']
            
            for i in integer_index:
                p_array[i] = torch.round(p_array[i])

            sd['p_array'] = p_array
            self.load_state_dict(sd)

    def get_weight_integer_index(self, threshold):
        
        index = []
        
        with torch.no_grad():            
            for i in range(self.num):                
                if torch.abs(self.p_array[i] - torch.round(self.p_array[i])) < threshold:
                    index.append(i)
            
        return index
                    
    def train(self, x, y, is_freeze_integer=False, total_epoch=1000000, lr=1e-4, integer_threshold=0.05, loss_params_file_name='loss_params_history.npy'):
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_history = np.zeros(total_epoch)
        p_history = np.zeros((total_epoch, self.num))
        
        time_start = time.time()
        integer_index = []
        for epoch in range(total_epoch):
                
            optimizer.zero_grad()
            loss = criterion(self(x), y)
            
            loss_history[epoch] = loss.item()
            p_history[epoch,:] = self.get_params()
            
            loss.backward()
            optimizer.step()
            
            self.enforce_weight_integer(integer_index)
                
            if epoch%10000 == 0:
                time_end = time.time()
                print("epoch: %d; MSE: %.2e; elapsed time: %.2f sec"%(epoch, loss_history[epoch], time_end-time_start))
                self.print_params()
                
            if (is_freeze_integer) and (epoch%10000 == 0) and (epoch>100) and (loss_history[epoch]>=loss_history[epoch-100]):
                integer_index = self.get_weight_integer_index(integer_threshold)
                print("integer_index: ", integer_index)

        # save loss and parameters history at the end of iteration
        with open(loss_params_file_name, 'wb') as f:
            np.savez(f, loss_history=loss_history, p_history=p_history)
            
    def read_loss_params(self, file_name='loss_params_history.npy'):
        
        with open(file_name, 'rb') as f:
            data = np.load(f)
            loss_history = data['loss_history']
            p_history = data['p_history']
            epoch_array = np.arange(loss_history.size)
            
        return epoch_array, loss_history, p_history

    # compute loss as a 1D function of p1 with fixed p2
    def compute_loss_1d(self, data_x, data_y, p1_array, p2_val):
        
        np1 = p1_array.size
        loss_1d = np.zeros(np1)

        criterion = torch.nn.MSELoss()
        
        iip1 = 0
        for p1 in p1_array:
            self.set_params(np.array([p1, p2_val]))
            loss = criterion(self(data_x), data_y)
            loss_1d[iip1] = loss.item()
            iip1 += 1
        return loss_1d

    # compute loss as a 2D function of p1 and p2
    def compute_loss_2d(self, data_x, data_y, p1_arr_1d, p2_arr_1d):
        
        np1, np2 = p1_arr_1d.size, p2_arr_1d.size
        p1_arr_2d, p2_arr_2d = np.meshgrid(p1_arr_1d, p2_arr_1d)
        loss_2d = np.zeros((np2, np1))

        criterion = torch.nn.MSELoss()

        iip2 = 0
        for p2 in p2_arr_1d:
            iip1 = 0
            for p1 in p1_arr_1d:
                self.set_params(np.array([p1, p2]))
                l = criterion(self(data_x), data_y)
                loss_2d[iip2, iip1] = l.item()
                iip1 += 1
            iip2 += 1

        return p1_arr_2d, p2_arr_2d, loss_2d


if __name__ == "__main__":

    # set random seed to make results reproducible
    rand_seed_num = 0
    torch.manual_seed(rand_seed_num)
        
    # generate training data
    Nx = 1000
    data_x = torch.rand(Nx, 1)
        
    c_array = np.array([1.0, 1e2])      # c1, c2 = 1.0, 1e2
    p_array = np.array([1.0, 2.0])      # p1, p2 = 1, 2
    
    data_y = torch.zeros_like(data_x)
    for i in range(c_array.size):
        data_y += c_array[i] * torch.pow(data_x, p_array[i])
    
    reg = Regression(c_array)        
    reg.train(data_x, data_y, is_freeze_integer=True, lr=1e-4, integer_threshold=0.05, total_epoch=200000)