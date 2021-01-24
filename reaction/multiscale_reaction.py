import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import chemical_example
import reaction_ODE_solver


class MultiscaleReaction(torch.nn.Module):
    
    def __init__(self, num_species, num_reaction):
        super(MultiscaleReaction, self).__init__()
        
        self.num_species  = num_species
        self.num_reaction = num_reaction
        
        # use one weight matrix to represent left and right stoichiometric coefficients
        # only applies to non-catalysis reaction
        self.w = torch.nn.Parameter(torch.zeros(self.num_species, self.num_reaction))
        self.b_r = torch.nn.Parameter(torch.zeros(1, self.num_reaction))
        self.b_f = torch.nn.Parameter(torch.zeros(1, self.num_reaction))
                            
        # initialize parameter
        for p in self.parameters():
            torch.nn.init.uniform_(p, -0.5, 0.5)
                
    def forward(self, c):
                
        # coefficient of reactant and product
        w_reactant = -torch.clamp(self.w, max=0)
        w_product = torch.clamp(self.w, min=0)
        
        # reactant rate of forward reaction and reverse reaction
        z_f = torch.exp((torch.mm(torch.log(c), w_reactant) + self.b_f))
        z_r = torch.exp((torch.mm(torch.log(c), w_product) + self.b_r))
        
        # dc/dt
        dc = torch.mm(z_f - z_r, torch.transpose(self.w, 0, 1))
        return dc
    
    def print_params(self, log_file):
        
        for name, param in self.named_parameters():
            if name == 'w':
                w = param.data.detach().numpy()
            
            elif name == 'b_f':
                b_f = param.data.detach().numpy()
                
            elif name == 'b_r':
                b_r = param.data.detach().numpy()

        with open(log_file, 'a') as f:
            f.write('weight (each column represent a reaction)\n')
            np.savetxt(f, w, fmt='%.5f', newline='\n')
        
        with open(log_file, 'a') as f:
            f.write('k_f\n')
            np.savetxt(f, np.exp(b_f), fmt='%.6e', newline='\n')
        
        with open(log_file, 'a') as f:
            f.write('k_r\n')
            np.savetxt(f, np.exp(b_r), fmt='%.6e', newline='\n')
            f.write('\n')
    
    # rearrange weight and bias such that 
    # b_f is always larger than b_r, and b_f is in descending order
    # this should be called at the end of iteration
    def rearrange_params(self):
        
        with torch.no_grad():
            
            sd = self.state_dict()
            w = sd['w']
            b_f = sd['b_f']
            b_r = sd['b_r']
            for index in range(self.num_reaction):                
                if b_f[0,index] < b_r[0,index]:
                    b_r[0, index], b_f[0, index] = b_f[0,index].item(), b_r[0,index].item()
                    w[:,index] = -w[:,index]
            
            _, indices = torch.sort(b_f[0,:], descending=True)            
            w, b_f, b_r = w[:,indices], b_f[:,indices], b_r[:,indices]

            sd['w'] = w
            sd['b_f'] = b_f
            sd['b_r'] = b_r
            self.load_state_dict(sd)
            
    # compute Lp norm of weight in all layers
    def Lp_norm_params(self, p):
        Lp_reg = torch.tensor(0., requires_grad=True)
        for _, param in self.named_parameters():
            Lp_reg = Lp_reg + torch.norm(param, p)
        return Lp_reg
    
    # round weight to the closest integer
    def enforce_weight_integer(self, integer_index):
        
        with torch.no_grad():
            
            sd = self.state_dict()
            w = sd['w']
            w[:, integer_index] = torch.round(w[:, integer_index])
            sd['w'] = w
            self.load_state_dict(sd)

    # random reinitialization of non-integer weights
    def random_reinitial_weight_non_integer(self, integer_index):
        
        with torch.no_grad():
            
            sd = self.state_dict()
            w = sd['w']
            for i in range(self.num_reaction):
                if not (i in integer_index):
                    w[:, i] = torch.rand(self.num_species) * 4 - 2       # random number in [-2, 2]
            sd['w'] = w
            self.load_state_dict(sd)
                
    # return if all the weights are integer (error less than a given threshold)
    def is_weight_integer(self, threshold):

        err = self.w - torch.round(self.w)        
        if torch.max(torch.abs(err)) < threshold:
            return True
        else:
            return False
        
    def get_weight_integer_index(self, threshold):
        
        index = []
        for i in range(self.num_reaction):
            
            err = self.w[:,i] - torch.round(self.w[:,i])
            
            # avoid put weights all close to zero to integer index set
            if (torch.max(torch.abs(err)) < threshold) and (torch.norm(torch.round(self.w[:,i])) > 0.1):                
                index.append(i)
                        
        return index
    
    # avoid the case when two weight integer vector are the same
    # if this case, then delete one integer index from the integer list and reinitialize another reaction weight
    def delete_duplicate_weight_integer(self, integer_index):
                
        with torch.no_grad():
            
            is_while_loop = True
            while is_while_loop:
                
                is_while_loop = False
                for i in integer_index:
                    for j in integer_index:
                        is_same_integer_vec = (
                            (torch.norm(torch.round(self.w[:,i]) - torch.round(self.w[:,j])) < 1e-3)
                            or
                            (torch.norm(torch.round(self.w[:,i]) + torch.round(self.w[:,j])) < 1e-3)
                            )
                        if (not(i==j)) and is_same_integer_vec:
                            self.w[:,j] = torch.rand(self.num_species)
                            integer_index.remove(j)
                            is_while_loop = True
        
        return integer_index
                        
    def train(self, c_train, dc_train, c_validate, dc_validate, lr=1e-3, total_epoch=100000, batch_size=128, regularization=1e-8, regularization_norm=[1,2], is_freeze_integer=True, integer_freq=10000, integer_threshold=0.05, log_file=None):

        if log_file==None:
            log_file = "training.log"
        print("training log is written into file: " + log_file)
            
        # write basic parameters into file
        with open(log_file, 'w') as f:
            f.write('learning_rate:         {:e} \n'.format(lr))            
            f.write('total_epoch:           {:d} \n'.format(total_epoch))
            f.write('batch_size:            {:d} \n'.format(batch_size))
            f.write('regularization norm:   ')
            for item in regularization_norm:
                f.write('{:d} '.format(item))
            f.write('\n')
            f.write('log_file:              {:s} \n'.format(log_file))
            f.write("----------------------------------------------------------------\n")

        # make sure that c is strictly greater than zero
        eps = 1e-30
        c_train    = torch.clamp(c_train,    min=eps, max=None)
        c_validate = torch.clamp(c_validate, min=eps, max=None)
                                 
        # print initial parameter
        with open(log_file, 'a') as f:
            f.write('initial parameters:\n')
        self.print_params(log_file)
           
        total_data_num = c_train.size()[0]
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        loss_history = np.zeros(total_epoch)
        train_error = np.zeros(total_epoch)
        validation_error = np.zeros(total_epoch)

        time_start = time.time()
        integer_index = []
        is_all_weight_integer = False
        for epoch in range(total_epoch):

            permutation = torch.randperm(total_data_num)

            for i in range(0, total_data_num, batch_size):
                
                optimizer.zero_grad()

                indices = permutation[i:i+batch_size]
            
                loss = criterion(self.forward(c_train[indices]), dc_train[indices])                
                if not(is_all_weight_integer):
                    for reg_norm in regularization_norm:
                        loss += regularization * self.Lp_norm_params(reg_norm)
                
                loss_history[epoch] += loss.item()
                loss.backward()
                    
                optimizer.step()
                if len(integer_index) > 0:
                    self.enforce_weight_integer(integer_index)

            # record relative l2 and linf error in training and validation data
            train_error[epoch] = get_relative_error(self.forward(c_train), dc_train)
            validation_error[epoch] = get_relative_error(self.forward(c_validate), dc_validate)
                    
            # at the end of iteration, rearrange order of parameters
            if epoch==(total_epoch-1):
                self.rearrange_params()
                
            if (epoch%10000 == 0) or (epoch==(total_epoch-1)):
                time_end = time.time()
                with open(log_file, 'a') as f:
                    f.write("epoch: %d; MSE: %.2e; train err: %.2e; validation err: %.2e; elapsed time: %.2f sec\n"%(epoch, loss_history[epoch], train_error[epoch], validation_error[epoch], time_end-time_start))
                self.print_params(log_file)
                
            # if loss not decrease
            if (not(is_all_weight_integer)) and (epoch%10000==0) and (epoch>10000) and (loss_history[epoch-1000] < loss_history[epoch]) and (is_freeze_integer):
                
                index_list = self.get_weight_integer_index(integer_threshold)
                integer_index.extend(index_list)
                integer_index = list(set(integer_index))    # get rid of possible duplicate index
                
                integer_index = self.delete_duplicate_weight_integer(integer_index)
                with open(log_file, 'a') as f:
                    f.write("----------------------------------------------------------------\n")
                    f.write("loss not decrease\n")
                    f.write("integer index set ")
                    for item in integer_index:
                        f.write("%d "%(item))
                    f.write("\n")
                    f.write("reinitialize non-integer weight\n")
                    f.write("----------------------------------------------------------------\n")
                self.random_reinitial_weight_non_integer(integer_index)

            # if all the weights are integer, then freeze all the weights and only update reaction rates
            if (not(is_all_weight_integer)) and (epoch%10000 == 0) and (self.is_weight_integer(1e-3))  and (is_freeze_integer):
                
                is_all_weight_integer = True
                with open(log_file, 'a') as f:
                    f.write("----------------------------------------------------------------\n")
                    f.write("all stoichiometric coefficients reach integer\n")
                    f.write("now only update reaction rate\n")
                    f.write("----------------------------------------------------------------\n")
                
                # round all the weights to be integer
                integer_index = range(self.num_reaction)
                self.enforce_weight_integer(integer_index)
                
                self.w.requires_grad = False
                
            if epoch%10000 == 0:
                self.save_model('net_para_epoch_' + str(epoch) + '.pt')
            
        # save loss history at the end of iteration
        with open('loss_history.npy', 'wb') as f:
            np.savez(f, loss_history=loss_history, 
                     train_error=train_error,
                     validation_error=validation_error)
        
    # assign exact parameters in chemical reaction ODE into neural network
    # only useful in debug
    def assign_exact_params(self, reaction_rate_f, reaction_rate_r, stoichiometric_coefficient):
                
        sd = self.state_dict()
        sd['w'] = torch.from_numpy(np.transpose(stoichiometric_coefficient)).float()
        sd['b_f'] = torch.log(torch.from_numpy(np.reshape(reaction_rate_f, (1, self.num_reaction))).float())
        sd['b_r'] = torch.log(torch.from_numpy(np.reshape(reaction_rate_r, (1, self.num_reaction))).float())
        self.load_state_dict(sd)

    # assign exact weights parameters in chemical reaction ODE into neural network
    def assign_exact_weight_params(self, stoichiometric_coefficient):
                
        sd = self.state_dict()
        sd['w'] = torch.from_numpy(np.transpose(stoichiometric_coefficient)).float()
        self.load_state_dict(sd)
    
    # compute error between exact and numerical reaction rates
    def compute_error_reaction_rate(self, reaction_rate_f_exact, reaction_rate_r_exact, log_file):
            
        rate_f_exact = torch.from_numpy(reaction_rate_f_exact).float()
        rate_f_exact = torch.reshape(rate_f_exact, (1, self.num_reaction))
        
        rate_r_exact = torch.from_numpy(reaction_rate_r_exact).float()
        rate_r_exact = torch.reshape(rate_r_exact, (1, self.num_reaction))
        
        with torch.no_grad():
            
            relative_err_k_f = torch.abs(rate_f_exact - torch.exp(self.b_f))/torch.exp(self.b_f)
            relative_err_k_r = torch.abs(rate_r_exact - torch.exp(self.b_r))/torch.exp(self.b_r)

            relative_err_k_f = relative_err_k_f.numpy()
            relative_err_k_r = relative_err_k_r.numpy()
            
            err_k_f = torch.abs(rate_f_exact - torch.exp(self.b_f))
            err_k_r = torch.abs(rate_r_exact - torch.exp(self.b_r))

            err_k_f = err_k_f.numpy()
            err_k_r = err_k_r.numpy()
            
            rate_f_exact = rate_f_exact.numpy()
            rate_r_exact = rate_r_exact.numpy()
                        
        with open(log_file, 'a') as f:
            
            f.write('k_f (exact)\n')
            np.savetxt(f, rate_f_exact, fmt='%.6e', newline='\n')
            f.write('k_r (exact)\n')
            np.savetxt(f, rate_r_exact, fmt='%.6e', newline='\n')
            
            f.write('error in k_f\n')
            np.savetxt(f, err_k_f, fmt='%.6e', newline='\n')
            f.write('error in k_r\n')
            np.savetxt(f, err_k_r, fmt='%.6e', newline='\n')
            
            f.write('relative error in k_f\n')
            np.savetxt(f, relative_err_k_f, fmt='%.6e', newline='\n')
            f.write('relative error in k_r\n')
            np.savetxt(f, relative_err_k_r, fmt='%.6e', newline='\n')
            f.write('\n')
                    
    # save model into a file
    def save_model(self, file_name):
        
        torch.save({
            'num_species':          self.num_species,
            'num_reaction':         self.num_reaction,
            'model_state_dict':     self.state_dict(),
            }, file_name)


# load neural network from parameter file
def load_model(file_name):
    
    checkpoint = torch.load(file_name)
    
    num_species       = checkpoint['num_species']
    num_reaction      = checkpoint['num_reaction']
    
    model = MultiscaleReaction(num_species, num_reaction)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
        
# compute relative L_inf error between two tensors
def get_relative_error(num, exa):
    
    with torch.no_grad():        
        err = torch.max(num - exa) / torch.max(exa)
    
    return err.item()        

# add noise in tensor
def add_noise(c, noise_level):
    
    c_noise = c * (1 + 2 * noise_level * (torch.rand(c.shape) - 0.5))
    return c_noise


if __name__ == "__main__":
    
    # set random seed to make results reproducible
    rand_seed_num = 1
    np.random.seed(rand_seed_num)
    torch.manual_seed(rand_seed_num)
    
    example = 3
    [num_species, num_reaction, 
     reaction_rate_f, reaction_rate_r, 
     stoichiometric_coefficient] =  chemical_example.chemical_ODE_examples(example)

    # generate data from ODE solver
    reaction_ODE = reaction_ODE_solver.ReactionODE(num_species, num_reaction)
    reaction_ODE.init_reversible_reaction(reaction_rate_f, reaction_rate_r, stoichiometric_coefficient)

    num_init = 100
    c0 = np.random.rand(num_init, num_species)
    t_np = np.linspace(0.0, 1.0, 10)
    # t_np = np.logspace(-3, 8, num=200)

    c_np, dc_np = reaction_ODE.solve_ODE(c0, t_np, method='Radau')
        
    # # load data
    # data = np.load('training_data.npy')
    # c_np, dc_np = data['c_np'], data['dc_np']

    # save data
    training_data_file = 'training_data.npy'
    print('training data is saved in file: ' + training_data_file)
    with open(training_data_file, 'wb') as f:
        np.savez(f, c0=c0, t_np=t_np, c_np=c_np, dc_np=dc_np)

    print("c min: ", np.amin(c_np))
    print("c max: ", np.amax(c_np))

    num_time_steps = int(c_np.size / num_species)

    # transform numpy to tensor
    c = torch.from_numpy(c_np).float()
    c = torch.reshape(c, (num_time_steps, num_species))
    
    dc = torch.from_numpy(dc_np).float()
    dc = torch.reshape(dc, (num_time_steps, num_species))

    # add noise in training data
    noise_level = 0.0
    dc = add_noise(dc, noise_level)

    # random permutation all the data
    total_data_num = c.size()[0]
    permutation = torch.randperm(total_data_num)
    c, dc = c[permutation], dc[permutation]
    
    # training data, take 80% of data
    training_dat_ratio = 0.8
    training_data_num = int(total_data_num * training_dat_ratio)
    c_train, dc_train = c[0:training_data_num, :], dc[0:training_data_num, :]
    
    # validation data, take 20% of data
    validation_data_num = total_data_num - training_data_num
    c_validate, dc_validate = c[training_data_num:, :], dc[training_data_num:, :]
    
    # machine learning model
    model = MultiscaleReaction(num_species, num_reaction)
    
    # # load model from file
    # model = load_model(net_param_file)
    
    # # only for debug
    # model.assign_exact_params(reaction_rate_f, reaction_rate_r, stoichiometric_coefficient)    
    # model.assign_exact_weight_params(stoichiometric_coefficient)

    model.train(c_train, dc_train,
                c_validate, dc_validate,
                lr=1e-3, 
                total_epoch=1000000,
                batch_size=num_time_steps,
                regularization=1e-8, regularization_norm=[1, 2], 
                integer_threshold=0.05, log_file='print.log')

    model.compute_error_reaction_rate(reaction_rate_f, reaction_rate_r, log_file='print.log')