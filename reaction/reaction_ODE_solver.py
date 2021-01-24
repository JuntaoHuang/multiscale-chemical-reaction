import numpy as np
import sys
from scipy.integrate import solve_ivp

# generate data for chemical reaction ODE
class ReactionODE():
    
    def __init__(self, num_species, num_reaction):
        
        self.num_species  = num_species
        self.num_reaction = num_reaction
        
    def init_reversible_reaction(self, reaction_rate_f, reaction_rate_r, stoichiometric_coefficient):

        self.reaction_rate_f = reaction_rate_f
        self.reaction_rate_r = reaction_rate_r
        
        # take negative part as coefficient on the left hand side (reactant)
        # take positive part as coefficient on the right hand side (product)
        self.stoichiometric_coefficient_left  = -np.clip(stoichiometric_coefficient, a_min=None, a_max=0)
        self.stoichiometric_coefficient_right = np.clip(stoichiometric_coefficient, a_min=0, a_max=None)
        
    # compute rhs of chemical reaction ODE
    def dc(self, t, c):
                
        r = np.zeros(self.num_reaction)
        for i_reaction in range(self.num_reaction):
            
            r_tmp_f = self.reaction_rate_f[i_reaction]
            r_tmp_r = self.reaction_rate_r[i_reaction]
            for j_species in range(self.num_species):
                r_tmp_f *= c[j_species] ** self.stoichiometric_coefficient_left[i_reaction, j_species]
                r_tmp_r *= c[j_species] ** self.stoichiometric_coefficient_right[i_reaction, j_species]
            r[i_reaction] = r_tmp_f - r_tmp_r
        
        dc_dt = np.zeros(self.num_species)
        for j_species in range(self.num_species):
            for i_reaction in range(self.num_reaction):
                dc_dt[j_species] += r[i_reaction] * (self.stoichiometric_coefficient_right[i_reaction, j_species] - self.stoichiometric_coefficient_left[i_reaction, j_species])
                
        return dc_dt
    
    def solve_ODE(self, u_init, t_np, method='RK45'):
        
        num_init = int(u_init.size / self.num_species)
        u_init = np.reshape(u_init, (num_init, self.num_species))
        
        u_np = np.zeros((num_init, t_np.size, self.num_species))
        for i_init in range(num_init):
            t_span = [0., np.amax(t_np)]
            sol = solve_ivp(self.dc, t_span, u_init[i_init], t_eval=t_np, method=method)
            u_np[i_init,:,:] = np.transpose(sol.y)
        
        # make sure numerical solution is strictly positive
        u_np = np.clip(u_np, a_min=1e-30, a_max=None)
        
        # compute du/dt
        du_np = np.zeros((num_init, t_np.size, self.num_species))
        for i_init in range(num_init):
            for j_time_step in range(t_np.size):
                du_np[i_init, j_time_step, :] = self.dc(0.0, u_np[i_init, j_time_step, :])

        # if np.amin(u_np) < 0.:
        #     sys.exit("Error in ODE solver: negative value generated")
        if np.isnan(du_np).any():
            sys.exit("Error in ODE solver: nan generated")
            
        return u_np, du_np