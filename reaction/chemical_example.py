import numpy as np

# list some examples for chemical reaction ODE
def chemical_ODE_examples(example):

    # example 1: hypothetical stiff reaction network
    # see reaction 2 in Lu and Law. "On the applicability of directed relation graphs to the reduction of reaction mechanisms." Combustion and Flame 146.3 (2006): 472-483.
    if example == 1:
        
        num_species = 3
        num_reaction = 2
        
        k1, k2 = 1.0, 1e3
        reaction_rate_f = np.array([k1, k2])
        reaction_rate_r = np.array([k1, k2])
        
        stoichiometric_coefficient = np.zeros((num_reaction, num_species))
        
        # x1 <-> x2
        stoichiometric_coefficient[0, 0] = -1
        stoichiometric_coefficient[0, 1] = 1
        
        # x2 <-> x3
        stoichiometric_coefficient[1, 1] = -1
        stoichiometric_coefficient[1, 2] = 1
        
    # example 2: Michaelis–Menten kinetics
    elif example == 2:

        num_species = 4
        num_reaction = 2
        
        k_f, k_r, k_cat = 1e6, 1e3, 1e1

        reaction_rate_f = np.array([k_f, k_cat])
        reaction_rate_r = np.array([k_r, 0.0])

        stoichiometric_coefficient  = np.zeros((num_reaction, num_species))

        # E, S, ES, P correpond to 0, 1, 2, 3
        # E + S <-> ES
        stoichiometric_coefficient[0, 0] = -1
        stoichiometric_coefficient[0, 1] = -1
        stoichiometric_coefficient[0, 2] = 1

        # ES -> E + P
        stoichiometric_coefficient[1, 2] = -1
        stoichiometric_coefficient[1, 0] = 1
        stoichiometric_coefficient[1, 3] = 1
    
    # example 3: hydrogen oxidation reaction
    # see section 9 in Chiavazzo and Karlin. "Quasi-equilibrium grid algorithm: Geometric construction for model reduction." Journal of Computational Physics 227.11 (2008): 5535-5560.
    elif example == 3:
        
        # H_2, O_2, H_2O, H, O, OH correpond to 0, 1, 2, 3, 4, 5
        num_species = 6
        num_reaction = 6
        
        reaction_rate_f = np.array([2, 1, 1, 1e3, 1e3, 1e2])
        reaction_rate_r = np.array([216, 337.5, 1400, 10800, 33750, 0.7714285714285716])

        stoichiometric_coefficient  = np.zeros((num_reaction, num_species))

        # H_2 -> 2 H            (x1 <-> 2 * x4)
        stoichiometric_coefficient[0, 0] = -1
        stoichiometric_coefficient[0, 3] = 2

        # O_2 -> 2 O            (x2 <-> 2 * x5)
        stoichiometric_coefficient[1, 1] = -1
        stoichiometric_coefficient[1, 4] = 2

        # H_2O -> H + OH        (x3 <-> x4 + x6)
        stoichiometric_coefficient[2, 2] = -1
        stoichiometric_coefficient[2, 3] = 1
        stoichiometric_coefficient[2, 5] = 1

        # H_2 + O -> H + OH     (x1 + x5 <-> x4 + x6)
        stoichiometric_coefficient[3, 0] = -1
        stoichiometric_coefficient[3, 3] = 1
        stoichiometric_coefficient[3, 4] = -1
        stoichiometric_coefficient[3, 5] = 1
        
        # O_2 + H -> O + OH     (x2 + x4 <-> x5 + x6)
        stoichiometric_coefficient[4, 1] = -1
        stoichiometric_coefficient[4, 3] = -1
        stoichiometric_coefficient[4, 4] = 1
        stoichiometric_coefficient[4, 5] = 1
        
        # H_2 + O -> H_2O       (x1 + x5 <-> x3)
        stoichiometric_coefficient[5, 0] = -1
        stoichiometric_coefficient[5, 2] = 1
        stoichiometric_coefficient[5, 4] = -1

    else:
        print("example not implemented")

    reaction_rate_f, reaction_rate_r, stoichiometric_coefficient = rearrange_params(num_reaction, reaction_rate_f, reaction_rate_r, stoichiometric_coefficient)
    
    return num_species, num_reaction, reaction_rate_f, reaction_rate_r, stoichiometric_coefficient


def rearrange_params(num_reaction, reaction_rate_f, reaction_rate_r, stoichiometric_coefficient):
        
    for index in range(num_reaction):
        if reaction_rate_f[index] < reaction_rate_r[index]:
            reaction_rate_f[index], reaction_rate_r[index] = reaction_rate_r[index], reaction_rate_f[index]
            stoichiometric_coefficient[index,:] = -stoichiometric_coefficient[index,:]
    
    indices = np.argsort(-reaction_rate_f)
    stoichiometric_coefficient = stoichiometric_coefficient[indices,:]
    reaction_rate_f, reaction_rate_r = reaction_rate_f[indices], reaction_rate_r[indices]

    return reaction_rate_f, reaction_rate_r, stoichiometric_coefficient