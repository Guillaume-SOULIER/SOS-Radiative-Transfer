## Van der Hulst extraction

def vdh(list, nb_angles, nb_layers):
    idx = int((nb_angles-1)/10)
    return list[[0, idx, idx*3, idx*5, idx*7, idx*9, idx*10]]

def In_up_down(I, nb_angles, nb_layers):
    I_up = I[0,nb_angles:] # extracted at top of atmosphere (t=0)
    I_down = I[nb_layers-1,:nb_angles] # extracted at surface level (t=tauStar)
    I_down[:] = I_down[::-1] # sorting value from [-1; 0] to [0; 1]
    return I_up, I_down