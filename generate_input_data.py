import numpy as np
from full_data_extract import family_X, family_Y, len_family

family_names = ['analog', 'phase', 'qam', 'apsk']

# No. of training examples
total = 65536   # 2 to the power 16
pure  = 9832    # approx 15%
mix_2 = 16384   # approx 25%
mix_3 = 19660   # approx 30%
mix_4 = 19660   # approx 30%


X = {}
Y_special = {}
Y_router = []

for target_family in family_names:
    X_fam = []
    Y_targ_fam = []

    for _ in range(pure):
        idx = np.random.randint(len_family[target_family])
        X_fam.append(family_X[target_family][idx])
        Y_targ_fam.append(family_Y[target_family][idx])
        Y_router.append([family_names.index(target_family)])


    for _ in range(mix_2):
        chosen_fam_arr = np.random.choice(family_names, size=1, replace=False)
        chosen_fam_arr = np.append(chosen_fam_arr, target_family)

        ind1 = np.random.randint(len_family[chosen_fam_arr[0]])
        ind2 = np.random.randint(len_family[chosen_fam_arr[1]])

        sig1 = family_X[chosen_fam_arr[0]][ind1]
        sig2 = family_X[chosen_fam_arr[1]][ind2]
        mix_sig = sig1 + sig2
        X_fam.append(mix_sig)

        # multi‑hot label by adding two one hot encodings if both signals are from same family 
        label = None
        if chosen_fam_arr[0] == target_family:
            label = family_Y[chosen_fam_arr[0]][ind1] + family_Y[chosen_fam_arr[1]][ind2]
            label = np.clip(label, 0, 1)  # in case of same signal type, adding will give 2 as encoding in one column

            Y_targ_fam.append(label)

        else:
            Y_targ_fam.append(family_Y[chosen_fam_arr[1]][ind2])
        
        rout_label = []
        for fam in chosen_fam_arr:
            rout_label.append(family_names.index(fam))
        Y_router.append(rout_label)


    for _ in range(mix_3):
        chosen_fam_arr = np.random.choice(family_names, size=2, replace=False).tolist()
        chosen_fam_arr = np.append(chosen_fam_arr, target_family)

        ind1 = np.random.randint(len_family[chosen_fam_arr[0]])
        ind2 = np.random.randint(len_family[chosen_fam_arr[1]])
        ind3 = np.random.randint(len_family[chosen_fam_arr[2]])

        sig1 = family_X[chosen_fam_arr[0]][ind1]
        sig2 = family_X[chosen_fam_arr[1]][ind2]
        sig3 = family_X[chosen_fam_arr[2]][ind3]
        mix_sig = sig1 + sig2 + sig3
        X_fam.append(mix_sig)

        idx_list = [ind1, ind2, ind3]
        label = None
        for i in range(len(chosen_fam_arr)):
            fam = chosen_fam_arr[i]
            idx = idx_list[i]
            if fam == target_family:
                if label is None:
                    label = family_Y[fam][idx]
                else:
                    label += family_Y[fam][idx]
        if label is None:
            label = family_Y[chosen_fam_arr[2]][ind3] 

        label = np.clip(label, 0, 1)
        Y_targ_fam.append(label)

        rout_label = []
        for fam in chosen_fam_arr:
            rout_label.append(family_names.index(fam))
        Y_router.append(rout_label)

    for _ in range(mix_4):
        chosen_fam_arr = np.random.choice(family_names, size=3, replace=False).tolist()
        chosen_fam_arr = np.append(chosen_fam_arr, target_family)

        ind1 = np.random.randint(len_family[chosen_fam_arr[0]])
        ind2 = np.random.randint(len_family[chosen_fam_arr[1]])
        ind3 = np.random.randint(len_family[chosen_fam_arr[2]])
        ind4 = np.random.randint(len_family[chosen_fam_arr[3]])

        sig1 = family_X[chosen_fam_arr[0]][ind1]
        sig2 = family_X[chosen_fam_arr[1]][ind2]
        sig3 = family_X[chosen_fam_arr[2]][ind3]
        sig4 = family_X[chosen_fam_arr[3]][ind4]
        mix_sig = sig1 + sig2 + sig3 + sig4
        X_fam.append(mix_sig)

       
        idx_list = [ind1, ind2, ind3, ind4]
        label = None
        for i in range(len(chosen_fam_arr)):
            fam = chosen_fam_arr[i]
            idx = idx_list[i]
            if fam == target_family:
                if label is None:
                    label = family_Y[fam][idx]
                else:
                    label += family_Y[fam][idx]
        if label is None:
            label = family_Y[chosen_fam_arr[3]][ind4]

        label = np.clip(label, 0, 1)
        Y_targ_fam.append(label)

        rout_label = []
        for fam in chosen_fam_arr:
            rout_label.append(family_names.index(fam))
        Y_router.append(rout_label)


    X[target_family] = np.array(X_fam)
    Y_special[target_family] = np.array(Y_targ_fam)

print(X['analog'].shape)
print(Y_special['qam'].shape)

