import numpy as np
from full_data_extract import family_X, family_Y, len_family

family_names = ['analog', 'phase', 'qam', 'apsk']

# No. of training examples
total = 65536   # 2 to the power 16
pure = 9832     # approx 15%
mix_2 = 16384   # approx 25%
mix_3 = 19660   # approx 30%
mix_4 = 19660   # approx 30%

X_analog = []
Y_analog = []

X_pure = []
for _ in range(pure):
    ind = np.random.randint(len_family['analog'])
    X_analog.append(family_X['analog'][ind])
    Y_analog.append(family_Y['analog'][ind])

X_mix2 = []
for _ in range(mix_2):
    chosen_fam_arr = np.random.choice(family_names, size=1, replace=False)
    chosen_fam_arr = np.append(chosen_fam_arr, 'analog')

    ind1 = np.random.randint(len_family[chosen_fam_arr[0]])
    ind2 = np.random.randint(len_family[chosen_fam_arr[1]])

    sig1 = family_X[chosen_fam_arr[0]][ind1]
    sig2 = family_X[chosen_fam_arr[1]][ind2]
    mix_sig = sig1 + sig2

    X_analog.append(mix_sig)
    Y_analog.append(family_Y[chosen_fam_arr[1]][ind2])

X_mix3 = []
for _ in range(mix_3):
    chosen_fam_arr = np.random.choice(family_names, size=2, replace=False).tolist()
    chosen_fam_arr = np.append(chosen_fam_arr, 'analog')

    ind1 = np.random.randint(len_family[chosen_fam_arr[0]])
    ind2 = np.random.randint(len_family[chosen_fam_arr[1]])
    ind3 = np.random.randint(len_family[chosen_fam_arr[2]])

    sig1 = family_X[chosen_fam_arr[0]][ind1]
    sig2 = family_X[chosen_fam_arr[1]][ind2]
    sig3 = family_X[chosen_fam_arr[2]][ind3]

    mix_sig = sig1 + sig2 + sig3

    X_analog.append(mix_sig)
    Y_analog.append(family_Y[chosen_fam_arr[2]][ind3])

X_mix4 = []
for _ in range(mix_3):
    chosen_fam_arr = np.random.choice(family_names, size=3, replace=False).tolist()
    chosen_fam_arr = np.append(chosen_fam_arr, 'analog')

    ind1 = np.random.randint(len_family[chosen_fam_arr[0]])
    ind2 = np.random.randint(len_family[chosen_fam_arr[1]])
    ind3 = np.random.randint(len_family[chosen_fam_arr[2]])
    ind4 = np.random.randint(len_family[chosen_fam_arr[3]])

    sig1 = family_X[chosen_fam_arr[0]][ind1]
    sig2 = family_X[chosen_fam_arr[1]][ind2]
    sig3 = family_X[chosen_fam_arr[2]][ind3]
    sig4 = family_X[chosen_fam_arr[3]][ind4]

    mix_sig = sig1 + sig2 + sig3 + sig4

    X_analog.append(mix_sig)
    Y_analog.append(family_Y[chosen_fam_arr[3]][ind4])


X_analog = np.array(X_analog)
Y_analog = np.array(Y_analog)

print(X_analog.shape)
print(Y_analog.shape)

