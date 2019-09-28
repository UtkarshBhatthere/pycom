import numpy as np
import matplotlib.pyplot as plt

#Constants.
denominator = np.sqrt(2)

# TARC = ( sqrt((S(1,1) + S(1,2)*(exp(i*__theta)))^(2) + ( S(2,1) + S(2,2)*(exp(i*__theta)) )^(2)) / sqrt(2) )
def TARC_func(s11, s12, s21, s22, theta):
    temp = list()
    for i in range(0, len(s11)):
        val = (np.sqrt(s11[i]+s12[i]*np.exp(1j*theta)**2 +(s21[i] + s22[i]*np.exp(1j*theta)**2)))/denominator
        temp.append(val)
    return(temp)

#  MEG_ONE = 0.5*(1 - abs(S(1,1))^(2) - abs(S(1,2))^(2))
#  MEG_TWO = 0.5*(1 - abs(S(1,2))^(2) - abs(S(2,2))^(2))
#  MEG_diff = MEG1 - MEG2
# def MEG_ONE(s11, s12):
#     temp = list()
#     for i in range(0, len(s12)):
#         temp.append((0.5*(1 - np.abs(s11[i]**2) - np.abs(s12[i])**2))[0])
#     return temp

# def MEG_TWO(s12, s22):
#     temp = list()
#     for i in range(0, len(s12)):
#         temp.append((0.5*(1 - np.abs(s12[i]**2) - np.abs(s22)[i]**2))[1])
#     return temp

# def MEG_diff(meg1, meg2):
#     return([meg1[i] - meg2[i] for i in range(0, len(meg1))])



def to_db(x):
    return([-20*np.log10(np.sqrt(np.real(val)**2 + np.imag(val)**2)) for val in x])

S11 = np.loadtxt(open("S11.csv", "rb"), delimiter=",", skiprows=1)
S12 = np.loadtxt(open("S12.csv", "rb"), delimiter=",", skiprows=1)
S21 = np.loadtxt(open("S21.csv", "rb"), delimiter=",", skiprows=1)
S22 = np.loadtxt(open("S22.csv", "rb"), delimiter=",", skiprows=1)

# Frequency list for x_axis.
freq = [i/10 for i in range(10, 151)]
s11_db = [s11[1] for s11 in S11]
s12_db = [s12[1] for s12 in S12]
s21_db = [s21[1] for s21 in S21]
s22_db = [s22[1] for s22 in S22]

unit_theta = np.pi/12 # This is 15 degrees, all our required values would be mulitples of these.
theta_list = [unit_theta*i for i in range(1, 13)]

TARC_all = list(list())

for __theta in theta_list:
    for_theta = TARC_func(s11_db, s12_db, s21_db, s22_db, __theta)
    TARC_all.append(to_db(for_theta))

# meg_one = MEG_ONE(s11_db, s12_db)
# meg_two = MEG_TWO(s12_db, s22_db)


# output = [{'label' : "MEG1", 'x' : freq, 'y' : meg_one},
#           {'label' : "MEG2", 'x' : freq, 'y' : meg_two},
#           {'label' : "MEG1 - MEG2", 'x' : freq, 'y' : MEG_diff(meg_one, meg_two)}]

output = [{'label' : f"Theta : {(i+1)*15}°", 'x' : freq, 'y' : TARC_all[i]} for i in range(0, len(TARC_all))]

fig, ax = plt.subplots()
for op in output:
    ax.plot(op['x'], np.real(op['y']), label=op['label'])
    # ax.plot(op['x'], np.imag(op['y'], label=op['label'])

ax.legend()
ax.set_title("MEG")
plt.show()