from numpy import sqrt, imag, real, exp, abs, log10, loadtxt, pi
import matplotlib.pyplot as plt

class compy:

    denominator = sqrt(2)
    unit_theta = pi/12 # 15 degrees.
    theta_list = [pi/12*i for i in range(1, 13)]

    # Output Variables.
    TARC_all = list(list())

    def __init__(self):
        #Fetching the CSV File data to internal variables.
        self.S11 = loadtxt(open("S11.csv", "rb"), delimiter=",", skiprows=1)
        self.S12 = loadtxt(open("S12.csv", "rb"), delimiter=",", skiprows=1)
        self.S21 = loadtxt(open("S21.csv", "rb"), delimiter=",", skiprows=1)
        self.S22 = loadtxt(open("S22.csv", "rb"), delimiter=",", skiprows=1)

        #Fetching Frequency data from variable.
        self.freq = [i/10 for i in range(10, 151)]#  [s11[0] for s11 in self.S11]

        #Fetching S param Data from internal variables.
        self.s11 = [s11[1] for s11 in self.S11]
        self.s12 = [s12[1] for s12 in self.S12]
        self.s21 = [s21[1] for s21 in self.S21]
        self.s22 = [s22[1] for s22 in self.S22]

    def to_db(self, x):
        return([-20*log10(sqrt(real(val)**2 + imag(val)**2)) for val in x])

    def TARC_func(self, s11, s12, s21, s22, theta):
        temp = list()
        for i in range(0, len(s11)):
            val = (sqrt(s11[i]+s12[i]*exp(1j*theta)**2 +(s21[i] + s22[i]*exp(1j*theta)**2)))/self.denominator
            temp.append(val[0])
        return(temp)

    def TARC_all(self):
        for __theta in self.theta_list:
            for_theta = self.TARC_func(self.s11, self.s12, self.s21, self.s22, __theta)
            self.TARC_all.append(self.to_db(for_theta))
        
    
    def output(self):
        self.TARC_all()
        op_values = [{'label' : f"Theta : {(i+1)*15}°", 'x' : self.freq, 'y' : self.tarc_op_all_theta[i]} for i in range(0, len(self.tarc_op_all_theta))]
        fig, ax = plt.subplots()
        for op in op_values:
            ax.plot(op['x'], real(op['y']), label=op['label'])
            # ax.plot(op['x'], imag(op['y'], label=op['label'])

        ax.legend()
        ax.set_title("MEG")
        plt.show()

obj = compy()
obj.output()