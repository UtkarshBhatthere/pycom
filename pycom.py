from numpy import sqrt, imag, real, exp, abs, log10, loadtxt, pi
import matplotlib.pyplot as plt

class pycom:

    #Funtional Data Members.
    path = str()
    method_token = dict()

    #Computational Data Members.
    denominator = sqrt(2)
    unit_theta = pi/6 # 15 degrees.
    theta_list = [pi/9*i for i in range(1, 10)]


    # Output Variables.
    TARC_all = list(list())
    MEG1 = list()
    MEG2 = list()
    MEG_diff = list()

    #
    def __init__(self, path_str: str):
        ''' Initialiser for the pycom Class, does all the lazy work preparation.'''

        #Fetching the CSV File data to internal variables. Skip rows skips the Column title.
        self.S11 = loadtxt(open(path_str+"S11.csv", "rb"), delimiter=",", skiprows=1)
        self.S12 = loadtxt(open(path_str+"S12.csv", "rb"), delimiter=",", skiprows=1)
        self.S21 = loadtxt(open(path_str+"S21.csv", "rb"), delimiter=",", skiprows=1)
        self.S22 = loadtxt(open(path_str+"S22.csv", "rb"), delimiter=",", skiprows=1)

        #Fetching Frequency data from variable.
        self.freq = [i/10 for i in range(10, 151)]#  [s11[0] for s11 in self.S11]

        #Fetching S param Data from internal variables.
        self.s11 = [s11[1] for s11 in self.S11]
        self.s12 = [s12[1] for s12 in self.S12]
        self.s21 = [s21[1] for s21 in self.S21]
        self.s22 = [s22[1] for s22 in self.S22]

    def to_db(self, x):
        '''Converts the passed Array to corresponding Db values for Graphical Analysis.'''
        
        return([-20*log10(sqrt(real(val)**2 + imag(val)**2)) for val in x])

    def TARC_func(self, s11, s12, s21, s22, theta):
        '''Computes the TARC value corresponding to the passed S Params and Theta Value.'''
        
        temp = list()
        for i in range(0, len(s11)):
            val = (sqrt(s11[i]+s12[i]*exp(1j*theta)**2 +(s21[i] + s22[i]*exp(1j*theta)**2)))/self.denominator
            temp.append(val)
        return(temp)

    def TARC_all_compute(self, clear_first = True):
        '''Computes the TARC function over the internal S-Params and Thetas in Theta List'''

        if(clear_first):
            self.TARC_all.clear()

        for __theta in self.theta_list:
            for_theta = self.TARC_func(self.s11, self.s12, self.s21, self.s22, __theta)
            self.TARC_all.append(for_theta)
        
        op_values = [{'label' : f"Theta : {(i+1)*20}°", 'x' : self.freq, 'y' : self.TARC_all[i]} for i in range(0, len(self.TARC_all))]
        
        return (op_values)


    def MEG1_compute(self, clear_first = True):
        '''Computes the MEG1 Value over the provided S Param Values.'''
        
        if(clear_first):
            self.MEG1.clear()

        for i in range(0, len(self.s12)):
            self.MEG1.append((0.5*(1 - abs(self.s11[i]**2) - abs(self.s12[i])**2)))

    def MEG2_compute(self, clear_first = True):
        '''Computes the MEG2 Value over the provided S Param Values.'''

        if(clear_first):
            self.MEG2.clear()

        for i in range(0, len(self.s12)):
            self.MEG2.append((0.5*(1 - abs(self.s12[i]**2) - abs(self.s22)[i]**2)))
        return(self.freq, self.MEG2)

    def MEG_diff_compute(self, clear_first = True):
        '''Computes the MEG Value difference over the pre calculated MEG1 and MEG2.'''

        if(clear_first):
            self.MEG_diff.clear()

        if(len(self.MEG1) is 0 or len(self.MEG1) is 0):
            self.MEG1_compute()
            self.MEG2_compute()

        self.MEG_diff = [self.MEG1[i] - self.MEG2[i] for i in range(0, len(self.MEG1))]
        op_values = [
            {'label' : "MEG1", 'x' : self.freq, 'y' : self.MEG1},
            {'label' : "MEG2", 'x' : self.freq, 'y' : self.MEG2},
            {'label' : "MEG1 - MEG2", 'x' : self.freq, 'y' : self.MEG_diff}
        ]

        return op_values
        

    method_token = {
        "TARC" : TARC_all_compute,
        "MEG"  : MEG_diff_compute
    }

    def output(self, property = "TARC"):

        """Generates the Output for the desired property.

        If the argument `property` isn't passed in, TARC is computed.

        Parameters
        ----------
        Property : str, optional
            The Desired property to be computed.It serves as a token for the 
            method to be executed. Options are TARC, MEG, ECC, CCL, DG, and CM.
        """

        op_values = self.method_token[property](self, clear_first = True)
        fig, ax = plt.subplots()
        for op in op_values:
            ax.plot(op['x'], real(op['y']), label=op['label'])
        ax.legend()
        ax.set_title(property)
        plt.show()