from numpy import imag, real, conj, abs, log10, loadtxt, savetxt, absolute, column_stack
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from cmath import sqrt, log, exp, pi, rect, e
import csv
class pycom:

    #Funtional Data Members.
    path            = str()
    method_token    = dict()
    project_name = str()

    #Storage Data members
    s11 = list()
    s12 = list()
    s21 = list()
    s22 = list()

    #Computational Data Members.
    theta_list  = [pi/12*i for i in range(1, 10)] # 15 degrees.


    # Output Variables.
    TARC_all    = list(list())
    MEG1        = list()
    MEG2        = list()
    MEG_diff    = list()
    ECC         = list()
    DG          = list()
    CCL         = list()

    #
    def __init__(self, project_name: str):
        ''' Initialiser for the pycom Class, does all the lazy work preparation.'''
        self.project_name = project_name
        path_str = f"Characteristics/{project_name}/"
        #Fetching the Real S params to internal variables. Skip rows skips the Column title.
        self.S11_mag = loadtxt(open(path_str+"S11_mag.csv", "rb"), delimiter=",", skiprows=1)
        self.S12_mag = loadtxt(open(path_str+"S12_mag.csv", "rb"), delimiter=",", skiprows=1)
        self.S21_mag = loadtxt(open(path_str+"S21_mag.csv", "rb"), delimiter=",", skiprows=1)
        self.S22_mag = loadtxt(open(path_str+"S22_mag.csv", "rb"), delimiter=",", skiprows=1)

        #Fetching the Imag S params to internal variables. Skip rows skips the Column title.
        self.S11_arg = loadtxt(open(path_str+"S11_ang_rad.csv", "rb"), delimiter=",", skiprows=1)
        self.S12_arg = loadtxt(open(path_str+"S12_ang_rad.csv", "rb"), delimiter=",", skiprows=1)
        self.S21_arg = loadtxt(open(path_str+"S21_ang_rad.csv", "rb"), delimiter=",", skiprows=1)
        self.S22_arg = loadtxt(open(path_str+"S22_ang_rad.csv", "rb"), delimiter=",", skiprows=1)

        #Fetching Frequency data from variable.
        self.freq = [s11[0] for s11 in self.S11_mag]

        #Fetching S param Data from internal variables.
        for i in range(0, len(self.S11_mag)):
            self.s11.append( self.S11_mag[i][1]*(e**(1j*self.S11_arg[i][1])))
            self.s12.append( self.S12_mag[i][1]*(e**(1j*self.S12_arg[i][1])))
            self.s21.append( self.S21_mag[i][1]*(e**(1j*self.S21_arg[i][1])))
            self.s22.append( self.S22_mag[i][1]*(e**(1j*self.S22_arg[i][1])))

    def to_db(self, x):
        '''Converts the passed Array to corresponding Db values for Graphical Analysis.'''
        
        return([20*log10(sqrt(real(val)**2 + imag(val)**2)) for val in x])

    def TARC_func(self, s11, s12, s21, s22, theta):
        '''Computes the TARC value corresponding to the passed S Params and Theta Value.'''
        
        temp = list()
        for i in range(0, len(s11)):
            temp.append((sqrt(abs(self.s11[i] + self.s12[i]*e**(1j*theta))**2 + abs(self.s21[i] + self.s22[i]*e**(1j*theta))**2) / sqrt(2)))
        return(temp)

    def TARC_all_compute(self, clear_first = True):
        '''Computes the TARC function over the internal S-Params and Thetas in Theta List'''

        if(clear_first):
            self.TARC_all.clear()

        modded = list()

        for __theta in self.theta_list:
            for_theta = self.TARC_func(self.s11, self.s12, self.s21, self.s22, __theta)
            modded.clear() 
            # for x in self.to_db(for_theta):
            #     modded.append(x-1)
            self.TARC_all.append(self.to_db(for_theta))
        
        op_values = [{'label' : f"Theta-{(i+1)*15}°", 'x' : self.freq, 'y' : self.TARC_all[i]} for i in range(0, len(self.TARC_all))]
        # op_values.append({'label' : "-10 dB Threshold", 'linestyle' : '-.', 'x' : self.freq, 'y' : [-10 for i in range(0, len(self.freq))]})
        
        return (op_values)


    def MEG1_compute(self, clear_first = True):
        '''Computes the MEG1 Value over the provided S Param Values.'''
        
        if(clear_first):
            self.MEG1.clear()

        for i in range(0, len(self.s12)):
            self.MEG1.append((0.5*(1 - abs(self.s11[i]**2) - abs(self.s12[i])**2)))
        
        return(self.freq, self.MEG1)

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

        meg_one = self.to_db(self.MEG1)
        meg_two = self.to_db(self.MEG2)

        self.MEG_diff = [meg_one[i] - meg_two[i] for i in range(0, len(self.MEG1))]
        op_values = [
            {'label' : "MEG1", 'x' : self.freq, 'y' : meg_one},
            {'label' : "MEG2", 'x' : self.freq, 'y' : meg_two},
            {'label' : "MEG1 - MEG2", 'x' : self.freq, 'y' : self.MEG_diff}
        ]

        return op_values

    def ECC_compute(self, clear_first = True):
        '''Computes the Envelope Correlation Coefficient over the provided S Param Values.'''
        if(clear_first):
            self.ECC.clear()

        for i in range(0, len(self.s12)):
            self.ECC.append((abs(conj(self.s11[i])*self.s12[i]+conj(self.s21[i])*self.s22[i]))**2/((1-abs(self.s11[i])**2-abs(self.s21[i])**2)*(1-abs(self.s22[i])**2-abs(self.s12[i])**2)))

        op_values = [{'label' : "ECC", 'x' : self.freq, 'y' : self.ECC}]
        return op_values
    
    def DG_compute(self, clear_first = True):
        '''Computes the Diversity Gain Value over the provided S Param Values.'''
        temp = list()
        if(clear_first):
            self.DG.clear()

        if(len(self.ECC) == 0):
            self.ECC_compute()

        for i in range(0, len(self.s12)):
            # val = 10 * sqrt(1 - ( (abs(self.s11[i]*self.s12[i] + self.s21[i]*self.s22[i])**2) / ( ( 1 - abs(self.s11[i])**2 - abs(self.s21[i])**2) * (1 - abs(self.s22[i])**2 - abs(self.s12[i])**2) ) )**2 )
            # temp.append(val)
            temp.append(10*sqrt(1 - self.ECC[i]**2))
        self.DG = temp
        op_values = [{'label' : "DG", 'x' : self.freq, 'y' : self.DG}]
        return op_values

    def CCL_compute(self, clear_first = True):
        '''Computes the Channel Capacity Loss over the provided S param Values.'''
        temp = list()
        if(clear_first):
            self.CCL.clear()

        for i in range(0, len(self.s12)):
            val = -1*log( (1 - (abs(self.s11[i])**(2) + abs(self.s11[i])**(2)))*(1 - (abs(self.s22[i])**(2) + abs(self.s22[i])**(2))) - ( -1*((conj(self.s11[i])*self.s12[i]) + (conj(self.s21[i])*self.s12[i])) )*( -1*((conj(self.s11[i])*self.s21[i]) + (conj(self.s12[i])*self.s21[i])) ), 2)
            temp.append(val)
        self.CCL = temp
        #savetxt("CCL.csv", self.freq, self.CCL, delimiter=',')
        
        op_values = [{'label' : "CCL", 'x' : self.freq, 'y' : self.CCL}] 
        print(type(self.CCL))      
        return op_values
    
    def S_Params(self, clear_first = True):
        ''' Prepares the S params and pushes them for the output '''
        op_values = [
            {'label' : "S(1,1)", 'x' : self.freq, 'y' : self.s11},
            {'label' : "S(1,2)", 'x' : self.freq, 'y' : self.s12},
            {'label' : "S(2,1)", 'x' : self.freq, 'y' : self.s21},
            {'label' : "S(2,2)", 'x' : self.freq, 'y' : self.s22}
        ]
        return op_values

    method_token = {
        "TARC" : TARC_all_compute,
        "MEG"  : MEG_diff_compute,
        "ECC"  : ECC_compute,
        "DG"   : DG_compute,
        "CCL"  : CCL_compute,
        "SP"   : S_Params
    }

    def output(self, property = "TARC", save = True):

        """Generates the Output for the desired property.

        If the argument `property` isn't passed in, TARC is computed.

        Parameters
        ----------
        Property : str, optional
            The Desired property to be computed.It serves as a token for the 
            method to be executed. Options are TARC, MEG, ECC, CCL, DG, and SP.
        """

        op_values = self.method_token[property](self, clear_first = True)
        for label in op_values:
            filename = f"{property}_{label['label']}.csv"
            pathname = f"CSV_Output/{self.project_name}"
            freq = numpy.array([real(x) for x in label['x']])
            mag  = numpy.array(label['y'])
            dataFrame = pd.DataFrame({"freq(in Ghz)" : freq, f"{property}_{label['label']}" : mag})
            dataFrame.to_csv(f"{pathname}/{filename}", index=False)

        print(len(op_values))
        fig, ax = plt.subplots()
        for op in op_values:
            ax.plot(op['x'], real(op['y']), label=op['label'])
        ax.legend()
        ax.set_title(property)
        if save == False:
            plt.show()
        else:
            plt.savefig(f"Graph_Output/{property}.png")