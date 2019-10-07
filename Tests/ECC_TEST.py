import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from pycom import pycom

if __name__ == "__main__":
    obj = pycom("Characteristics/MIMO1/")
    obj.output("ECC", False)