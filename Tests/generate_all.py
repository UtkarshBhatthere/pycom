import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from pycom import pycom

tests = ["TARC", "MEG", "ECC", "DG", "CCL"]

if __name__ == "__main__":
    obj = pycom("Characteristics/MIMO1/")
    for test in tests:
        obj.output(test)