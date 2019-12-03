from pycom import pycom

tests = ["TARC", "MEG", "ECC", "DG", "CCL", "SP"]

if __name__ == "__main__":
    obj = pycom("MIMO2")
    for test in tests:
        obj.output(test)