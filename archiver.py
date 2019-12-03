from os import path, mkdir
import shutil

class archiver:

    def check_CSV_files(self, out_path : str, projectname : str, verification_list : list):
        '''
        Checks the presence of all the files passed in as a list and generates a report.
        '''
        print("Checking Files")
        stat = list()
        for files in verification_list:
            filepath = f"{out_path}/{projectname}/CSV_Outpath/{files}"
            stat.append(f"File '{files}' exists : {path.exists(filepath)}")
        return stat

    def make_path(pathname : str):
        if(not path.exists(pathname)):
            mkdir(pathname)

    def __init__(self, out_path : str, projectname : str, verification_list : list):

        # Generating report list
        # report = self.check_CSV_files(out_path, projectname, verification_list)
        # with open(f"{out_path}/{projectname}_report.txt", 'w+') as report_file:
        #     for file_stat in report:
        #         report_file.write(f"{file_stat}\n")
        #     report_file.write("This report is automatically generated by Archiver module in pycom")

        # Compressing the CSV_Output folder
        shutil.make_archive(f"{out_path}/ZIP_Output/{projectname}/CSV", 'zip', f"{out_path}/CSV_Output/{projectname}")
        shutil.make_archive(f"{out_path}/ZIP_Output/{projectname}/Graphs", 'zip', f"{out_path}/Graph_Output/{projectname}")