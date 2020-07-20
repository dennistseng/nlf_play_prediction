"""
 Python Script:
  Combine/Merge multiple CSV files using the Pandas library
"""
from os import chdir
from glob import glob
import pandas as pdlib
import os

# Produce a single CSV after combining all files
def produceOneCSV(list_of_files, file_out):
   # Consolidate all CSV files into one object
   result_obj = pdlib.concat([pdlib.read_csv(file) for file in list_of_files])
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")

# Move to the path that holds our CSV files
csv_file_path = 'c:/Users/halfc/NFL/Data/'
chdir(csv_file_path)

# List all CSV files in the working dir
list_of_files = [file for file in glob('*.csv')]
print(list_of_files)

file_out = "ConsolidateOutput.csv"
produceOneCSV(list_of_files, file_out)