"""
 Python Script:
  Combine/Merge multiple CSV files using the Pandas library
"""
from os import chdir
from glob import glob
import pandas as pdlib
import os

# Produce a single CSV after combining all files
def produceOneCSV(list_of_files, file_out, year):
   # Consolidate all CSV files into one object
   result_obj = pdlib.concat([pdlib.read_excel(file) for file in list_of_files])
   result_obj['year'] = '20' + str(year[0])
   result_obj = result_obj[['Team', 'Position', 'Overall', 'year']]
   # Convert the above object into a csv file and export
   result_obj.to_csv(file_out, index=False, encoding="utf-8")

# Move to the path that holds our CSV files
csv_file_path = 'c:/Users/halfc/NFL/Madden/2012/'
chdir(csv_file_path)

# List all CSV files in the working dir
list_of_files = [file for file in glob('*.xlsx')]
print(list_of_files)

a = list_of_files[0]
a = a.replace("_"," ")
a = a.replace("."," ")
a = a.replace(")"," ")
year = [int(i) for i in a.split() if i.isdigit()] 



file_out = "ConsolidateOutput" + str(year[0]) + ".csv"
produceOneCSV(list_of_files, file_out, year)