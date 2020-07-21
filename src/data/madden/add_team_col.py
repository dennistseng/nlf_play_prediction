# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 00:51:29 2020

@author: halfc
"""


from os import chdir
from glob import glob
import pandas as pd
import os
import numpy as np

# Move to the path that holds our CSV files
csv_file_path = 'c:/Users/halfc/NFL/Madden/2012/'
chdir(csv_file_path)

list_of_files = [file for file in glob('*.xlsx')]
print(list_of_files)

for f in list_of_files:
    xl = pd.ExcelFile(f)
    sheet_name = xl.sheet_names[0]
    table = xl.parse(sheet_name)
    table = table[(table['Name'] != 'Name') & (table['Name'] != np.nan)]
    table = table.dropna(how="all")
    table['Team'] = sheet_name
    table.to_excel('1'+f)
    