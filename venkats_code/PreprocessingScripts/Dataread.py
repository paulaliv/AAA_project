import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/home/vsayyalasomayajula/Documents/Proteomics/PlasmaAAA/')

raw_data_file = 'PrOEF-241125-OPL3026-VLHJ-Protein.Profiling-Human.Plasma.(Abdominal.Aortic.Aneurysm)-v1.3.xlsx'

RawProteinData = pd.read_excel(raw_data_file, sheet_name='protein report')

print(RawProteinData.head(10))