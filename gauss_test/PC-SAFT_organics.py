import numpy as np
import pandas as pd



dataset_kwargs = {"dtype": 'str',
                  "sep": ';',
                  "header": 0,
                  "float_precision": 'round_trip',
                  "na_values": 'none'}

raw_data_lit = pd.read_csv('./input/inputs_purecomp_ECFP16384_5.csv',**dataset_kwargs,index_col=0)
pc_params = raw_data_lit.iloc[:,-3:]
pc_params = pc_params.astype('float32')
mm= raw_data_lit.iloc[:,0]
mm= mm.astype('float32')
comp = list(raw_data_lit.index)

raw_data_ml = pd.read_csv('./code/output_test.csv',**dataset_kwargs,index_col=None)
pc_params_ml = raw_data_ml.iloc[:,1:4]
pc_params_ml = pc_params_ml.astype('float32')
comp_ml = raw_data_ml.iloc[:,10]

keys = comp
values = mm
dict_comp = dict(zip(keys,values))
mm_ml = []
for key in comp_ml:
    mm_ml.append(round(dict_comp[key],2))


def create_textcode(comp,params,mm):
    comp = comp.replace(" ","_")
    comp = comp.replace(",",".")
    m_seg = params[0]/mm
    string=(f"\tELSEIF(compo(i).EQ.'{comp}') THEN \n"
            f"\t\tmm(i)     = {mm:.2f}d0\n"
            f"\t\tm_seg(i) = mm(i) * {m_seg:.6f}d0\n"
            f"\t\td00ij(i, 1, i, 1) = {params[1]:.3f}d0\n"
            f"\t\tuij(i, 1, i, 1) = {params[2]:.3f}d0\n"
            f"\t\tbind(i, 1, 1) = 1.d0\n"
            f"                          \n")
    return string

for i in range(len(comp)):
    string = create_textcode(comp[i],pc_params.iloc[i,:],mm[i])
    if i==0:
        with open("./code/PC-SAFT_organics.inc", 'w') as file:
            file.write(string)
    else:
        with open("./code/PC-SAFT_organics.inc", 'a') as file:
            file.write(string)

for i in range(len(comp_ml)):
    string = create_textcode(comp_ml[i],pc_params_ml.iloc[i,:],mm_ml[i])
    if i==0:
        with open("./code/PC-SAFT_organics_ml.inc", 'w') as file:
            file.write(string)
    else:
        with open("./code/PC-SAFT_organics_ml.inc", 'a') as file:
            file.write(string)




