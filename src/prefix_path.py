full_data_path = "/home/smivys/Sorbonne/108/m2_internship/DHF/data/Data_2017_2019_(1).csv"
redundant_cols = ['ID BN ','STT','Full name','Ward','District','Hospitalization day','Discharge day','Patient No',\
                  'Storage No','Characteristic fever '] 
missing_2017 = ['Level of temp','Duration fever'] 
hematoma_cols = ['Tourniquet test','Petechiae','Purpura','Ecchymoses','Muscle bleeding']
bleeding_cols = ['Gum','Nose','Gastrointestine','Urology','Vaginal']
fillna_values = {'pregnancy': 0, 'Tourniquet test': 0}