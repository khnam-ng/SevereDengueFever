full_data_path = "/home/smivys/Sorbonne/108/m2_internship/DHF/data/Data_2017_2019_(1).csv"
redundant_cols = ['ID BN ','STT','Full name','Blood group','Ward','District','Hospitalization day','Discharge day','Patient No',\
                  'Storage No','Temperature','Characteristic fever ']
redundant_rows = [143, 144, 145, 225, 226, 227, 262] 
additional_diseases = ['Diabetes','HBP','Hepatitis','Gastritis','pregnancy','Other']
missing_2017 = ['Level of temp','Duration fever'] 
hematoma_cols = ['Tourniquet test','Petechiae','Purpura','Ecchymoses','Muscle bleeding']
bleeding_cols = ['Gum','Nose','Gastrointestine','Urology','Vaginal']
fillna_values = {'pregnancy': 0, 'Tourniquet test': 0}