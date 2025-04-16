FULL_DATA_PATH = "/home/smivys/Sorbonne/108/m2_internship/DHF/data/Data_2017_2019_(1).csv"
REDUNDANT_COLS = ['ID BN ','STT','Full name','Age','Blood group','Ward','District','Hospitalization day','Discharge day','Patient No',\
                  'Storage No','Temperature','Characteristic fever ']
REDUNDANT_ROWS = [143, 144, 145, 225, 226, 227, 250, 262] 
ADDITIONAL_DISEASES = ['Diabetes','HBP','Hepatitis','Gastritis','pregnancy','Other']
MISSING_2017 = ['Level of temp','Duration fever'] 
HEMATOMA_COLS = ['Tourniquet test','Petechiae','Purpura','Ecchymoses','Muscle bleeding']
BLEEDING_COLS = ['Gum','Nose','Gastrointestine','Urology','Vaginal']
FILLNA_VALUES = {'pregnancy': 0, 'Tourniquet test': 0}

HEART_INDEX = ['Pulse rate','Systolic BP','Diastolic BP']