import pandas as pd


######################################################################
## Hoa: 08.11.2018 Version 1 : load_data_from_csv.py
######################################################################
# process_LUZ()
# Reads a *.csv file containing irradiation data from a weatherstation
# near Lucern 47°02'11.0"N 8°18'04.0"E at 454 m.
#
# Data contained:
#  - stn: Stations id
#  - time: date
#  - gor000za: Global irradiation standart deviation in W/m²
#  - gre000z0: mean global irradiation of 10 minutes
#####################################################################
# process_SODA()
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 08.11.2018 : first add
#
######################################################################

path_data = r'irradiation_luz_2017_2018.csv'


def process_LUZ(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
    #print(df.head(n=20))
    return df

def process_SODA(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=37)
    period = df['Observation period']
    print(period.head(n=20))
    #df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
    #print(df.head(n=20))
    return df

'''
def main():
    try:
        process_LUZ(path_data)
        print('done')

    except Exception as e:
       print('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()
'''