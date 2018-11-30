import pandas as pd


######################################################################
## Hoa: 08.11.2018 Version 1 : load_data_from_csv.py
######################################################################
# process_LUZ()
# Reads a *.csv file containing irradiation data from a weatherstation
# near Lucern 47°02'11.0"N 8°18'04.0"E at 454 m.
#
# Data in solar irradiance contained:
#  - stn: Stations id
#  - time: date
#  - gor000za: Global irradiation standart deviation in W/m²
#  - gre000z0: mean global irradiation of 10 minutes
#
# Data in precipitation contained:
#  - stn: Stations id
#  - time: date
#  - rhh150dx: Max hourly precipitation sum of the day in mm
#  - rka150d0: Precipitation in mm of the day
#  - rhhtimdx: Time of max hourly precipitation sum of the day
#####################################################################
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 08.11.2018 : first add
#
######################################################################

path_irradi = r'irradiation_luz_2017_2018.csv'
path_precip = r'../ddddweather_data/precipitation_luz_2017_2018.csv'

def process_LUZ_dur(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d', utc=True)
    df['time'] = df['time'].dt.tz_localize('UTC')           # Fuer HSLU Laptop einkommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)

    #df.to_csv('luzout.csv')
    return df

def process_LUZ_Precip(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d', utc=True)
    #df['time'] = df['time'].dt.tz_localize('UTC')          # PC_Home auskommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['rka150d0'] = pd.to_numeric(df['rka150d0'], errors='coerce')

    #df.to_csv('luzout.csv')
    return df

def process_LUZ_Irrad(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', utc=True)
    df['time'] = df['time'].dt.tz_localize('UTC')           # Fuer HSLU Laptop einkommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['gre000z0'] = pd.to_numeric(df['gre000z0'], errors='coerce')

    #df.to_csv('sodaout_10min.csv')
    return df

'''
def main():
    try:
        df = process_LUZ_Precip(path_precip)
        #df = process_LUZ_Irrad(path_irradi)
        print(df.head(n=10))


    except Exception as e:
       print('Error in MAIN: {}'.format(e))

if __name__ == '__main__':
    main()
'''
