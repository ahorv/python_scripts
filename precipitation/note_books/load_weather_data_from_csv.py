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
#
# Data in wind speed contained:
#  - stn: Stations id
#  - time: date
#  - fkl010d1: Max wind gust (Wind Böen) in m/s
#  - fu3010d0: Daily mean of wind speed (skalar) in km/h
#  - dkl010d0: Wind direction; daly mean in [°]
#####################################################################
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 08.11.2018 : first add
#
######################################################################

path_irradi = r'../weather_data/sunshine_duration_2017_2018.csv'
path_precip = r'../weather_data/precipitation_luz_2017_2018.csv'
path_wind   = r'../weather_data/wind_speed_luz_2017_2018.csv'

def process_LUZ_dur(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d', utc=True)
    #df['time'] = df['time'].dt.tz_localize('UTC')           # Fuer HSLU Laptop einkommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)

    #df.to_csv('luzout.csv')
    return df

def process_LUZ_Precip(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d', utc=True)
    #df['time'] = df['time'].dt.tz_localize('UTC')          # Fuer HSLU Laptop einkommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['rka150d0'] = pd.to_numeric(df['rka150d0'], errors='coerce')

    #df.to_csv('luzout.csv')
    return df

def process_LUZ_Irrad(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M', utc=True)
    #df['time'] = df['time'].dt.tz_localize('UTC')           # Fuer HSLU Laptop einkommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['gre000z0'] = pd.to_numeric(df['gre000z0'], errors='coerce')

    #df.to_csv('sodaout_10min.csv')
    return df

def process_LUZ_wind_data(csv_file):
    df = pd.read_csv(csv_file, sep=';',index_col = False, header=0)
    df.set_index(pd.DatetimeIndex(df['time']))
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d', utc=True)
    #df['time'] = df['time'].dt.tz_localize('UTC')           # Fuer HSLU Laptop einkommentieren
    df['time'] = df['time'].dt.tz_convert('Europe/Zurich')  # converted to central europe time respecting daylight saving
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df['fkl010d1'] = pd.to_numeric(df['fkl010d1'], errors='coerce')
    df['fu3010d0'] = pd.to_numeric(df['fu3010d0'], errors='coerce')
    df['dkl010d0'] = pd.to_numeric(df['dkl010d0'], errors='coerce')

    return df

'''
def main():
    try:
        #df = process_LUZ_Precip(path_precip)
        #df = process_LUZ_Irrad(path_irradi)
        df = process_LUZ_wind_data(path_wind)
        print(df.head(n=10))


    except Exception as e:
       print('Error in MAIN: {}'.format(e))

if __name__ == '__main__':
    main()
'''
