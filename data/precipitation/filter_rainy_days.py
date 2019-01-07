import pandas as pd


######################################################################
## Hoa: 05.11.2018 Version 1 : rainy_day.py
######################################################################
# Reaads a *.csv file containing precipitation data.
# Dates of rainy days are filtered and collected in a separate file.
#
# Data contained:
#  - stn: Stations id
#  - time: date
#  - rhh150dx: precipitation sum of the day in mm
#  - rka150d0: one days precipitation in mm
#  - rhhtimdx: time of max precipitation in one hour per day
#
# New /Changes:
# ----------------------------------------------------------------------
#
# 05.11.2018 : first add
#
######################################################################

path_data = r'precipitation_LUZ_2017_2018_data.txt'


def filter_rainy_days_csv(path):
    df = pd.read_csv(path, sep=';')
    #print(df)
    no_rain = ['-']
    rainy_days = df[~df.rhhtimdx.isin(no_rain)]
    rainy = rainy_days[['time', 'rhhtimdx']]

    rainy['rhhtimdx'] = rainy['rhhtimdx'].map(lambda x: x.split(':')[0])
    rainy['rhhtimdx'] = rainy.rhhtimdx.astype(int)
    #print(rainy)
    rainy_images = rainy.loc[(rainy['rhhtimdx'] >= 9) & (rainy['rhhtimdx'] <= 15)] # & (rainy['rhhtimdx'] <= 9)

    rainy_images.time.to_csv('rainy_days.csv', index=False)

def is_rainy_day(date = '20180316'):
    rainy = False
    df = pd.read_csv('rainy_days.csv',index_col=False, squeeze=True, header=0)
    rainy = int(date) in df.values
    return rainy

def main():
    try:
        #filter_rainy_days_csv(path_data)
        #print('done')

        date = '20180316'
        rainy = is_rainy_day()
        print('Is {} a rainy day: {}'.format(date, rainy))


    except Exception as e:
       print('MAIN: {}'.format(e))


if __name__ == '__main__':
    main()