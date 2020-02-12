from sklearn import preprocessing
import pandas as pd
import pickle
import time


start_date = "/2015-02-10"
end_date = "/2020-02-10"
M = 1258  # Total no. of rows
url1 = "https://www.nasdaq.com/api/v1/historical/"
url2 = "/stocks" + start_date + end_date

symbols_list = []
symbols_file_name = "symbols.txt"
with open(symbols_file_name, 'r') as symbols_file:
    for line in symbols_file:
        symbols_list.append(line.rstrip('\n'))
D = pd.read_csv(url1 + symbols_list[0] + url2, usecols=[0])
counter = 0  # Number of valid stocks' 5-year data
for s in symbols_list:
    print(s)
    time.sleep(1)
    try:
        T = pd.read_csv(url1 + s + url2, usecols=[1])
    except:
        print("\tERROR: Can't load file.")
        continue
    try:
        T[" Close/Last"] = T[" Close/Last"].replace('[\$,]', '', regex=True).astype(float)
    except:
        print("\tERROR: Problem with file.")
        continue
    T[" Close/Last"] = preprocessing.scale(T[" Close/Last"])  # Scale data to zero mean and unit variance
    T = T.rename(columns={' Close/Last': s})  # Column name = stock symbol
    if T.shape[0] == M:  # Getting only full 5-year data
        D = pd.concat([D, T], axis=1)
        counter += 1

print("==== Total number of stocks: " + str(counter) + " ====")

filename = "D"  # Create time-series data file
outfile = open(filename, 'wb')
pickle.dump(D, outfile)
outfile.close()
