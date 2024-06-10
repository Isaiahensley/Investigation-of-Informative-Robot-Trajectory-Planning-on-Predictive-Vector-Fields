import netCDF4
import numpy as np
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

# Sets size for the empty veloU and veloV
m, n = 30, 30

class Data:
    def __init__(self, lat, lon, time, u, v, temp, salt):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.u = u
        self.v = v
        self.temp = temp
        self.salt = salt

    def __repr__(self):
        return f"time={self.time} lat={self.lat} lon={self.lon} u={self.u:.4f} v={self.v:.4f} temp={self.temp:.4f} salt={self.salt:.4f}"

    def to_dict(self):
        return {'time': self.time, 'lat': self.lat, 'lon': self.lon, 'u': self.u, 'v': self.v, 'temp': self.temp, 'salt': self.salt}

def load_dataset(files):

    for index, filename in tqdm(enumerate(sorted(files))):
        nc = netCDF4.Dataset(str(filename))
        # #time=string[-5:-3]
        time = filename.name
        timestamp = int(re.findall("\d+", filename.name)[0])
        # Store variables from dataset file locally
        longitude = nc.variables['lon']
        latitude = nc.variables['lat']
        depth = nc.variables['depth']
        u = nc.variables['u']
        v = nc.variables['v']
        temp = nc.variables['temp']  # Add temperature variable
        salt = nc.variables['salt']  # Add salt variable

        # m = n = 30
        d = 0
        LONG = np.array(longitude[:])
        LATT = np.array(latitude[:])
        for i in range(m):
            for j in range(n):
                lat = LATT[i]
                lon = LONG[j]
                uu = u[0][d][i][j]
                vv = v[0][d][i][j]
                temp_val = temp[0][d][i][j]
                salt_val = salt[0][d][i][j]
                row = Data(lat, lon, timestamp, uu, vv, temp_val, salt_val)
                yield row

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # files.upload()
    path = Path('Testing_Target')
    files = path.glob("*.nc")
    dataset = []
    for data in load_dataset(files):
        # Extract year, month, day, and hour from timestamp
        year = data.time // 1000000  # Year is the first six digits
        month_day_hour = data.time % 1000000  # The rest are combined as month, day, and hour
        month = month_day_hour // 10000
        day_hour = month_day_hour % 10000
        day = day_hour // 100
        hour = day_hour % 100

        # Add extracted components to the data dictionary
        data_dict = data.to_dict()
        data_dict.update({'year': year, 'month': month, 'day': day, 'hour': hour})

        dataset.append(data_dict)

    df = pd.DataFrame(dataset)

    # Rearrange columns and remove 'time'
    df = df[['year', 'month', 'day', 'hour', 'lat', 'lon', 'u', 'v', 'temp', 'salt']]

    df.to_csv('Testing_Target.csv', index=False)