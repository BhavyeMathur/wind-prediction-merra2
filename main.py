from modules import *   # TODO - better import statement


east_wind = MERRA2Dataset("data/wind/YAVG", "U", is_float16=True)
const_asm = MERRA2Dataset("data/raw/MERRA2_101.const_2d_asm_Nx.00000000.nc4", "FRLAND")

plot_dataset(east_wind, time="4:30 01-01", lev=35)

# east_wind_0101 = east_wind.load(time="01:30 01-01-YAVG", lev=0)
# print(east_wind_0101.shape)
