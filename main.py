from modules import *   # TODO - better import statement


east_wind = MERRA2Dataset("data/wind/YAVG", "U", is_float16=True)

constants_ds = "data/raw/MERRA2_101.const_2d_asm_Nx.00000000.nc4"
fraction_land = MERRA2Dataset(constants_ds, "FRLAND")

plot_dataset(east_wind, time="01:30 01-01", lev=35, projection=projections.PlateCarree(),
             nightshade=True, coastlines=True)
