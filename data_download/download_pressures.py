import cdsapi

c = cdsapi.Client()

for year in ['1980', '1981']:
    for pressure in ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']:

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                pressure
            ],
            'year': year,
            'month': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            'day': ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
            'time': [
                '00:00', '06:00', '12:00', '18:00'
            ],
            },
        '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/' + year + '_p' + pressure  +'.nc'
        )
