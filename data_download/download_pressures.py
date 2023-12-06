import cdsapi

c = cdsapi.Client()

for year in ['1980', '1981']:

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
                '1000','925','850','700','600','500','400','300','250','200','150','100','50',
            ],
            'year': year,
            'month': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            'day': [
                '19','20'
            ],
            'time': [
                '00:00', '06:00', '12:00', '18:00'
            ],
        },
        '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/' + year + '.nc'
    )