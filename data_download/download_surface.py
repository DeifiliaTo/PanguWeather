import cdsapi

c = cdsapi.Client()

for year in ['1980', '1981']:

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'mean_sea_level_pressure', '10m_u_component_of_wind','10m_v_component_of_wind', '2m_temperature',
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
        '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/single_' + year + '.nc'
    )

