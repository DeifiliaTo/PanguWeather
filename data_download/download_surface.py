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
            'month': [ '1', '2', '3', '4', '5', '6'],
            'day': ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],
            'time': [
                '00:00', '06:00', '12:00', '18:00'
            ],
        },
        '/hkfs/work/workspace/scratch/ke4365-pangu/PANGU_ERA5_data_v0/single_' + year +'_0' + '.nc'
    )

