"""
Download ERA5-Land hourly weather data for WildfireSpreadTS fires.

Uses the Copernicus CDS API to download ERA5-Land hourly data directly,
bypassing GEE entirely. Much faster than GEE export tasks.

For each fire, downloads all days in a single CDS request (or monthly chunks
if the date range spans multiple months), then slices into per-day NetCDF files.

Variables (6):
  - 2m_temperature (K)
  - 2m_dewpoint_temperature (K)
  - 10m_u_component_of_wind (m/s)
  - 10m_v_component_of_wind (m/s)
  - total_precipitation (m)
  - surface_pressure (Pa)

Output structure:
  {output_dir}/{year}/{fire_name}/{date}.nc
  Each file: (24, 6, lat, lon) — 24 hours, 6 variables, ~10x10 grid cells

Prerequisites:
  pip install cdsapi xarray netcdf4
  Set up ~/.cdsapirc with your CDS credentials:
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
"""
import argparse
import datetime
import os
from collections import defaultdict

import cdsapi
import numpy as np
import xarray as xr
import yaml
import tqdm


CDS_DATASET = 'reanalysis-era5-land'
CDS_VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'total_precipitation',
    'surface_pressure',
]
ALL_HOURS = [f'{h:02d}:00' for h in range(24)]


def get_date_range(start, end, buffer_days=4):
    """Return list of dates from start-buffer to end+buffer."""
    buf = datetime.timedelta(days=buffer_days)
    first = start - buf
    last = end + buf
    return [first + datetime.timedelta(days=i)
            for i in range((last - first).days + 1)]


def group_dates_by_month(dates):
    """Group dates by (year, month) for chunked CDS requests."""
    groups = defaultdict(list)
    for d in dates:
        groups[(d.year, d.month)].append(d)
    return groups


def download_era5_for_fire(client, config, fire_name, output_dir, buffer_days=4):
    """Download ERA5-Land hourly data for one fire event."""
    loc = config[fire_name]
    lat = loc['latitude']
    lon = loc['longitude']
    start = loc['start']
    end = loc['end']
    rect_size = config.get('rectangular_size', 0.5)
    year = config.get('year', start.year)

    fire_dir = os.path.join(output_dir, str(year), fire_name)
    os.makedirs(fire_dir, exist_ok=True)

    dates = get_date_range(start, end, buffer_days)

    # Check which dates are already downloaded
    existing = set()
    for d in dates:
        nc_path = os.path.join(fire_dir, f'{d.strftime("%Y-%m-%d")}.nc')
        if os.path.exists(nc_path):
            existing.add(d)
    dates = [d for d in dates if d not in existing]
    if not dates:
        return 0

    # Area: [North, West, South, East]
    area = [
        lat + rect_size,
        lon - rect_size,
        lat - rect_size,
        lon + rect_size,
    ]

    # Group by month (CDS works best with monthly chunks)
    monthly = group_dates_by_month(dates)
    n_downloaded = 0

    for (yr, mo), month_dates in sorted(monthly.items()):
        day_list = sorted(set(d.day for d in month_dates))

        tmp_file = os.path.join(fire_dir, f'_tmp_{yr}_{mo:02d}.nc')

        try:
            client.retrieve(
                CDS_DATASET,
                {
                    'variable': CDS_VARIABLES,
                    'year': str(yr),
                    'month': f'{mo:02d}',
                    'day': [f'{d:02d}' for d in day_list],
                    'time': ALL_HOURS,
                    'area': area,
                    'data_format': 'netcdf',
                },
                tmp_file,
            )
        except Exception as e:
            print(f"    CDS request failed for {fire_name} {yr}-{mo:02d}: {e}")
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            continue

        # Split into per-day files
        try:
            ds = xr.open_dataset(tmp_file)
            for d in month_dates:
                day_str = d.strftime('%Y-%m-%d')
                day_data = ds.sel(time=day_str)

                # Verify we got 24 hours
                if 'time' in day_data.dims and day_data.sizes['time'] == 24:
                    out_path = os.path.join(fire_dir, f'{day_str}.nc')
                    day_data.to_netcdf(out_path)
                    n_downloaded += 1
                elif 'time' not in day_data.dims:
                    # Single timestep selected, might happen at boundaries
                    pass
                else:
                    out_path = os.path.join(fire_dir, f'{day_str}.nc')
                    day_data.to_netcdf(out_path)
                    n_downloaded += 1

            ds.close()
        except Exception as e:
            print(f"    Failed to split {fire_name} {yr}-{mo:02d}: {e}")
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    return n_downloaded


def main():
    parser = argparse.ArgumentParser(
        description='Download ERA5-Land hourly data for WildfireSpreadTS fires')
    parser.add_argument('--config', type=str, required=True,
                        help='Fire config YAML (e.g., config/us_fire_2021_1e7.yml)')
    parser.add_argument('--output_dir', type=str, default='data/era5',
                        help='Output directory (default: data/era5)')
    parser.add_argument('--buffer_days', type=int, default=4,
                        help='Days before/after fire dates (default: 4)')
    parser.add_argument('--skip_fires', type=int, default=0,
                        help='Number of fires to skip (for resuming)')
    parser.add_argument('--max_fires', type=int, default=None,
                        help='Max fires to process (default: all)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    client = cdsapi.Client()

    metadata_keys = {'output_bucket', 'rectangular_size', 'year'}
    fire_names = [k for k in config.keys() if k not in metadata_keys]
    fire_names = fire_names[args.skip_fires:]
    if args.max_fires:
        fire_names = fire_names[:args.max_fires]

    total_downloaded = 0
    for fire_name in tqdm.tqdm(fire_names, desc='Downloading ERA5'):
        try:
            n = download_era5_for_fire(
                client, config, fire_name, args.output_dir,
                buffer_days=args.buffer_days,
            )
            total_downloaded += n
            if n > 0:
                tqdm.tqdm.write(f"  {fire_name}: {n} days downloaded")
        except Exception as e:
            tqdm.tqdm.write(f"  {fire_name}: FAILED - {e}")

    print(f"\nDone. Downloaded {total_downloaded} day-files.")


if __name__ == '__main__':
    main()
