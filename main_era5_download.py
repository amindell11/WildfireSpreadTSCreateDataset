"""
Download ERA5-Land hourly weather data for WildfireSpreadTS fires.

Uses the Copernicus CDS API to download ERA5-Land hourly data directly,
bypassing GEE entirely. Writes output directly to GCS and checks GCS
for existing files to enable resume without local storage.

Variables (6):
  - 2m_temperature (K)
  - 2m_dewpoint_temperature (K)
  - 10m_u_component_of_wind (m/s)
  - 10m_v_component_of_wind (m/s)
  - total_precipitation (m)
  - surface_pressure (Pa)

Output structure (in GCS):
  gs://{bucket}/{prefix}/{year}/{fire_name}/{date}.nc
  Each file: 6 variables x 24 hours, ~10x10 grid cells

Prerequisites:
  pip install cdsapi xarray netcdf4 cfgrib eccodes google-cloud-storage
  Set up ~/.cdsapirc with your CDS credentials
"""
import argparse
import datetime
import io
import os
import tempfile
from collections import defaultdict

import cdsapi
import numpy as np
import xarray as xr
import yaml
import tqdm
from google.cloud import storage as gcs_storage


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


def build_existing_set(gcs_bucket, gcs_prefix):
    """Scan GCS for existing .nc files. Returns set of (fire_name, date_str)."""
    client = gcs_storage.Client()
    bucket = client.bucket(gcs_bucket)
    existing = set()
    print(f"Scanning gs://{gcs_bucket}/{gcs_prefix}/ for existing files...")
    for blob in bucket.list_blobs(prefix=gcs_prefix):
        if blob.name.endswith('.nc'):
            parts = blob.name.replace(gcs_prefix + '/', '').split('/')
            if len(parts) == 3:
                year, fire_name, fname = parts
                date_str = fname.replace('.nc', '')
                existing.add((fire_name, date_str))
    print(f"  Found {len(existing)} existing files in GCS")
    return existing


def upload_nc_to_gcs(gcs_bucket, gcs_prefix, year_label, fire_name, date_str, ds):
    """Write xarray Dataset as NetCDF directly to GCS."""
    client = gcs_storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob_path = f"{gcs_prefix}/{year_label}/{fire_name}/{date_str}.nc"
    blob = bucket.blob(blob_path)

    buf = io.BytesIO()
    ds.to_netcdf(buf)
    buf.seek(0)
    blob.upload_from_file(buf)


def download_era5_for_fire(client, config, fire_name, gcs_bucket, gcs_prefix,
                           existing_set, buffer_days=4):
    """Download ERA5-Land hourly data for one fire event, upload to GCS."""
    loc = config[fire_name]
    lat = loc['latitude']
    lon = loc['longitude']
    start = loc['start']
    end = loc['end']
    rect_size = config.get('rectangular_size', 0.5)
    year = config.get('year', start.year)

    dates = get_date_range(start, end, buffer_days)

    # Filter out dates already in GCS
    dates = [d for d in dates
             if (fire_name, d.strftime('%Y-%m-%d')) not in existing_set]
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

        tmp_fd, tmp_file = tempfile.mkstemp(suffix='.grib')
        os.close(tmp_fd)

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
                    'download_format': 'unarchived',
                },
                tmp_file,
            )
        except Exception as e:
            print(f"    CDS request failed for {fire_name} {yr}-{mo:02d}: {e}")
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            continue

        try:
            # Detect format from header
            with open(tmp_file, 'rb') as fh:
                header = fh.read(8)

            if header[:4] == b'GRIB':
                ds = xr.open_dataset(tmp_file, engine='cfgrib')
            elif header[:4] == b'\x89HDF':
                ds = xr.open_dataset(tmp_file, engine='h5netcdf')
            elif header[:3] == b'CDF':
                ds = xr.open_dataset(tmp_file, engine='netcdf4')
            elif header[:2] == b'PK':
                import zipfile
                extract_dir = tmp_file + '_ext'
                os.makedirs(extract_dir, exist_ok=True)
                with zipfile.ZipFile(tmp_file, 'r') as zf:
                    zf.extractall(extract_dir)
                nc_files = [os.path.join(extract_dir, f)
                            for f in os.listdir(extract_dir)
                            if f.endswith('.nc')]
                ds = xr.open_dataset(nc_files[0])
            else:
                ds = xr.open_dataset(tmp_file)

            # cfgrib uses 'valid_time' instead of 'time'
            time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

            for d in month_dates:
                day_str = d.strftime('%Y-%m-%d')
                day_data = ds.sel({time_dim: day_str})

                n_times = day_data.sizes.get(time_dim, 0)
                if n_times >= 1:
                    upload_nc_to_gcs(gcs_bucket, gcs_prefix, year,
                                     fire_name, day_str, day_data)
                    n_downloaded += 1

            ds.close()
        except Exception as e:
            print(f"    Failed to split {fire_name} {yr}-{mo:02d}: {e}")
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            # Clean up extract dir if it exists
            extract_dir = tmp_file + '_ext'
            if os.path.exists(extract_dir):
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)

    return n_downloaded


def main():
    parser = argparse.ArgumentParser(
        description='Download ERA5-Land hourly data for WildfireSpreadTS fires')
    parser.add_argument('--config', type=str, required=True,
                        help='Fire config YAML (e.g., config/us_fire_2021_1e7.yml)')
    parser.add_argument('--gcs_bucket', type=str, required=True,
                        help='GCS bucket name for output')
    parser.add_argument('--gcs_prefix', type=str, default='WildfireSpreadTS_ERA5',
                        help='GCS prefix (default: WildfireSpreadTS_ERA5)')
    parser.add_argument('--buffer_days', type=int, default=4,
                        help='Days before/after fire dates (default: 4)')
    parser.add_argument('--skip_fires', type=int, default=0,
                        help='Number of fires to skip (for resuming)')
    parser.add_argument('--max_fires', type=int, default=None,
                        help='Max fires to process (default: all)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cds_client = cdsapi.Client()

    # Check GCS for existing files
    existing = build_existing_set(args.gcs_bucket, args.gcs_prefix)

    metadata_keys = {'output_bucket', 'rectangular_size', 'year'}
    fire_names = [k for k in config.keys() if k not in metadata_keys]
    fire_names = fire_names[args.skip_fires:]
    if args.max_fires:
        fire_names = fire_names[:args.max_fires]

    total_downloaded = 0
    for fire_name in tqdm.tqdm(fire_names, desc='Downloading ERA5'):
        try:
            n = download_era5_for_fire(
                cds_client, config, fire_name,
                args.gcs_bucket, args.gcs_prefix,
                existing, buffer_days=args.buffer_days,
            )
            total_downloaded += n
            if n > 0:
                tqdm.tqdm.write(f"  {fire_name}: {n} days downloaded")
        except Exception as e:
            tqdm.tqdm.write(f"  {fire_name}: FAILED - {e}")

    print(f"\nDone. Downloaded {total_downloaded} day-files to "
          f"gs://{args.gcs_bucket}/{args.gcs_prefix}/")


if __name__ == '__main__':
    main()
