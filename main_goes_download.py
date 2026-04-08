"""
Download GOES-16 FDCF (Full Disk fire detection) data for WildfireSpreadTS fires.

Downloads from the public AWS S3 bucket (no auth needed). Uses a date-first
strategy: for each unique date-hour across all fires, downloads one full-disk
file and crops all active fires from it. This avoids redundant downloads since
many fires overlap in time (~12 fires per date on average).

Writes output directly to GCS and checks GCS for existing files to enable
resume without local storage.

Output structure (in GCS):
  gs://{bucket}/{prefix}/{year}/{fire_name}/{date}.npz
  Each file contains:
    'data': (24, 3, H, W) float32 — 3 channels x 24 hours
    'hours': int32 array of hours with valid data
  Channels: Mask (fire confidence), Power (FRP MW), Area (km^2)

Prerequisites:
  pip install s3fs xarray h5netcdf numpy pyyaml tqdm google-cloud-storage
"""
import argparse
import datetime
import io
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import s3fs
import xarray as xr
import yaml
import tqdm
from google.cloud import storage as gcs_storage


S3_BUCKET = 'noaa-goes16'
PRODUCT = 'ABI-L2-FDCF'
GOES_VARS = ['Mask', 'Power', 'Area']

GOES_LON_ORIGIN = -75.0
SAT_HEIGHT = 35786023.0
R_EQ = 6378137.0
R_POL = 6356752.31414


def latlon_to_goes_xy(lat, lon):
    """Convert lat/lon (degrees) to GOES fixed grid x/y (radians)."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lon0_rad = np.radians(GOES_LON_ORIGIN)
    geocentric_lat = np.arctan((R_POL**2 / R_EQ**2) * np.tan(lat_rad))
    r_earth = R_POL / np.sqrt(
        1 - (R_EQ**2 - R_POL**2) / R_EQ**2 * np.cos(geocentric_lat)**2)
    sx = SAT_HEIGHT - r_earth * np.cos(geocentric_lat) * np.cos(lon_rad - lon0_rad)
    sy = -r_earth * np.cos(geocentric_lat) * np.sin(lon_rad - lon0_rad)
    sz = r_earth * np.sin(geocentric_lat)
    x = np.arcsin(-sy / np.sqrt(sx**2 + sy**2 + sz**2))
    y = np.arctan(sz / sx)
    return x, y


def build_existing_set(gcs_bucket, gcs_prefix):
    """Scan GCS for existing .npz files to skip. Returns set of 'year/fire/date' keys."""
    client = gcs_storage.Client()
    bucket = client.bucket(gcs_bucket)
    existing = set()
    print(f"Scanning gs://{gcs_bucket}/{gcs_prefix}/ for existing files...")
    for blob in bucket.list_blobs(prefix=gcs_prefix):
        if blob.name.endswith('.npz'):
            # e.g. "WildfireSpreadTS_GOES/2021/fire_123/2021-08-01.npz"
            parts = blob.name.replace(gcs_prefix + '/', '').split('/')
            if len(parts) == 3:
                year, fire_name, fname = parts
                date_str = fname.replace('.npz', '')
                existing.add((fire_name, date_str))
    print(f"  Found {len(existing)} existing files in GCS")
    return existing


def upload_to_gcs(gcs_bucket, gcs_prefix, year_label, fire_name, date_str, data, hours):
    """Write npz directly to GCS."""
    client = gcs_storage.Client()
    bucket = client.bucket(gcs_bucket)
    blob_path = f"{gcs_prefix}/{year_label}/{fire_name}/{date_str}.npz"
    blob = bucket.blob(blob_path)

    buf = io.BytesIO()
    np.savez_compressed(buf, data=data, hours=hours)
    buf.seek(0)
    blob.upload_from_file(buf)


def build_fire_index(configs):
    """Build mapping from date -> list of (fire_name, year, bbox)."""
    date_fires = defaultdict(list)

    for config in configs:
        year_label = config.get('year', 2020)
        rect_size = config.get('rectangular_size', 0.5)
        metadata_keys = {'output_bucket', 'rectangular_size', 'year'}

        for fire_name in config:
            if fire_name in metadata_keys:
                continue
            loc = config[fire_name]
            lat, lon = loc['latitude'], loc['longitude']

            x1, y1 = latlon_to_goes_xy(lat - rect_size, lon - rect_size)
            x2, y2 = latlon_to_goes_xy(lat + rect_size, lon + rect_size)
            bbox = (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))

            start, end = loc['start'], loc['end']
            buf = datetime.timedelta(days=4)
            first_day = start - buf
            last_day = end + buf

            for i in range((last_day - first_day).days + 1):
                date = first_day + datetime.timedelta(days=i)
                date_fires[date].append((fire_name, year_label, bbox))

    return date_fires


def find_goes_file(fs, year, doy, hour):
    """Find first available GOES FDCF file for a given hour."""
    prefix = f'{S3_BUCKET}/{PRODUCT}/{year}/{doy:03d}/{hour:02d}/'
    try:
        files = [f for f in fs.ls(prefix) if f.endswith('.nc')]
        return files[0] if files else None
    except (FileNotFoundError, PermissionError):
        return None


def process_one_hour(fs, s3_path, fires_for_date, hour):
    """Download one full-disk file to temp, crop for all fires."""
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.nc')
        os.close(tmp_fd)
        fs.get(s3_path, tmp_path)

        ds = xr.open_dataset(tmp_path, engine='h5netcdf')
        x_vals = ds['x'].values
        y_vals = ds['y'].values
        results = {}

        for fire_name, year_label, bbox in fires_for_date:
            x_lo, x_hi, y_lo, y_hi = bbox

            xi = np.where((x_vals >= x_lo) & (x_vals <= x_hi))[0]
            yi = np.where((y_vals >= y_lo) & (y_vals <= y_hi))[0]

            if len(xi) == 0 or len(yi) == 0:
                continue

            xs, xe = int(xi[0]), int(xi[-1]) + 1
            ys, ye = int(yi[0]), int(yi[-1]) + 1

            channels = []
            for var in GOES_VARS:
                if var in ds:
                    arr = ds[var].values[ys:ye, xs:xe].astype(np.float32)
                    arr = np.nan_to_num(arr, nan=0.0)
                else:
                    arr = np.zeros((ye - ys, xe - xs), dtype=np.float32)
                channels.append(arr)

            results[fire_name] = np.stack(channels, axis=0)

        ds.close()
        return results

    except Exception:
        return {}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description='Download GOES-16 fire data for WildfireSpreadTS fires')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                        help='Fire config YAMLs (e.g., config/us_fire_2021_1e7.yml)')
    parser.add_argument('--gcs_bucket', type=str, required=True,
                        help='GCS bucket name for output')
    parser.add_argument('--gcs_prefix', type=str, default='WildfireSpreadTS_GOES',
                        help='GCS prefix (default: WildfireSpreadTS_GOES)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel download threads (default: 4)')
    args = parser.parse_args()

    configs = []
    for cfg_path in args.configs:
        with open(cfg_path, 'r', encoding='utf8') as f:
            configs.append(yaml.load(f, Loader=yaml.FullLoader))

    fs = s3fs.S3FileSystem(anon=True)

    # Build index: date -> fires that need GOES data for that date
    print("Building fire date index...")
    date_fires = build_fire_index(configs)
    total_pairs = sum(len(v) for v in date_fires.values())
    print(f"  {len(date_fires)} unique dates, {total_pairs} fire-date pairs")

    # Check GCS for existing files
    existing = build_existing_set(args.gcs_bucket, args.gcs_prefix)

    # Filter out dates where all fires already have data in GCS
    dates_to_process = []
    for date, fires in sorted(date_fires.items()):
        date_str = date.strftime('%Y-%m-%d')
        needed = [(fn, yl, bb) for fn, yl, bb in fires
                  if (fn, date_str) not in existing]
        if needed:
            dates_to_process.append((date, needed))

    print(f"  {len(dates_to_process)} dates need processing "
          f"({len(date_fires) - len(dates_to_process)} already done)")

    n_workers = args.workers

    for date, fires in tqdm.tqdm(dates_to_process, desc='Processing dates'):
        date_str = date.strftime('%Y-%m-%d')
        doy = date.timetuple().tm_yday

        # Find all available S3 paths for this day's 24 hours
        hour_paths = {}
        for hour in range(24):
            s3_path = find_goes_file(fs, date.year, doy, hour)
            if s3_path is not None:
                hour_paths[hour] = s3_path

        # Download + crop hours in parallel
        hour_results = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(process_one_hour, fs, s3_path, fires, hour): hour
                for hour, s3_path in hour_paths.items()
            }
            for future in as_completed(futures):
                hour = futures[future]
                try:
                    hour_results[hour] = future.result()
                except Exception:
                    pass

        # Write per-fire output files directly to GCS
        for fire_name, year_label, bbox in fires:
            hourly = {}
            for hour, results in hour_results.items():
                if fire_name in results:
                    hourly[hour] = results[fire_name]

            if not hourly:
                continue

            spatial_shape = next(iter(hourly.values())).shape[1:]
            full_day = np.zeros((24, 3) + spatial_shape, dtype=np.float32)
            for h, data in hourly.items():
                if data.shape[1:] == spatial_shape:
                    full_day[h] = data

            upload_to_gcs(
                args.gcs_bucket, args.gcs_prefix,
                year_label, fire_name, date_str,
                full_day,
                np.array(sorted(hourly.keys()), dtype=np.int32),
            )

    # Summary
    final_existing = build_existing_set(args.gcs_bucket, args.gcs_prefix)
    print(f"\nDone. {len(final_existing)} total GOES day-files in "
          f"gs://{args.gcs_bucket}/{args.gcs_prefix}/")


if __name__ == '__main__':
    main()
