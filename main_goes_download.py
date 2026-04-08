"""
Download GOES-16 FDCF (Full Disk fire detection) data for WildfireSpreadTS fires.

Downloads from the public AWS S3 bucket (no auth needed). Uses a date-first
strategy: for each unique date-hour across all fires, downloads one full-disk
file and crops all active fires from it. This avoids redundant downloads since
many fires overlap in time (~12 fires per date on average).

Each full-disk file is ~3MB and covers the entire Western Hemisphere.
1,123 unique dates x 24 hours = ~27K files, ~79GB raw, but streamed and
discarded after cropping.

Output structure:
  {output_dir}/{year}/{fire_name}/{date}.npz
  Each file contains:
    'data': (24, 3, H, W) float32 — 3 channels x 24 hours
    'hours': int32 array of hours with valid data
  Channels: Mask (fire confidence), Power (FRP MW), Area (km^2)

Prerequisites:
  pip install s3fs xarray h5netcdf numpy pyyaml tqdm
"""
import argparse
import datetime
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import s3fs
import xarray as xr
import yaml
import tqdm


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


def build_fire_index(configs):
    """Build mapping from date -> list of (fire_name, year, bbox) that need data.

    Returns:
        date_fires: dict mapping date -> list of (fire_name, year_label,
                    (x_lo, x_hi, y_lo, y_hi), fire_dir)
    """
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


def process_one_hour(fs, s3_path, fires_for_date, hour, output_dir):
    """Download one full-disk file to temp, crop for all fires.

    Downloads to a temp file first, then opens locally for fast partial reads.
    This is ~2x faster than streaming from S3 with .load().

    Args:
        fires_for_date: list of (fire_name, year_label, bbox)

    Returns:
        dict mapping fire_name -> (3, H, W) array for this hour
    """
    tmp_path = None
    try:
        # Download to temp file for fast partial reads
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
    parser.add_argument('--output_dir', type=str, default='data/goes',
                        help='Output directory (default: data/goes)')
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
    print(f"  {len(date_fires)} unique dates, "
          f"{sum(len(v) for v in date_fires.values())} fire-date pairs")

    # Filter out dates where all fires already have output files
    dates_to_process = []
    for date, fires in sorted(date_fires.items()):
        date_str = date.strftime('%Y-%m-%d')
        needed = []
        for fire_name, year_label, bbox in fires:
            fire_dir = os.path.join(args.output_dir, str(year_label), fire_name)
            out_path = os.path.join(fire_dir, f'{date_str}.npz')
            if not os.path.exists(out_path):
                needed.append((fire_name, year_label, bbox))
        if needed:
            dates_to_process.append((date, needed))

    print(f"  {len(dates_to_process)} dates need processing "
          f"({len(date_fires) - len(dates_to_process)} already done)")

    # Accumulators: fire_name -> date_str -> {hour: (3, H, W)}
    fire_hourly = defaultdict(lambda: defaultdict(dict))
    fire_meta = {}  # fire_name -> year_label

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
        hour_results = {}  # hour -> {fire_name: (3,H,W)}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(process_one_hour, fs, s3_path, fires, hour,
                            args.output_dir): hour
                for hour, s3_path in hour_paths.items()
            }
            for future in as_completed(futures):
                hour = futures[future]
                try:
                    hour_results[hour] = future.result()
                except Exception:
                    pass

        # Write per-fire output files for this date
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

            fire_dir = os.path.join(args.output_dir, str(year_label), fire_name)
            os.makedirs(fire_dir, exist_ok=True)
            out_path = os.path.join(fire_dir, f'{date_str}.npz')
            np.savez_compressed(
                out_path,
                data=full_day,
                hours=np.array(sorted(hourly.keys()), dtype=np.int32),
            )

    # Summary
    total_files = 0
    for root, dirs, files in os.walk(args.output_dir):
        total_files += sum(1 for f in files if f.endswith('.npz'))
    print(f"\nDone. {total_files} total GOES day-files in {args.output_dir}")


if __name__ == '__main__':
    main()
