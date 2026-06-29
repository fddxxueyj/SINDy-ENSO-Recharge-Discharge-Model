import os
import urllib.request
from urllib.error import HTTPError, URLError

save_dir = r"E:\data"
os.makedirs(save_dir, exist_ok=True)

base_url = "https://ftp.cpc.ncep.noaa.gov/godas/monthly"

start_year = 1980
end_year = 2025

whether_download = True

if whether_download:
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):

            yyyymm = f"{year}{month:02d}"

            for suffix in [".grb", ".grb.inv"]:

                filename = f"godas.M.{yyyymm}{suffix}"
                save_path = os.path.join(save_dir, filename)

                if os.path.isfile(save_path):
                    print(f"Skip: {filename}")
                    continue

                url = f"{base_url}/{filename}"

                try:
                    print(f"Downloading: {filename}")
                    urllib.request.urlretrieve(url, save_path)
                    print(f"OK: {filename}")

                except HTTPError as e:
                    print(f"HTTP Error {e.code}: {filename}")

                except URLError as e:
                    print(f"URL Error: {filename}")
                    print(e.reason)

                except Exception as e:
                    print(f"FAILED: {filename}")
                    print(e)