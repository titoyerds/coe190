import rasterio
from s2dr3 import inferutils

# 1. specify parameters
lonlat = (122.859543, 8.141217)   # your AOI centre
date   = '2025-10-21'      # image date
out_dir = '/path/to/output'

# 2. run inference
inferutils.test(lonlat=lonlat, date=date, output_dir=out_dir)

# 3. load result
res_path = f"{out_dir}/MS_{lonlat[0]}_{lonlat[1]}_{date}.tif"
with rasterio.open(res_path) as src:
    img = src.read()        # shape: bands × height × width
    profile = src.profile

# 4. integrate into your segmentation pipeline
# e.g., convert bands to RGB, prepare input tile, etc.

# 5. optional: profile check
print("Resolution:", profile['transform'][0], profile['transform'][4])