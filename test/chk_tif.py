import rasterio

file_path = 'input.tif'

with rasterio.open(file_path) as src:
    # Setting masked=True automatically applies a mask over the -99999 NoData values
    band1_masked = src.read(1, masked=True)
    
    # Calculate directly using numpy masked array methods
    valid_count = band1_masked.count()
    total_sum = band1_masked.sum()

print(f"Total valid pixels : {valid_count}")
print(f"Sum of valid pixels: {total_sum}")
