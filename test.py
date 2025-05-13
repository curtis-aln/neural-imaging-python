import psutil
print(f"RAM used: {psutil.virtual_memory().used / 1024**3:.2f} GB")
print(f"Input data size: {input_data.nbytes / 1024**2:.2f} MB")
