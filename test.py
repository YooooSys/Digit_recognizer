import time

for i in range(10):
    print(f"Processing step {i + 1}... \n \n", flush=True)
    time.sleep(1)  # Simulate a delay
print("Script completed!")