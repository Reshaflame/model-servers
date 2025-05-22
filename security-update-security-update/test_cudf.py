import cudf

print("[cuDF] Testing cuDF DataFrame creation on GPU...")

# Create a simple GPU DataFrame
df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df)

print("[cuDF] ✅ cuDF is working and using GPU!")
