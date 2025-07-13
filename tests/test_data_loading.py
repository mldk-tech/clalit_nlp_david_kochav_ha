import json

# Test data loading
with open('../doctor_appointment_summaries.json', 'r') as f:
    data = json.load(f)

print(f"Total records loaded: {len(data)}")
print(f"First record: {data[0]}")
print(f"Last record: {data[-1]}")

# Check for any issues
for i, record in enumerate(data):
    if not all(key in record for key in ['id', 'doctor_id', 'summary', 'future_outcome']):
        print(f"Record {i} is missing required fields: {record}")
        break
else:
    print("All records have required fields")

# Count unique doctor IDs
doctor_ids = set(record['doctor_id'] for record in data)
print(f"Unique doctor IDs: {sorted(doctor_ids)}") 