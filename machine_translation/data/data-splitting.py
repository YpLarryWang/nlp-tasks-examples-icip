import os
import csv
import random
import shutil

# Read the original CSV file
data = []
with open('m2v_trans_data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip the header
    for row in reader:
        data.append(row)

# Set the random seed
seed = 42

# Use the same seed to ensure the same results
random.seed(seed)

# Shuffle the data
random.shuffle(data)

# Calculate the size of each part
total_rows = len(data)
part_size = total_rows // 10

if not os.path.exists('split_data'):
    os.mkdir('split_data')

# Create subdirectories for each part
for i in range(10):
    subdir = f'split_data/part{i + 1}'
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    # Split the data into ten parts
    start_index = i * part_size
    end_index = (i + 1) * part_size
    if i == 9:
        end_index = total_rows  # Adjust the end index for the last part

    part_data = data[start_index:end_index]

    # Split the part data into train, valid, and test sets
    train_data = part_data[:int(len(part_data) * 0.8)]
    valid_data = part_data[int(len(part_data) * 0.8):int(len(part_data) * 0.9)]
    test_data = part_data[int(len(part_data) * 0.9):]

    # Write the train data to a separate CSV file
    train_filename = os.path.join(subdir, 'train.csv')
    with open(train_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write the header
        writer.writerows(train_data)

    # Write the valid data to a separate CSV file
    valid_filename = os.path.join(subdir, 'valid.csv')
    with open(valid_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write the header
        writer.writerows(valid_data)

    # Write the test data to a separate CSV file
    test_filename = os.path.join(subdir, 'test.csv')
    with open(test_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Write the header
        writer.writerows(test_data)
