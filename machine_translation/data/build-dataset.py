import os
import csv

# Open the output file in write mode
with open('m2v_trans_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Classical', 'Modern', 'Source'])  # Write header

    # Traverse the directory hierarchy
    for root, dirs, files in os.walk('Classical-Modern/双语数据/'):
        for file in files:
            if file == 'bitext.txt':
                # Read the contents of bitext.txt
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as bitext_file:
                    content = bitext_file.read()

                # Process the contents and extract sentence pairs
                sentences = content.strip().split('\n\n')
                for sentence_pair in sentences:
                    sentences_split = sentence_pair.split('\n')
                    classical = sentences_split[0].split('：')[1]
                    modern = sentences_split[1].split('：')[1]
                    source = os.path.relpath(root, 'Classical-Modern/双语数据/')

                    # Write the sentence pair directly to the output file
                    writer.writerow([classical, modern, source])
