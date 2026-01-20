import struct
import os

label_file = r'd:\University\BS CS (5th)\Maching Learning\Project\ML Project-Antigravity\Emnist\train-labels.idx1-ubyte'

try:
    with open(label_file, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        print(f"Magic: {magic}, Num labels: {num}")
        
        # Read first 1000 labels to get a sense
        labels = list(f.read(num))
        print(f"Min label: {min(labels)}")
        print(f"Max label: {max(labels)}")
        print(f"Unique labels: {set(labels)}")
except Exception as e:
    print(f"Error reading file: {e}")
