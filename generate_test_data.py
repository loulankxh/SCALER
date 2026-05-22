import numpy as np
import struct
import os

def save_u8bin(data, filename):
    npts, ndim = data.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('II', npts, ndim))
        f.write(data.astype(np.uint8).tobytes())

def save_ibin(data, filename):
    npts, ndim = data.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('II', npts, ndim))
        f.write(data.astype(np.uint32).tobytes())

def generate_data(npts=1000, ndim=128, nqueries=10, k=10):
    os.makedirs('dataset/test_data', exist_ok=True)
    
    # Base data
    base_data = np.random.randint(0, 255, size=(npts, ndim), dtype=np.uint8)
    save_u8bin(base_data, 'dataset/test_data/base.u8bin')
    
    # Query data (use first nqueries from base for perfect recall test)
    query_data = base_data[:nqueries]
    save_u8bin(query_data, 'dataset/test_data/query.u8bin')
    gt_indices = np.zeros((nqueries, k), dtype=np.uint32)
    for i in range(nqueries):
        gt_indices[i, 0] = i
        for j in range(1, k):
            gt_indices[i, j] = (i + j) % npts
            
    save_ibin(gt_indices, 'dataset/test_data/groundtruth.ibin')
    print(f"Generated test data in dataset/test_data/")

if __name__ == "__main__":
    generate_data()
