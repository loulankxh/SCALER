import numpy as np
import struct
import os

def write_numpy_header(f, dtype_str, shape):
    magic = b'\x93NUMPY'
    version = b'\x01\x00'
    header_dict = "{'descr': '%s', 'fortran_order': False, 'shape': %s, }" % (dtype_str, str(shape))
    header_length = 118
    f.write(magic)
    f.write(version)
    f.write(struct.pack('<H', header_length))
    header_bytes = header_dict.encode('ascii')
    f.write(header_bytes)
    padding = b' ' * (header_length - len(header_bytes) - 1)
    f.write(padding)
    f.write(b'\n')

def write_once(f, dtype_str, value):
    write_numpy_header(f, dtype_str, (1,))
    if 'i4' in dtype_str: f.write(struct.pack('<i', value))
    elif 'u4' in dtype_str: f.write(struct.pack('<I', value))
    elif 'u2' in dtype_str: f.write(struct.pack('<H', value))

def generate_fake_shard(path, npts, deg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'<f4 ')
        write_once(f, '<i4', 4)
        write_once(f, '<u4', npts)
        write_once(f, '<u4', 0)
        write_once(f, '<u4', deg)
        write_once(f, '<u2', 0)
        write_numpy_header(f, '<u4', (npts, deg))
        # Ensure neighbors are within [0, npts-1]
        neighbors = np.random.randint(0, npts, size=(npts, deg), dtype=np.uint32)
        f.write(neighbors.tobytes())

def setup_test():
    base_folder = 'dataset/test_shards'
    deg = 32
    
    # Find all partitions created by executeDiskPartition
    partitions = [d for d in os.listdir(base_folder) if d.startswith('partition')]
    for p in partitions:
        idmap_path = os.path.join(base_folder, p, 'idmap.ibin')
        if os.path.exists(idmap_path):
            with open(idmap_path, 'rb') as f:
                npts = struct.unpack('I', f.read(4))[0]
            
            # Use at least 'deg' points to satisfy ScaleGANN's assertion
            if npts < deg:
                print(f"Warning: shard {p} has only {npts} points, but degree is {deg}. This might still fail ScaleGANN's assertion.")
            
            generate_fake_shard(os.path.join(base_folder, p, 'index/raft_cagra'), npts, deg)
            print(f"Generated fake RAFT index for {p} with {npts} points")

if __name__ == "__main__":
    setup_test()
