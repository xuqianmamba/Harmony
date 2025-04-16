import numpy as np

def save_fvecs(filename, vectors):
    with open(filename, 'wb') as f:
        for vector in vectors:
            f.write(np.array([len(vector)], dtype=np.int32).tobytes())
            f.write(vector.astype(np.float32).tobytes())

dimensions = [64, 128, 256, 512]
sizes = [250_000, 500_000, 750_000, 1_000_000]

for dim in dimensions:
    for size in sizes:
        vectors = np.random.normal(loc=0.0, scale=1.0, size=(size, dim))

        filename = f'gaussian_{dim}dim_{size}.fvecs'
        save_fvecs(filename, vectors)
        
        print(f"Saved {size} vectors of dimension {dim} to {filename}")
