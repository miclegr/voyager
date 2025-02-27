import voyager
import numpy as np

params = (
    [256],
    [1024],
    [voyager.Space.Euclidean, voyager.Space.InnerProduct, voyager.Space.Cosine],
    [voyager.StorageDataType.E4M3, voyager.StorageDataType.Float8, voyager.StorageDataType.Float32],
    [24],
)

num_dimensions: int = 256
num_elements: int = 1024
space: voyager.Space = voyager.Space.Euclidean
storage_data_type: voyager.StorageDataType = voyager.StorageDataType.Float32
ef_construction: float = 24
generator = np.random.default_rng(seed=1234)
input_data = generator.random((num_elements, num_dimensions)).astype(np.float32) * 2 - 1

if storage_data_type == voyager.StorageDataType.Float8:
    input_data = np.round(input_data * 127) / 127

index = voyager.Index(
    space=space,
    num_dimensions=num_dimensions,
    ef_construction=ef_construction,
    M=20,
    storage_data_type=storage_data_type,
    random_seed=4321,
)

index.add_items(input_data, num_threads=1)

