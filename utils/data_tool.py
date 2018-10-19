import numpy as np


def generate_batch_indices_shuffle(total_data_len, batch_size):
    batch_num = np.ceil(total_data_len / batch_size)
    batch_num = np.int(batch_num)

    indices = np.arange(total_data_len)
    np.random.shuffle(indices)
    for start_i in range(batch_num):
        end_i = start_i + 1
        if end_i == batch_num:
            yield indices[start_i*batch_size:]
        else:
            yield indices[start_i*batch_size: end_i*batch_size]


def generate_batch_indices_sequence(total_data_len, batch_size):
    batch_num = np.ceil(total_data_len / batch_size)
    batch_num = np.int(batch_num)

    while True:
        indices = np.arange(total_data_len)

        for start_i in range(batch_num):
            end_i = start_i + 1
            if end_i == batch_num:
                yield indices[start_i * batch_size:]
            else:
                yield indices[start_i * batch_size: end_i * batch_size]


def create_validation_split(num_data,  num_validation, seed=None):
    if isinstance(num_validation, float):
        num_validation = int(num_validation * num_data)

    random_generator = np.random.RandomState(seed=seed)
    validation_y = random_generator.choice(
        num_data, num_validation, replace=False)

    validation_mask = np.full((num_data,), False, bool)
    validation_mask[validation_y] = True
    training_mask = np.logical_not(validation_mask)

    return np.where(training_mask)[0], validation_y


def data_scale(data, scale):
    return np.multiply(data, scale)


