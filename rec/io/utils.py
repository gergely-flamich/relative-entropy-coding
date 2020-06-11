import numpy as np
import struct

from rec.io.entropy_coding import ArithmeticCoder


def write_compressed_code(file_path,
                          seed,
                          image_shape,
                          block_size,
                          block_indices,
                          max_index,
                          num_aux_var_counts_file=None,
                          index_counts_file=None):
    if len(image_shape) != 3:
        raise ValueError(f"Image shape must be rank 3, but was {image_shape}!")

    img_h, img_w, img_c = image_shape

    num_res_blocks = len(block_indices)

    # We need to store the sizes of the blocks
    num_blocks = list(map(len, block_indices))

    # Get the number of auxiliary variables drawn in each block
    num_aux_vars = [list(map(len, block)) for block in block_indices]

    block_indices_flattened = [np.concatenate(block, axis=0) for block in block_indices]

    # If we don't have counts, we assume a uniform probability mass
    if index_counts_file is None:
        index_counts = np.ones(max_index + 1, dtype=np.int32)

        # This makes all symbols relative to the EOF symbol much more likely
        index_counts[1:] += 1000
    else:
        index_counts = np.load(index_counts_file)

    if num_aux_var_counts_file is None:
        num_aux_var_counts = []
        num_aux_var_maxes = []

        for nav in num_aux_vars:
            nav_max = np.max(nav)
            counts = np.ones(nav_max + 2, dtype=np.int32)
            counts[1:] += 100

            num_aux_var_counts.append(counts)
            num_aux_var_maxes.append(int(nav_max))

    else:
        num_aux_var_counts = np.load(num_aux_var_counts_file, allow_pickle=True)
        num_aux_var_maxes = [-1] * num_res_blocks

    num_aux_var_coders = [ArithmeticCoder(nav_counts, precision=32) for nav_counts in num_aux_var_counts]
    index_coder = ArithmeticCoder(index_counts, precision=32)

    def to_message(msg):
        return np.concatenate([np.array(msg) + 1, [0]], axis=0)

    def from_message(msg):
        return np.array(msg)[:-1] - 1

    # Note the leading 1s: this is to ensure that if we have leading 0s in the original codes, we do not lose
    # them during int conversion
    num_aux_var_codes = ['1' + ''.join(nav_coder.encode(to_message(nav))) for nav, nav_coder in
                         zip(num_aux_vars, num_aux_var_coders)]
    index_codes = ['1' + ''.join(index_coder.encode(to_message(index))) for index in block_indices_flattened]

    # Number of bytes required to store the codes
    num_aux_var_codelengths = [len(c) // 8 + (1 if len(c) % 8 != 0 else 0) for c in num_aux_var_codes]
    index_codelengths = [len(c) // 8 + (1 if len(c) % 8 != 0 else 0) for c in index_codes]

    print(f"Num res blocks: {num_res_blocks}")
    print(f"Block sizes: {num_blocks}")
    print(f"num aux var lengths: {num_aux_var_codelengths}")
    print(f"index codelengths: {index_codelengths}")

    print(f"Static header size: {struct.calcsize(f'IIIIIHHHH')}")

    header_format = f'IIIIIHHHH{num_res_blocks}I{num_res_blocks}I{num_res_blocks}I{num_res_blocks}I'
    header = struct.pack(header_format,
                         seed,
                         block_size,
                         max_index,
                         img_h,
                         img_w,
                         img_c,
                         1 - int(num_aux_var_counts_file is None),
                         1 - int(index_counts_file is None),
                         num_res_blocks,
                         *num_blocks,
                         *num_aux_var_codelengths,
                         *index_codelengths,
                         *num_aux_var_maxes)

    with open(file_path, 'wb') as rec_file:
        rec_file.write(header)

        for nav_code, nav_codelength in zip(num_aux_var_codes, num_aux_var_codelengths):
            nav_bytes = int(nav_code, 2).to_bytes(length=nav_codelength, byteorder='big')
            rec_file.write(nav_bytes)

        for index_code, index_codelength in zip(index_codes, index_codelengths):
            index_bytes = int(index_code, 2).to_bytes(length=index_codelength, byteorder='big')
            rec_file.write(index_bytes)


def read_compressed_code(file_path,
                         static_header_size=28,
                         num_aux_var_counts_file=None,
                         index_counts_file=None
                         ):
    """
    The static header size is determined using calcsize() in the function above
    :param file_path:
    :param static_header_size:
    :return:
    """

    with open(file_path, 'rb') as rec_file:
        header = rec_file.read(static_header_size)

        header_info = struct.unpack(f'IIIIIHHHH', header)

        seed = header_info[0]
        block_size = header_info[1]
        max_index = header_info[2]
        image_shape = tuple(header_info[3:6])
        use_num_aux_var_counts_file = bool(header_info[6])
        use_index_counts_file = bool(header_info[7])
        num_res_blocks = header_info[8]

        if use_index_counts_file and index_counts_file is None:
            raise ValueError("The compressed file is using empirical index counts, but no counts file was supplied!")

        if use_num_aux_var_counts_file and use_num_aux_var_counts_file is None:
            raise ValueError(
                "The compressed file is using empirical num_aux_var counts, but no counts file was supplied!")

        dynamic_header_format = f"{num_res_blocks}I{num_res_blocks}I{num_res_blocks}I{num_res_blocks}I"
        dynamic_header_bytes = struct.calcsize(dynamic_header_format)

        dynamic_header = rec_file.read(dynamic_header_bytes)

        dynamic_header_info = struct.unpack(dynamic_header_format, dynamic_header)

        num_blocks = dynamic_header_info[0:num_res_blocks]
        num_aux_var_codelengths = dynamic_header_info[num_res_blocks: 2 * num_res_blocks]
        index_codelengths = dynamic_header_info[2 * num_res_blocks: 3 * num_res_blocks]
        num_aux_var_maxes = dynamic_header_info[3 * num_res_blocks:]

        num_aux_var_codes = []
        index_codes = []

        # Read in the number of aux variable codes in each block
        for i in range(num_res_blocks):
            nav_code = rec_file.read(num_aux_var_codelengths[i])
            nav_code = int.from_bytes(nav_code, byteorder='big')
            nav_code = bin(nav_code)[3:]

            num_aux_var_codes.append(nav_code)

        # Read in the index codes in each block
        for i in range(num_res_blocks):
            index_code = rec_file.read(index_codelengths[i])
            index_code = int.from_bytes(index_code, byteorder='big')
            index_code = bin(index_code)[3:]

            index_codes.append(index_code)

    # We now create the arithmetic coders to decode the num_aux_vars and indices
    if use_index_counts_file:
        index_counts = np.load(index_counts_file)
    else:
        index_counts = np.ones(max_index + 1, dtype=np.int32)

        # This makes all symbols relative to the EOF symbol much more likely
        index_counts[1:] += 1000

    if use_num_aux_var_counts_file:
        num_aux_var_counts = np.load(num_aux_var_counts_file, allow_pickle=True)
    else:
        num_aux_var_counts = []

        for nav_max in num_aux_var_maxes:
            counts = np.ones(nav_max + 2, dtype=np.int32)
            counts[1:] += 100

            num_aux_var_counts.append(counts)

    # Create the arithmetic coders
    num_aux_var_coders = [ArithmeticCoder(nav_counts, precision=32) for nav_counts in num_aux_var_counts]
    index_coder = ArithmeticCoder(index_counts, precision=32)

    def from_message(msg):
        return np.array(msg)[:-1] - 1

    num_aux_vars = [from_message(nav_coder.decode_fast(nav_code))
                    for nav_coder, nav_code in zip(num_aux_var_coders, num_aux_var_codes)]

    block_indices_flattened = [from_message(index_coder.decode_fast(index_code)) for index_code in index_codes]

    # Recover block indices
    block_indices = []
    for block_num_aux_vars, block_index_vec in zip(num_aux_vars, block_indices_flattened):
        indices = []

        index_bounds = np.cumsum(np.concatenate([[0], block_num_aux_vars], axis=0))

        for i in range(1, len(index_bounds)):
            indices.append(block_index_vec[index_bounds[i - 1]:index_bounds[i]].tolist())

        block_indices.append(indices)

    return seed, image_shape, block_size, block_indices