def conv_result_size(size, kernel, stride, padding, n=1):
    for _ in range(n):
        size = (size - kernel + padding * 2 + stride) // stride
    return size
