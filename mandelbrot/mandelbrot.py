

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import List, Callable, TypeVar

T = TypeVar("T")
A = TypeVar("A", bound=list)


width = 960
height = 960
MAX_ITERS = 200

min_x = -2.0
max_x = 0.6
min_y = -1.5
max_y = 1.5

def timeit(method: Callable[..., T]) -> Callable[..., T]:
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time() - start
        print(f"{method.__name__}: {end * 1e3}ms")
        return result
    return timed

def mandelbrot_kernel(value: complex) -> int:
    z = value
    for i in range(MAX_ITERS):
        z = z ** 2 + value
        if z.imag**2 + z.real**2 > 4:
            return i
    return MAX_ITERS

def mandelbrot() -> np.ndarray:
    t = np.zeros((height, width), np.float64)
    @timeit
    def mandelbrot_set():
        dx = (max_x - min_x) / width
        dy = (max_y - min_y) / height

        y = min_y
        for row in range(height):
            x = min_x
            for col in range(width):
                t[row, col] = mandelbrot_kernel(complex(x, y))
                x += dx
            y += dy
    mandelbrot_set()
    return t

def plot(numpy_array: np.ndarray) -> None:
    scale = 10
    dpi = 64
    fig = plt.figure(1, [scale, scale * height // width], dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)
    light = colors.LightSource(315, 10, 0, 1, 1, 0)
    image = light.shade(numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig("mandelbrot.png")

plot(mandelbrot())
