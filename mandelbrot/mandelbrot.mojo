from benchmark import Benchmark
from complex import ComplexSIMD, ComplexFloat64
from sys import argv
from math import iota
from python import Python
from runtime.llcl import num_cores, Runtime
from algorithm import parallelize, vectorize
from tensor import Tensor
from utils.index import Index

alias float_type = DType.float64
alias simd_width = simdwidthof[float_type]()

alias width = 960
alias height = 960
alias MAX_ITERS = 200

alias min_x = -2.0
alias max_x = 0.6
alias min_y = -1.5
alias max_y = 1.5

def show_plot(tensor: Tensor[float_type], filename: String = "mandelbrot.png"):
    alias scale = 10
    alias dpi = 64

    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")
    colors = Python.import_module("matplotlib.colors")

    numpy_array = np.zeros((height, width), np.float64)

    for row in range(height):
        for col in range(width):
            numpy_array.itemset((col, row), tensor[col, row])

    fig = plt.figure(1, [scale, scale * height // width], dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], False, 1)
    light = colors.LightSource(315, 10, 0, 1, 1, 0)

    image = light.shade(numpy_array, plt.cm.hot, colors.PowerNorm(0.3), "hsv", 0, 0, 1.5)
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(filename)

fn mandelbrot_kernel(c: ComplexFloat64) -> Int:
    var z = c
    for i in range(MAX_ITERS):
        z = z * z + c
        if z.squared_norm() > 4:
            return i
    return MAX_ITERS


fn trivial(filename: String = "mandelbrot_trivial.png"):
    # create a matrix. Each element of the matrix corresponds to a pixel
    var t = Tensor[float_type](height, width)
    @parameter
    fn bench():
        let dx = (max_x - min_x) / width
        let dy = (max_y - min_y) / height

        var y = min_y
        for row in range(height):
            var x = min_x
            for col in range(width):
                t[Index(row, col)] = mandelbrot_kernel(ComplexFloat64(x, y))
                x += dx
            y += dy

    let trivial = Benchmark().run[bench]() / 1e6
    
    try:
        _ = show_plot(t, filename)
    except e:
        print("failed to show plot:", e.value)
        

    print(filename, ":", trivial, "ms")

fn mandelbrot_kernel_SIMD[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[float_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    var z = ComplexSIMD[float_type, simd_width](0, 0)
    var iters = SIMD[float_type, simd_width](0)

    var in_set_mask: SIMD[DType.bool, simd_width] = True
    for i in range(MAX_ITERS):
        if not in_set_mask.reduce_or():
            break
        in_set_mask = z.squared_norm() <= 4
        iters = in_set_mask.select(iters + 1, iters)
        z = z.squared_add(c)

    return iters

fn vectorized(filename: String = "mandelbrot_vectorized.png"):
    let t = Tensor[float_type](height, width)

    @parameter
    fn worker(row: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we oeprate on a `simd_width` vector of pixels."""
            let cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            let cy = min_y + row * scale_y
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            t.data().simd_store[simd_width](row * width + col, mandelbrot_kernel_SIMD[simd_width](c))

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[simd_width, compute_vector](width)

    @parameter
    fn bench[simd_width: Int]():
        for row in range(height):
            worker(row)

    let vectorized = Benchmark().run[bench[simd_width]]() / 1e6
    print(filename, ":", vectorized, "ms")

    try:
        _ = show_plot(t, filename)
    except e:
        print("failed to show plot:", e.value)

fn parallelized(filename: String = "mandelbrot_parallel.png"):
    let t = Tensor[float_type](height, width)

    @parameter
    fn worker(row: Int):
        let scale_x = (max_x - min_x) / width
        let scale_y = (max_y - min_y) / height

        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we oeprate on a `simd_width` vector of pixels."""
            let cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            let cy = min_y + row * scale_y
            let c = ComplexSIMD[float_type, simd_width](cx, cy)
            t.data().simd_store[simd_width](row * width + col, mandelbrot_kernel_SIMD[simd_width](c))

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[simd_width, compute_vector](width)

    
    with Runtime() as rt:
        @parameter
        fn bench_parallel[simd_width: Int]():
            parallelize[worker](rt, height, 5 * num_cores())

        alias simd_width = simdwidthof[DType.float64]()
        let parallelized = Benchmark().run[bench_parallel[simd_width]]() / 1e6
        print(filename, ":", parallelized, "ms")

    try:
        _ = show_plot(t, filename)
    except e:
        print("failed to show plot:", e.value)

fn main():
    let args = argv()
    var s: String = ""
    if len(args) == 2:
        s = String(args[1]) + "_"
    trivial(s + "mandelbrot_trivial.png")
    vectorized(s + "mandelbrot_vectorized.png")
    parallelized(s + "mandelbrot_parallel.png")
    