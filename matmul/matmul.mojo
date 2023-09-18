from benchmark import Benchmark
from sys.intrinsics import strided_load
from utils.list import VariadicList
from math import div_ceil, min
from memory import memset_zero
from memory.unsafe import DTypePointer
from random import rand, random_float64
from sys.info import simdwidthof
from os import getenv
from runtime.llcl import Runtime
from python.python import Python
from algorithm import vectorize
from algorithm import parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import vectorize_unroll
from autotune import autotune, search
from time import now
from memory.unsafe import Pointer

# TODO: make generic as soon as it is supported
# alias matmul_fn_sig_type = fn[DATATYPE](Matrix[datatype], Matrix[datatype], Matrix[datatype], Runtime) -> None
alias DATATYPE = DType.float32
alias matmul_fn_sig_type = fn(Matrix[DATATYPE], Matrix[DATATYPE], Matrix[DATATYPE], Runtime) -> None

struct Matrix[datatype: DType]:
    var rows: Int
    var cols: Int
    var data: DTypePointer[datatype]

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = DTypePointer[datatype].alloc(cols * rows)
        memset_zero(self.data, cols * rows)

    fn __del__(owned self):
        self.data.free()
    
    fn zero(inout self):
        memset_zero(self.data, self.cols * self.rows)

    fn randomize(inout self):
        rand(self.data, self.cols * self.rows)
    
    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> SIMD[datatype, 1]:
        return self.load[1](row, col)

    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) -> SIMD[datatype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)
    
    
    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: SIMD[datatype, 1]):
        return self.store[1](y, x, val)

    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[datatype, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)


    fn __str__(inout self):
        for row in range(self.rows):
            var s = String("")
            for col in range(self.cols):
                s += String(self[col, row]) + "        "
            print(s)
    
    

fn ljust(s: String, n: Int) -> String:
    var acc = s
    for _ in range(n - len(s)):
        acc += String(" ")
    return acc

fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)
@always_inline
fn benchmark[
    datatype: DType,
    func: fn (Matrix[datatype], Matrix[datatype], Matrix[datatype]) -> None,
    prefix: String
](M: Int, N: Int, K: Int, base_gflops: Float64):
    var C = Matrix[datatype](M, N)
    C.zero()
    var A = Matrix[datatype](M, K)
    var B = Matrix[datatype](K, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    # Prevent the matrices from being freed before the benchmark run
    _ = (A, B, C)
    let gflops = ((2 * M * N * K) / secs) / 1e9
    let speedup: Float64 = gflops / base_gflops
    # print(gflops, "GFLOP/s", speedup, " speedup")
    print(ljust(prefix, 15) + ": ", gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")


@always_inline
fn benchmark_parallel[
    datatype: DType,
    func: fn (Matrix[datatype], Matrix[datatype], Matrix[datatype], Runtime) -> None,
    prefix: String
](M: Int, N: Int, K: Int, base_gflops: Float64):
    var C = Matrix[datatype](M, N)
    C.zero()
    var A = Matrix[datatype](M, K)
    var B = Matrix[datatype](K, N)

    with Runtime() as rt:
        @always_inline
        @parameter
        fn test_fn():
            _ = func(C, A, B, rt)

        let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
        # Prevent the matrices from being freed before the benchmark run
        _ = (A, B, C)
        let gflops = ((2 * M * N * K) / secs) / 1e9
        let speedup: Float64 = gflops / base_gflops
        # print(gflops, "GFLOP/s", speedup, " speedup")
        print(ljust(prefix, 15) + ": ", gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")

fn matmul_naive[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype]):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

fn matmul_vectorized_0[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype]):
    alias nelts = simdwidthof[datatype]()
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols, nelts):
                C.store[nelts](m,nv, C.load[nelts](m,nv) + A[m,k] * B.load[nelts](k,nv))
        
            for n in range(nelts*(C.cols//nelts), C.cols):
                C[m,n] += A[m,k] * B[k,n]

fn matmul_vectorized_1[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype]):
    alias nelts = simdwidthof[datatype]()
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)

fn matmul_parallelized[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype], rt: Runtime):
    alias nelts = simdwidthof[datatype]()
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)
        
    parallelize[calc_row](rt, C.rows)

fn matmul_tiled_parallelized[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype], rt: Runtime):
    alias nelts = simdwidthof[datatype]()
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n + x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))
                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](rt, C.rows)

fn matmul_tiled_unrolled_parallelized[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype], rt: Runtime):
    alias nelts = simdwidthof[datatype]()
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n+x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x//nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts*tile_size, tile_size](A.cols, C.cols)
      
    parallelize[calc_row](rt, C.rows)

@adaptive
fn matmul_autotune_impl[datatype: DType](C: Matrix[datatype], A: Matrix[datatype], B: Matrix[datatype], rt: Runtime):
    alias nelts = simdwidthof[datatype]()
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n+x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        # Instead of hardcoding to tile_size = 4, search for the fastest 
        # tile size by evaluting this function as tile size varies.
        alias tile_size = autotune(1, 2, 4, 8, 16, 32)
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)
      
    parallelize[calc_row](rt, C.rows)

# TODO: make generic as soon as it is supported
# fn matmul_evaluator[datatype: DType](funcs: Pointer[matmul_fn_sig_type[datatype]], size: Int) -> Int:
fn matmul_evaluator[datatype: DType](funcs: Pointer[matmul_fn_sig_type], size: Int) -> Int:
    print("matmul_evaluator, number of candidates: ", size)

    let eval_begin: Int = now()

    # This size is picked at random, in real code we could use a real size
    # distribution here.
    let M = 512
    let N = 512
    let K = 512
    print("Optimizing for size:", M, "x", N, "x", K)

    var best_idx: Int = -1
    var best_time: Int = -1

    alias eval_iterations = 10
    alias eval_samples = 10
    # TODO: make generic as soon as it is supported
    # var C = Matrix[datatype](M, N)
    # var A = Matrix[datatype](M, K)
    # var B = Matrix[datatype](K, N)
    # let Cptr = Pointer[Matrix[datatype]].address_of(C).address
    # let Aptr = Pointer[Matrix[datatype]].address_of(A).address
    # let Bptr = Pointer[Matrix[datatype]].address_of(B).address
    var C = Matrix[DATATYPE](M, N)
    var A = Matrix[DATATYPE](M, K)
    var B = Matrix[DATATYPE](K, N)
    let Cptr = Pointer[Matrix[DATATYPE]].address_of(C).address
    let Aptr = Pointer[Matrix[DATATYPE]].address_of(A).address
    let Bptr = Pointer[Matrix[DATATYPE]].address_of(B).address
    with Runtime() as rt:
        # Find the function that's the fastest on the size we're optimizing for
        for f_idx in range(size):
            let func = funcs.load(f_idx)

            @always_inline
            @parameter
            fn wrapper():
                func(C, A, B, rt)
            let cur_time = Benchmark(1, 100_000, 500_000_000, 1000_000_000).run[wrapper]()

            if best_idx < 0:
                best_idx = f_idx
                best_time = cur_time
            if best_time > cur_time:
                best_idx = f_idx
                best_time = cur_time

        let eval_end: Int = now()
        # Prevent matrices from being destroyed before we finished benchmarking them.
        _ = A.data
        _ = B.data
        _ = C.data
        print("Time spent in matmul_evaluator, ms:", (eval_end - eval_begin) // 1000000)
        print("Best candidate idx:", best_idx)
        return best_idx

# TODO: make generic as soon as it is supported
fn matmul_autotune[datatype: DType](C: Matrix[DATATYPE], A: Matrix[DATATYPE], B: Matrix[DATATYPE], rt: Runtime):
    alias best_impl: matmul_fn_sig_type
    search[
        matmul_fn_sig_type,
        VariadicList(matmul_autotune_impl[DATATYPE].__adaptive_set),
        matmul_evaluator[DATATYPE] -> best_impl
    ]()
    # Run the best candidate
    return best_impl(C, A, B, rt)

fn main() raises:
    Python.add_to_path(getenv("PWD"))
    let python_matmul: PythonObject = Python.import_module("matmul")
    let python_gflops = python_matmul.benchmark_matmul_python(128, 128, 128).to_float64()
    let numpy_gflops = python_matmul.benchmark_matmul_numpy(128, 128, 128, python_gflops).to_float64()
    benchmark[DATATYPE, matmul_naive[DATATYPE], "naive"](512, 512, 512, python_gflops)
    benchmark[DATATYPE, matmul_vectorized_0[DATATYPE], "vectorized_0"](512, 512, 512, python_gflops)
    benchmark[DATATYPE, matmul_vectorized_1[DATATYPE], "vectorized_1"](512, 512, 512, python_gflops)
    benchmark_parallel[DATATYPE, matmul_parallelized[DATATYPE], "parallel"](512, 512, 512, python_gflops)
    benchmark_parallel[DATATYPE, matmul_tiled_parallelized[DATATYPE], "tiled"](512, 512, 512, python_gflops)
    benchmark_parallel[DATATYPE, matmul_tiled_unrolled_parallelized[DATATYPE], "unrolled"](512, 512, 512, python_gflops)
    benchmark_parallel[DATATYPE, matmul_autotune[DATATYPE], "autotune"](512, 512, 512, python_gflops)
