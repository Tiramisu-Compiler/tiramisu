import tensor_comprehensions as tc
import torch
import time

# TODO: This is not full sgemm spec: C <= A x B + C
lang = """
def sgemm(float(M, K) A, float(K, N) B) -> (C) {
    C(m, n) += A(m, k) * B(k, n)
}
"""

M, K, N = 3072, 3072, 3072

def test(options, n):
    # runs given options n times and returns running times
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    sgemm = tc.define(lang, name="sgemm")
    times = []
    A_, B_, C_ = torch.randn(M, K), torch.randn(K, N), torch.randn(M, N)
    print("")
    for i in range(n + 1):
        print("\033[Frunning test: {}/{}".format(i, n))
        A, B, C = A_.clone(), B_.clone(), C_.clone()
        t1 = time.perf_counter()
        A_cuda, B_cuda, C_cuda = A.cuda(), B.cuda(), C.cuda()
        torch.cuda.synchronize()
        sgemm(A_cuda, B_cuda, outputs=C_cuda, options=options)
        torch.cuda.synchronize()
        C_res = C_cuda.cpu()
        torch.cuda.synchronize()
        if i > 0: # The first run is warmup
            times.append(time.perf_counter() - t1)
    return times

def autotune(cache_file='tc_cache'):
    print("Starting autotune")
    A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()
    sgemm = tc.define(lang, name="sgemm")
    best_opts = sgemm.autotune(A, B,
            cache=cache_file,
            generations=25,
            pop_size=50,
            crossover_rate=70,
            number_elites=5,
            gpus="1,2,3")
    print("Done autotune")
    print(sorted(test(best_opts, 20))[10])
    return best_opts

def load_cache(cache_file='tc_cache'):
    A, B = torch.randn(M, K).cuda(), torch.randn(K, N).cuda()
    sgemm = tc.define(lang, name="sgemm")
    # Couldn't find a reasonable way to load cache:
    return sgemm.autotune(A, B, cache=cache_file, generations=0)

# autotune()
print("naive:", sorted(test(tc.Options("naive"), 10))[5])
print("autotuned:", sorted(test(load_cache(), 100))[50])
