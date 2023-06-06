import dpnp, numpy, dpctl.tensor as dpt

size_1d, size_2d = (10**7,), (10**4, 10**4)
# size_1d, size_2d = (10**6,), (10**3, 10**3)
na = numpy.random.uniform(-10**4, 10**4, size=size_1d)
nb = numpy.random.uniform(-10**4, 10**4, size=size_1d)

types = [dpnp.bool, dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64, dpnp.complex64, dpnp.complex128]
# types = [dpnp.bool, dpnp.int32, dpnp.int64, dpnp.float32, dpnp.complex64]
devices = ['cpu', 'opencl:gpu', 'level_zero:gpu']

a = dpnp.array(na)
b = dpnp.array(nb)

# build all kernels
for d in devices:
    for dt in types:
        a = dpnp.array(na, dtype=dt, device=d)
        b = dpnp.array(nb, dtype=dt, device=d)
        dpnp_r = dpnp.divide(a, b, legacy=True)

        a = dpt.asarray(na, dtype=dt, device=d)
        b = dpt.asarray(nb, dtype=dt, device=d)
        dpt_r = dpt.divide(a, b)

# start measurements
print("\n=================== Test 1d ===================")
for dt in types:
    print("\n\ntype =", numpy.dtype(dt))
    for d in devices:
        print("\ndevice =", d)
        a = dpnp.array(na, dtype=dt, device=d)
        b = dpnp.array(nb, dtype=dt, device=d)
        print("dpnp leagcy time")
        %timeit _ = dpnp.divide(a, b, legacy=True)

        print("dpnp new impl time")
        %timeit _ = dpnp.divide(a, b, legacy=False)

        a = dpt.asarray(na, dtype=dt, device=d)
        b = dpt.asarray(nb, dtype=dt, device=d)
        print("dpt time")
        %timeit _ = dpt.divide(a, b)

    print("\nnumpy time")
    na_ = numpy.array(na, dtype=dt)
    nb_ = numpy.array(nb, dtype=dt)
    %timeit _ = numpy.divide(na_, nb_)

print("\n=================== Test 1d reverse ===================")
for dt in types:
    print("\n\ntype =", numpy.dtype(dt))
    for d in devices:
        print("\ndevice =", d)
        a = dpnp.array(na, dtype=dt, device=d)
        b = dpnp.array(nb, dtype=dt, device=d)
        print("dpnp leagcy time")
        %timeit _ = dpnp.divide(a[::-1], b[::-1], legacy=True)

        print("dpnp new impl time")
        %timeit _ = dpnp.divide(a[::-1], b[::-1], legacy=False)

        a = dpt.asarray(na, dtype=dt, device=d)
        b = dpt.asarray(nb, dtype=dt, device=d)
        print("dpt time")
        %timeit _ = dpt.divide(a[::-1], b[::-1])

    print("\nnumpy time")
    na_ = numpy.array(na, dtype=dt)
    nb_ = numpy.array(nb, dtype=dt)
    %timeit _ = numpy.divide(na_[::-1], nb_[::-1])

print("\n=================== Test 2d ===================")
na = numpy.random.uniform(-10**4, 10**4, size=size_2d)
nb = numpy.random.uniform(-10**4, 10**4, size=size_2d)
for dt in types:
    print("\n\ntype =", numpy.dtype(dt))
    for d in devices:
        print("\ndevice =", d)
        a = dpnp.array(na, dtype=dt, device=d)
        b = dpnp.array(nb, dtype=dt, device=d)
        print("dpnp leagcy time")
        %timeit _ = dpnp.divide(a, b, legacy=True)

        print("dpnp new impl time")
        %timeit _ = dpnp.divide(a, b, legacy=False)

        a = dpt.asarray(na, dtype=dt, device=d)
        b = dpt.asarray(nb, dtype=dt, device=d)
        print("dpt time")
        %timeit _ = dpt.divide(a, b)

    print("\nnumpy time")
    na_ = numpy.array(na, dtype=dt)
    nb_ = numpy.array(nb, dtype=dt)
    %timeit _ = numpy.divide(na_, nb_)

print("\n=================== Test 2d transpose ===================")
for dt in types:
    print("\n\ntype =", numpy.dtype(dt))
    for d in devices:
        print("\ndevice =", d)
        a = dpnp.array(na, dtype=dt, device=d)
        b = dpnp.array(nb, dtype=dt, device=d)
        print("dpnp leagcy time")
        %timeit _ = dpnp.divide(a.get_array().mT, b.get_array().mT, legacy=True)

        print("dpnp new impl time")
        %timeit _ = dpnp.divide(a.get_array().mT, b.get_array().mT, legacy=False)

        a = dpt.asarray(na, dtype=dt, device=d)
        b = dpt.asarray(nb, dtype=dt, device=d)
        print("dpt time")
        %timeit _ = dpt.divide(a.mT, b.mT)

    print("\nnumpy time")
    na_ = numpy.array(na, dtype=dt)
    nb_ = numpy.array(nb, dtype=dt)
    %timeit _ = numpy.divide(na_.T, nb_.T)

print("\n=================== Test 2d with scalat (broadcast) ===================")
for dt in types:
    print("\n\ntype =", numpy.dtype(dt))
    for d in devices:
        print("\ndevice =", d)
        a = dpnp.array(na, dtype=dt, device=d)
        print("dpnp leagcy time")
        %timeit _ = dpnp.divide(a, 7, legacy=True)

        print("dpnp new impl time")
        %timeit _ = dpnp.divide(a, 7, legacy=False)

        a = dpt.asarray(na, dtype=dt, device=d)
        print("dpt time")
        %timeit _ = dpt.divide(a, 7)

    print("\nnumpy time")
    na_ = numpy.array(na, dtype=dt)
    %timeit _ = numpy.divide(na_, 7)
