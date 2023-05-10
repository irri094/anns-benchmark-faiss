import subprocess

subprocess.call(['mkdir', '-p', 'results'])

filename = 'sift-128-euclidean.hdf5'

subprocess.call(['python3', 'flatl2.py', filename, str(50)])
subprocess.call(['python3', 'flatl2.py', filename, str(100)])

for nbits in [32, 64, 128, 256, 512, 1024, 2048]:
    for k in [50, 100]:
        ret = subprocess.call(['python3', 'lsh.py', filename, str(nbits), str(k)])
        if ret != 0:
            print("error with lsh.py at", nbits, k)

for ncenter in [512, 1024, 2048, 4096]:
    for nprobe in [10, 50, 100]:
        for k in [50, 100]:
            ret = subprocess.call(['python3', 'ivfpq.py', filename, str(ncenter), str(nprobe), str(k)])
            if ret != 0:
                print("error with ivfpq.py at", ncenter, nprobe, k)

for M in [8, 16, 32]:
    for efCons in [200, 400]:
        for efSear in [20, 40, 80]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'hnsw.py', filename, str(M), str(efCons), str(efSear), str(k)])
                if ret != 0:
                    print("error with hnsw.py at", M, efCons, efSear, k)

for ncenter in [512, 1024, 2048]:
    for M in [16, 32]:
        for nprobe in [30, 60]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'ivfhnsw.py', filename, str(ncenter), str(M), str(nprobe), str(k)])
                if ret != 0:
                    print("error with ivfhnsw.py at", ncenter, M, nprobe, k)

filename = 'gist-960-euclidean.hdf5'

subprocess.call(['python3', 'flatl2.py', filename, str(50)])
subprocess.call(['python3', 'flatl2.py', filename, str(100)])

for nbits in [256, 512, 1024, 2048, 4096]:
    for k in [50, 100]:
        ret = subprocess.call(['python3', 'lsh.py', filename, str(nbits), str(k)])
        if ret != 0:
            print("error with lsh.py at", nbits, k)

for ncenter in [512, 1024, 2048, 4096]:
    for nprobe in [10, 50, 100]:
        for k in [50, 100]:
            ret = subprocess.call(['python3', 'ivfpq.py', filename, str(ncenter), str(nprobe), str(k)])
            if ret != 0:
                print("error with ivfpq.py at", ncenter, nprobe, k)

for M in [16, 32]:
    for efCons in [100, 200]:
        for efSear in [20, 40, 80]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'hnsw.py', filename, str(M), str(efCons), str(efSear), str(k)])
                if ret != 0:
                    print("error with hnsw.py at", M, efCons, efSear, k)

for ncenter in [512, 1024, 2048]:
    for M in [16, 32]:
        for nprobe in [30, 60]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'ivfhnsw.py', filename, str(ncenter), str(M), str(nprobe), str(k)])
                if ret != 0:
                    print("error with ivfhnsw.py at", ncenter, M, nprobe, k)

filename = 'mnist-784-euclidean.hdf5'

subprocess.call(['python3', 'flatl2.py', filename, str(50)])
subprocess.call(['python3', 'flatl2.py', filename, str(100)])

for nbits in [32, 64, 128, 256, 512, 1024, 2048]:
    for k in [50, 100]:
        ret = subprocess.call(['python3', 'lsh.py', filename, str(nbits), str(k)])
        if ret != 0:
            print("error with lsh.py at", nbits, k)

for ncenter in [512, 1024, 2048, 4096]:
    for nprobe in [10, 50, 100]:
        for k in [50, 100]:
            ret = subprocess.call(['python3', 'ivfpq.py', filename, str(ncenter), str(nprobe), str(k)])
            if ret != 0:
                print("error with ivfpq.py at", ncenter, nprobe, k)

for M in [8, 16, 32]:
    for efCons in [200, 400]:
        for efSear in [20, 40, 80]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'hnsw.py', filename, str(M), str(efCons), str(efSear), str(k)])
                if ret != 0:
                    print("error with ivfhnsw.py at", M, efCons, efSear, k)

for ncenter in [512, 1024, 2048, 4096]:
    for M in [16, 32]:
        for nprobe in [30, 60]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'ivfhnsw.py', filename, str(ncenter), str(M), str(nprobe), str(k)])
                if ret != 0:
                    print("error with ivfhnsw.py at", ncenter, M, nprobe, k)