import subprocess

subprocess.call(['mkdir', '-p', 'results'])

filename = 'sift-128-euclidean.hdf5'

for ncentroid in [256, 512, 1024]:
    for pqparam in [32, 64]:
        for nprobe in [30, 50, 70]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'ivfadc.py', filename, str(ncentroid), str(pqparam), str(nprobe), str(k)])
                if ret != 0:
                    print("error with ivfadc.py")

for pqparam in [32, 64]:
    for nprobe in [30, 50, 70]:
        for k in [50, 100]:
            ret = subprocess.call(['python3', 'imiadc.py', filename, str(pqparam), str(nprobe), str(k)])
            if ret != 0:
                print("error with imiadc.py")

filename = 'gist-960-euclidean.hdf5'

for ncentroid in [256, 512, 1024]:
    for pqparam in [32, 64]:
        for nprobe in [30, 50, 70]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'ivfadc.py', filename, str(ncentroid), str(pqparam), str(nprobe), str(k)])
                if ret != 0:
                    print("error with ivfadc.py")

for pqparam in [32, 64]:
    for nprobe in [30, 50, 70]:
        for k in [50, 100]:
            ret = subprocess.call(['python3', 'imiadc.py', filename, str(pqparam), str(nprobe), str(k)])
            if ret != 0:
                print("error with imiadc.py")

filename = 'mnist-784-euclidean.hdf5'

for ncentroid in [256, 512]:
    for pqparam in [16, 28]:
        for nprobe in [30, 50, 70]:
            for k in [50, 100]:
                ret = subprocess.call(['python3', 'ivfadc.py', filename, str(ncentroid), str(pqparam), str(nprobe), str(k)])
                if ret != 0:
                    print("error with ivfadc.py")

for pqparam in [16, 28]:
    for nprobe in [30, 50, 70]:
        for k in [50, 100]:
            ret = subprocess.call(['python3', 'imiadc.py', filename, str(pqparam), str(nprobe), str(k)])
            if ret != 0:
                print("error with imiadc.py")
