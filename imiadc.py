import time
import faiss
import h5py
import os
import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    pqparam = int(sys.argv[2])
    nprobe = int(sys.argv[3])
    k = int(sys.argv[4])

    print(dataset, pqparam, nprobe, k, " IMIADC")

    fulpath = 'datasets/' + dataset

    f = h5py.File(fulpath, 'r')

    xb = f['train'][:]
    xq = f['test'][:]
    gtD = f['distances'][:]
    gtI = f['neighbors'][:]

    # set xq as first element of xq
    # xq = xq[0:1]

    # print("xb shape is ", xb.shape)
    # print("xq shape is ", xq.shape)
    # print("gtD shape is ", gtD.shape)
    # print("gtI shape is ", gtI.shape)

    d = xb.shape[1]

    # imi 2x8 means 2^(16) (most probably)
    consparam = "OPQ"+str(pqparam)+",IMI2x8,"+"PQ"+str(pqparam)    
    # IVF X _ HNSW Y => IVF has X centroids, hnsw has maxdeg = Y
    faiss_index = faiss.index_factory(d, consparam)
    imi = faiss.extract_index_ivf(faiss_index)
    imi.nprobe = nprobe

    print(faiss_index.is_trained)
    # start = time.time()
    # faiss_index.train(xb)
    # end = time.time()
    # print("time to train ", end-start)

    start = time.time()
    if not faiss_index.is_trained:
        faiss_index.train(xb)
    faiss_index.add(xb)
    end = time.time()
    traintime = end - start
    print("time to train and add is ", end - start)
    # print("faiss_index.ntotal is ", faiss_index.ntotal)
    # print(faiss_index.is_trained)

    start = time.time()
    D, I = faiss_index.search(xq, k)
    end = time.time()
    querytime = end - start
    print("time to search is ", end - start)

    cnt = 0
    for i in range(0, len(xq)):
        for j in range(0, k):
            if I[i][j] in gtI[i][0:k]:
                cnt += 1

    recall_score = cnt / (len(xq) * k)
    print("recall score is ", recall_score)

    memory_in_MB = faiss_index.ntotal * faiss_index.d * 4 / (1024*1024)

    qps = len(xq) / querytime
    dataset = dataset.split('-')[0]
    # append row in csv file
    filename = 'results/' + dataset +'-IMIADC' + '.csv'
    with open(filename, 'a') as fd:
        fd.write( str(traintime) + ',' + str(qps) + ',' + str(memory_in_MB) + ',' + str(recall_score) + ','  + str(pqparam)+','+ str(nprobe)+','+ str(k) +'\n')