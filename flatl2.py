import time
import faiss
import h5py
import os
import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    k = int(sys.argv[2])
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

    faiss_index = faiss.IndexFlatL2(d)

    print(faiss_index.is_trained)
    # start = time.time()
    # faiss_index.train(xb)
    # end = time.time()
    # print("time to train ", end-start)

    start = time.time()
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
    filename = 'results/' + dataset +'-FLATL2' + '.csv'
    with open(filename, 'a') as fd:
        fd.write(str(traintime) + ',' + str(qps) + ',' + str(memory_in_MB) + ',' + str(recall_score) + ',' + str(k) +'\n')
