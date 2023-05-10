import time
import faiss
import h5py
import os
import sys

if __name__ == '__main__':

    dataset = sys.argv[1]
    nbits = int(sys.argv[2])
    k = int(sys.argv[3])

    print(dataset, nbits, k, " LSH")

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

    # nbits = int(d/2)
    # res = faiss.StandardGpuResources() 
    faiss_index = faiss.IndexLSH(d, nbits)
    # faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    start = time.time()    
    if not faiss_index.is_trained:
        # print("training")
        faiss_index.train(xb)
    else:
        # print("is trained")
        pass

    faiss_index.add(xb)
    end = time.time()
    traintime = end - start
    print("time to train and add is ", traintime)


    start = time.time()
    D, I = faiss_index.search(xq, k)
    end = time.time()
    querytime = end - start
    print("time to search is ", querytime)

    cnt = 0
    for i in range(0, len(xq)):
        for j in range(0, k):
            if I[i][j] in gtI[i][0:k]:
                cnt += 1

    recall_score = cnt / (len(xq) * k)
    print("recall score is ", recall_score)


    faiss.write_index(faiss_index, './temp.index')
    filesz = os.path.getsize('./temp.index') / (1024 * 1024)
    os.remove('./temp.index')
    print("index size = ", filesz, " MB")

    qps = len(xq) / querytime
    dataset = dataset.split('-')[0]
    # append row in csv file
    filename = 'results/' + dataset +'-LSH' + '.csv'
    with open(filename, 'a') as fd:
        fd.write(str(nbits) + ',' + str(traintime) + ',' + str(qps) + ',' + str(filesz) + ',' + str(recall_score)+ ',' + str(k) +'\n')

