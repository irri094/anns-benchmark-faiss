import matplotlib.pyplot as plt
import numpy as np
import subprocess

saiz = 1

def make_plots(dataset, k):
    # read results/dataset-FLATL2.csv
    dir = 'results/'+dataset+'-FLATL2.csv'
    dataflat = np.genfromtxt(dir, delimiter=',')
    # get rows where last column is k
    dataflat = dataflat[dataflat[:, -1] == k]
    # read results/dataset-LSH.csv
    dir = 'results/'+dataset+'-LSH.csv'
    datalsh = np.genfromtxt(dir, delimiter=',')
    datalsh = datalsh[datalsh[:, -1] == k]
    # read results/dataset-IVFPQ.csv
    dir = 'results/'+dataset+'-IVFPQ.csv'
    dataivfpq = np.genfromtxt(dir, delimiter=',')
    dataivfpq = dataivfpq[dataivfpq[:, -1] == k]
    # read results/dataset-HNSW.csv
    dir = 'results/'+dataset+'-HNSW.csv'
    datahnsw = np.genfromtxt(dir, delimiter=',')
    datahnsw = datahnsw[datahnsw[:, -1] == k]
    # read results/dataset-IVFHNSW.csv
    dir = 'results/'+dataset+'-IVFHNSW.csv'
    dataivfhnsw = np.genfromtxt(dir, delimiter=',')
    dataivfhnsw = dataivfhnsw[dataivfhnsw[:, -1] == k]
    # read results/dataset-IVFADC.csv
    dir = 'results/'+dataset+'-IVFADC.csv'
    dataivfadc = np.genfromtxt(dir, delimiter=',')
    dataivfadc = dataivfadc[dataivfadc[:, -1] == k]
    # read results/dataset-IMIADC.csv
    dir = 'results/'+dataset+'-IMIADC.csv'
    dataimiadc = np.genfromtxt(dir, delimiter=',')
    dataimiadc = dataimiadc[dataimiadc[:, -1] == k]

    # rewrite the above code with s = saiz parameter
    plt.scatter(dataflat[:, 3], dataflat[:, 1], label='FLATL2', s=saiz)
    plt.scatter(datalsh[:, 4], datalsh[:, 2], label='LSH', s=saiz)
    plt.scatter(dataivfpq[:, 5], dataivfpq[:, 3], label='IVFADC', s=saiz)
    plt.scatter(datahnsw[:, 6], datahnsw[:, 4], label='HNSW', s=saiz)
    plt.scatter(dataivfhnsw[:, 6], dataivfhnsw[:, 4], label='IVFHNSW', s=saiz)
    plt.scatter(dataimiadc[:, 3], dataimiadc[:, 1], label='IMIADC', s=saiz)

    plt.xlabel('Recall')
    plt.ylabel('QPS (1/s)')
    plt.title('Recall vs QPS for '+str(dataset)+' dataset with k='+str(k))
    plt.legend()
    plt.savefig('plots/'+str(dataset)+'-@'+str(k)+'-recall-vs-qps.png')
    plt.clf()


    # rewrite the above code with s = saiz parameter
    plt.scatter(dataflat[:, 3], dataflat[:, 0], label='FLATL2', s=saiz)
    plt.scatter(datalsh[:, 4], datalsh[:, 1], label='LSH', s=saiz)
    plt.scatter(dataivfpq[:, 5], dataivfpq[:, 2], label='IVFADC', s=saiz)
    plt.scatter(datahnsw[:, 6], datahnsw[:, 3], label='HNSW', s=saiz)
    plt.scatter(dataivfhnsw[:, 6], dataivfhnsw[:, 3], label='IVFHNSW', s=saiz)
    plt.scatter(dataimiadc[:, 3], dataimiadc[:, 0], label='IMIADC', s=saiz)


    plt.xlabel('Recall')
    plt.ylabel('Train Time (s)')
    plt.title('Recall vs Train Time for '+str(dataset)+' dataset with k='+str(k))
    plt.legend()
    plt.savefig('plots/'+str(dataset)+'-@'+str(k)+'-recall-vs-train-time.png')
    plt.clf()


    # rewrite the above code with s = saiz parameter
    plt.scatter(dataflat[:, 3], dataflat[:, 2], label='FLATL2', s=saiz)
    plt.scatter(datalsh[:, 4], datalsh[:, 3], label='LSH', s=saiz)
    plt.scatter(dataivfpq[:, 5], dataivfpq[:, 4], label='IVFADC', s=saiz)
    plt.scatter(datahnsw[:, 6], datahnsw[:, 5], label='HNSW', s=saiz)
    plt.scatter(dataivfhnsw[:, 6], dataivfhnsw[:, 5], label='IVFHNSW', s=saiz)
    plt.scatter(dataimiadc[:, 3], dataimiadc[:, 2], label='IMIADC', s=saiz)

    plt.xlabel('Recall')
    plt.ylabel('Index Size (MB)')
    plt.title('Recall vs Index Size for '+dataset+' dataset with k='+str(k))
    plt.legend()
    

    plt.savefig('plots/'+str(dataset)+'-@'+str(k)+'-recall-vs-index-size.png')
    plt.clf()

if __name__ == '__main__':
    import sys
    saiz = int(sys.argv[1])
    subprocess.call(['rm', '-rf', 'plots'])
    subprocess.call(['mkdir', 'plots'])
    for k in [50, 100]:
        for dataset in ['sift', 'gist', 'mnist']:
            make_plots(dataset, k) 
