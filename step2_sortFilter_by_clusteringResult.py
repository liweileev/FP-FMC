import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Sort Filters by Clustering Result.')

parser.add_argument(
    '--clustering_dir',
    type=str,
    required=True,
    help='load the clustering results from the specified directory')
parser.add_argument(
    '--num_clustering',
    type=int,
    default=128,
    help='How many clustering results are used for sorting.')
args = parser.parse_args()

arch = args.clustering_dir.split("/")[-3]
files = os.listdir(args.clustering_dir)
files.sort()
print(arch)

for file in files:
    filename = file.split('.')[0]
    print(f"process: {filename}.")
    clusters = np.load(args.clustering_dir+file)[0:args.num_clustering,:]
    frequency = np.zeros(clusters.shape[1], dtype=int)

    for (idx1,cluster1) in enumerate(clusters):
        for (idx2, cluster2) in enumerate(clusters):
            if idx1==idx2 :
                pass
            else:
                # generate sets of each cluster
                partition1 = []
                partition2 = []
                for i in np.sort(np.unique(cluster1)):
                    partition1.append(np.where(cluster1 == i)[0])
                    partition2.append(np.where(cluster2 == i)[0])
                # print("partition1:", partition1)
                # print("partition2:", partition2)

                # get the max intersection for each set
                maxInterNum_by_cluster = []
                maxInter = []
                for x in partition1:
                    _max = 0
                    _set = np.array([])
                    for y in partition2:
                        intersect_count = np.intersect1d(x, y).shape[0]
                        if intersect_count == _max:
                            _set = np.append(_set, y)
                        elif intersect_count > _max:
                            _max = intersect_count
                            _set = np.array(y)
                    maxInterNum_by_cluster.append(_max)
                    maxInter.append(_set)

                # set value to each filter
                out = [0] * cluster1.shape[0]
                for (clusterid, (x, y, z)) in enumerate(zip(partition1, maxInterNum_by_cluster, maxInter)):
                    if clusterid==0:
                        # label the outliers as -1
                        for _n in x:
                            out[_n] = -1
                    else:
                        # label the others: the number of intersection elements
                        for _n in x:
                            if (_n in z) and (y > out[_n]):
                                out[_n] = y
                frequency += out
                # print(maxInterNum_by_cluster)
                # print(maxInter)

    os.makedirs(f"output/step2_sort/{arch}/samples={args.num_clustering}/", exist_ok=True)
    np.save(f"output/step2_sort/{arch}/samples={args.num_clustering}/{filename}_R.npy", frequency)
    np.save(f"output/step2_sort/{arch}/samples={args.num_clustering}/{filename}.npy", np.argsort(frequency))
#     # print(frequency)
#     # print(np.argsort(frequency))