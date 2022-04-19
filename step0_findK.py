import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

from models.vgg import vgg_16_bn
from models.googlenet import googlenet
from models.densenet import densenet
from models.resnet import resnet_56,resnet_110
import utils.common as utils

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings 
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Args
parser = argparse.ArgumentParser(description='Find the appropriate K for feature maps of each conv layer.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/data2/dataset/',
    help='dataset path')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn','googlenet','densenet','resnet_56','resnet_110'),
    help='The architecture to prune')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./pretrain/',
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--batch_calcK',
    type=int,
    default=1,
    help='The num of batch to cluster feature maps.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size for finding best K.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
args = parser.parse_args()

# Device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    
# Model
print('==> Building model..')
net = eval(args.arch)()
net = net.to(device)
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

if args.pretrain_dir:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if args.arch=='vgg_16_bn' or args.arch=='resnet_56':
        # checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)
        checkpoint = torch.load(args.pretrain_dir+args.arch+'.pt', map_location=device)
    else:
        checkpoint = torch.load(args.pretrain_dir+args.arch+'.pt')
    
    if args.arch=='densenet' or args.arch=='resnet_110':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(checkpoint['state_dict'])
else:
    print('please speicify a pretrain model ')
    raise NotImplementedError

# Global variable
samples = args.batch_size*args.batch_calcK
np_samples = (samples) * np.ones(samples, dtype=int)
batch_idx = 0
bestKs = np.array([], dtype=int)
bestK = -1

# get output feature map of certain layer via hook
def get_feature_hook(self, input, output):
    global batch_idx
    global bestKs
    global bestK

    B,C,W,H = output.shape
    output_max_bymap, _ = torch.max(output.view(B, C, W*H), dim=2, keepdim=True) 
    output_norm_bymap = (output.view(B, C, W*H)/output_max_bymap).nan_to_num() # avoid 0/0
    
    for (i, b) in enumerate(range(B)):
        # T-SNE to 2D space
        featureMap_numpy = output_norm_bymap[b].cpu().numpy()
        featureMap_embedded = TSNE(n_components=2, learning_rate='auto',perplexity=1.0).fit_transform(featureMap_numpy)
        # remove the outlier if exist
        outliers = utils.detect_outliers(featureMap_embedded)
        featureMap_embedded_rest = np.delete(featureMap_embedded, outliers, axis=0)
        # find the most appropriate K
        ss = []
        if featureMap_embedded_rest.shape[0]>=64:
            max_K = featureMap_embedded_rest.shape[0]//2
        else:
            max_K = featureMap_embedded_rest.shape[0]
        for k in range(2, max_K):
            clustering = KMeans(n_clusters=k).fit(featureMap_embedded_rest)
            ss.append(silhouette_score(featureMap_embedded_rest, clustering.labels_))
        current_bestK = np.argmax(np.array(ss))+2
        bestKs = np.append(bestKs, current_bestK)
        print(f"{batch_idx*B+i+1}/{B*args.batch_calcK} best K: {current_bestK}")

# get output feature map of certain layer via hook
def get_feature_hook_googlenet(self, input, output):
    global batch_idx
    global bestKs
    global bestK
    global googlenet_start
    global googlenet_end

    output_ = output[:,googlenet_start:googlenet_end, :, :]
    B,C,W,H = output_.shape
    output_max_bymap, _ = torch.max(output_.view(B, C, W*H), dim=2, keepdim=True) 
    output_norm_bymap = (output_.view(B, C, W*H)/output_max_bymap).nan_to_num() # avoid 0/0
    
    for (i, b) in enumerate(range(B)):
        # T-SNE to 2D space
        featureMap_numpy = output_norm_bymap[b].cpu().numpy()
        featureMap_embedded = TSNE(n_components=2, learning_rate='auto',perplexity=1.0).fit_transform(featureMap_numpy)
        # remove the outlier if exist
        outliers = utils.detect_outliers(featureMap_embedded)
        featureMap_embedded_rest = np.delete(featureMap_embedded, outliers, axis=0)
        # find the most appropriate K
        ss = []
        for k in range(2,featureMap_embedded_rest.shape[0]//2):
            clustering = KMeans(n_clusters=k).fit(featureMap_embedded_rest)
            ss.append(silhouette_score(featureMap_embedded_rest, clustering.labels_))
        current_bestK = np.argmax(np.array(ss))+2
        bestKs = np.append(bestKs, current_bestK)
        print(f"{batch_idx*B+i+1}/{B*args.batch_calcK} best K: {current_bestK}")


# get output feature map of certain layer via hook
def get_feature_hook_densenet(self, input, output):
    global batch_idx
    global bestKs
    global bestK

    output_ = output[:,output.shape[1]-12:, :, :]
    B,C,W,H = output_.shape
    output_max_bymap, _ = torch.max(output_.view(B, C, W*H), dim=2, keepdim=True) 
    output_norm_bymap = (output_.view(B, C, W*H)/output_max_bymap).nan_to_num() # avoid 0/0
    
    for (i, b) in enumerate(range(B)):
        # T-SNE to 2D space
        featureMap_numpy = output_norm_bymap[b].cpu().numpy()
        featureMap_embedded = TSNE(n_components=2, learning_rate='auto',perplexity=1.0).fit_transform(featureMap_numpy)
        # remove the outlier if exist
        outliers = utils.detect_outliers(featureMap_embedded)
        featureMap_embedded_rest = np.delete(featureMap_embedded, outliers, axis=0)
        # find the most appropriate K
        ss = []
        for k in range(2,featureMap_embedded_rest.shape[0]):
            clustering = KMeans(n_clusters=k).fit(featureMap_embedded_rest)
            ss.append(silhouette_score(featureMap_embedded_rest, clustering.labels_))
        current_bestK = np.argmax(np.array(ss))+2
        bestKs = np.append(bestKs, current_bestK)
        print(f"{batch_idx*B+i+1}/{B*args.batch_calcK} best K: {current_bestK}")


def inference():
    net.eval()
    global batch_idx

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use the first {limit} batches to estimate the average clustering results.
            if batch_idx >= args.batch_calcK:
                # output the classification precision
                break
            
            net(inputs.to(device))

print('==> Start clustering..')
print(f'{args.batch_calcK} batch ({samples} inputs) to find appropriate K for k-means..')

# googlenet global variable
googlenet_start = 0
googlenet_end = 0

if args.arch=='vgg_16_bn':
    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg
    
    Ks = np.array([], dtype=int)
    # calculate by the feature maps
    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        print(bestKs)
        bestK = np.argmax(np.bincount(np.array(bestKs)))
        print(f"for conv {i+1}/{len(relucfg)}: the most appropriate K is {bestK}")
        print("====================")
        Ks = np.append(Ks,bestK)

        # save the bestKs to csv file
        os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{i+1}", exist_ok=True)
        
        bestks_samples = np.column_stack((bestKs, np_samples))
        np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{i+1}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
        
        # reset
        bestKs = np.array([], dtype=int)
        bestK = -1
    
    # save the best Ks to the disk
    print(Ks)
    np.save(f'output/step0_appropriateK/{args.arch}/samples={samples}.npy', Ks)

elif args.arch=='googlenet':
    Ks = np.array([], dtype=int)
    # calculate by the feature maps
    cov_list=['pre_layers',
              'inception_a3',
              'maxpool1',
              'inception_a4',
              'inception_b4',
              'inception_c4',
              'inception_d4',
              'maxpool2',
              'inception_a5',
              'inception_b5',
              ]

    # branch type
    tp_list=['n1x1','n3x3','n5x5','pool_planes']
    for idx, cov in enumerate(cov_list):
        cov_layer=eval('net.'+cov)
        if idx>0:
            for idx1,tp in enumerate(tp_list):
                if idx1==3:
                    googlenet_start = sum(net.filters[idx-1][:-1])
                    googlenet_end = sum(net.filters[idx-1][:])
                    
                    handler = cov_layer.register_forward_hook(get_feature_hook_googlenet)
                    inference()
                    handler.remove()
                    
                else:
                    googlenet_start = sum(net.filters[idx-1][:idx1])
                    googlenet_end = sum(net.filters[idx-1][:idx1+1])
                    
                    handler = cov_layer.register_forward_hook(get_feature_hook_googlenet)
                    inference()
                    handler.remove()
                
                print(bestKs)
                bestK = np.argmax(np.bincount(np.array(bestKs)))
                print(f"for conv {1+(idx-1)*4+idx1}/37: the most appropriate K is {bestK}")
                print("====================")
                Ks = np.append(Ks,bestK)

                # save the bestKs to csv file
                os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{(idx+1)}{tp}", exist_ok=True)
                
                bestks_samples = np.column_stack((bestKs, np_samples))
                np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{(idx+1)}{tp}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
                
                # reset
                bestKs = np.array([], dtype=int)
                bestK = -1
        else:
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {idx+1}/37: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)

            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{idx+1}", exist_ok=True)
            
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{idx+1}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1
    
    # save the all best Ks to the disk
    print(Ks)
    np.save(f'output/step0_appropriateK/{args.arch}/samples={samples}.npy', Ks)

elif args.arch=='densenet':
    Ks = np.array([], dtype=int)
    # Densenet per block & transition
    for i in range(3):
        dense = eval('net.dense%d' % (i + 1))
        for j in range(12):
            cov_layer = dense[j].relu
            if j==0:
                handler = cov_layer.register_forward_hook(get_feature_hook)
            else:
                handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {(13*i+j+1)}/39: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)

            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{(13*i+j+1)}", exist_ok=True)
            
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{(13*i+j+1)}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1

        if i<2:
            trans=eval('net.trans%d' % (i + 1))
            cov_layer = trans.relu
    
            handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {13 * (i+1)}/39: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)

            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{13 * (i+1)}", exist_ok=True)
            
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{13 * (i+1)}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1

    cov_layer = net.relu
    handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
    inference()
    handler.remove()

    print(bestKs)
    bestK = np.argmax(np.bincount(np.array(bestKs)))
    print(f"for conv 39/39: the most appropriate K is {bestK}")
    print("====================")
    Ks = np.append(Ks,bestK)

    # save the bestKs to csv file
    os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_39", exist_ok=True)
    
    bestks_samples = np.column_stack((bestKs, np_samples))
    np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_39/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
    
    # reset
    bestKs = np.array([], dtype=int)
    bestK = -1
    
    # save the best Ks to the disk
    print(Ks)
    np.save(f'output/step0_appropriateK/{args.arch}/samples={samples}.npy', Ks)

elif args.arch=='resnet_56':
    Ks = np.array([], dtype=int)

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    print(bestKs)
    bestK = np.argmax(np.bincount(np.array(bestKs)))
    print(f"for conv 1/55: the most appropriate K is {bestK}")
    print("====================")
    Ks = np.append(Ks,bestK)

    # save the bestKs to csv file
    os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_1", exist_ok=True)
    
    bestks_samples = np.column_stack((bestKs, np_samples))
    np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_1/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
    
    # reset
    bestKs = np.array([], dtype=int)
    bestK = -1
    
    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {cnt+1}/55: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)
            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}", exist_ok=True)
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1

            
            cnt+=1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {cnt+1}/55: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)
            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}", exist_ok=True)
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1

            cnt += 1

    # save the best Ks to the disk
    print(Ks)
    np.save(f'output/step0_appropriateK/{args.arch}/samples={samples}.npy', Ks)


elif args.arch=='resnet_110':
    Ks = np.array([], dtype=int)

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    print(bestKs)
    bestK = np.argmax(np.bincount(np.array(bestKs)))
    print(f"for conv 1/109: the most appropriate K is {bestK}")
    print("====================")
    Ks = np.append(Ks,bestK)

    # save the bestKs to csv file
    os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_1", exist_ok=True)
    
    bestks_samples = np.column_stack((bestKs, np_samples))
    np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_1/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
    
    # reset
    bestKs = np.array([], dtype=int)
    bestK = -1
    
    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {cnt+1}/109: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)
            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}", exist_ok=True)
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1

            
            cnt+=1

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            print(bestKs)
            bestK = np.argmax(np.bincount(np.array(bestKs)))
            print(f"for conv {cnt+1}/109: the most appropriate K is {bestK}")
            print("====================")
            Ks = np.append(Ks,bestK)
            # save the bestKs to csv file
            os.makedirs(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}", exist_ok=True)
            bestks_samples = np.column_stack((bestKs, np_samples))
            np.savetxt(f"output/step0_appropriateK/{args.arch}/conv_{cnt+1}/samples={samples}.csv", bestks_samples, fmt="%d", delimiter=",")
            # reset
            bestKs = np.array([], dtype=int)
            bestK = -1

            cnt += 1

    # save the best Ks to the disk
    print(Ks)
    np.save(f'output/step0_appropriateK/{args.arch}/samples={samples}.npy', Ks)