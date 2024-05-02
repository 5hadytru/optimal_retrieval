import numpy as np
import os, time, sys
import torch
import faiss
import pprint
import pickle
import faiss.contrib.torch_utils


def store_IndexIVF(pth, pq:bool, vec_type, embeddings:np.ndarray, dim, nlist, m, bits, nprobe):
    print("nlist", nlist, "m", m, "bits", bits, "nprobe", nprobe)

    start = time.time()
    quantizer = faiss.IndexFlatL2(dim)  # this remains the same
    if pq:
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, bits)
    else:
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.nprobe = nprobe
    index.train(embeddings)
    print(f"Took {time.time() - start}s to train approx index")

    start = time.time()
    index.add(embeddings)
    print(f"Took {time.time() - start}s to add data to approx index")

    index_file = os.path.join(pth, f"feat_IVF{'PQ' if pq else 'Flat'}_{nlist}_{m}_{bits}_{nprobe}_{vec_type}.index")
    faiss.write_index(index, index_file)
    print("Stored index")


def store_ground_truth_index(pth, vec_type, embeddings:np.ndarray, dim, metric):
    start = time.time()
    if metric == "L2":
        index = faiss.IndexFlatL2(dim)
    elif metric == "IP":
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"Took {time.time() - start}s to add data to ground truth index")

    index_file = os.path.join(pth, f"{metric}_{vec_type}.index")
    faiss.write_index(index, index_file)
    print("Stored index")


def recall(true_indices:np.ndarray, approx_indices:np.ndarray):
    matches = sum([len(np.intersect1d(true_row, approx_row)) for true_row, approx_row in zip(true_indices, approx_indices)])
    total_true_neighbors = true_indices.size
    return matches / total_true_neighbors


def test_gpu_indexes(pth, vec_type):
    gpu_resources = faiss.StandardGpuResources()
    ground_truth_index_pth = os.path.join(pth, f"L2_{vec_type}.index")
    ground_truth_index = faiss.read_index(ground_truth_index_pth)
    ground_truth_index = faiss.index_cpu_to_gpu(gpu_resources, 0, ground_truth_index)

    approx_index_files = [
        os.path.join(pth, filename) for filename in os.listdir(pth) if filename.endswith(".index") and "L2" not in filename 
    ]

    get_nprobe = lambda fname: int(fname.split("_")[-2])

    query_vectors = (torch.ones((72, 768), dtype=torch.float32, device=torch.device(0)) - 0.5) * 2
    k = 8
    n_tries = 10
    nq = 4

    gt_indices = [ground_truth_index.search(query_vectors[i*nq:(i+1)*nq], k)[1] for i in range(n_tries)]

    del ground_truth_index

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for fname in approx_index_files:
        try:
            approx_index = faiss.read_index(fname)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            approx_index.nprobe = min(get_nprobe(fname), 2048)
            approx_index = faiss.index_cpu_to_gpu(gpu_resources, 0, approx_index, co)
        except:
            continue

        times = []
        recalls = []
        for i in range(n_tries):
            start.record()
            _, approx_indices = approx_index.search(query_vectors[i*nq:(i+1)*nq], k)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000)

            r = recall(gt_indices[i].cpu().numpy(), approx_indices.cpu().numpy())
            recalls.append(r)

        print(f"{fname}: {np.mean(times):.3f}, {np.mean(recalls):.3f}")


if __name__ == '__main__':
    import cupy as cp
    import torch
    from pylibraft.common import DeviceResources
    from pylibraft.neighbors.brute_force import knn

    n_samples = 600000
    n_features = 768
    n_queries = 12

    with cp.cuda.Device(2):
        dataset = torch.randn((n_samples, 3, n_features),
                                        dtype=torch.float32, device=torch.device(2))
        # Search using the built index
        queries = torch.randn((n_queries, n_features),
                                        dtype=torch.float32, device=torch.device(2))
        
        print(dataset.device, queries.device)

        k = 128
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for i in range(5):
            idx = torch.arange(100000).to(torch.device(2))
            start.record()
            distances, neighbors = knn(dataset[idx, :, :], queries, k)
            end.record()
            torch.cuda.synchronize()
            print(start.elapsed_time(end) / 1000)
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)

    exit(0)

    # coreset_path = "../data/coresets/laion400m-data_0.4/CLIP_ViT_B_16_2B"

    # vec_type = sys.argv[1]
    # # metric_type = sys.argv[2]
    # # max_ms = int(sys.argv[3])
    # embeddings = np.load(f"{coreset_path}/feat_{vec_type}.npy")

    # print("Got embeddings")

    # faiss.IndexPQ

    # start = time.time()
    # d = 512
    # # nlist = 16384
    # m = 256   
    # nbits = int(sys.argv[2])                  
    # index = faiss.IndexPQ(d, m, nbits)
    # index.train(embeddings)
    # print(f"Took {time.time() - start}s to train approx index")

    # start = time.time()
    # index.add(embeddings)
    # print(f"Took {time.time() - start}s to add data to approx index")

    # index_file = os.path.join(coreset_path, f"feat_{vec_type}_PQ_{m}_{nbits}.index")
    # faiss.write_index(index, index_file)

    # gpu_resources = faiss.StandardGpuResources()
    # ground_truth_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

    vec_type = sys.argv[1]
    generating = bool(int(sys.argv[2]))
    coreset_path = "../data/coresets/O365_VG/OWL_B16/"

    if generating:
        index_type = sys.argv[3]
        full_features = torch.load(os.path.join(coreset_path, f"features_TDS.pth"))
        
        indices_pth = f"idx_o365_0.9_VG_0.0_N50.pkl"
        with open(indices_pth, "rb") as f:
            subset_indices = pickle.load(f)

        feature_dim = full_features.tensors[1].size(-1)
        embeddings = full_features.tensors[1][subset_indices].view(-1, feature_dim).numpy()

        print(embeddings.shape, embeddings.dtype)

        # idx mapping
        # idx = 353636
        # n_tok = 3
        # f1 = full_features.tensors[1][subset_indices][idx // n_tok, idx % n_tok,:].numpy()
        # f2 = embeddings[idx]
        # print((f1 - f2).sum())

        dim = embeddings.shape[1]
        print(f"Keys shape: {embeddings.shape}")

        if index_type.startswith('IVF'):
            nlist, m, bits, nprobe = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
            store_IndexIVF(coreset_path, index_type.endswith("PQ"), vec_type, embeddings, dim, nlist, m, bits, nprobe)
        elif index_type in ["L2", "IP"]:
            store_ground_truth_index(coreset_path, vec_type, embeddings, dim, index_type)
        else:
            raise Exception
    else:
        # ground_truth_index_pth = os.path.join(coreset_path, f"L2_{vec_type}.index")
        # ground_truth_index = faiss.read_index(ground_truth_index_pth)
        # ground_truth_index2 = faiss.read_index(ground_truth_index_pth)

        # query_vectors = (torch.rand((72, 512), device=torch.device("cuda:0"), dtype=torch.float32) - 0.5) * 2
        # query_vectors = query_vectors.cpu().numpy()

        # start = time.time()
        # _, indices = ground_truth_index.search(query_vectors, 1)
        # _, indices = ground_truth_index2.search(query_vectors, 1)

        # print(f"Took {time.time()-start}s to search")

        test_gpu_indexes(coreset_path, vec_type)
