import requests, sys, os
from multiprocessing import Pool


def get_file_size(url):
    return int(os.popen(f"curl -sI {url} | grep -i Content-Length | awk '{{print $2}}'").read().strip())

def supports_partial_content(url):
    headers = {"Range": "bytes=0-1"}
    response = requests.head(url, headers=headers)
    
    if response.status_code == 206:
        return True
    
    if "Accept-Ranges" in response.headers and response.headers["Accept-Ranges"] == "bytes":
        return True
    
    return False

def download_chunk(args):
    url, start, end, output_file = args
    os.system(f"srun -N 1 -n 1 curl -L --range {start}-{end} '{url}' -o {output_file}")

def parallel_download(url, *, output_file, file_size, num_chunks):
    chunk_size = file_size // num_chunks

    # Download chunks in parallel
    with Pool(processes=num_chunks) as pool:
        pool.map(download_chunk, [(url, i * chunk_size, (i + 1) * chunk_size - 1 if i < num_chunks - 1 else "", f"{output_file}.part{i}") for i in range(num_chunks)])

    print("Completed chunk downloads")

    # Combine downloaded chunks into a single file
    os.system(f"cat {' '.join([f'{output_file}.part{i}' for i in range(num_chunks)])} > {output_file}")
    os.system(f"rm {' '.join([f'{output_file}.part{i}' for i in range(num_chunks)])}")


def main():
    url = sys.argv[1]

    result = supports_partial_content(url)

    if result:
        print("The URL supports partial content requests.")
    else:
        print("The URL does not support partial content requests.")
        return

    file_size = get_file_size(url)
    print(f"File size: {file_size} bytes")

    output_file = sys.argv[2]
    num_chunks = int(sys.argv[3]) # should equal slurm node count

    parallel_download(url, output_file=output_file, file_size=file_size, num_chunks=num_chunks)

if __name__ == '__main__':
    main()


"""

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz -o mini_iNat2021_train.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz -o mini_iNat2021_train_ann.json.tar.gz

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz -o iNat2021_val.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz -o iNat2021_val_ann.json.tar.gz

wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz -o iNat2021_test.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz -o iNat2021_test_ann.tar.gz

tar -xzf mini_iNat2021_train.tar.gz
tar -xzf mini_iNat2021_train_ann.json.tar.gz
tar -xzf iNat2021_val.tar.gz
tar -xzf iNat2021_val_ann.json.tar.gz
tar -xzf iNat2021_test.tar.gz
tar -xzf iNat2021_test_ann.tar.gz

"""