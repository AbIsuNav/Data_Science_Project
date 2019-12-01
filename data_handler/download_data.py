import urllib.request
import os
import tarfile


def download_data(download_path):
    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]
    if not os.path.isdir(download_path):
        os.makedirs(download_path)
        print(f'In [download_data]: created path: "{download_path}"')

    for idx, link in enumerate(links):
        file_name = download_path + '/' + 'images_%02d.tar.gz' % idx

        # do not download the file if already exists
        if os.path.exists(file_name):
            print(f'In [download_data]: "{file_name}" already exists...')
            continue

        print(f'In [download_data]: downloading at "{file_name}"...')
        urllib.request.urlretrieve(link, file_name)  # download the zip file
    print("In [download_data]: download complete. \n")


def extract_data(archive_path, extract_path):
    if not os.path.isdir(extract_path):
        os.makedirs(extract_path)
        print(f'In [extract_data]: created path: "{extract_path}". Extracting all the archive files...')

    for fname in os.listdir(archive_path):
        file_path = f'{archive_path}/{fname}'

        tar = tarfile.open(file_path, 'r:gz')
        tar.extractall(path=extract_path)

        print(f'In [extract_data]: extracted "{archive_path}/{fname}" to "{extract_path}"...')
        tar.close()
