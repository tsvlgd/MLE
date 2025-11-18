import tarfile
with tarfile.open('../aclImdb_v1.tar.gz', 'r:gz') as tar:
    tar.extractall()