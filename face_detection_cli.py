def conceptual_captions(*, data_dir="conceptual_captions", num_train, num_val):
    def iter_index(index_path):
        with open(index_path) as f:
            for line in f:
                caption, url = line.strip().split('\t')
                yield caption, url

    def download_image_urls(data_dir, urls):
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=100)

        def save_image(url):
            hash = hashlib.sha1(url.encode())
            # Name the files after the hash of the URL.
            file_path = data_dir / f'{hash.hexdigest()}.jpeg'
            if file_path.exists():
                # Only download each file once.
                return file_path

            try:
                result = requests.get(url, timeout=5)
            except Exception:
                file_path = None
            else:
                file_path.write_bytes(result.content)
            return file_path

        result = []
        out_paths = ex.map(save_image, urls)
        for file_path in tqdm.tqdm(out_paths, total=len(urls)):
            result.append(file_path)

        return result

    def ds_from_index_file(index_path, data_dir, count):
        data_dir.mkdir(exist_ok=True)
        index = list(itertools.islice(iter_index(index_path), count))
        captions = [caption for caption, url in index]
        urls = [url for caption, url in index]

        paths = download_image_urls(data_dir, urls)

        new_captions = []
        new_paths = []
        for cap, path in zip(captions, paths):
            if path is None:
                # Download failed, so skip this pair.
                continue
            new_captions.append(cap)
            new_paths.append(path)

        new_paths = [str(p) for p in new_paths]

        ds = tf.data.Dataset.from_tensor_slices((new_paths, new_captions))
        ds = ds.map(lambda path, cap: (path, cap[tf.newaxis]))  # 1 caption per image
        return ds

    data_dir = pathlib.Path(data_dir)
    train_index_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv',
        cache_subdir=data_dir,
        cache_dir='.')

    val_index_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv',
        cache_subdir=data_dir,
        cache_dir='.')

    train_raw = ds_from_index_file(train_index_path, data_dir=data_dir / 'train', count=num_train)
    test_raw = ds_from_index_file(val_index_path, data_dir=data_dir / 'val', count=num_val)

    return train_raw, test_raw
