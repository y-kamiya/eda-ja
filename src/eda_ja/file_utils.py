from __future__ import annotations

import functools
import gzip
import os
import shutil
import tempfile
from typing import IO

import requests
from tqdm.auto import tqdm


def get_from_cache(local_path: str, url: str) -> str:
    if os.path.exists(local_path):
        return local_path

    cache_dir = os.path.dirname(local_path)
    os.makedirs(cache_dir, exist_ok=True)

    download_path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(download_path):
        temp_file_manager = functools.partial(
            tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
        )
        with temp_file_manager() as temp_file:
            http_get(url, temp_file)

        os.replace(temp_file.name, download_path)

        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(download_path, 0o666 & ~umask)

    if download_path.endswith(".gz"):
        decompress(download_path, local_path)
    else:
        shutil.move(download_path, local_path)

    return local_path


def http_get(url: str, temp_file: IO[str]) -> None:
    res = requests.get(url, stream=True)
    res.raise_for_status()

    length = res.headers.get("Content-Length")
    content_length = 0 if length is None else int(length)
    progress = tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=content_length,
        desc="Downloading",
    )

    for chunk in res.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)

    progress.close()


def decompress(src_path: str, tgt_path: str):
    with gzip.open(src_path, mode="rb") as fs:
        contents = fs.read()

    with open(tgt_path, mode="wb") as ft:
        ft.write(contents)
