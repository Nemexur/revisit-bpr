from abc import ABC, abstractmethod
import contextlib
import os
from pathlib import Path
import re


class S3Object(ABC):
    @abstractmethod
    def upload(self, s3_client) -> None:
        pass

    @abstractmethod
    def load(self, s3_client) -> None:
        pass


class S3File(S3Object):
    def __init__(
        self, bucket: str, dst: str, src: str, exist_ok: bool = False, overwrite: bool = False
    ) -> None:
        self._bucket = bucket
        self._dst = dst
        self._src = src
        self._exist_ok = exist_ok
        self._overwrite = overwrite

    def load(self, s3_client) -> None:
        os.makedirs(os.path.dirname(self._dst), exist_ok=True)
        file_exists = os.path.exists(self._dst)
        if file_exists and not self._exist_ok:
            raise FileExistsError(f"File exists: {self._dst}")
        if file_exists and not self._overwrite:
            return
        s3_client.download_file(Bucket=self._bucket, Key=self._src, Filename=self._dst)

    def upload(self, s3_client) -> None:
        file_exists = False
        with contextlib.suppress(s3_client.exceptions.ClientError):
            _ = s3_client.head_object(Bucket=self._bucket, Key=self._dst)
            file_exists = True
        if file_exists and not self._exist_ok:
            raise FileExistsError(f"File exists: {self._dst}")
        if file_exists and not self._overwrite:
            return
        s3_client.upload_file(Bucket=self._bucket, Key=self._dst, Filename=self._src)


class S3Directory(S3Object):
    def __init__(
        self, bucket: str, dst: str, src: str, exist_ok: bool = False, overwrite: bool = False
    ) -> None:
        self._bucket = bucket
        self._dst = dst
        self._src = src
        self._exist_ok = exist_ok
        self._overwrite = overwrite

    def load(self, s3_client) -> None:
        self._load(s3_client, src=self._src, initial_src=self._src)

    def upload(self, s3_client) -> None:
        self._upload(s3_client, src=self._src, initial_src=self._src)

    def _load(self, s3_client, src: str, initial_src: str = "") -> None:
        resp = s3_client.list_objects_v2(Bucket=self._bucket, Delimiter="/", Prefix=src)
        for sub_dir in resp.get("CommonPrefixes", []):
            sub_dir_path = sub_dir.get("Prefix")
            if sub_dir_path is None:
                continue
            self._load(s3_client, src=sub_dir_path, initial_src=initial_src)
        for file in resp.get("Contents", []):
            src = file.get("Key")
            if src is None or src.endswith("/"):
                continue
            s3_file = S3File(
                self._bucket,
                dst=os.path.join(self._dst, re.sub(rf"^{initial_src}/+", "", src)),
                src=src,
                exist_ok=self._exist_ok,
                overwrite=self._overwrite,
            )
            s3_file.load(s3_client)

    def _upload(self, s3_client, src: str, initial_src: str = "") -> None:
        def handler(entry: os.DirEntry) -> None:
            if entry.is_dir():
                self._upload(s3_client, src=entry.path, initial_src=initial_src)
                return
            s3_file = S3File(
                self._bucket,
                dst=os.path.join(self._dst, re.sub(rf"^{initial_src}/+", "", entry.path)),
                src=entry.path,
                exist_ok=self._exist_ok,
                overwrite=self._overwrite,
            )
            s3_file.upload(s3_client)

        with os.scandir(src) as it:
            for entry in it:
                handler(entry)


class S3FS:
    def __init__(self, bucket: str, s3_client) -> None:
        self._bucket = bucket
        self._s3_client = s3_client

    def exists(self, path: Path | str) -> bool:
        path = os.path.normpath(path)
        resp = self._s3_client.list_objects_v2(Bucket=self._bucket, Delimiter="/", Prefix=str(path))
        contents = {key for c in resp.get("Contents", []) if (key := c.get("Key")) is not None}
        prefixes = {
            re.sub(r"/$", "", p)
            for c in resp.get("CommonPrefixes", [])
            if (p := c.get("Prefix")) is not None
        }
        return len(contents) > 0 or path in prefixes

    def load(
        self, dst: Path | str, src: Path | str, exist_ok: bool = False, overwrite: bool = False
    ) -> None:
        if not self.exists(src):
            raise FileNotFoundError(f"File or directory not found: {src}")
        dst, src = os.path.normpath(dst), os.path.normpath(src)
        s3_obj = (
            S3Directory(self._bucket, dst=dst, src=src, exist_ok=exist_ok, overwrite=overwrite)
            if self._is_s3_dir(src)
            else S3File(self._bucket, dst=dst, src=src, exist_ok=exist_ok, overwrite=overwrite)
        )
        s3_obj.load(self._s3_client)

    def upload(
        self, dst: Path | str, src: Path | str, exist_ok: bool = False, overwrite: bool = False
    ) -> None:
        if not os.path.exists(src):
            raise FileNotFoundError(f"File or directory not found: {src}")
        dst, src = os.path.normpath(dst), os.path.normpath(src)
        s3_obj = (
            S3Directory(self._bucket, dst=dst, src=src, exist_ok=exist_ok, overwrite=overwrite)
            if os.path.isdir(src)
            else S3File(self._bucket, dst=dst, src=src, exist_ok=exist_ok, overwrite=overwrite)
        )
        s3_obj.upload(self._s3_client)

    def remove(self, path: Path | str) -> None:
        resp = self._s3_client.list_objects_v2(Bucket=self._bucket, Delimiter="/", Prefix=str(path))
        for sub_dir in resp.get("CommonPrefixes", []):
            sub_dir_path = sub_dir.get("Prefix")
            if sub_dir_path is None:
                continue
            self.remove(sub_dir_path)
        for c in resp.get("Contents", []):
            key = c.get("Key")
            if key is None or key.endswith("/"):
                continue
            self._s3_client.delete_object(Bucket=self._bucket, Key=key)

    def _is_s3_dir(self, dst: Path | str) -> bool:
        dst = os.path.normpath(dst)
        resp = self._s3_client.list_objects_v2(Bucket=self._bucket, Delimiter="/", Prefix=str(dst))
        contents = {key for c in resp.get("Contents", []) if (key := c.get("Key")) is not None}
        common_prefixes = resp.get("CommonPrefixes", [])
        file_condition = len(contents) == 1 or (dst in contents and len(common_prefixes) == 0)
        return not file_condition
