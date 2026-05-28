"""Minimal local type stub for the untyped ``piexif`` package.

Covers only the API surface used in this repo (``load``/``dump`` plus the
``ImageIFD`` tag ids), so strict pyright resolves the otherwise-Unknown values
that ``piexif.load`` returns instead of carrying type debt. piexif ships no
``py.typed`` marker; extend this stub if new piexif APIs are adopted.
"""

from typing import Any

class ImageIFD:
    Software: int
    Make: int
    Artist: int
    ImageDescription: int

class ExifIFD: ...
class GPSIFD: ...

def load(input_data: bytes | str, key_is_name: bool = ...) -> dict[str, Any]: ...
def dump(exif_dict_original: dict[str, Any]) -> bytes: ...
def insert(exif: bytes, image: str, new_file: str | None = ...) -> None: ...
def remove(src: str, new_file: str | None = ...) -> None: ...
