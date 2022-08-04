##
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MICROSOFT QUANTUM NON-COMMERCIAL License.
##
import configparser
import datetime
import functools
import json
import os
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

import adlfs
import fsspec
import ipywidgets as widgets
import pandas as pd
import xarray as xr
from dateutil.parser import ParserError
from IPython.display import display
from tqdm import tqdm
from urlpath import URL

from topogap_protocol.utils import TMP_DIR

PathLike = Union[str, Path, URL]


def load_dataset(
    fname: PathLike,
    cache_dir: Optional[Union[Path, str]] = TMP_DIR / "datalake_cache",
    return_copy: bool = True,
) -> xr.Dataset:
    if str(fname).startswith("abfs:"):
        assert not isinstance(fname, Path)
        STORAGE_OPTIONS = datalake_storage_options()
        if cache_dir is not None:
            fname = URL(fname).with_scheme("filecache")
            kwargs = dict(
                target_options=STORAGE_OPTIONS,
                cache_storage=str(cache_dir),
                target_protocol="abfs",
            )
        else:
            kwargs = STORAGE_OPTIONS

        with fsspec.open(fname, **kwargs) as f:
            ds = xr.open_dataset(f).load()
        return ds.copy() if return_copy else ds
    else:
        return xr.load_dataset(fname).load()


def datalake_fs():
    return adlfs.AzureBlobFileSystem(**datalake_storage_options())


def path_or_url(x):
    if str(x).startswith("abfs:"):
        assert not isinstance(x, Path)
        return URL(str(x))
    else:
        return Path(x).resolve()


def exists(fname: PathLike):
    if str(fname).startswith("abfs:"):
        assert not isinstance(fname, Path)
        fs = datalake_fs()
        return fs.exists(str(fname))
    else:
        return Path(fname).exists()


@functools.lru_cache(None)
def datalake_storage_options():
    try:
        return datalake_storage_options_from_environment()
    except KeyError:
        return datalake_storage_options_from_file()


@functools.lru_cache(None)
def datalake_storage_options_from_file(key: str = "majodata") -> Dict[str, str]:
    p = Path("~/.datalake.conf").expanduser()
    if not p.exists():
        msg = f"""
        The file {p} cannot be found. This file should contain
        your datalake credentials. An example of the file looks like
        ```
        [majodata]
        account_name = majodata
        account_key = tMe3gQyF1LyvPKTxtC/RCGNvA2lQSALreeQVYDjkqe/O82mXgK37nI6bbsrmSUraxYJ5icITaiDRGz2JfiLGa6==
        ```
        where you should change the `account_key` to the correct one.
        Obtain it by going to https://portal.azure.com → "Storage accounts"
        → "majodata" → "Access keys" and copy "key1" or "key2".
        """
        raise FileNotFoundError(textwrap.dedent(msg))
    config = configparser.ConfigParser()
    config.read(p)
    print("Using datalake_storage_options_from_file")
    return dict(config[key])


def datalake_storage_options_from_environment() -> Dict[str, str]:
    if (
        "AZURE_STORAGE_CLIENT_SECRET" in os.environ
        and "AZURE_STORAGE_CLIENT_ID" in os.environ
        and "AZURE_STORAGE_TENANT_ID" in os.environ
    ):
        # adlfs automatically picks these up
        return {"account_name": "majodata"}

    storage_options = {
        "account_name": os.environ["ACCOUNT_NAME"],
        "sas_token": "?" + os.environ["SAS_TOKEN"],
    }
    assert all(x != "" for x in storage_options.values())
    print("Using datalake_storage_options_from_environment")
    return storage_options


def download_latest_reports(folder: str = "reports") -> Dict[str, datetime.datetime]:
    """Download all latest reports that are generated by
    the `run_create_report.py` script and that have automatically been uploaded
    to the Data Lake."""
    fs = datalake_fs()
    remote_files = fs.listdir(r"majodata/report")
    remote_fnames = [f["name"] for f in remote_files]
    remote_fnames = [f for f in remote_fnames if "latest" in f]
    folder = Path(folder)
    local_fnames = [folder / Path(f).name for f in remote_fnames]
    folder.mkdir(exist_ok=True)
    for rfname, lfname in tqdm(
        zip(remote_fnames, local_fnames),
        desc="Downloading reports",
        total=len(local_fnames),
    ):
        fs.get_file(rfname, lfname)
    last_modified = [f["last_modified"] for f in remote_files]
    return dict(zip(local_fnames, last_modified))


@functools.lru_cache(None)
def get_datalake_summary(datelake_container: str = "majodata") -> pd.DataFrame:
    """Fetch information about the datasets on the Data Lake.

    These datasets have been uploaded via the CI pipelines that are attached
    to the Git LFS repositories on https://dev.azure.com/ms-quantum/Majodata/.
    """
    fs = datalake_fs()
    meta_fnames = fs.glob(f"{datelake_container}/*/meta.json")
    metas = []
    for meta_fname in meta_fnames:
        lpath = Path(TMP_DIR) / meta_fname.replace("/", "_")
        fs.get_file(meta_fname, lpath)
        with lpath.open() as f:
            meta = json.load(f)
        meta["datalake"] = os.path.dirname(meta_fname)
        metas.append(meta)

    nc_files = list(map(Path, fs.glob(f"{datelake_container}/*/*/*nc")))
    nc_files += list(map(Path, fs.glob(f"{datelake_container}/*/phase_two/*/*nc")))

    def split(p):
        if "phase_two" in p.parts:
            return "/".join(p.parts[-2:])
        return p.name

    mapping = defaultdict(lambda: defaultdict(set))
    for fname in nc_files:
        folder = "/".join(fname.parts[:2])
        mapping[folder][fname.parts[2]].add(split(fname))

    def is_valid(f):
        return Path(f).name in {
            "left.nc",
            "right.nc",
            "lc_lb.nc",
            "lc_rb.nc",
            "rc_lb.nc",
            "rc_rb.nc",
        }

    for meta in metas:
        fnames = mapping[meta["datalake"]]
        for folder, nc_fnames in fnames.items():
            if folder in ("calibration", "rf", "phase_two"):
                meta[folder] = [f for f in nc_fnames if is_valid(f)]

    df = pd.DataFrame(metas)
    df["datalake"] = r"abfs://" + df["datalake"].astype(str)

    def phase_one_complete(row):
        return isinstance(row.calibration, list) and isinstance(row.rf, list)

    def phase_two_complete(row):
        return isinstance(row.phase_two, list)

    def try_datetime(row):
        try:
            return pd.to_datetime(row)
        except ParserError:
            return row

    df.upload_date = pd.to_datetime(df.upload_date)
    df.date = df.date.apply(try_datetime)
    df["phase_one_complete"] = df.apply(phase_one_complete, axis=1)
    df["phase_two_complete"] = df.apply(phase_two_complete, axis=1)
    return df


def get_datalake_summary_widget(datelake_container: str = "majodata"):
    df = get_datalake_summary(datelake_container)
    out = widgets.Output()

    def view(device_name, data_type, uploaded_by):
        with out:
            out.clear_output(wait=True)
        device_name = (df.device_name == device_name) if device_name != "all" else True
        data_type = df.data_type == data_type
        uploaded_by = (df.uploaded_by == uploaded_by) if uploaded_by != "all" else True
        display(df[device_name & data_type & uploaded_by])

    display(
        widgets.interactive(
            view,
            device_name=widgets.Select(
                options=["all"] + df.device_name.unique().tolist(), rows=7
            ),
            uploaded_by=widgets.Select(
                options=["all"] + df.uploaded_by.unique().tolist()
            ),
            data_type=widgets.Select(options=df.data_type.unique().tolist()),
        )
    )