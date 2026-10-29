import os

import pytest

import yolo9.utils.general as general


def test_download_key_is_not_run_by_default(tmp_path):
    marker = tmp_path / 'pwned'
    data = tmp_path / 'data.yaml'
    data.write_text(f"path: {tmp_path}\ntrain: missing/images\nval: missing/images\nnames: [a]\n"
                    f"download: \"open({str(marker)!r}, 'w').close()\"\n")
    with pytest.raises(Exception, match='Dataset not found'):
        general.check_dataset(str(data))
    assert not os.path.exists(marker)
