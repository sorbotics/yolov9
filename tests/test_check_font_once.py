from unittest import mock

import yolo9.utils.general as general


def test_failed_font_download_is_attempted_once(tmp_path, monkeypatch):
    monkeypatch.setattr(general, 'CONFIG_DIR', tmp_path)  # no Arial.ttf here
    monkeypatch.setattr(general, '_FONT_DOWNLOAD_FAILED', set(), raising=False)
    with mock.patch.object(general.torch.hub, 'download_url_to_file', side_effect=OSError('offline')) as dl, \
            mock.patch.object(general.LOGGER, 'warning') as warn:
        for _ in range(5):
            general.check_font('Arial.ttf')
    assert dl.call_count == 1
    assert warn.call_count == 1
