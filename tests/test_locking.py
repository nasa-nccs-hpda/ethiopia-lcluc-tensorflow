import pytest
from ethiopia_lcluc_tensorflow.utils.locking import output_lock


def test_lock_excludes_second_worker_and_releases_after_failure(tmp_path):
    output = tmp_path / 'scene.tif'
    with pytest.raises(RuntimeError):
        with output_lock(output) as acquired:
            assert acquired
            with output_lock(output) as second:
                assert not second
            assert (tmp_path / 'scene.tif.lock').exists()
            raise RuntimeError('inference failed')
    assert not (tmp_path / 'scene.tif.lock').exists()
    with output_lock(output) as acquired:
        assert acquired
        output.write_bytes(b'completed')
    with output_lock(output) as acquired:
        assert not acquired
    assert output.read_bytes() == b'completed'
