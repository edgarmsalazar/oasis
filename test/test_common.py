import os

import numpy
import pytest

from oasis import common


def test_get_np_unit_dytpe():
    # Check for input type
    with pytest.raises(TypeError):
        common.get_np_unit_dtype(-1)

    with pytest.raises(TypeError):
        common.get_np_unit_dtype(1.0)

    # Check that the correct type is returned
    assert common.get_np_unit_dtype(65_535) == numpy.uint16
    assert common.get_np_unit_dtype(4_294_967_295) == numpy.uint32
    assert common.get_np_unit_dtype(4_294_967_296) == numpy.uint64
    assert common.get_np_unit_dtype(18_446_744_073_709_551_615) == numpy.uint64

    # Check that it overflows if the number is too large for unit64
    with pytest.raises(OverflowError):
        common.get_np_unit_dtype(18_446_744_073_709_551_616)


def test_mkdir():
    dirpath_fail = '/zzz/tmp_dir_test/'
    with pytest.raises(FileNotFoundError):
        common.mkdir(dirpath_fail, verbose=False)

    dirpath_pass = os.getcwd() + '/test/tmp_dir_test/'
    common.mkdir(dirpath_pass, verbose=True)
    assert os.path.exists(dirpath_pass)
    assert common.mkdir(dirpath_pass, verbose=False) == 1
    assert common.mkdir(dirpath_pass, verbose=True) == 1


    os.removedirs(dirpath_pass)


###
