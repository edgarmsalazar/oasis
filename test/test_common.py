import os

import numpy
import pytest

from oasis import common


class TestUintDtype:

    def test_wrong_dtype(self):
        # Check for input type
        with pytest.raises(TypeError):
            common.get_min_unit_dtype(-1)

        with pytest.raises(TypeError):
            common.get_min_unit_dtype(1.0)

    def test_correct_dtype(self):

        # Check that the correct type is returned
        assert common.get_min_unit_dtype(65_535) == numpy.uint16
        assert common.get_min_unit_dtype(4_294_967_295) == numpy.uint32
        assert common.get_min_unit_dtype(4_294_967_296) == numpy.uint64
        assert common.get_min_unit_dtype(18_446_744_073_709_551_615) == numpy.uint64

    def test_overflow(self):
        # Check that it overflows if the number is too large for unit64
        with pytest.raises(OverflowError):
            common.get_min_unit_dtype(18_446_744_073_709_551_616)

class TestMkdir:

    def test_fail(self):
        dirpath_fail = '/zzz/tmp_dir_test/'
        with pytest.raises(OSError):
            common.ensure_dir_exists(dirpath_fail, verbose=False)

    def test_pass(self):
        dirpath_pass = os.getcwd() + '/test/tmp_dir_test/'
        common.ensure_dir_exists(dirpath_pass, verbose=True)
        assert os.path.exists(dirpath_pass)

        os.removedirs(dirpath_pass)


###
