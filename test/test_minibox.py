import os

import numpy
import pytest

from oasis import coordinates, minibox, common


l_box = 100.
l_mb = 20.


def test_generate_mini_box_grid():
    """Check if `generate_mini_box_grid` creates a regular grid."""
    ids, centres = minibox.generate_mini_box_grid(boxsize=l_box, minisize=l_mb)

    assert len(ids) == len(centres)  # Length of arrays is the same
    # Number of elements is (l_box/l_mb)**3
    assert len(ids) == numpy.int_(numpy.ceil(l_box / l_mb))**3
    # First position is shifted by l_mb/2
    assert all(centres[0] == numpy.full(3, 0.5*l_mb))
    assert all(centres[100] == numpy.array([10., 10., 90.]))
    assert all(centres[44] == numpy.array([90., 70., 30.]))

    # Test when minisize is larger than boxsize
    with pytest.raises(ValueError, match='Mini box size cannot be larger than box size.'):
        minibox.generate_mini_box_grid(boxsize=1.0, minisize=5.0)

    # Test very small minisize
    ids, centres = minibox.generate_mini_box_grid(boxsize=1.0, minisize=0.1)
    assert len(ids) == 10**3  # Should create 10x10x10 grid
    
    # Test when minisize equals boxsize
    ids, centres = minibox.generate_mini_box_grid(boxsize=5.0, minisize=5.0)
    assert len(ids) == 1
    assert numpy.allclose(centres[0], [2.5, 2.5, 2.5])

    # Test with multiple box/minibox size combinations
    test_cases = [
        (10.0, 1.0),   # Perfect division
        (10.0, 3.0),   # Non-perfect division
        (5.5, 2.0),    # Fractional box size
        (1.0, 1.0),    # Single box
    ]
    
    for boxsize, minisize in test_cases:
        ids, centres = minibox.generate_mini_box_grid(boxsize=boxsize, minisize=minisize)
        
        # Calculate expected dimensions
        boxes_per_side = numpy.int_(numpy.ceil(boxsize / minisize))
        expected_count = boxes_per_side**3
        
        # Basic array properties
        assert len(ids) == len(centres), f"Length mismatch for case ({boxsize}, {minisize})"
        assert len(ids) == expected_count, f"Expected {expected_count} boxes, got {len(ids)}"
        assert centres.shape == (expected_count, 3), f"Centres shape incorrect for case ({boxsize}, {minisize})"
        
        # ID properties
        assert len(numpy.unique(ids)) == len(ids), "IDs are not unique"
        assert numpy.all(ids >= 0), "All IDs should be non-negative"
        assert numpy.max(ids) < boxes_per_side**3, "ID values exceed expected range"
        
        # Verify IDs are sorted (as per the function design)
        assert numpy.array_equal(ids, numpy.sort(ids)), "IDs should be sorted"
    
    # Validate that centres form a proper regular grid.
    
    # First position should be at (0.5*minisize, 0.5*minisize, 0.5*minisize)
    expected_first = numpy.full(3, 0.5 * minisize)
    assert numpy.allclose(centres[0], expected_first), \
        f"First centre should be {expected_first}, got {centres[0]}"
    
    # Check grid spacing
    # Extract unique coordinates along each axis
    x_coords = numpy.unique(centres[:, 0])
    y_coords = numpy.unique(centres[:, 1])
    z_coords = numpy.unique(centres[:, 2])
    
    # Verify we have the correct number of unique coordinates per axis
    assert len(x_coords) == boxes_per_side, f"Expected {boxes_per_side} unique x-coords, got {len(x_coords)}"
    assert len(y_coords) == boxes_per_side, f"Expected {boxes_per_side} unique y-coords, got {len(y_coords)}"
    assert len(z_coords) == boxes_per_side, f"Expected {boxes_per_side} unique z-coords, got {len(z_coords)}"
    
    # Verify regular spacing
    if boxes_per_side > 1:
        x_spacing = numpy.diff(x_coords)
        y_spacing = numpy.diff(y_coords)
        z_spacing = numpy.diff(z_coords)
        
        assert numpy.allclose(x_spacing, minisize), "X-coordinates not regularly spaced"
        assert numpy.allclose(y_spacing, minisize), "Y-coordinates not regularly spaced"
        assert numpy.allclose(z_spacing, minisize), "Z-coordinates not regularly spaced"
    
    # Verify coordinate bounds
    assert numpy.all(centres >= 0.5 * minisize), "Some centres are too close to origin"
    max_expected = (boxes_per_side - 0.5) * minisize
    assert numpy.all(centres <= max_expected), f"Some centres exceed expected bounds ({max_expected})"

    # Validate that the ID-to-coordinate mapping is consistent
    
    # Reconstruct expected coordinates from IDs using the inverse mapping
    # ID = k*1 + j*boxes_per_side + i*boxes_per_side^2
    # where k, j, i are 0-indexed grid positions
    
    for idx, (id_val, centre) in enumerate(zip(ids, centres)):
        # Decompose ID back to grid indices
        i = id_val // (boxes_per_side**2)
        remainder = id_val % (boxes_per_side**2)
        j = remainder // boxes_per_side
        k = remainder % boxes_per_side
        
        # Calculate expected centre from grid indices
        expected_centre = numpy.array([
            minisize * (k + 0.5),  # x
            minisize * (j + 0.5),  # y  
            minisize * (i + 0.5)   # z
        ])
        
        assert numpy.allclose(centre, expected_centre), \
            f"Centre mismatch at index {idx}: ID={id_val}, expected={expected_centre}, got={centre}"


def test_get_mini_box_id():
    """Check if `get_mini_box_id` generates the right IDs for each particle."""
    _, centres = minibox.generate_mini_box_grid(boxsize=l_box, minisize=l_mb)
    n_mb = numpy.int_(numpy.ceil(l_box / l_mb))**3

    # Partition box and retrive subbox ID for each particle. Only one particle
    # per subbox.
    box_ids = minibox.get_mini_box_id(centres, boxsize=l_box, minisize=l_mb)

    assert box_ids[0] == 0  # First particle is in the first box with ID = 0
    # Last particle is in the last box with ID = 999
    assert box_ids[-1] == n_mb - 1
    assert len(numpy.unique(box_ids)) == n_mb  # One particle per subbox

    box_ids = minibox.get_mini_box_id(centres[0], boxsize=l_box, minisize=l_mb)
    assert box_ids == 0

class TestGetMiniBoxId:
    """Test suite for minibox.get_mini_box_id function"""
    
    def test_single_position_origin(self):
        """Test single position at origin"""
        position = numpy.array([0.1, 0.1, 0.1])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(position, boxsize, minisize)
        expected = numpy.array([0])  # Should be in first mini box (0,0,0)
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_single_position_interior(self):
        """Test single position in interior"""
        position = numpy.array([1.5, 2.5, 3.5])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(position, boxsize, minisize)
        # Grid indices: (1, 2, 3), ID = 1*1 + 2*10 + 3*100 = 321
        expected = numpy.array([321])
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_multiple_positions(self):
        """Test multiple positions"""
        positions = numpy.array([
            [0.1, 0.1, 0.1],  # Grid (0,0,0) -> ID = 0
            [1.5, 0.1, 0.1],  # Grid (1,0,0) -> ID = 1
            [0.1, 1.5, 0.1],  # Grid (0,1,0) -> ID = 10
            [0.1, 0.1, 1.5]   # Grid (0,0,1) -> ID = 100
        ])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        expected = numpy.array([0, 1, 10, 100])
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_edge_cases_upper_boundary(self):
        """Test positions exactly at upper boundary"""
        positions = numpy.array([
            [10.0, 5.0, 5.0],  # Exactly at upper x boundary
            [5.0, 10.0, 5.0],  # Exactly at upper y boundary
            [5.0, 5.0, 10.0]   # Exactly at upper z boundary
        ])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        
        # Should be moved inward and placed in grid (9,5,5), (5,9,5), (5,5,9)
        expected = numpy.array([
            9 + 5*10 + 5*100,   # 559
            5 + 9*10 + 5*100,   # 595  
            5 + 5*10 + 9*100    # 955
        ])
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_edge_cases_lower_boundary(self):
        """Test positions exactly at lower boundary (zero)"""
        positions = numpy.array([
            [0.0, 0.5, 0.5],  # Exactly at lower x boundary
            [0.5, 0.0, 0.5],  # Exactly at lower y boundary
            [0.5, 0.5, 0.0]   # Exactly at lower z boundary
        ])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        
        # Should be moved inward and remain in grid (0,0,0)
        expected = numpy.array([0, 0, 0])
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_non_unit_minisize(self):
        """Test with non-unit mini box size"""
        position = numpy.array([3.0, 6.0, 9.0])
        boxsize = 12.0
        minisize = 2.0
        
        result = minibox.get_mini_box_id(position, boxsize, minisize)
        
        # Grid indices: floor([3.0, 6.0, 9.0] / 2.0) = [1, 3, 4]
        # n_cells_per_side = ceil(12.0 / 2.0) = 6
        # ID = 1*1 + 3*6 + 4*36 = 1 + 18 + 144 = 163
        expected = numpy.array([163])
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_fractional_minisize(self):
        """Test with fractional mini box size"""
        position = numpy.array([1.0, 1.0, 1.0])
        boxsize = 3.0
        minisize = 0.5
        
        result = minibox.get_mini_box_id(position, boxsize, minisize)
        
        # Grid indices: floor([1.0, 1.0, 1.0] / 0.5) = [2, 2, 2]
        # n_cells_per_side = ceil(3.0 / 0.5) = 6
        # ID = 2*1 + 2*6 + 2*36 = 2 + 12 + 72 = 86
        expected = numpy.array([86])
        
        numpy.testing.assert_array_equal(result, expected)
    
    def test_input_validation_minisize_too_large(self):
        """Test error when minisize > boxsize"""
        position = numpy.array([1.0, 1.0, 1.0])
        boxsize = 5.0
        minisize = 10.0
        
        with pytest.raises(ValueError, match="Mini box size cannot be larger than box size"):
            minibox.get_mini_box_id(position, boxsize, minisize)
    
    def test_output_shape_single_position(self):
        """Test output shape for single position"""
        position = numpy.array([1.0, 1.0, 1.0])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(position, boxsize, minisize)
        
        assert result.shape == (1,)
        assert isinstance(result, numpy.ndarray)
    
    def test_output_shape_multiple_positions(self):
        """Test output shape for multiple positions"""
        positions = numpy.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0]
        ])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        
        assert result.shape == (3,)
        assert isinstance(result, numpy.ndarray)
    
    def test_unique_ids_different_positions(self):
        """Test that different positions get different IDs"""
        positions = numpy.array([
            [0.5, 0.5, 0.5],  # Grid (0,0,0)
            [1.5, 0.5, 0.5],  # Grid (1,0,0)
            [0.5, 1.5, 0.5],  # Grid (0,1,0)
            [0.5, 0.5, 1.5]   # Grid (0,0,1)
        ])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        
        # All IDs should be unique
        assert len(numpy.unique(result)) == len(result)
    
    @pytest.mark.parametrize("boxsize,minisize", [
        (1.0, 0.1),
        (10.0, 1.0),
        (5.5, 0.7),
        (100.0, 3.3)
    ])
    def test_parametrized_box_sizes(self, boxsize, minisize):
        """Test with various box and mini sizes"""
        position = numpy.array([boxsize/2, boxsize/2, boxsize/2])
        
        result = minibox.get_mini_box_id(position, boxsize, minisize)
        
        # Should return a valid non-negative integer
        assert result.dtype == int
        assert result[0] >= 0
        assert len(result) == 1
    
    def test_corner_cases_near_boundaries(self):
        """Test positions very close to but not exactly at boundaries"""
        positions = numpy.array([
            [1e-10, 1.0, 1.0],      # Very close to zero
            [10.0 - 1e-10, 1.0, 1.0]  # Very close to upper boundary
        ])
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        
        # Should handle these gracefully
        assert len(result) == 2
        assert all(id >= 0 for id in result)
    
    def test_empty_input_handling(self):
        """Test behavior with empty input"""
        positions = numpy.empty((0, 3))
        boxsize = 10.0
        minisize = 1.0
        
        result = minibox.get_mini_box_id(positions, boxsize, minisize)
        
        assert result.shape == (0,)
        assert isinstance(result, numpy.ndarray)


def test_get_adjacent_mini_box_ids():
    """Check if the number of adjacent miniboxes is in fact 27."""

    adj_ids = minibox.get_adjacent_mini_box_ids(
        mini_box_id=0,
        boxsize=l_box,
        minisize=l_mb,
    )

    adj_ids_0 = [0, 1, 4, 5, 6, 9, 20, 21, 24, 25, 26, 29, 30, 31, 34, 45, 46,
                 49, 100, 101, 104, 105, 106, 109, 120, 121, 124]

    assert len(adj_ids) == 27
    assert all([i in adj_ids for i in adj_ids_0])


def test_generate_mini_box_ids():
    """Sort items into miniboxes according to their positions."""
    n_samples = 1000
    chunk_size = 10
    seed = 1234
    pos = coordinates.gen_data_pos_random(l_box, n_samples, seed)

    ids = minibox.generate_mini_box_ids(pos, l_box, l_mb, chunk_size)

    assert min(ids) == 0
    assert max(ids) <= numpy.int_(numpy.ceil(l_box / l_mb))**3
    assert len(ids) == n_samples


def test_get_chunks():
    ones = numpy.ones(100, dtype=numpy.uint8)
    with pytest.raises(IndexError):
        minibox.get_chunks(ids=ones, chunksize=1)

    nums = numpy.repeat(numpy.arange(10, dtype=numpy.uint8), 10)
    chunks = minibox.get_chunks(ids=nums, chunksize=15)
    assert len(chunks) == 11



def test_split_into_mini_boxes():
    # Create sinthetic data
    l_box = 500.
    l_mb = 100.
    n_points = numpy.int_(numpy.ceil(l_box / l_mb))**3
    pos = coordinates.gen_data_pos_regular(l_box, l_mb)
    vel = numpy.random.uniform(-1, 1, n_points)
    # Offset PIDs to avoid confusion with mini box ID.
    pid = numpy.arange(2000, 2000 + n_points)

    temp_dir = os.getcwd() + '/temp/'
    # common.mkdir(temp_dir)
    # minibox.split_box_into_mini_boxes(pos, vel, pid, temp_dir, l_box, l_mb, 
    #                                   chunksize=2*n_points)
    # os.removedirs(temp_dir)


def test_load_particles():
    ...


def test_load_seeds():
    ...

###
