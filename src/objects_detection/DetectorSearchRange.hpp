#ifndef DETECTORSEARCHRANGE_HPP
#define DETECTORSEARCHRANGE_HPP

#include <boost/cstdint.hpp>
#include <vector>

namespace doppia {

/// Helper class used by the detectors to define the set of scales to search
/// and where in the image search at each scale
class DetectorSearchRange {
public:

    /// detection window scale to consider
    float detection_window_scale;

    /// ratio used for the current detection
    /// ratio == width / height
    float detection_window_ratio;

    /// is this range itself scaled ?
    float range_scaling;

    /// is this range itself ratio-adjusted ?
    float range_ratio;

    /// when range_scaling == 1.0 the x and y coordinates are in the input image coordinates
    /// the x,y point corresponds the upper left position of the detection window
    /// this is _not_ the window center
    /// when range_scaling == 1.0 then scaled_max_x = original_max*range_scaling
    /// ( @warning max_x + scaled_detection_window_size may be out of range )
    boost::uint16_t min_x, max_x, min_y, max_y;

    DetectorSearchRange get_rescaled(const float scaling, const float ratio = 1.0f) const;

    bool operator==(const DetectorSearchRange &other) const;
};

/// we expect the search range to be ordered from smallest scale to largest scale
typedef std::vector<DetectorSearchRange> detector_search_ranges_t;

} // end of namespace doppia

#endif // DETECTORSEARCHRANGE_HPP
