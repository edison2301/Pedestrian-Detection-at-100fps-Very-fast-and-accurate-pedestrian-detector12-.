#include "DetectorSearchRange.hpp"

#include <stdexcept>
#include <cassert>

namespace doppia {

// no explicit constructor or destructor for DetectorSearchRange, this is essentially a simple struct

DetectorSearchRange DetectorSearchRange::get_rescaled(const float scaling, const float ratio) const
{
    if(scaling <= 0)
    {
        throw std::invalid_argument("DetectorSearchRange::get_rescaled expects a scaling factor > 0");
    }
    DetectorSearchRange scaled_range;

    scaled_range.range_scaling = range_scaling * scaling;
    scaled_range.range_ratio = range_ratio * ratio;
    scaled_range.detection_window_scale = detection_window_scale * scaling;
    scaled_range.detection_window_ratio = detection_window_ratio * ratio; // ratio is invariant to scaling
    scaled_range.min_x = min_x * scaling*ratio;
    scaled_range.max_x = max_x * scaling*ratio;
    scaled_range.min_y = min_y * scaling;
    scaled_range.max_y = max_y * scaling;

    assert(scaled_range.range_scaling > 0);
    return scaled_range;
}

bool DetectorSearchRange::operator==(const DetectorSearchRange &other) const
{
    const DetectorSearchRange &self = *this;
    bool ret = true;
    ret &= self.detection_window_scale == other.detection_window_scale;
    ret &= self.detection_window_ratio == other.detection_window_ratio;
    ret &= self.range_scaling == other.range_scaling;
    ret &= self.range_ratio == other.range_ratio;
    ret &= self.min_x == other.min_x;
    ret &= self.max_x == other.max_x;
    ret &= self.min_y == other.min_y;
    ret &= self.max_y == other.max_y;

    return ret;
}

} // end of namespace doppia
