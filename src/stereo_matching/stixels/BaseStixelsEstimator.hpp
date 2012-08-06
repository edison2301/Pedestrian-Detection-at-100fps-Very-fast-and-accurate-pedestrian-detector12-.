#ifndef BICLOP_BASESTIXELSESTIMATOR_HPP
#define BICLOP_BASESTIXELSESTIMATOR_HPP

#include "AbstractStixelsEstimator.hpp"

namespace doppia {

// forward declarations
class MetricStereoCamera;

/// Base class that implements some helper methods used by all stixels estimators
class BaseStixelsEstimator : public AbstractStixelsEstimator
{
protected:
    BaseStixelsEstimator(const MetricStereoCamera &camera,
                         const float expected_object_height,
                         const int minimum_object_height_in_pixels,
                         const int stixel_width);
    ~BaseStixelsEstimator();

public:
    int get_stixel_width() const;
    const std::vector<int> &get_v_given_disparity() const;
    const std::vector<int> &get_disparity_given_v() const;

protected:

    const MetricStereoCamera &stereo_camera;
    const float expected_object_height; ///< [meters]
    const int minimum_object_height_in_pixels; ///< [pixels]    
    const int stixel_width;


    GroundPlane the_ground_plane;
    GroundPlaneEstimator::line_t the_v_disparity_ground_line;

    /// for each disparity index stores the corresponding v values
    std::vector<int> v_given_disparity, disparity_given_v;
    void set_v_disparity_line_bidirectional_maps(const int num_rows, const int num_disparities);

    /// for each disparity get the minimum relevant v value
    /// using the ground plane and an expected maximum height
    std::vector<int> expected_v_given_disparity, top_v_for_stixel_estimation_given_disparity;
    void set_v_given_disparity(const int num_rows, const int num_disparities);


};

} // end namespace doppia

#endif // BICLOP_BASESTIXELSESTIMATOR_HPP
