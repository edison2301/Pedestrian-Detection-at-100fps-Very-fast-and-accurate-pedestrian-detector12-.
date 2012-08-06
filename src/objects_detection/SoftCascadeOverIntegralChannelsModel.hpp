#ifndef SOFTCASCADEOVERINTEGRALCHANNELSMODEL_HPP
#define SOFTCASCADEOVERINTEGRALCHANNELSMODEL_HPP

#include "SoftCascadeOverIntegralChannelsStage.hpp"
#include "SoftCascadeOverIntegralChannelsFastStage.hpp"
#include "SoftCascadeOverIntegralChannelsFastFractionalStage.hpp"

#include "helpers/geometry.hpp"

#include <vector>
#include <iosfwd>

// forward declaration
namespace doppia_protobuf {
class DetectorModel;
class SoftCascadeOverIntegralChannelsModel;
}

namespace doppia {


/// See "Robust Object Detection Via Soft Cascade", Bourdev and Brandt, CVPR 2005
/// See also FastestPedestrianDetectorInTheWest
class SoftCascadeOverIntegralChannelsModel
{
public:

    typedef SoftCascadeOverIntegralChannelsStage stage_t;
    typedef std::vector<stage_t> stages_t;

    // stage_t can be converted directly into fast_stage_t
    typedef SoftCascadeOverIntegralChannelsFastStage fast_stage_t;
    typedef std::vector<fast_stage_t> fast_stages_t;

    // stage_t can be converted directly into fast_fractional_stage_t
    typedef SoftCascadeOverIntegralChannelsFastFractionalStage fast_fractional_stage_t;
    typedef std::vector<fast_fractional_stage_t> fast_fractional_stages_t;

    typedef SoftCascadeOverIntegralChannelsStumpStage stump_stage_t;
    typedef std::vector<stump_stage_t> stump_stages_t;

    typedef geometry::point_xy<boost::uint16_t> model_window_size_t;
    typedef geometry::box<model_window_size_t> object_window_t;

    /// this constructor will copy the protobuf data into a more efficient data structure
    SoftCascadeOverIntegralChannelsModel(const doppia_protobuf::DetectorModel &model);

    SoftCascadeOverIntegralChannelsModel(
            const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &soft_cascade,
            const model_window_size_t &model_window_size, const object_window_t &object_window);

    ~SoftCascadeOverIntegralChannelsModel();

    /// returns a soft cascade where each feature has been rescaled
    stages_t get_rescaled_stages(const float relative_scale) const;

    /// returns a soft cascade where each feature has been rescaled
    fast_stages_t get_rescaled_fast_stages(const float relative_scale) const;

    /// returns a soft cascade where each feature has been rescaled
    fast_fractional_stages_t get_rescaled_fast_fractional_stages(const float relative_scale) const;

    /// returns a soft cascade where each feature has been rescaled
    stump_stages_t get_rescaled_stump_stages(const float relative_scale) const;

    stages_t &get_stages();
    const stages_t &get_stages() const;
    const fast_stages_t &get_fast_stages() const;
    const stump_stages_t &get_stump_stages() const;

    int get_shrinking_factor() const;

    /// Helper method that returns the cascade threshold of the last stage of the model
    float get_last_cascade_threshold() const;

    /// the detection window scale, with respect to the "canonical scale 1"
    float get_scale() const;
    const model_window_size_t &get_model_window_size() const;
    const object_window_t &get_object_window() const;

    bool has_soft_cascade() const;

protected:

    stages_t stages;
    fast_stages_t fast_stages;
    stump_stages_t stump_stages;

    int shrinking_factor;
    float scale;
    model_window_size_t model_window_size;
    object_window_t object_window;

    void set_stages_from_model(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &model);    
};

/// debugging helper
void print_detection_cascade_stages(std::ostream &, const SoftCascadeOverIntegralChannelsModel::stages_t &stages);
void print_detection_cascade_stages(std::ostream &, const SoftCascadeOverIntegralChannelsModel::fast_stages_t &stages);

/// Helper method that gives the crucial information for the FPDW implementation
/// these numbers are obtained via
/// doppia/src/test/objects_detection/test_objects_detection + plot_channel_statistics.py
/// method exposed for usage inside DetectorsComparisonTestApplication
float get_channel_scaling_factor(const boost::uint8_t channel_index,
                                 const float relative_scale);

/// small helper function
template<typename Box>
float rectangle_area(const Box &box)
{
    return (box.max_corner().x() - box.min_corner().x())*(box.max_corner().y() - box.min_corner().y());
}


} // end of namespace doppia

#endif // SOFTCASCADEOVERINTEGRALCHANNELSMODEL_HPP
