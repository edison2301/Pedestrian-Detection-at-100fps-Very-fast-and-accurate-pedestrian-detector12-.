#ifndef BICLOP_MULTISCALESINTEGRALCHANNELSMODEL_HPP
#define BICLOP_MULTISCALESINTEGRALCHANNELSMODEL_HPP

#include "SoftCascadeOverIntegralChannelsModel.hpp"


// forward declaration
namespace doppia_protobuf {
class MultiScalesDetectorModel;
}

namespace doppia {

/// This is the application specific parallel of the protocol buffer message MultiScalesDetectorModel
/// @see detector_model.proto
class MultiScalesIntegralChannelsModel
{
public:

    typedef SoftCascadeOverIntegralChannelsModel detector_t;
    typedef std::vector<detector_t> detectors_t;

    /// this constructor will copy the protobuf data into a more efficient data structure
    MultiScalesIntegralChannelsModel(const doppia_protobuf::MultiScalesDetectorModel &model);
    ~MultiScalesIntegralChannelsModel();

    const detectors_t& get_detectors() const;

    bool has_soft_cascade() const;

protected:

    detectors_t detectors;

    void sanity_check() const;
};

} // namespace doppia

#endif // BICLOP_MULTISCALESINTEGRALCHANNELSMODEL_HPP
