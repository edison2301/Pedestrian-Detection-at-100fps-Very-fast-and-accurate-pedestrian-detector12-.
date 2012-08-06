#ifndef BICLOP_BASEMULTISCALESINTEGRALCHANNELSDETECTOR_HPP
#define BICLOP_BASEMULTISCALESINTEGRALCHANNELSDETECTOR_HPP

#include "BaseIntegralChannelsDetector.hpp"
#include "MultiScalesIntegralChannelsModel.hpp"

#include <boost/program_options/variables_map.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {

/// Common code shared by MultiscalesIntegralChannelsDetector and GpuMultiscalesIntegralChannelsDetector
/// (declaring BaseIntegralChannelsDetector as virtual inheritance means that
/// children of BaseMultiscalesIntegralChannelsDetector must be also children of BaseIntegralChannelsDetector)
/// http://www.parashift.com/c++-faq-lite/multiple-inheritance.html#faq-25.9
class BaseMultiscalesIntegralChannelsDetector: public virtual BaseIntegralChannelsDetector
{

protected:
    /// the constructor is protected because this base class is should not be instanciated directly
    BaseMultiscalesIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p);
    ~BaseMultiscalesIntegralChannelsDetector();

protected:

    boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p;

    /// updates the values inside detection_cascade_per_scale
    /// this variant will also update search_ranges,
    /// (since we will be shifting the actual scales)
    void compute_scaled_detection_cascades();

};


} // end of namespace doppia

#endif // BICLOP_BASEMULTISCALESINTEGRALCHANNELSDETECTOR_HPP
