#ifndef BICLOP_GPUFASTESTPEDESTRIANDETECTORINTHEWEST_HPP
#define BICLOP_GPUFASTESTPEDESTRIANDETECTORINTHEWEST_HPP

#include "GpuIntegralChannelsDetector.hpp"
#include "BaseFastestPedestrianDetectorInTheWest.hpp"

namespace doppia {

/// This is is the GPU version of FastestPedestrianDetectorInTheWestV2,
/// this version exploits fractional channel features (using the GPU fast texture interpolation features)
/// @see FastestPedestrianDetectorInTheWestV2
class GpuFastestPedestrianDetectorInTheWest:
                public GpuIntegralChannelsDetector, public BaseFastestPedestrianDetectorInTheWest

{
public:

    /// Where we store the fractional cascade information
    /// @see GpuIntegralChannelsDetector
    typedef Cuda::DeviceMemoryLinear2D<fractional_cascade_stage_t> gpu_fractional_detection_cascade_per_scale_t;

    GpuFastestPedestrianDetectorInTheWest(
            const boost::program_options::variables_map &options,
            boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~GpuFastestPedestrianDetectorInTheWest();

protected:

    gpu_fractional_detection_cascade_per_scale_t gpu_fractional_detection_cascade_per_scale;
    void set_gpu_scale_detection_cascades();

    void set_gpu_scale_fractional_detection_cascades();

    /// computes the detections directly on GPU, avoiding the score image transfer
    void compute_detections_at_specific_scale_v1(const size_t search_range_index,
                                                 const bool first_call = false);



};

} // end of namespace doppia

#endif // BICLOP_GPUFASTESTPEDESTRIANDETECTORINTHEWEST_HPP
