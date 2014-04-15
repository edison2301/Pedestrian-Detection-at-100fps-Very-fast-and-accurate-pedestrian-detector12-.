#ifndef GPUINTEGRALCHANNELSDETECTOR_HPP
#define GPUINTEGRALCHANNELSDETECTOR_HPP

#include "integral_channels/GpuIntegralChannelsForPedestrians.hpp"

#include "BaseIntegralChannelsDetector.hpp"

#include "SoftCascadeOverIntegralChannelsModel.hpp"
#include "IntegralChannelsDetector.hpp"

#include "gpu/integral_channels_detector.cu.hpp" // for gpu_detections_t

//#include <cudatemplates/symbol.hpp>
#include <cudatemplates/devicememorypitched.hpp>
#include <cudatemplates/devicememorylinear.hpp>

#include <opencv2/core/version.hpp>
#if CV_MINOR_VERSION <= 3
#include <opencv2/gpu/gpu.hpp> // opencv 2.3
#else
#include <opencv2/core/gpumat.hpp> // opencv 2.4
#include <opencv2/gpu/gpu.hpp> // opencv 2.4, for CudaMem (copied by hand from the svn)
#endif


#include <boost/program_options.hpp>

#include <vector>

namespace doppia {

class SoftCascadeOverIntegralChannelsModel;
typedef doppia::objects_detection::gpu_detections_t gpu_detections_t;
typedef gpu_detections_t::Type gpu_detection_t;

/// This is the GPU variant of IntegralChannelsDetector
/// @see IntegralChannelsDetector
class GpuIntegralChannelsDetector: public virtual BaseIntegralChannelsDetector
{

public:

    //typedef Cuda::Symbol2D<cascade_stage_t> gpu_detection_cascade_per_scale_t;

    // using Pitched allocation for a custom type makes us run into troubles,
    // we used linear allocation, even is slightly less efficient GPU access (to be fixed...)
    //typedef Cuda::DeviceMemoryPitched2D<cascade_stage_t> gpu_detection_cascade_per_scale_t;
    typedef Cuda::DeviceMemoryLinear2D<cascade_stage_t> gpu_detection_cascade_per_scale_t;

    typedef Cuda::DeviceMemoryLinear2D<stump_cascade_stage_t> gpu_detection_stump_cascade_per_scale_t;

public:

    static boost::program_options::options_description get_args_options();

    GpuIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold, const int additional_border);
    ~GpuIntegralChannelsDetector();

    void set_image(const boost::gil::rgb8c_view_t &input_image);
    void compute();

protected:

    const bool frugal_memory_usage;

    boost::scoped_ptr<GpuIntegralChannelsForPedestrians> integral_channels_computer_p;

    cv::gpu::CudaMem input_rgb8_gpu_mem; // allows for faster data upload
    cv::Mat input_rgb8_gpu_mem_mat; // allows for faster data upload
    cv::gpu::GpuMat input_rgb8_gpu_mat, input_gpu_mat;

    /// helper vector avoid unecessary GPU re-allocations
    std::vector<cv::gpu::GpuMat> resized_input_gpu_matrices;
    size_t previous_resized_input_gpu_matrix_index;

    gpu_detection_cascade_per_scale_t gpu_detection_cascade_per_scale;
    gpu_detection_stump_cascade_per_scale_t gpu_detection_stump_cascade_per_scale;
    bool use_stump_cascades;
    virtual void set_gpu_scale_detection_cascades();

    gpu_detections_t gpu_detections;
    size_t num_gpu_detections;

    size_t get_input_width() const;
    size_t get_input_height() const;


    /// shared section of code between
    /// compute_detections_at_specific_scale_v0 and compute_detections_at_specific_scale_v1
    doppia::objects_detection::gpu_integral_channels_t &
    resize_input_and_compute_integral_channels(const size_t search_range_index,
                                               const bool first_call = false);


    /// computes the score image on GPU, transfer to CPU and find the detections
    void compute_detections_at_specific_scale_v0(const size_t search_range_index,
                                                 const bool save_score_image = false,
                                                 const bool first_call = false);

    /// computes the detections directly on GPU, avoiding the score image transfer
    virtual void compute_detections_at_specific_scale_v1(const size_t search_range_index,
                                                         const bool first_call);

    void collect_the_gpu_detections();


#if defined(BOOTSTRAPPING_LIB)
    friend class bootstrapping::FalsePositivesDataCollector;
    float current_image_scale;

    /// we keep the detections on the rescaled image coordinates,
    /// instead of mapping them back to the original image scale
    detections_t non_rescaled_detections;
#endif

};

} // end of namespace doppia

#endif // GPUINTEGRALCHANNELSDETECTOR_HPP
