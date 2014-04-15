#include "GpuFastestPedestrianDetectorInTheWest.hpp"

#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/copy.hpp>

namespace doppia {

GpuFastestPedestrianDetectorInTheWest::GpuFastestPedestrianDetectorInTheWest(
        const boost::program_options::variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold, const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   cascade_model_p,
                                   non_maximal_suppression_p, score_threshold, additional_border),
      GpuIntegralChannelsDetector(
          options,
          cascade_model_p, non_maximal_suppression_p,
          score_threshold, additional_border),
      BaseFastestPedestrianDetectorInTheWest(options)
{

    return;
}


GpuFastestPedestrianDetectorInTheWest::~GpuFastestPedestrianDetectorInTheWest()
{
    // nothing to do here
    return;
}


const bool use_fractional_features = false;
//const bool use_fractional_features = true;


void GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_detection_cascades()
{
    if(use_fractional_features)
    { // we state fractional cascade stages, instead of the usual cascade stages
        set_gpu_scale_fractional_detection_cascades();
    }
    else
    {
        GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades();
    }
    return;
}


/// This function is a copy+paste+minor_edits of GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades
void GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_fractional_detection_cascades()
{
    // FIXME copy and paste is bad ? Making this a templated function would be better ?
    // (seems more complication for little benefit. Applying the "three is too much" rule,
    // right now, only two copies, so it is ok)

    if(fractional_detection_cascade_per_scale.empty())
    {
        throw std::runtime_error(
                    "GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_detection_fractional_cascades called, but "
                    "fractional_detection_cascade_per_scale is empty");
    }

    const size_t
            cascades_length = fractional_detection_cascade_per_scale[0].size(),
            num_cascades = fractional_detection_cascade_per_scale.size();

    Cuda::HostMemoryHeap2D<fractional_cascade_stage_t>
            cpu_fractional_detection_cascade_per_scale(cascades_length, num_cascades);

    for(size_t cascade_index=0; cascade_index < num_cascades; cascade_index+=1)
    {
        if(cascades_length != fractional_detection_cascade_per_scale[cascade_index].size())
        {
            throw std::invalid_argument("Current version of GpuFastestPedestrianDetectorInTheWest requires "
                                        "multiscales models with equal number of weak classifiers");
            // FIXME how to fix this ?
            // Using cascade_score_threshold to stop when reached the "last stage" ?
            // Adding dummy stages with zero weight ?
        }

        for(size_t stage_index =0; stage_index < cascades_length; stage_index += 1)
        {
            const size_t index = stage_index + cascade_index*cpu_fractional_detection_cascade_per_scale.stride[0];
            cpu_fractional_detection_cascade_per_scale[index] = \
                    fractional_detection_cascade_per_scale[cascade_index][stage_index];
        } // end of "for each stage in the cascade"
    } // end of "for each cascade"

    if(false and ((num_cascades*cascades_length) > 0))
    {
        printf("GpuFastestPedestrianDetectorInTheWest::set_gpu_scale_fractional_detection_cascades "
               "Cascade 0, stage 0 cascade_threshold == %3.f\n",
               cpu_fractional_detection_cascade_per_scale[0].cascade_threshold);
    }

    gpu_fractional_detection_cascade_per_scale.alloc(cascades_length, num_cascades);
    Cuda::copy(gpu_fractional_detection_cascade_per_scale, cpu_fractional_detection_cascade_per_scale);

    return;
}



void GpuFastestPedestrianDetectorInTheWest::compute_detections_at_specific_scale_v1(
        const size_t search_range_index,
        const bool first_call)
{

    if(use_fractional_features == false)
    {
        GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v1(
                    search_range_index, first_call);
        return;
    }

    doppia::objects_detection::gpu_integral_channels_t &integral_channels =
            resize_input_and_compute_integral_channels(search_range_index, first_call);

    const ScaleData &scale_data = extra_data_per_scale[search_range_index];

    // const stride_t &actual_stride = scale_data.stride;
    // on current GPU code the stride is ignored, and all pixels of each scale are considered (~x/y_stride == 1E-10)
    // FIXME either consider the strides (not a great idea, using stixels is better), or print a warning at run time

    // compute the scores --
    {
        // compute the detections, and keep the results on the gpu memory
        doppia::objects_detection::integral_channels_detector(
                    integral_channels,
                    search_range_index,
                    scale_data.scaled_search_range,
                    gpu_fractional_detection_cascade_per_scale,
                    score_threshold, use_the_detector_model_cascade,
                    gpu_detections, num_gpu_detections);
    }

    // ( the detections will be colected after iterating over all the scales )

#if defined(BOOTSTRAPPING_LIB)
    current_image_scale = 1.0f/search_ranges[search_range_index].detection_window_scale;
#endif

    return;
}



} // end of namespace doppia
