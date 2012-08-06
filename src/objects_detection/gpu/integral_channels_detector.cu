#include "integral_channels_detector.cu.hpp"

#include "helpers/gpu/cuda_safe_call.hpp"

#include "cudatemplates/array.hpp"
#include "cudatemplates/symbol.hpp"
#include "cudatemplates/copy.hpp"

#include <cudatemplates/hostmemoryheap.hpp>

#include <boost/cstdint.hpp>

#include <stdexcept>



namespace {

/// small helper function that computes "static_cast<int>(ceil(static_cast<float>(total)/grain))", but faster
static inline int div_up(const int total, const int grain)
{
    return (total + grain - 1) / grain;
}


} // end of anonymous namespace


namespace doppia {
namespace objects_detection {

using namespace cv::gpu;

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 1>::Texture gpu_integral_channels_1d_texture_t;
gpu_integral_channels_1d_texture_t integral_channels_1d_texture;

typedef Cuda::DeviceMemory<gpu_integral_channels_t::Type, 2>::Texture gpu_integral_channels_2d_texture_t;
gpu_integral_channels_2d_texture_t integral_channels_2d_texture;

// global variable to switch from using 1d textures to using 2d textures
// On visics-gt680r 1d texture runs at ~4.8 Hz; 2d texture runs at ~4.4 Hz
//const bool use_2d_texture = false;
const bool use_2d_texture = true;

/// this method will do zero memory checks, the user is responsible of avoiding out of memory accesses
inline
__device__
float get_feature_value_global_memory(const IntegralChannelsFeature &feature,
                                      const int x, const int y,
                                      const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride,
            top_left_index     = (x + box.min_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            top_right_index    = (x + box.max_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            bottom_left_index  = (x + box.min_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset,
            bottom_right_index = (x + box.max_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset;

    const gpu_integral_channels_t::Type
            a = integral_channels.data[top_left_index],
            b = integral_channels.data[top_right_index],
            c = integral_channels.data[bottom_right_index],
            d = integral_channels.data[bottom_left_index];

    const float feature_value = a +c -b -d;

    return feature_value;
}


inline
__device__
float get_feature_value_tex1d(const IntegralChannelsFeature &feature,
                              const int x, const int y,
                              const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride;
    const size_t
            top_left_index     = (x + box.min_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            top_right_index    = (x + box.max_corner().x()) + ((y + box.min_corner().y())*row_stride) + channel_offset,
            bottom_left_index  = (x + box.min_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset,
            bottom_right_index = (x + box.max_corner().x()) + ((y + box.max_corner().y())*row_stride) + channel_offset;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_1d_texture_t &tex = integral_channels_1d_texture;
#define tex integral_channels_1d_texture
    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    // tex1Dfetch should be used to access linear memory (not text1D)
    const float
            a = tex1Dfetch(tex, top_left_index),
            b = tex1Dfetch(tex, top_right_index),
            c = tex1Dfetch(tex, bottom_right_index),
            d = tex1Dfetch(tex, bottom_left_index);
#undef tex


    const float feature_value = a +c -b -d;

    return feature_value;
}



/// This is a dumb compability code, IntegralChannelsFractionalFeature is meant to be used with
/// get_feature_value_tex2d
inline
__device__
float get_feature_value_tex1d(const IntegralChannelsFractionalFeature &feature,
                              const int x, const int y,
                              const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    const IntegralChannelsFractionalFeature::rectangle_t &box = feature.box;

    const size_t
            &channel_stride = integral_channels.stride[1],
            &row_stride = integral_channels.stride[0];

    // if x or y are too high, some of these indices may be fall outside the channel memory
    const size_t
            channel_offset = feature.channel_index*channel_stride;
    const size_t
            top_left_index =
            (x + size_t(box.min_corner().x())) + ((y + size_t(box.min_corner().y()))*row_stride) + channel_offset,
            top_right_index =
            (x + size_t(box.max_corner().x())) + ((y + size_t(box.min_corner().y()))*row_stride) + channel_offset,
            bottom_left_index =
            (x + size_t(box.min_corner().x())) + ((y + size_t(box.max_corner().y()))*row_stride) + channel_offset,
            bottom_right_index =
            (x + size_t(box.max_corner().x())) + ((y + size_t(box.max_corner().y()))*row_stride) + channel_offset;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_1d_texture_t &tex = integral_channels_1d_texture;
#define tex integral_channels_1d_texture
    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    // tex1Dfetch should be used to access linear memory (not text1D)
    const float
            a = tex1Dfetch(tex, top_left_index),
            b = tex1Dfetch(tex, top_right_index),
            c = tex1Dfetch(tex, bottom_right_index),
            d = tex1Dfetch(tex, bottom_left_index);
#undef tex
    const float feature_value = a +c -b -d;

    return feature_value;
}



template <typename FeatureType>
inline
__device__
float get_feature_value_tex2d(const FeatureType &feature,
                              const int x, const int y,
                              const gpu_3d_integral_channels_t::KernelConstData &integral_channels)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory

    const size_t integral_channels_height = integral_channels.size[1];
    const float y_offset = y + feature.channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    const typename FeatureType::rectangle_t &box = feature.box;

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    const float
            a = tex2D(tex, x + box.min_corner().x(), box.min_corner().y() + y_offset), // top left
            b = tex2D(tex, x + box.max_corner().x(), box.min_corner().y() + y_offset), // top right
            c = tex2D(tex, x + box.max_corner().x(), box.max_corner().y() + y_offset), // bottom right
            d = tex2D(tex, x + box.min_corner().x(), box.max_corner().y() + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;

    return feature_value;
}


template <typename FeatureType>
inline
__device__
float get_feature_value_tex2d(const FeatureType &feature,
                              const int x, const int y,
                              const gpu_2d_integral_channels_t::KernelConstData &integral_channels)
{
    // if x or y are too high, some of these indices may be fall outside the channel memory

    //const size_t integral_channels_height = integral_channels.size[1];
    const size_t integral_channels_height = 120; // FIXME HARDCODED TEST (will only work with shrinking factor 4 and images of size 640x480
    const float y_offset = y + feature.channel_index*integral_channels_height;

    // in CUDA 5 (4.2 ?) references to textures are not allowed, we use macro work around
    //    gpu_integral_channels_2d_texture_t &tex = integral_channels_2d_texture;
#define tex integral_channels_2d_texture

    const typename FeatureType::rectangle_t &box = feature.box;

    //const gpu_integral_channels_t::Type  // could cause overflows during a + c
    const float
            a = tex2D(tex, x + box.min_corner().x(), box.min_corner().y() + y_offset), // top left
            b = tex2D(tex, x + box.max_corner().x(), box.min_corner().y() + y_offset), // top right
            c = tex2D(tex, x + box.max_corner().x(), box.max_corner().y() + y_offset), // bottom right
            d = tex2D(tex, x + box.min_corner().x(), box.max_corner().y() + y_offset); // bottom left
#undef tex

    const float feature_value = a +c -b -d;

    return feature_value;
}


template <typename FeatureType, bool should_use_2d_texture>
inline
__device__
float get_feature_value(const FeatureType &feature,
                        const int x, const int y,
                        const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    // default implementation (hopefully optimized by the compiler)
    if (should_use_2d_texture)
    {
        return get_feature_value_tex2d(feature, x, y, integral_channels);
    }
    else
    {
        //return get_feature_value_global_memory(feature, x, y, integral_channels);
        return get_feature_value_tex1d(feature, x, y, integral_channels);
    }

    //return 0;
}


template <>
inline
__device__
float get_feature_value<IntegralChannelsFractionalFeature, true>(
        const IntegralChannelsFractionalFeature &feature,
        const int x, const int y,
        const gpu_integral_channels_t::KernelConstData &integral_channels)
{
    // should_use_2d_texture == true
    return get_feature_value_tex2d(feature, x, y, integral_channels);
}




inline
__device__
bool evaluate_decision_stump(const DecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    if(feature_value >= stump.feature_threshold)
    {
        return stump.larger_than_threshold;
    }
    else
    {
        return not stump.larger_than_threshold;
    }
}


inline
__device__
bool evaluate_decision_stump(const SimpleDecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold);
}


inline
__device__
float evaluate_decision_stump(const DecisionStumpWithWeights &stump,
                              const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold)? stump.weight_true_leaf : stump.weight_false_leaf;
}

inline
__device__
bool evaluate_decision_stump(const SimpleFractionalDecisionStump &stump,
                             const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold);
}


inline
__device__
float evaluate_decision_stump(const FractionalDecisionStumpWithWeights &stump,
                              const float &feature_value)
{
    // uses >= to be consistent with Markus Mathias code
    // uses >= to be consistent with Markus Mathias code
    return (feature_value >= stump.feature_threshold)? stump.weight_true_leaf : stump.weight_false_leaf;
}


template<typename CascadeStageType>
inline
__device__
void update_detection_score(
        const int x, const int y,
        const CascadeStageType &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_score)
{
    const typename CascadeStageType::weak_classifier_t &weak_classifier = stage.weak_classifier;
    typedef typename CascadeStageType::weak_classifier_t::feature_t feature_t;

    // level 1 nodes evaluation returns a boolean value,
    // level 2 nodes evaluation returns directly the float value to add to the score

    const float level1_feature_value =
            get_feature_value<feature_t, use_2d_texture>(
                weak_classifier.level1_node.feature, x, y, integral_channels);

    // On preliminary versions,
    // evaluating the level2 features inside the if/else
    // runs slightly faster than evaluating all of them beforehand; 4.35 Hz vs 4.55 Hz)
    // on the fastest version (50 Hz or more) evaluating all three features is best

    const bool use_if_else = false;
    if(not use_if_else)
    { // this version is faster

        const float level2_true_feature_value =
                get_feature_value<feature_t, use_2d_texture>(
                    weak_classifier.level2_true_node.feature, x, y, integral_channels);

        const float level2_false_feature_value =
                get_feature_value<feature_t, use_2d_texture>(
                    weak_classifier.level2_false_node.feature, x, y, integral_channels);

        current_score +=
                (evaluate_decision_stump(weak_classifier.level1_node, level1_feature_value)) ?
                    evaluate_decision_stump(weak_classifier.level2_true_node, level2_true_feature_value) :
                    evaluate_decision_stump(weak_classifier.level2_false_node, level2_false_feature_value);
    }
    else
    {
        if(evaluate_decision_stump(weak_classifier.level1_node, level1_feature_value))
        {
            const float level2_true_feature_value =
                    get_feature_value<feature_t, use_2d_texture>(
                        weak_classifier.level2_true_node.feature, x, y, integral_channels);

            current_score += evaluate_decision_stump(weak_classifier.level2_true_node, level2_true_feature_value);
        }
        else
        {
            const float level2_false_feature_value =
                    get_feature_value<feature_t, use_2d_texture>(
                        weak_classifier.level2_false_node.feature, x, y, integral_channels);

            current_score +=evaluate_decision_stump(weak_classifier.level2_false_node, level2_false_feature_value);
        }

    }

    return;
}



template<>
inline
__device__
void update_detection_score<SoftCascadeOverIntegralChannelsStumpStage>(
        const int x, const int y,
        const SoftCascadeOverIntegralChannelsStumpStage &stage,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        float &current_detection_score)
{
    const typename SoftCascadeOverIntegralChannelsStumpStage::weak_classifier_t &weak_classifier = stage.weak_classifier;
    typedef SoftCascadeOverIntegralChannelsStumpStage::weak_classifier_t::feature_t feature_t;

    const float feature_value =
            get_feature_value<feature_t, use_2d_texture>(
                weak_classifier.feature, x, y, integral_channels);

    current_detection_score += evaluate_decision_stump(weak_classifier, feature_value);

    return;
}

// FIXME these should be templated options, selected at runtime
//const bool use_hardcoded_cascade = true;
const bool use_hardcoded_cascade = false;
//const int hardcoded_cascade_start_stage = 100;
const int hardcoded_cascade_start_stage = 500;


/// this kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is a mirror of the CPU method compute_cascade_stage_on_row(...) inside IntegralChannelsDetector.cpp
/// @see IntegralChannelsDetector

template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_kernel(
        const int search_range_width, const int search_range_height,
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const size_t scale_index,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        PtrElemStepf detection_scores)
{
    const int
            x = blockIdx.x * blockDim.x + threadIdx.x,
            //y = blockIdx.y;
            y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x >= search_range_width) or ( y >= search_range_height))
    {
        // out of area of interest
        return;
    }

    //const bool print_cascade_scores = false; // just for debugging

    // retrieve current score value
    float detection_score = 0; //detection_scores_row[x];

    const size_t
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    for(size_t stage_index=0; stage_index < cascade_length; stage_index+=1)
    {
        const size_t index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        update_detection_score(x, y, stage, integral_channels, detection_score);

        // (printfs in cuda code requires at least -arch=sm_20) (we use -arch=sm_21)
        // should not print too much from the GPU, to avoid opencv's timeouts
        /*if(print_cascade_scores
                and (y == 52) and (x == 4)
                //and (y == 203) and (x == 101)
                and (stage_index < 10))
        {

            const DetectionCascadeStageType::weak_classifier_t &weak_classifier = stage.weak_classifier;
            const float level1_feature_value =
                    get_feature_value_tex1d(weak_classifier.level1_node.feature,
                                            x, y, integral_channels);

            const float level2_true_feature_value =
                    get_feature_value_tex1d(weak_classifier.level2_true_node.feature,
                                            x, y, integral_channels);

            const float level2_false_feature_value =
                    get_feature_value_tex1d(weak_classifier.level2_false_node.feature,
                                            x, y, integral_channels);

            printf("Cascade score at (%i, %i),\tstage %i == %2.3f,\tthreshold == %.3f,\t"
                   "level1_feature == %.3f,\tlevel2_true_feature == %4.3f,\tlevel2_false_feature == %4.3f\n",
                   x, y, stage_index,
                   detection_score, stage.cascade_threshold,
                   level1_feature_value, level2_true_feature_value, level2_false_feature_value);

            printf("level1 threshold %.3f, level2_true threshold %.3f, level2_false threshold %.3f\n",
                   weak_classifier.level1_node.feature_threshold,
                   weak_classifier.level2_true_node.feature_threshold,
                   weak_classifier.level2_false_node.feature_threshold);
        }*/



        if((not use_the_model_cascade) and
           use_hardcoded_cascade and (stage_index > hardcoded_cascade_start_stage) and (detection_score < 0))
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            break;
        }

        if(use_the_model_cascade and detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"
            break;
        }

    } // end of "for each stage"


    float* detection_scores_row = detection_scores.ptr(y);
    detection_scores_row[x] = detection_score; // save the updated score
    //detection_scores_row[x] = cascade_length; // just for debugging
    return;
}


/// type int because atomicAdd does not support size_t
__device__ int num_gpu_detections[1];
Cuda::Symbol<int, 1> num_gpu_detections_symbol(Cuda::Size<1>(1), num_gpu_detections);
int num_detections_int;
Cuda::HostMemoryReference1D<int> num_detections_host_ref(1, &num_detections_int);


void move_num_detections_from_cpu_to_gpu(size_t &num_detections)
{ // move num_detections from CPU to GPU --
    num_detections_int = static_cast<int>(num_detections);
    Cuda::copy(num_gpu_detections_symbol, num_detections_host_ref);
    return;
}

void move_num_detections_from_gpu_to_cpu(size_t &num_detections)
{ // move (updated) num_detections from GPU to CPU
    Cuda::copy(num_detections_host_ref, num_gpu_detections_symbol);
    if(num_detections_int < static_cast<int>(num_detections))
    {
        throw std::runtime_error("Something went terribly wrong when updating the number of gpu detections");
    }
    num_detections = static_cast<size_t>(num_detections_int);
    return;
}



template<typename ScaleType>
inline
__device__
void add_detection(
        gpu_detections_t::KernelData &gpu_detections,
        const int x, const int y, const ScaleType scale_index,
        const float detection_score)
{
    gpu_detection_t detection;
    detection.scale_index = static_cast<boost::int16_t>(scale_index);
    detection.x = static_cast<boost::int16_t>(x);
    detection.y = static_cast<boost::int16_t>(y);
    detection.score = detection_score;

    const size_t detection_index = atomicAdd(num_gpu_detections, 1);
    if(detection_index < gpu_detections.size[0])
    {
        // copy the detection into the global memory
        gpu_detections.data[detection_index] = detection;
    }
    else
    {
        // we drop out of range detections
    }

    return;
}


/// this kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is a mirror of the CPU method compute_cascade_stage_on_row(...) inside IntegralChannelsDetector.cpp
/// @see IntegralChannelsDetector
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_kernel(
        const int search_range_width, const int search_range_height,
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const size_t scale_index,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            x = blockIdx.x * blockDim.x + threadIdx.x,
            //y = blockIdx.y;
            y = blockIdx.y * blockDim.y + threadIdx.y;

    if((x >= search_range_width) or ( y >= search_range_height))
    {
        // out of area of interest
        return;
    }

    //const bool print_cascade_scores = false; // just for debugging

    // retrieve current score value
    float detection_score = 0;

    const size_t
            cascade_length = detection_cascade_per_scale.size[0],
            //cascade_length = 1000, // FIXME MEGA HARCODED VALUE DANGER DANGER <<<<<<<<<<<<<<<<
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    for(size_t stage_index=0; stage_index < cascade_length; stage_index+=1)
    {
        const size_t index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        update_detection_score(x, y, stage, integral_channels, detection_score);

        if((not use_the_model_cascade) and
           use_hardcoded_cascade and (stage_index > hardcoded_cascade_start_stage) and (detection_score < 0))
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            break;
        }

        if(use_the_model_cascade and detection_score < stage.cascade_threshold)
        {
            // this is not an object of the class we are looking for
            // do an early stop of this pixel
            detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"
            break;
        }

    } // end of "for each stage"


    // >= to be consistent with Markus's code
    if(detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);
    }

    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 1d binding
void bind_integral_channels_to_1d_texture(gpu_integral_channels_t &integral_channels)
{

    //integral_channels_texture.filterMode = cudaFilterModeLinear; // linear interpolation of the values
    integral_channels_1d_texture.filterMode = cudaFilterModePoint; // normal access to the values
    //integral_channels.bindTexture(integral_channels_texture);

    // cuda does not support binding 3d memory data.
    // We will hack this and bind the 3d data, as if it was 1d data,
    // and then have ad-hoc texture access in the kernel
    // (if interpolation is needed, will need to do a different 2d data hack
    const cudaChannelFormatDesc texture_channel_description = \
            cudaCreateChannelDesc<gpu_integral_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
       or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc failed");
    }

    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // FIXME add this 3D to 1D strategy into cudatemplates
    CUDA_CHECK(cudaBindTexture(0, integral_channels_1d_texture, integral_channels.getBuffer(),
                               texture_channel_description, integral_channels.getBytes()));

    cuda_safe_call( cudaGetLastError() );

    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 2d binding
void bind_integral_channels_to_2d_texture(gpu_3d_integral_channels_t &integral_channels)
{ // 3d integral channels case

    // linear interpolation of the values, only valid for floating point types
    //integral_channels_2d_texture.filterMode = cudaFilterModeLinear;
    integral_channels_2d_texture.filterMode = cudaFilterModePoint; // normal access to the values
    //integral_channels.bindTexture(integral_channels_2d_texture);

    // cuda does not support binding 3d memory data.
    // We will hack this and bind the 3d data, as if it was 2d data,
    // and then have ad-hoc texture access in the kernel
    const cudaChannelFormatDesc texture_channel_description = cudaCreateChannelDesc<gpu_3d_integral_channels_t::Type>();

    if(texture_channel_description.f == cudaChannelFormatKindNone
       or texture_channel_description.f != cudaChannelFormatKindUnsigned )
    {
        throw std::runtime_error("cudaCreateChannelDesc seems to have failed");
    }

    if(false)
    {
        printf("texture_channel_description.x == %i\n", texture_channel_description.x);
        printf("texture_channel_description.y == %i\n", texture_channel_description.y);
        printf("texture_channel_description.z == %i\n", texture_channel_description.z);
        printf("texture_channel_description.w == %i\n", texture_channel_description.w);
    }

    // Layout.size is width, height, num_channels
    const size_t
            integral_channels_width = integral_channels.getLayout().size[0],
            integral_channels_height = integral_channels.getLayout().size[1],
            num_integral_channels = integral_channels.getLayout().size[2],
            //channel_stride = integral_channels.getLayout().stride[1],
            row_stride = integral_channels.getLayout().stride[0],
            pitch_in_bytes = row_stride * sizeof(gpu_3d_integral_channels_t::Type);

    if(false)
    {
        printf("image width/height == %zi, %zi; row_stride == %zi\n",
               integral_channels.getLayout().size[0], integral_channels.getLayout().size[1],
               integral_channels.getLayout().stride[0]);

        printf("integral_channels size / channel_stride == %.3f\n",
               integral_channels.getLayout().stride[2] / float(integral_channels.getLayout().stride[1]) );
    }

    // FIXME add this 3D to 2D strategy into cudatemplates
    CUDA_CHECK(cudaBindTexture2D(0, integral_channels_2d_texture, integral_channels.getBuffer(),
                                 texture_channel_description,
                                 integral_channels_width, integral_channels_height*num_integral_channels,
                                 pitch_in_bytes));

    cuda_safe_call( cudaGetLastError() );
    return;
}


/// helper method to map the device memory to the specific texture reference
/// This specific implementation will do a 2d binding
void bind_integral_channels_to_2d_texture(gpu_2d_integral_channels_t &integral_channels)
{ // 2d integral channels case

    // linear interpolation of the values, only valid for floating point types
    //integral_channels_2d_texture.filterMode = cudaFilterModeLinear;
    integral_channels_2d_texture.filterMode = cudaFilterModePoint; // normal access to the values

    // integral_channels_height == (channel_height*num_channels) + 1
    // 2d to 2d binding
    integral_channels.bindTexture(integral_channels_2d_texture);

    cuda_safe_call( cudaGetLastError() );
    return;
}



void bind_integral_channels_texture(gpu_integral_channels_t &integral_channels)
{
    if(use_2d_texture)
    {
        bind_integral_channels_to_2d_texture(integral_channels);
    }
    else
    {
        bind_integral_channels_to_1d_texture(integral_channels);
    }

    return;
}


void unbind_integral_channels_texture()
{

    if(use_2d_texture)
    {
        cuda_safe_call( cudaUnbindTexture(integral_channels_2d_texture) );
    }
    else
    {
        cuda_safe_call( cudaUnbindTexture(integral_channels_1d_texture) );
    }
    cuda_safe_call( cudaGetLastError() );

    return;
}


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_impl(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        cv::gpu::DevMem2Df& detection_scores)
{

    if((search_range.min_x != 0) or (search_range.min_y != 0))
    {
        printf("search_range.min_x/y == (%i, %i)\n", search_range.min_x, search_range.min_y);
        throw std::runtime_error("integral_channels_detector(...) expect search_range.min_x/y values to be zero");
    }

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;
    const int width = search_range.max_x, height = search_range.max_y;

    if((width > detection_scores.cols) or (height > detection_scores.rows))
    {
        printf("search_range.max_x/y == (%i, %i)\n", search_range.max_x, search_range.max_y);
        printf("detection_scores.cols/rows == (%i, %i)\n", detection_scores.cols, detection_scores.rows);
        throw std::runtime_error("integral_channels_detector(...) expects "
                                 "detections_scores to be larger than the search_range.max_x/y values");
    }


    if(false)
    {
        printf("detection_cascade_per_scale.size[0] == %zi\n", detection_cascade_per_scale.size[0]);
        printf("detection_cascade_per_scale.stride[0] == %zi\n", detection_cascade_per_scale.stride[0]);
        printf("detection_cascade_per_scale.size[1] == %zi\n", detection_cascade_per_scale.size[1]);
    }


    //const int nthreads = 256; dim3 block_dimensions(nthreads, 1);
    //dim3 block_dimensions(16, 16);
    //dim3 block_dimensions(32, 32);
    //const int nthreads = 320; // we optimize for images of width 640 pixel
    dim3 block_dimensions(32, 10);
    dim3 grid_dimensions(div_up(width, block_dimensions.x), div_up(height, block_dimensions.y));

    // bind the integral_channels_texture
    bind_integral_channels_texture(integral_channels);

    if(use_the_model_cascade)
    {

        integral_channels_detector_kernel
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       detection_scores);
    }
    else
    {
        integral_channels_detector_kernel
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       detection_scores);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    unbind_integral_channels_texture();
    return;
}


void integral_channels_detector(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        cv::gpu::DevMem2Df& detection_scores)
{
    integral_channels_detector_impl(integral_channels,
                                    search_range_index,
                                    search_range,
                                    detection_cascade_per_scale,
                                    use_the_model_cascade,
                                    detection_scores);
    return;
}


void integral_channels_detector(
        gpu_integral_channels_t &integral_channels,
        const size_t search_range_index,
        const doppia::DetectorSearchRange &search_range,
        gpu_detection_stump_cascade_per_scale_t &detection_cascade_per_scale,
        const bool use_the_model_cascade,
        cv::gpu::DevMem2Df& detection_scores)
{
    integral_channels_detector_impl(integral_channels,
                                    search_range_index,
                                    search_range,
                                    detection_cascade_per_scale,
                                    use_the_model_cascade,
                                    detection_scores);
    return;
}

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

/// this method directly adds elements into the gpu_detections vector
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_impl(gpu_integral_channels_t &integral_channels,
                                     const size_t search_range_index,
                                     const doppia::DetectorSearchRange &search_range,
                                     GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
                                     const float score_threshold,
                                     const bool use_the_model_cascade,
                                     gpu_detections_t& gpu_detections,
                                     size_t &num_detections)
{

    if((search_range.min_x != 0) or (search_range.min_y != 0))
    {
        printf("search_range.min_x/y == (%i, %i)\n", search_range.min_x, search_range.min_y);
        throw std::runtime_error("integral_channels_detector(...) expect search_range.min_x/y values to be zero");
    }

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;
    const int width = search_range.max_x, height = search_range.max_y;

    //const int nthreads = 320; // we optimize for images of width 640 pixel
    //dim3 block_dimensions(32, 10);
    //const int block_x = std::max(4, width/5), block_y = std::max(1, 256/block_x);

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            block_x = 16,
            block_y = num_threads / block_x;
    dim3 block_dimensions(block_x, block_y);

    dim3 grid_dimensions(div_up(width, block_dimensions.x), div_up(height, block_dimensions.y));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    if(use_the_model_cascade)
    {
        integral_channels_detector_kernel
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       score_threshold,
                                                       gpu_detections);
    }
    else
    {
        integral_channels_detector_kernel
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>>
                                                      (width, height,
                                                       integral_channels,
                                                       search_range_index,
                                                       detection_cascade_per_scale,
                                                       score_threshold,
                                                       gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::DetectorSearchRange &search_range,
                                gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{    
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, search_range, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}

void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::DetectorSearchRange &search_range,
                                gpu_detection_stump_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, search_range, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}


void integral_channels_detector(gpu_integral_channels_t &integral_channels,
                                const size_t search_range_index,
                                const doppia::DetectorSearchRange &search_range,
                                gpu_fractional_detection_cascade_per_scale_t &detection_cascade_per_scale,
                                const float score_threshold,
                                const bool use_the_model_cascade,
                                gpu_detections_t& gpu_detections,
                                size_t &num_detections)
{
    // call the templated generic implementation
    integral_channels_detector_impl(integral_channels, search_range_index, search_range, detection_cascade_per_scale,
                                    score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


/// This kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is based on integral_channels_detector_kernel,
/// but does detections for all scales instead of a single scale
/// @see IntegralChannelsDetector
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v0(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const int max_search_range_min_x, const int max_search_range_min_y,
        const int max_search_range_width, const int max_search_range_height,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y;

    if((delta_x >= max_search_range_width) or (delta_y >= max_search_range_height))
    {
        // out of area of interest
        return;
    }

    const int
            x = max_search_range_min_x + delta_x,
            y = max_search_range_min_y + delta_y;

    float max_detection_score = -1E10; // initialized with a very negative value
    int max_detection_scale_index = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    const int
            cascade_length = detection_cascade_per_scale.size[0],
            num_scales = scales_data.size[0];

    for(int scale_index=0; scale_index < num_scales; scale_index +=1)
    {
        // when using softcascades, __syncthread here is a significant slow down, so not using it

        // we copy the search stage from global memory to thread memory
        // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
        const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
        // (in current code, we ignore the gpu_scale_datum_t stride value)


        // we order the if conditions putting most likely ones first
        if( (y > search_range.max_corner().y())
            or (y < search_range.min_corner().y())
            or (x < search_range.min_corner().x())
            or (x > search_range.max_corner().x()) )
        {
            // current pixel is out of this scale search range, we skip computations
            continue;
        }

        //bool should_skip_scale = false;

        // retrieve current score value
        float detection_score = 0;

        const int scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        for(int stage_index=0; stage_index < cascade_length; stage_index+=1)
        {
            const int index = scale_offset + stage_index;

            // we copy the cascade stage from global memory to thread memory
            // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
            const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

            update_detection_score(x, y, stage, integral_channels, detection_score);

            if((not use_the_model_cascade) and
               use_hardcoded_cascade and (stage_index > hardcoded_cascade_start_stage) and (detection_score < 0))
            {
                // this is not an object of the class we are looking for
                // do an early stop of this pixel

                // FIXME this is an experiment
                // since this is a really bad detection, we also skip one scale
                //scale_index += 1;
                //scale_index += 3;
                //scale_index += 10;
                //should_skip_scale = true;
                break;
            }

            if(use_the_model_cascade
               //and scale_index > 100 // force evaluation of the first 100 stages
               and detection_score < stage.cascade_threshold)
            {
                // this is not an object of the class we are looking for
                // do an early stop of this pixel
                detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"

                // FIXME this is an experiment
                // since this is a really bad detection, we also skip one scale
                //if(stage_index < hardcoded_cascade_start_stage)
                {
                    //scale_index += 1;
                    //scale_index += 3;
                }

                const bool use_hardcoded_scale_skipping_from_cascade = false;
                if(use_hardcoded_scale_skipping_from_cascade)
                {

                    // These are the "good enough" values (7.6% curve in INRIA, slightly worse than best results)
                    // these values give a slowdown instead of the desired speed-up
                    /*                    if(stage_index < 10)
                    {
                        scale_index += 5;
                    }
                    else  if(stage_index < 20)
                    {
                        scale_index += 2;
                    }
                    else if(stage_index < 75)
                    {
                        scale_index += 1;
                    }*/

                    // This provide overlap with FPDW, speed up to 17 Hz from 15 Hz
                    if(stage_index < 20)
                    {
                        scale_index += 10;
                    }
                    else if(stage_index < 100)
                    {
                        scale_index += 4;
                    }
                    else if(stage_index < 200)
                    {
                        scale_index += 2;
                    }
                    else if(stage_index < 300)
                    {
                        scale_index += 1;
                    }

                }

                break;
            }


        } // end of "for each stage"

        if(detection_score > max_detection_score)
        {
            max_detection_score = detection_score;
            max_detection_scale_index = scale_index;
        }

        const bool use_hardcoded_scale_skipping = false;
        if(use_hardcoded_scale_skipping)
        {
            // these values are only valid when not using a soft-cascade

            // these are the magic numbers for the INRIA dataset,
            // when using 2011_11_03_1800_full_multiscales_model.proto.bin
            if(detection_score < -0.3)
            {
                scale_index += 11;
            }
            else if(detection_score < -0.2)
            {
                scale_index += 4;
            }
            else if(detection_score < -0.1)
            {
                scale_index += 2;
            }
            else if(detection_score < 0)
            {
                scale_index += 1;
            }
            else
            {
                // no scale jump
            }

        }

        /*
        // we only skip the scale if all the pixels in the warp agree
        if(__all(should_skip_scale))
        {
            // FIXME this is an experiment
            //scale_index += 1;
            scale_index += 3;
            //scale_index += 10;
        }
*/

    } // end of "for each scale"


    // >= to be consistent with Markus's code
    if(max_detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, max_detection_scale_index, max_detection_score);

    } // end of "if detection score is high enough"

    return;
} // end of integral_channels_detector_over_all_scales_kernel




/// This kernel is called for each position where we which to detect objects
/// we assume that the border effects where already checked when computing the DetectorSearchRange
/// thus we do not do any checks here.
/// This kernel is based on integral_channels_detector_kernel,
/// but does detections for all scales instead of a single scale
/// @see IntegralChannelsDetector
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
// _v1 is slower than _v0
void integral_channels_detector_over_all_scales_kernel_v1(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const int max_search_range_min_x, const int max_search_range_min_y,
        const int max_search_range_width, const int max_search_range_height,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            thread_id = blockDim.x*threadIdx.y + threadIdx.x;


    const int num_threads = 192;
    __shared__  DetectionCascadeStageType cascade_stages[num_threads];

    const int
            x = max_search_range_min_x + delta_x,
            y = max_search_range_min_y + delta_y;

    float max_detection_score = -1E10; // initialized with a very negative value
    int max_detection_scale_index = -1;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    const int
            cascade_length = detection_cascade_per_scale.size[0],
            num_scales = scales_data.size[0];
    for(int scale_index=0; scale_index < num_scales; scale_index +=1)
    {
        // FIXME _sync, will not work as it is when skipping the scales
        // should create next_scale_to_evaluate variable; if (scale_index < next_scale_to_evaluate) continue;
        __syncthreads(); // all threads should move from one scale to the next at the same peace
        // (adding this syncthreads allowed to move from 1.53 Hz to 1.59 Hz)

        // we copy the search stage from global memory to thread memory
        // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
        const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
        // (in current code, we ignore the gpu_scale_datum_t stride value)


        const int scale_offset = scale_index * detection_cascade_per_scale.stride[0];

        const bool inside_search_range = ( (y >= search_range.min_corner().y())
                                           and (y < search_range.max_corner().y())
                                           and (x >= search_range.min_corner().x())
                                           and (x < search_range.max_corner().x()) );

        bool should_update_score = inside_search_range;
        //bool should_skip_scale = false;

        // retrieve current score value
        float detection_score = 0;

        if(inside_search_range == false)
        {
            detection_score = -1E10; // we set to a very negative value
        }

        int max_loaded_stage = 0, stage_index_modulo = 0;

        for(int stage_index=0; stage_index < cascade_length; stage_index+=1, stage_index_modulo+=1)
        {

            if(stage_index >= max_loaded_stage)
            { // time to reload the next batch of cascade stages
                //__syncthreads();

                const int thread_stage_index = max_loaded_stage + thread_id;
                if(thread_stage_index < cascade_length)
                {
                    cascade_stages[thread_id] = detection_cascade_per_scale.data[scale_offset + thread_stage_index];
                }

                __syncthreads();
                max_loaded_stage += num_threads;
                stage_index_modulo = 0;
            }


            if(should_update_score)
            {
                // copying the stage is faster than referring to the shared memory
                const DetectionCascadeStageType stage = cascade_stages[stage_index_modulo];
                //const DetectionCascadeStageType stage = detection_cascade_per_scale.data[scale_offset + stage_index];

                update_detection_score(x, y, stage, integral_channels, detection_score);


                if(use_the_model_cascade and detection_score < stage.cascade_threshold)
                {
                    // this is not an object of the class we are looking for
                    // do an early stop of this pixel
                    detection_score = -1E5; // since re-ordered classifiers may have a "very high threshold in the middle"

                    // FIXME this is an experiment
                    // since this is a really bad detection, we also skip one scale
                    //if(stage_index < hardcoded_cascade_start_stage)
                    {
                        //scale_index += 1;
                        //scale_index += 3;
                    }

                    should_update_score = false;
                } // end of "if crossed detection cascade"

            } // end of "should update score"

        } // end of "for each stage"

        if(detection_score > max_detection_score)
        {
            max_detection_score = detection_score;
            max_detection_scale_index = scale_index;
        }

        const bool use_hardcoded_scale_skipping = false;
        if(use_hardcoded_scale_skipping)
        {
            // these values are only valid when not using a soft-cascade

            // these are the magic numbers for the INRIA dataset,
            // when using 2011_11_03_1800_full_multiscales_model.proto.bin
            if(detection_score < -0.3)
            {
                scale_index += 11;
            }
            else if(detection_score < -0.2)
            {
                scale_index += 4;
            }
            else if(detection_score < -0.1)
            {
                scale_index += 2;
            }
            else if(detection_score < 0)
            {
                scale_index += 1;
            }
            else
            {
                // no scale jump
            }

        }

        /*
        // we only skip the scale if all the pixels in the warp agree
        if(__all(should_skip_scale))
        {
            // FIXME this is an experiment
            //scale_index += 1;
            scale_index += 3;
            //scale_index += 10;
        }
*/

    } // end of "for each scale"


    // >= to be consistent with Markus's code
    if(max_detection_scale_index >= 0 and max_detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, max_detection_scale_index, max_detection_score);

    } // end of "if detection score is high enough"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v1


/// _v2 is significantly faster than _v0
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v2(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const int max_search_range_min_x, const int max_search_range_min_y,
        const int /*max_search_range_width*/, const int /*max_search_range_height*/,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            x = max_search_range_min_x + delta_x,
            y = max_search_range_min_y + delta_y;

    float max_detection_score = -1E10; // initialized with a very negative value
    int max_detection_scale_index = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    const int
            cascade_length = detection_cascade_per_scale.size[0],
            num_scales = scales_data.size[0];

    for(int scale_index=0; scale_index < num_scales; scale_index +=1)
    {
        // when using softcascades, __syncthreads here is a significant slow down, so not using it

        // we copy the search stage from global memory to thread memory
        // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
        const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
        // (in current code, we ignore the gpu_scale_datum_t stride value)


        // we order the if conditions putting most likelly ones first
        if( (y > search_range.max_corner().y())
            or (y < search_range.min_corner().y())
            or (x < search_range.min_corner().x())
            or (x > search_range.max_corner().x()) )
        {
            // current pixel is out of this scale search range, we skip computations
            // (nothing to do here)
        }
        else
        { // inside search range

            // retrieve current score value
            float detection_score = 0;

            const int scale_offset = scale_index * detection_cascade_per_scale.stride[0];

            for(int stage_index=0; stage_index < cascade_length; stage_index+=1)
            {
                const int index = scale_offset + stage_index;

                // we copy the cascade stage from global memory to thread memory
                // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
                const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

                if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
                {
                    update_detection_score(x, y, stage, integral_channels, detection_score);
                }
                else
                {
                    // detection score is below cascade threshold,
                    // we are not interested on this object
                    detection_score = -1E5; // set to a value lower than score_threshold
                    break;
                }

            } // end of "for each stage"

            if(detection_score > max_detection_score)
            {
                max_detection_score = detection_score;
                max_detection_scale_index = scale_index;
            }

        } // end of "inside search range or not"

    } // end of "for each scale"


    // >= to be consistent with Markus's code
    if(max_detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, max_detection_scale_index, max_detection_score);

    } // end of "if detection score is high enough"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v2


/// this function will evaluate the detection score of a specific window,
/// it is assumed that border checks have already been done
/// this method will directly call add_detection(...) if relevant
/// used in integral_channels_detector_over_all_scales_kernel_v3 and superior
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
inline
__device__
void compute_specific_detection(
        const int x, const int y, const int scale_index,
        const float score_threshold,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData &detection_cascade_per_scale,
        gpu_detections_t::KernelData &gpu_detections)
{
    const int
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    // retrieve current score value
    float detection_score = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    int stage_index=0;
    for(; stage_index < cascade_length; stage_index+=1)
    {
        const int index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        if(use_the_model_cascade and (detection_score <= stage.cascade_threshold))
        {
            // detection score is below cascade threshold,
            // we are not interested on this object
            break;
        }

        update_detection_score(x, y, stage, integral_channels, detection_score);

    } // end of "for each stage"

    // >= to be consistent with Markus's code
    if((detection_score >= score_threshold) and (stage_index >= cascade_length))
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);

    } // end of "if detection score is high enough"

    return;
}


/// _v3 should only be used with integral_channels_detector_over_all_scales_impl_v1
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v3(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            delta_x = blockIdx.x * blockDim.x + threadIdx.x,
            delta_y = blockIdx.y * blockDim.y + threadIdx.y,
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t::search_range_t search_range = scales_data.data[scale_index].search_range;
    // (in current code, we ignore the gpu_scale_datum_t stride value)

    const int
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        compute_specific_detection<use_the_model_cascade, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v3



/// _v4 should only be used with integral_channels_detector_over_all_scales_impl_v2
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
__global__
void integral_channels_detector_over_all_scales_kernel_v4_xy_stride(
        const gpu_integral_channels_t::KernelConstData integral_channels,
        const gpu_scales_data_t::KernelConstData scales_data,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData detection_cascade_per_scale,
        const float score_threshold,
        gpu_detections_t::KernelData gpu_detections)
{
    const int
            scale_index = blockIdx.z * blockDim.z + threadIdx.z,
            num_scales = scales_data.size[0];

    if(scale_index >= num_scales)
    {
        // out of scales range
        // (nothing to do here)
        return;
    }

    // we copy the search stage from global memory to thread memory
    // (here using copy or reference seems to make no difference, 11.82 Hz in both cases)
    const gpu_scale_datum_t scale_datum = scales_data.data[scale_index];
    const gpu_scale_datum_t::search_range_t &search_range = scale_datum.search_range;
    const gpu_scale_datum_t::stride_t &stride = scale_datum.stride;

    const int
            delta_x = (blockIdx.x * blockDim.x + threadIdx.x)*stride.x(),
            delta_y = (blockIdx.y * blockDim.y + threadIdx.y)*stride.y(),
            x = search_range.min_corner().x() + delta_x,
            y = search_range.min_corner().y() + delta_y;

    // we order the if conditions putting most likelly ones first
    if( (y > search_range.max_corner().y()) or (x > search_range.max_corner().x()) )
    {
        // current pixel is out of this scale search range, we skip computations
        // (nothing to do here)
    }
    else
    { // inside search range

        compute_specific_detection<use_the_model_cascade, DetectionCascadeStageType>
                (x, y, scale_index, score_threshold,
                 integral_channels, detection_cascade_per_scale, gpu_detections);

    } // end of "inside search range or not"

    return;
} // end of integral_channels_detector_over_all_scales_kernel_v4


/// this method directly adds elements into the gpu_detections vector
template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v0(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;
    const int
            max_search_range_min_x = max_search_range.min_corner().x(),
            max_search_range_min_y = max_search_range.min_corner().y(),
            max_search_range_width = max_search_range.max_corner().x() - max_search_range_min_x,
            max_search_range_height = max_search_range.max_corner().y() - max_search_range_min_y;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            block_y = 2, block_x = num_threads / block_y; // slightly faster than block_y = 4

    dim3 block_dimensions(block_x, block_y);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // _v1 is slower than _v0; we should use _v0
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v2
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          max_search_range_min_x, max_search_range_min_y,
                                                          max_search_range_width, max_search_range_height,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v2
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          max_search_range_min_x, max_search_range_min_y,
                                                          max_search_range_width, max_search_range_height,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
}


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v1(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            block_z = 8,
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4

    dim3 block_dimensions(block_x, block_y, block_z);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y),
                         div_up(num_scales, block_z));


    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v3
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v3
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v1


template<typename GpuDetectionCascadePerScaleType>
void integral_channels_detector_over_all_scales_impl_v2_xy_stride(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        GpuDetectionCascadePerScaleType &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{

    typedef typename GpuDetectionCascadePerScaleType::Type CascadeStageType;

    // CUDA occupancy calculator pointed out
    // 192 (or 256) threads as a sweet spot for the current setup (revision 1798:ebfd7914cdfd)
    const int
            num_scales = scales_data.getNumElements(),
            //block_z = 8,
            block_z = 1,
            num_threads = 192, // ~4.8 Hz
            //num_threads = 256, // ~4.5 Hz
            // we want to keep the vertical elements of the block low so that we can efficiently search
            // in the scales that have strong vertical constraints
            //block_y = 4, block_x = num_threads / block_y; // runs at ~ 15 Hz too
            //block_y = 2, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4
            block_y = 16, block_x = num_threads / (block_y * block_z); // slightly faster than block_y = 4

    // FIXME should use the stride information when setting the block sizes ?

    dim3 block_dimensions(block_x, block_y, block_z);

    dim3 grid_dimensions(div_up(max_search_range_width, block_dimensions.x),
                         div_up(max_search_range_height, block_dimensions.y),
                         div_up(num_scales, block_z));

    // prepare variables for kernel call --
    bind_integral_channels_texture(integral_channels);
    move_num_detections_from_cpu_to_gpu(num_detections);

    // call the GPU kernel --
    // v1 is slower than v0; v3 is the fastest, we should use v3
    // v4 considers the strides, making it faster than v4 (when using strides > 1)
#if defined(USE_GENERATED_CODE)
    static bool first_call = true;
    if(first_call)
    {
        printf("Using integral_channels_detector_over_all_scales_kernel_v4_generated\n");
        first_call = false;
    }

    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v4_generated
                <true>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          score_threshold,
                                                          gpu_detections);



    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v4_generated
                <false>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          score_threshold,
                                                          gpu_detections);
    }
#else
    if(use_the_model_cascade)
    {
        integral_channels_detector_over_all_scales_kernel_v4_xy_stride
                <true, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
    else
    {
        integral_channels_detector_over_all_scales_kernel_v4_xy_stride
                <false, CascadeStageType>
                <<<grid_dimensions, block_dimensions>>> (
                                                          integral_channels,
                                                          scales_data,
                                                          detection_cascade_per_scale,
                                                          score_threshold,
                                                          gpu_detections);
    }
#endif // end of "use generated detector files or not"

    cuda_safe_call( cudaGetLastError() );
    cuda_safe_call( cudaDeviceSynchronize() );

    // clean-up variables after kernel call --
    unbind_integral_channels_texture();
    move_num_detections_from_gpu_to_cpu(num_detections);

    return;
} // end of integral_channels_detector_over_all_scales_impl_v2_xy_stride





/// this function will evaluate the detection score of a specific window,
/// it is assumed that border checks have already been done
/// this method will directly call add_detection(...) if relevant
/// used in integral_channels_detector_over_all_scales_kernel_v6_two_pixels_per_thread
/// will modify should_evaluate_neighbour based on the hints from the score progress
template <bool use_the_model_cascade, typename DetectionCascadeStageType>
inline
__device__
void compute_specific_detection(
        const int x, const int y, const int scale_index,
        const float score_threshold, bool &should_evaluate_neighbour,
        const gpu_integral_channels_t::KernelConstData &integral_channels,
        const typename Cuda::DeviceMemory<DetectionCascadeStageType, 2>::KernelConstData &detection_cascade_per_scale,
        gpu_detections_t::KernelData &gpu_detections)
{
    const int
            cascade_length = detection_cascade_per_scale.size[0],
            scale_offset = scale_index * detection_cascade_per_scale.stride[0];

    // retrieve current score value
    float detection_score = 0;

    // (we use int instead of size_t, as indicated by Cuda Best Programming Practices, section 6.3)
    // (using int of size_t moved from 1.48 Hz to 1.53 Hz)

    int stage_index = 0;
    for(; stage_index < cascade_length; stage_index+=1)
    {
        const int index = scale_offset + stage_index;

        // we copy the cascade stage from global memory to thread memory
        // (when using a reference code runs at ~4.35 Hz, with copy it runs at ~4.55 Hz)
        const DetectionCascadeStageType stage = detection_cascade_per_scale.data[index];

        if((not use_the_model_cascade) or detection_score > stage.cascade_threshold)
        {
            update_detection_score(x, y, stage, integral_channels, detection_score);
        }
        else
        {
            // detection score is below cascade threshold,
            // we are not interested on this object
            detection_score = -1E5; // set to a value lower than score_threshold
            break;
        }

    } // end of "for each stage"

    // if we got far in the cascade, then we know that this was "close to be a detection"
    should_evaluate_neighbour = stage_index > (cascade_length / 3);

    // >= to be consistent with Markus's code
    if(detection_score >= score_threshold)
    {
        // we got a detection
        add_detection(gpu_detections, x, y, scale_index, detection_score);

    } // end of "if detection score is high enough"

    return;
}



void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{
    // call the templated generic implementation
    // v1 is faster than v0 (?)
    // v2 handles the strides (v0 and v1 omit them), reaches ~19 Hz in monocular mode, ~57 Hz with ground plane and strides (4,8)
    // v3 handles strides and is fully coalesced, reaches ~18 Hz in monocular mode, ~54 Hz with ground plane and strides (4,8)
    // v2 in unuk reaches 55 Hz, v3 56 Hz (monocular mode)
    //integral_channels_detector_over_all_scales_impl_v0(
    //integral_channels_detector_over_all_scales_impl_v1(
    integral_channels_detector_over_all_scales_impl_v2_xy_stride( // this is the version you want
                integral_channels,
                max_search_range, max_search_range_width, max_search_range_height,
                scales_data,
                detection_cascade_per_scale,
                score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}


void integral_channels_detector_over_all_scales(
        gpu_integral_channels_t &integral_channels,
        gpu_scale_datum_t::search_range_t &max_search_range,
        const int max_search_range_width, const int max_search_range_height,
        gpu_scales_data_t &scales_data,
        gpu_fractional_detection_cascade_per_scale_t &detection_cascade_per_scale,
        const float score_threshold,
        const bool use_the_model_cascade,
        gpu_detections_t& gpu_detections,
        size_t &num_detections)
{
    // call the templated generic implementation
    // v1 is faster than v0 (?)
    // v2 handles the strides (v0 and v1 omit them)
    // v3 handles strides but is slower than v2
    //integral_channels_detector_over_all_scales_impl_v0(
    //integral_channels_detector_over_all_scales_impl_v1(
    integral_channels_detector_over_all_scales_impl_v2_xy_stride(
                integral_channels,
                max_search_range, max_search_range_width, max_search_range_height,
                scales_data,
                detection_cascade_per_scale,
                score_threshold, use_the_model_cascade, gpu_detections, num_detections);
    return;
}

} // end of namespace objects_detection
} // end of namespace doppia

#include "integral_channels_detector_with_stixels.cuda_include"

