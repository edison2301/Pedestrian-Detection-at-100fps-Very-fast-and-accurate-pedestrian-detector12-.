#include "GpuIntegralChannelsDetector.hpp"

#include "gpu/integral_channels_detector.cu.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <cudatemplates/hostmemoryheap.hpp>
#include <cudatemplates/copy.hpp>

#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>

#include <opencv2/highgui/highgui.hpp>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "GpuIntegralChannelsDetector");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "GpuIntegralChannelsDetector");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "GpuIntegralChannelsDetector");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "GpuIntegralChannelsDetector");
}

} // end of anonymous namespace


namespace doppia {


using namespace std;
using namespace boost;
using namespace boost::program_options;

typedef GpuIntegralChannelsDetector::cascade_stages_t cascade_stages_t;
typedef cascade_stages_t::value_type cascade_stage_t;

typedef AbstractObjectsDetector::detections_t detections_t;
typedef AbstractObjectsDetector::detection_t detection_t;
typedef GpuIntegralChannelsDetector::detection_window_size_t detection_window_size_t;
/*typedef IntegralChannelsDetector::detections_scores_t detections_scores_t;
typedef IntegralChannelsDetector::stages_left_t stages_left_t;
typedef IntegralChannelsDetector::stages_left_in_the_row_t stages_left_in_the_row_t;

typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;
*/

options_description
GpuIntegralChannelsDetector::get_args_options()
{
    options_description desc("GpuIntegralChannelsDetector options");

    desc.add_options()

            ("objects_detector.gpu.frugal_memory_usage", value<bool>()->default_value(false),
             "By default we use as much GPU memory as useful for speeding things up. "
             "If frugal memory usage is enabled, we will reduce the memory usage, "
             "at the cost of longer computation time.")

            ;

    return desc;
}


GpuIntegralChannelsDetector::GpuIntegralChannelsDetector(
        const variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold, const int additional_border)
    : BaseIntegralChannelsDetector(options, cascade_model_p, non_maximal_suppression_p,
                                   score_threshold, additional_border),
      frugal_memory_usage(get_option_value<bool>(options, "objects_detector.gpu.frugal_memory_usage")),
      previous_resized_input_gpu_matrix_index(0),
      use_stump_cascades(false)
{

    // create the integral channels computer
    integral_channels_computer_p.reset(new GpuIntegralChannelsForPedestrians());

    // this number is fixed, but it is not very sensitive
    // at 50 scales, we usually have 0~300 detections.
    // If we ever approach the maximum, we could dynamically schedule GPU->CPU downloads of the detections,
    // so as to empty this buffer before calling the detector GPU kernel
    //const size_t max_num_gpu_detections = 1E4; // enough for shrinking factor 4
    const size_t max_num_gpu_detections = 1E7; // enough for shrinking factor 1
    gpu_detections.alloc(max_num_gpu_detections);

    if(true)
    {
        log_info() << "cv::gpu::CudaMem::canMapHostMemory() == " << cv::gpu::CudaMem::canMapHostMemory() << std::endl;
    }
    return;
}



GpuIntegralChannelsDetector::~GpuIntegralChannelsDetector()
{
    // nothing to do here
    return;
}


void GpuIntegralChannelsDetector::set_image(const boost::gil::rgb8c_view_t &input_view)
{
    const bool input_dimensions_changed =
            ((input_gpu_mat.cols != input_view.width())
             or (input_gpu_mat.rows != input_view.height()));

    // transfer image into GPU --
    {
        boost::gil::opencv::ipl_image_wrapper input_ipl =
                boost::gil::opencv::create_ipl_image(input_view);

        cv::Mat input_mat(input_ipl.get());

        //printf("input_ipl.get()->nChannels == %i\n", input_ipl.get()->nChannels);
        //printf("input_mat.type() == %i =?= %i\n", input_mat.type(), CV_8UC3);
        //printf("input_mat.channels() == %i\n", input_mat.channels());
        //printf("input_mat (height, width) == (%i, %i)\n", input_mat.rows, input_mat.cols);

        const bool use_cuda_write_combined = true;
        if(use_cuda_write_combined)
        {
            if((input_rgb8_gpu_mem.rows != input_mat.rows) or (input_rgb8_gpu_mem.cols != input_mat.cols))
            {
                // lazy allocate the cuda memory
                // using WRITE_COMBINED, in theory allows for 40% speed-up in the upload,
                // (but reading this memory from host will be _very slow_)
                // tests on the laptop show no speed improvement (maybe faster on desktop ?)
                input_rgb8_gpu_mem.create(input_mat.size(), input_mat.type(), cv::gpu::CudaMem::ALLOC_WRITE_COMBINED);
                input_rgb8_gpu_mem_mat = input_rgb8_gpu_mem.createMatHeader();
            }

            input_mat.copyTo(input_rgb8_gpu_mem_mat); // copy to write_combined host memory
            input_rgb8_gpu_mat.upload(input_rgb8_gpu_mem_mat);  // fast transfer from CPU to GPU
        }
        else
        {
            input_rgb8_gpu_mat.upload(input_mat);  // from CPU to GPU
        }

        //printf("input_rgb8_gpu_mat.channels() == %i\n", input_rgb8_gpu_mat.channels());

        // most tasks in GPU are optimized for CV_8UC1 and CV_8UC4, so we set the input as such
        cv::gpu::cvtColor(input_rgb8_gpu_mat, input_gpu_mat, CV_RGB2RGBA); // GPU type conversion

        if(input_gpu_mat.type() != CV_8UC4)
        {
            throw std::runtime_error("cv::gpu::cvtColor did not work as expected");
        }
    }

    // set default search range --
    if(input_dimensions_changed or search_ranges.empty())
    {
        log_debug() << boost::str(boost::format("::set_image resizing the search_ranges using input size (%i,%i)")
                                  % input_view.width() % input_view.height()) << std::endl;

        compute_search_ranges(input_view.dimensions(),
                              scale_one_detection_window_size,
                              search_ranges);

        // update the detection cascades
        compute_scaled_detection_cascades();
        set_gpu_scale_detection_cascades();

        // update additional, input size dependent, data
        compute_extra_data_per_scale(input_view.width(), input_view.height());

        // resize helper array
        resized_input_gpu_matrices.resize(search_ranges.size());
    } // end of "set default search range"

    return;
}


void GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades()
{
    if(detection_cascade_per_scale.empty())
    {
        throw std::runtime_error("GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades called, but "
                                 "detection_cascade_per_scale is empty");
    }

    // set the gpu_detection_cascade_per_scale --
    if((detection_cascade_per_scale.empty() == false)
       and (detection_cascade_per_scale.front().empty() == false))
    {
        const size_t
                cascades_length = detection_cascade_per_scale[0].size(),
                num_cascades = detection_cascade_per_scale.size();

        Cuda::HostMemoryHeap2D<cascade_stage_t> cpu_detection_cascade_per_scale(cascades_length, num_cascades);

        for(size_t cascade_index=0; cascade_index < num_cascades; cascade_index+=1)
        {
            if(cascades_length != detection_cascade_per_scale[cascade_index].size())
            {
                throw std::invalid_argument("Current version of GpuIntegralChannelsDetector requires "
                                            "multiscales models with equal number of weak classifiers");
                // FIXME how to fix this ?
                // Using cascade_score_threshold to stop when reached the "last stage" ?
                // Adding dummy stages with zero weight ?
            }

            for(size_t stage_index =0; stage_index < cascades_length; stage_index += 1)
            {
                const size_t index = stage_index + cascade_index*cpu_detection_cascade_per_scale.stride[0];
                cpu_detection_cascade_per_scale[index] = detection_cascade_per_scale[cascade_index][stage_index];
            } // end of "for each stage in the cascade"
        } // end of "for each cascade"

        if(false and ((num_cascades*cascades_length) > 0))
        {
            printf("GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades "
                   "Cascade 0, stage 0 cascade_threshold == %3.f\n",
                   cpu_detection_cascade_per_scale[0].cascade_threshold);
        }

        gpu_detection_cascade_per_scale.alloc(cascades_length, num_cascades);
        Cuda::copy(gpu_detection_cascade_per_scale, cpu_detection_cascade_per_scale);
    }

    // set the gpu_stump_detection_cascade_per_scale --
    if((detection_stump_cascade_per_scale.empty() == false)
       and (detection_stump_cascade_per_scale.front().empty() == false))
    {
        const size_t
                cascades_length = detection_stump_cascade_per_scale[0].size(),
                num_cascades = detection_stump_cascade_per_scale.size();

        use_stump_cascades = true;

        Cuda::HostMemoryHeap2D<stump_cascade_stage_t> cpu_detection_stump_cascade_per_scale(cascades_length, num_cascades);

        for(size_t cascade_index=0; cascade_index < num_cascades; cascade_index+=1)
        {
            if(cascades_length != detection_stump_cascade_per_scale[cascade_index].size())
            {
                throw std::invalid_argument("Current version of GpuIntegralChannelsDetector requires "
                                            "multiscales models with equal number of weak classifiers");
                // FIXME how to fix this ?
                // Using cascade_score_threshold to stop when reached the "last stage" ?
                // Adding dummy stages with zero weight ?
            }

            for(size_t stage_index =0; stage_index < cascades_length; stage_index += 1)
            {
                const size_t index = stage_index + cascade_index*cpu_detection_stump_cascade_per_scale.stride[0];
                cpu_detection_stump_cascade_per_scale[index] = detection_stump_cascade_per_scale[cascade_index][stage_index];
            } // end of "for each stage in the cascade"
        } // end of "for each cascade"

        if(false and ((num_cascades*cascades_length) > 0))
        {
            printf("GpuIntegralChannelsDetector::set_gpu_scale_detection_cascades "
                   "Cascade 0, stump stage 0 cascade_threshold == %3.f\n",
                   cpu_detection_stump_cascade_per_scale[0].cascade_threshold);
        }

        //printf("(re)Allocating %zix%zi cascade stumps\n", cascades_length, num_cascades);
        gpu_detection_stump_cascade_per_scale.alloc(cascades_length, num_cascades);
        Cuda::copy(gpu_detection_stump_cascade_per_scale, cpu_detection_stump_cascade_per_scale);
    }

    return;
}


size_t GpuIntegralChannelsDetector::get_input_width() const
{
    return input_gpu_mat.cols;
}


size_t GpuIntegralChannelsDetector::get_input_height() const
{
    return input_gpu_mat.rows;
}


void GpuIntegralChannelsDetector::compute()
{

    detections.clear();
    num_gpu_detections = 0; // no need to clean the buffer

    // some debugging variables
    static bool first_call = true;

    assert(integral_channels_computer_p);
    assert(gpu_detection_cascade_per_scale.getBuffer() != NULL);

    const bool use_v1 = true;
    //const bool use_v1 = false;

    // on Jabbah v0 runs at 2.3 Hz, and v1 at 2.55 Hz
    if(use_v1)
    {
        // for each range search
        for(size_t search_range_index=0; search_range_index < search_ranges.size(); search_range_index +=1)
        {
            compute_detections_at_specific_scale_v1(search_range_index, first_call);
        } // end of "for each search range"

        collect_the_gpu_detections();
    }
    else
    { // we use v0

        const bool save_score_image = false;
        //const bool save_score_image = true;

        // for each range search
        for(size_t search_range_index=0; search_range_index < search_ranges.size(); search_range_index +=1)
        {
            compute_detections_at_specific_scale_v0(search_range_index, save_score_image, first_call);
        } // end of "for each search range"

        if(save_score_image)
        {
            // stop everything
            throw std::runtime_error("Stopped the program so we can debug it. "
                                     "See the scores_at_*.png score images");
        }
    }

    log_info() << "number of raw (before non maximal suppression) detections on this frame == "
               << detections.size() << std::endl;

    // windows size adjustment should be done before non-maximal suppression
    if(this->resize_detection_windows)
    {
        (*model_window_to_object_window_converter_p)(detections);
    }

    compute_non_maximal_suppresion();

    first_call = false;

    return;
}


/// this is a variant of collect_the_detections(...) inside IntegralChannelsDetector.cpp
/// we use a cv::Mat instead of boost::multi_array
void collect_the_detections(
        const ScaleData &scale_data,
        const detection_window_size_t &original_detection_window_size,
        const cv::Mat &detections_scores_mat,
        const float detection_score_threshold,
        detections_t &detections,
        detections_t *non_rescaled_detections_p)
{

    const DetectorSearchRange &search_range = scale_data.scaled_search_range;
    const BaseIntegralChannelsDetector::stride_t &stride = scale_data.stride;

    using boost::math::iround;

    assert(search_range.range_scaling > 0);
    //log_debug() << "search_range.range_scaling == " << search_range.range_scaling << std::endl;

    // printf("original_col == col*%.3f\n", 1/search_range.range_scaling); // just for debugging

    for(uint16_t row=search_range.min_y; row < search_range.max_y; row += stride.y())
    {
        for(uint16_t col=search_range.min_x; col < search_range.max_x; col += stride.x())
        {
            const float &detection_score = detections_scores_mat.at<float>(row, col);

            // >= to be consistent with Markus's code
            if(detection_score >= detection_score_threshold)
            { // we got a detection, yey !

                add_detection(col, row, detection_score, scale_data, detections);

#if defined(BOOTSTRAPPING_LIB)
                if(non_rescaled_detections_p != NULL)
                {
                    add_detection_for_bootstrapping(col, row, detection_score,
                                                    original_detection_window_size,
                                                    *non_rescaled_detections_p);
                }
#endif
            }
            else
            {
                // not a detection, nothing to do
            }

        } // end of "for each column in the search range"
    } // end of "for each row in the search range"

    return;
}


doppia::objects_detection::gpu_integral_channels_t &
GpuIntegralChannelsDetector::resize_input_and_compute_integral_channels(
        const size_t search_range_index, const bool first_call)
{

    GpuIntegralChannelsForPedestrians &integral_channels_computer = *integral_channels_computer_p;

    // resize the image --
    size_t input_gpu_matrix_index = search_range_index;

    if(frugal_memory_usage)
    {
        // instead of keeping multiple resized matrices in GPU,
        // (to avoid dynamic memory allocation when the input image has stable size)
        // we keep only one, which we resize for each scale (slower, but less memory usage)
        input_gpu_matrix_index = 0; // index 0, means the first element
        // resized_input_gpu_matrices.empty() is expected to be false at this stage of the code.
    }

    cv::gpu::GpuMat &resized_input_gpu_mat = resized_input_gpu_matrices[input_gpu_matrix_index];
    {

        const image_size_t &scaled_input_image_size = extra_data_per_scale[search_range_index].scaled_input_image_size;

        if((scaled_input_image_size.x() == 0) or (scaled_input_image_size.y() == 0))
        {
            // empty image to process, nothing to do here
            // return;
            throw std::invalid_argument("Input scale and ratio ranges (and channel shrinking factor), "
                                        "generates rescaled images with zero pixels. This is not good.");
        }
        //else if(false and first_call)
        else if(true and first_call)
        //else if(true or first_call)
        {
            log_info()
                    << "scaled_x == " << scaled_input_image_size.x()
                    << ", scaled_y == " << scaled_input_image_size.y() << std::endl;
        }
        else if(scaled_input_image_size.x() < 5)
        {
            const DetectorSearchRange &original_search_range = search_ranges[search_range_index];
            printf("current scale == %.3f, current ratio == %.3f, current image width == %i, resized image width == %zi\n",
                   original_search_range.detection_window_scale,
                   original_search_range.detection_window_ratio,
                   input_gpu_mat.cols, scaled_input_image_size.x());
            throw std::invalid_argument("Input scale and ratio ranges (and channel shrinking factor), "
                                        "generates rescaled images with width < 5. This is too narrow.");
        }


        if(search_range_index > 0)
        { // if not on the first DetectorSearchRange
            // (when search_range_index ==0 we assume a new picture is being treated)

            const cv::gpu::GpuMat &previous_resized_gpu_mat = \
                    resized_input_gpu_matrices[previous_resized_input_gpu_matrix_index];

            if((scaled_input_image_size.x() == static_cast<size_t>(previous_resized_gpu_mat.cols))
               and (scaled_input_image_size.y() == static_cast<size_t>(previous_resized_gpu_mat.rows)))
            {
                if(first_call)
                {
                    log_debug() << "Skipped integral channels computation for search range index "
                                << search_range_index
                                << " (since redundant with previous computed one)"<< std::endl;
                }
                // current required scale, match the one already computed in the integral_channels_computer
                // no need to recompute the integral_channels, we provide the current result
                return integral_channels_computer.get_gpu_integral_channels();
            }
        }

        cv::gpu::resize(input_gpu_mat, resized_input_gpu_mat,
                        cv::Size(scaled_input_image_size.x(), scaled_input_image_size.y()));
    }


    // compute integral channels --
    {
        integral_channels_computer.set_image(resized_input_gpu_mat);
        integral_channels_computer.compute();
    }

    previous_resized_input_gpu_matrix_index = input_gpu_matrix_index;
    return integral_channels_computer.get_gpu_integral_channels();
}


void GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v0(
        const size_t search_range_index,
        const bool save_score_image,
        const bool first_call)
{
    doppia::objects_detection::gpu_integral_channels_t &integral_channels =
            resize_input_and_compute_integral_channels(search_range_index, first_call);

    const DetectorSearchRange &original_search_range = search_ranges[search_range_index];
    //const cascade_stages_t &cascade_stages = detection_cascade_per_scale[search_range_index];
    const detection_window_size_t &detection_window_size = detection_window_size_per_scale[search_range_index];
    const ScaleData &scale_data = extra_data_per_scale[search_range_index];


#if defined(BOOTSTRAPPING_LIB)
    current_image_scale = 1.0f/original_search_range.detection_window_scale;
    detections_t *non_rescaled_detections_p = &non_rescaled_detections;
#else
    detections_t *non_rescaled_detections_p = NULL;
#endif


    // compute the scores --
    int alloc_type = cv::gpu::CudaMem::ALLOC_PAGE_LOCKED;
    if(cv::gpu::CudaMem::canMapHostMemory())
    {
        alloc_type = cv::gpu::CudaMem::ALLOC_ZEROCOPY;
    }

    // adding max(128, ...) seems to be required to use the host memory<->cpu memory mapping
    // smaller memory allocations raise an exception when calling createGpuMatHeader
    // CUDA @ VISICS seems not happy with page_locked or zero_copy allocations,
    // with sizes not power of 2. We thus increase the size to match a power of 2.
    cv::gpu::CudaMem detection_scores_cuda_memory(
                max<uint16_t>(128, pow(2, ceil(log(scale_data.scaled_search_range.max_y)/log(2)))),
                max<uint16_t>(128, pow(2, ceil(log(scale_data.scaled_search_range.max_x)/log(2)))),
                cv::DataType<float>::type, alloc_type);
    cv::Mat detection_scores_mat;

    {
        cv::gpu::GpuMat detection_scores_gpu_mat = detection_scores_cuda_memory.createGpuMatHeader();
        cv::gpu::DevMem2Df detection_scores_device_memory = detection_scores_gpu_mat;

        // compute the scores
        if(use_stump_cascades)
        {
            doppia::objects_detection::integral_channels_detector(
                        integral_channels,
                        search_range_index,
                        scale_data.scaled_search_range,
                        gpu_detection_stump_cascade_per_scale,
                        use_the_detector_model_cascade,
                        detection_scores_device_memory);
        }
        else
        {
            doppia::objects_detection::integral_channels_detector(
                        integral_channels,
                        search_range_index,
                        scale_data.scaled_search_range,
                        gpu_detection_cascade_per_scale,
                        use_the_detector_model_cascade,
                        detection_scores_device_memory);
        }
        detection_scores_mat = detection_scores_cuda_memory.createMatHeader();

        if(false)
        { // just for debugging
            printf("detection_scores_cuda_memory.cols/rows == (%i, %i), step == %zi\n",
                   detection_scores_cuda_memory.cols, detection_scores_cuda_memory.rows, detection_scores_cuda_memory.step);

            printf("detection_scores_mat.cols/rows == (%i, %i), step == %zi, %zi\n",
                   detection_scores_mat.cols, detection_scores_mat.rows, detection_scores_mat.step[0], detection_scores_mat.step[1]);

            printf("detection_scores_device_memory.cols/rows == (%i, %i), step == %zi\n",
                   detection_scores_device_memory.cols, detection_scores_device_memory.rows, detection_scores_device_memory.step);
        }
    }

    // collect the detection windows --
    {
        collect_the_detections(
                    scale_data, detection_window_size,
                    detection_scores_mat, score_threshold,
                    detections, non_rescaled_detections_p);
    }


    if(save_score_image)
    {
        // we crop the detection_scores_mat due to the max(128, scale_data.scaled_search_range.max_x/y)
        /*cv::Mat scores_mat = detection_scores_mat(cv::Rect(0, 0,
                                                           scale_data.scaled_search_range.max_x,
                                                           scale_data.scaled_search_range.max_y));*/
        cv::Mat scores_mat = detection_scores_mat; // FIXME just for debugging

        if(scores_mat.empty() == false)
        {
            printf("scores_mat(0,0) == %.3f\n", scores_mat.at<float>(0,0));
        }

        cv::Mat normalized_scores;
        double min_score, max_score;
        cv::minMaxLoc(scores_mat, &min_score, &max_score);
        cv::normalize(scores_mat, normalized_scores, 255, 0, cv::NORM_MINMAX);

        const float original_detection_window_scale = original_search_range.detection_window_scale;
        const string filename = str(format("gpu_scores_at_%.2f.png") % original_detection_window_scale);
        cv::imwrite(filename, normalized_scores);
        log_info() << "Created debug file " << filename << std::endl;
        log_info() << str(format("Scores in %s are in the range (min, max) == (%.3f, %.3f)")
                          % filename % min_score % max_score) << std::endl;

        if(false and min_score > 0)
        {
            throw std::runtime_error("Min score should be 0 sometimes. Something is fishy in this test.");
        }
    }

    return;
}


void GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v1(
        const size_t search_range_index,
        const bool first_call)
{

    doppia::objects_detection::gpu_integral_channels_t &integral_channels =
            resize_input_and_compute_integral_channels(search_range_index, first_call);

    const ScaleData &scale_data = extra_data_per_scale[search_range_index];

    // const stride_t &actual_stride = scale_data.stride;
    // on current GPU code the stride is ignored, and all pixels of each scale are considered (~x/y_stride == 1E-10)
    // FIXME either consider the strides (not a great idea, using stixels is better), or print a warning at run time

    // compute the scores --
    if(use_stump_cascades)
    {
        // compute the detections, and keep the results on the gpu memory
        doppia::objects_detection::integral_channels_detector(
                    integral_channels,
                    search_range_index,
                    scale_data.scaled_search_range,
                    gpu_detection_stump_cascade_per_scale, score_threshold, use_the_detector_model_cascade,
                    gpu_detections, num_gpu_detections);
    }
    else
    {
        // compute the detections, and keep the results on the gpu memory
        doppia::objects_detection::integral_channels_detector(
                    integral_channels,
                    search_range_index,
                    scale_data.scaled_search_range,
                    gpu_detection_cascade_per_scale, score_threshold, use_the_detector_model_cascade,
                    gpu_detections, num_gpu_detections);
    }

    // ( the detections will be colected after iterating over all the scales )

#if defined(BOOTSTRAPPING_LIB)
    throw std::runtime_error("GpuIntegralChannelsDetector::compute_detections_at_specific_scale_v1 "
                             "should not be used inside bootstrapping_lib, use v0 instead");
#endif
    return;
}



void GpuIntegralChannelsDetector::collect_the_gpu_detections()
{

    if(num_gpu_detections == 0)
    {
        // nothing to do,
        //calling Cuda::copy for 0 elements will fail
        return;
    }

    // transfering the detections from gpu to cpu --
    if(num_gpu_detections > gpu_detections.getSize())
    {
        log_warning() << boost::str(
                             boost::format("num_gpu_detections == %i, but max_num_gpu_detections == %i.\n"
                                           "There are more detections than the allocated memory buffer.")
                             % num_gpu_detections %  gpu_detections.getSize()) << std::endl;

        throw std::runtime_error("Received more gpu detection than the allocated memory buffer");
    }

    // we expect num_retreivable_detections == num_gpu_detections, but we still handle the corner cases
    const size_t num_retreivable_detections = std::min(gpu_detections.getSize(), num_gpu_detections);
    std::vector<gpu_detection_t> gpu_detections_on_cpu(num_retreivable_detections);

    Cuda::HostMemoryReference1D<gpu_detection_t> host_ref(gpu_detections_on_cpu.size(), gpu_detections_on_cpu.data());

    // we copy the first num_gpu_detections elements
    Cuda::copy(host_ref, gpu_detections, Cuda::Size<1>(0), Cuda::Size<1>(0), Cuda::Size<1>(num_retreivable_detections));

    // converting the gpu data into actual detection windows --
    BOOST_FOREACH(const gpu_detection_t &gpu_detection, gpu_detections_on_cpu)
    {
        const ScaleData &scale_data = extra_data_per_scale[gpu_detection.scale_index];

        add_detection(gpu_detection.x, gpu_detection.y, gpu_detection.score,
                      scale_data, detections);

    } // end of "for each gpu detection"

    return;
}



} // end of namespace doppia
