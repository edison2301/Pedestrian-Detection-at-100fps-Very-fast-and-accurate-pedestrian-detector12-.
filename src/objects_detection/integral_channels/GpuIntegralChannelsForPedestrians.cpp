#include "GpuIntegralChannelsForPedestrians.hpp"


#if not defined(USE_GPU)
#error "Should use -DUSE_GPU to compile this file"
#endif

#include "gpu/integral_channels.cu.hpp"
#include "gpu/shrinking.cu.hpp"

#include "helpers/Log.hpp"
#include "helpers/gpu/cuda_safe_call.hpp"

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/extension/opencv/ipl_image_wrapper.hpp>
#include <boost/type_traits/is_same.hpp>

#include <cudatemplates/hostmemoryreference.hpp>
#include <cudatemplates/copy.hpp>
#include <cudatemplates/devicememoryreference.hpp>
//#include <cudatemplates/gilreference.hpp>
//#include <cudatemplates/devicememorylinear.hpp>
#include <cudatemplates/stream.hpp>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp> // for debugging purposes

#include <numeric>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "IntegralChannelsForPedestrians");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "IntegralChannelsForPedestrians");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "IntegralChannelsForPedestrians");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "IntegralChannelsForPedestrians");
}

} // end of anonymous namespace

namespace cv
{ // to define get_slice(...) we need to fix OpenCv

//template<> class DataDepth<boost::uint32_t> { public: enum { value = CV_32S, fmt=(int)'i' }; };

template<> class DataType<boost::uint32_t>
{
public:
    typedef boost::uint32_t value_type;
    typedef value_type work_type;
    typedef value_type channel_type;
    typedef value_type vec_type;
    enum { generic_type = 0, depth = DataDepth<channel_type>::value, channels = 1,
           fmt=DataDepth<channel_type>::fmt,
           type = CV_MAKETYPE(depth, channels) };
};


} // end of namespace cv

namespace doppia {

template<typename ChannelsType>
cv::gpu::GpuMat get_slice(ChannelsType &channels, const size_t slice_index)
{
    // GpuMat(rows, cols, ...)
    return cv::gpu::GpuMat(channels.getSlice(slice_index).size[1], channels.getSlice(slice_index).size[0],
                           cv::DataType<typename ChannelsType::Type>::type,
                           channels.getSlice(slice_index).getBuffer(),
                           channels.getSlice(slice_index).getPitch() );
}

} // end of namespace doppia


namespace doppia {

using namespace cv;
using namespace cv::gpu;

typedef boost::gil::rgb8_pixel_t pixel_t;

// this can only be defined in the *.cu files
// Cuda::Array2D<pixel_t>::Texture input_image_texture;

class GpuIntegralChannelsForPedestriansImplementation
{

public:

    typedef GpuIntegralChannelsForPedestrians::input_image_view_t input_image_view_t;

    //typedef Cuda  ::Array2D<pixel_t> array_t;
    typedef GpuIntegralChannelsForPedestrians::gpu_channels_t  gpu_channels_t;

    typedef GpuIntegralChannelsForPedestrians::gpu_3d_integral_channels_t  gpu_3d_integral_channels_t;
    typedef GpuIntegralChannelsForPedestrians::gpu_2d_integral_channels_t  gpu_2d_integral_channels_t;
    typedef GpuIntegralChannelsForPedestrians::gpu_integral_channels_t  gpu_integral_channels_t;

    typedef cv::gpu::FilterEngine_GPU filter_t;
    typedef cv::Ptr<filter_t> filter_shared_pointer_t;

public:

    GpuIntegralChannelsForPedestriansImplementation(GpuIntegralChannelsForPedestrians &);
    ~GpuIntegralChannelsForPedestriansImplementation();

    void set_image(const boost::gil::rgb8c_view_t &input_view);

    /// keep a reference to an existing GPU image
    /// we assume that the given gpu image will not change during the compute calls
    void set_image(const cv::gpu::GpuMat &input_image);

    void compute_v0();
    void compute_v1();

    const gpu_channels_t& get_gpu_channels() const;

    /// returns a non-const reference because cuda structures do not play nice with const-ness
    gpu_integral_channels_t& get_gpu_integral_channels();

protected:

    GpuIntegralChannelsForPedestrians &parent;

    void compute_smoothed_image_v0();
    void compute_hog_channels_v0();
    void compute_luv_channels_v0();
    void resize_and_integrate_channels_v0();
    void resize_and_integrate_channels_v1();
    void resize_and_integrate_channels_v2();

    void shrink_channel_v0(GpuMat &feature_channel, GpuMat &shrunk_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);

    void shrink_channel_v1(GpuMat &feature_channel, GpuMat &shrunk_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);

    void shrink_channel_v2(GpuMat &feature_channel, GpuMat &shrunk_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);

    void shrink_channel_v3(GpuMat &feature_channel, GpuMat &shrunk_channel,
                           const int shrinking_factor, cv::gpu::Stream &stream);


    void compute_hog_and_luv_channels_v1();


    void allocate_channels(const size_t image_width, const size_t image_height);

    input_image_view_t::point_t input_size, shrunk_channel_size;

    GpuMat
    input_rgb8_gpu_mat,
    input_gpu_mat,
    smoothed_input_gpu_mat,
    hog_input_gpu_mat,
    luv_gpu_mat;

    /// helper variable to avoid doing memory re-allocations for each channel shrinking
    GpuMat shrink_channel_buffer_a, shrink_channel_buffer_b;

    /// helper variable to avoid doing memory re-allocations for each channel index integral computation
    GpuMat integral_channel_buffer_a, integral_channel_buffer_b;

    /// the size of one_half and shrunk_channel is the same of all channels,
    /// thus we can reuse the same matrices
    GpuMat one_half, shrunk_channel;

    filter_shared_pointer_t pre_smoothing_filter_p;

    //array_t device_image_array;

    /// these are the non-shrunk  channels
    gpu_channels_t channels;

    /// these are the shrunk channels
    gpu_channels_t shrunk_channels;

    /// integral channels are computed over the shrinked channels
    gpu_integral_channels_t integral_channels;
};

typedef doppia::GpuIntegralChannelsForPedestriansImplementation implementation_t;
typedef GpuIntegralChannelsForPedestriansImplementation::filter_shared_pointer_t filter_shared_pointer_t;


filter_shared_pointer_t
create_pre_smoothing_gpu_filter()
{
    const int binomial_filter_degree = 1;
    //const int binomial_filter_degree = 2;

    const bool copy_data = true;
    const cv::Mat binomial_kernel_1d = cv::Mat(get_binomial_kernel_1d(binomial_filter_degree), copy_data);
    filter_shared_pointer_t pre_smoothing_filter_p =
            cv::gpu::createSeparableLinearFilter_GPU(CV_8UC4, CV_8UC4, binomial_kernel_1d, binomial_kernel_1d);

    // using the 2d kernel is significantly faster than using the linear separable kernel
    // (at least for binomial_filter_degree == 1)
    // on OpenCv 2.3 svn trunk @ yildun, when detecting over 52 scales,
    // we get 0.89 Hz with 1d kernel and 0.98 Hz with 2d kernel
    // ( 2.45 Hz versus 2.55 Hz on jabbah)
    const bool use_2d_kernel = true;
    if(use_2d_kernel)
    {
        const cv::Mat binomial_kernel_2d =  binomial_kernel_1d * binomial_kernel_1d.t();
        pre_smoothing_filter_p =
                cv::gpu::createLinearFilter_GPU(CV_8UC4, CV_8UC4, binomial_kernel_2d);
    }

    return pre_smoothing_filter_p;
}


implementation_t::GpuIntegralChannelsForPedestriansImplementation(GpuIntegralChannelsForPedestrians &parent_)
    : parent(parent_)
{

    pre_smoothing_filter_p = create_pre_smoothing_gpu_filter();
    return;
}

implementation_t::~GpuIntegralChannelsForPedestriansImplementation()
{
    // nothing to do here
    return;
}

void implementation_t::set_image(const boost::gil::rgb8c_view_t &input_view)
{

    static int warnings_count = 0;

    if(warnings_count <50){
        log_warning() << "Called GpuIntegralChannelsForPedestriansImplementation::set_image(...), "
                         "which provides a slow implementation (see GpuIntegralChannelsDetector::set_image instead)"
                      << std::endl;
        warnings_count +=1;
    }
    else if(warnings_count==50){
        warnings_count +=1;
        log_warning() << "Too many warnings, going silent!"
                      << std::endl;

    }else{
        //nothing to do
    }

    // transfer image into GPU --
    boost::gil::opencv::ipl_image_wrapper input_ipl =
            boost::gil::opencv::create_ipl_image(input_view);

    cv::Mat input_mat(input_ipl.get());

    input_rgb8_gpu_mat.upload(input_mat);  // from CPU to GPU

    // most tasks in GPU are optimized for CV_8UC1 and CV_8UC4, so we set the input as such
    cv::gpu::cvtColor(input_rgb8_gpu_mat, input_gpu_mat, CV_RGB2RGBA); // GPU type conversion

    allocate_channels(input_view.width(), input_view.height());
    /*
    Cuda::HostMemoryReference2D<pixel_t> host_image(
                input_view.width(), input_view.height(),
                boost::gil::interleaved_view_get_raw_data(input_view));

    if((device_image_array.size[0] != input_view.width())
            or (device_image_array.size[1] != input_view.height()))
    {
        device_image_array.realloc(input_view.width(), input_view.height());
    }

    // move data from CPU to GPU
    Cuda::copy(device_image, host_image);

    input_image_texture.filterMode = cudaFilterModeLinear; // versus cudaFilterModePoint
    device_image.bindTexture(input_image_texture);
*/


    /*
    Cuda::GilReference2D<pixel_t>::gil_image_t input_image(the_input_view.dimensions());

    right_image(right.dimensions());
    boost::gil::copy_pixels(left, boost::gil::view(left_image));
    boost::gil::copy_pixels(right, boost::gil::view(right_image));

    Cuda::GilReference2D<uchar1>::gil_image_t disparity_visualization(left.dimensions());

    Cuda::GilReference2D<pixel_t> left_ref(left_image), right_ref(right_image);
    Cuda::GilReference2D<uchar1> disparity_visualization_ref(disparity_visualization);

    const bool be_verbose = first_disparity_map_computation;
    const float computation_time_in_seconds =
            winner_take_all::gpu_stereo(left_ref, right_ref, gpu_method, gpu_options, disparity_visualization_ref, be_verbose);

    if(first_disparity_map_computation)
    {
        printf("winner_take_all::gpu_stereo computation time %.5f [s] == %.1f [fps]\n", computation_time_in_seconds, 1.0 / computation_time_in_seconds);
    }

    //gil::png_write_view("output_gpu_disparity_map.visualization.png", gil::const_view(disparity_visualization));
    boost::gil::copy_pixels(gil::const_view(disparity_visualization), this->disparity_map_view);
*/

    return;
}

void implementation_t::set_image(const cv::gpu::GpuMat &input_image)
{
    if(input_image.type() != CV_8UC4)
    {
        printf("GpuIntegralChannelsForPedestriansImplementation::set_image input_image.type() == %i\n",
               input_image.type());
        printf("CV_8UC1 == %i, CV_8UC3 == %i,  CV_16UC3 == %i,  CV_8UC4 == %i\n",
               CV_8UC1, CV_8UC3, CV_16UC3, CV_8UC4);
        throw std::runtime_error("OpenCv gpu module handles color images as CV_8UC4, received an unexpected GpuMat type");
    }

    const bool input_size_changed = (input_gpu_mat.cols != input_image.cols) or (input_gpu_mat.rows != input_image.rows);
    input_gpu_mat = input_image;

    if(input_size_changed)
    {
        allocate_channels(input_image.cols, input_image.rows);
    }
    else
    {
        // no need to reallocated, only need to reset a few matrices

        // we reset the content of the channels to zero
        // this is particularly important for the HOG channels where not all the pixels will be set
        channels.initMem(0);
    }
    return;
}

template <typename T>
void allocate_integral_channels(const size_t /*shrunk_channel_size_x*/,
                                const size_t /*shrunk_channel_size_y*/,
                                const size_t /*num_channels*/,
                                T &/*integral_channels*/)
{
    throw std::runtime_error("Called alloc_integral_channels<> with an unhandled integral channel type");
    return;
}

template <>
void allocate_integral_channels<implementation_t::gpu_3d_integral_channels_t>(
        const size_t shrunk_channel_size_x,
        const size_t shrunk_channel_size_y,
        const size_t num_channels,
        implementation_t::gpu_3d_integral_channels_t &integral_channels)
{
    integral_channels.alloc(shrunk_channel_size_x+1, shrunk_channel_size_y+1, num_channels);
    // size0 == x/cols, size1 == y/rows, size2 == num_channels
    return;
}


template <>
void allocate_integral_channels<implementation_t::gpu_2d_integral_channels_t>(
        const size_t shrunk_channel_size_x,
        const size_t shrunk_channel_size_y,
        const size_t num_channels,
        implementation_t::gpu_2d_integral_channels_t &integral_channels)
{
    integral_channels.alloc(shrunk_channel_size_x+1, (shrunk_channel_size_y*num_channels)+1);
    // size0 == x/cols, size1 == y/rows, size2 == num_channels
    return;
}


void implementation_t::allocate_channels(const size_t image_width, const size_t image_height)
{
    // 6 gradients orientations, 1 gradient intensity, 3 LUV color channels
    const int num_channels = 10;

    input_size.x = image_width;
    input_size.y = image_height;

    //channel_size = input_image.dimensions() / resizing_factor;
    // +resizing_factor/2 to round-up
    if(parent.resizing_factor == 4)
    {
        shrunk_channel_size.x = (( (image_width+1)/2) + 1)/2;
        shrunk_channel_size.y = (( (image_height+1)/2) + 1)/2;
    }
    else if(parent.resizing_factor == 2)
    {
        shrunk_channel_size.x = (image_width+1) / 2;
        shrunk_channel_size.y = (image_height+1) / 2;
    }
    else
    {
        shrunk_channel_size = input_size;
    }

    static int num_calls = 0;
    if(num_calls < 100)
    { // we only log the first N calls
        log_debug() << "Input image dimensions (" << image_width << ", " << image_height << ")" << std::endl;
        log_debug() << "Shrunk Channel size (" << shrunk_channel_size.x << ", " << shrunk_channel_size.y << ")" << std::endl;
        num_calls += 1;
    }

    if(shrunk_channel_size.x == 0 or shrunk_channel_size.y == 0)
    {
        log_error() << "Input image dimensions (" << image_width << ", " << image_height << ")" << std::endl;
        throw std::runtime_error("Input image for GpuIntegralChannelsForPedestriansImplementation::set_image "
                                 "was too small");
    }

    // FIXME should allocate only once (for the largest size)
    // and then reuse the allocated memory with smaller images
    // e.g. allocate only for "bigger images", and reuse memory for smaller images
    // Cuda::DeviceMemoryPitched3D<> for allocations and Cuda::DeviceMemoryReference3D<> on top, or similar..

    // allocate the channel images, the first dimension goes last
    channels.alloc(input_size.x, input_size.y, num_channels);
    // if not using shrunk_channels, this allocation should be commented out (since it takes time)
    shrunk_channels.alloc(shrunk_channel_size.x, shrunk_channel_size.y, num_channels);

    allocate_integral_channels(shrunk_channel_size.x, shrunk_channel_size.y, num_channels,
                               integral_channels);


    // we reset the content of the channels to zero
    // this is particularly important for the HOG channels where not all the pixels will be set
    channels.initMem(0);

    const bool print_slice_size = false;
    if(print_slice_size)
    {
        printf("channels.getSlice(0).size[0] == %zi\n", channels.getSlice(0).size[0]);
        printf("channels.getSlice(0).size[1] == %zi\n", channels.getSlice(0).size[1]);
        printf("channels.getSlice(9).size[0] == %zi\n", channels.getSlice(9).size[0]);
        printf("channels.getSlice(9).size[1] == %zi\n", channels.getSlice(9).size[1]);
        printf("input_size.y == %zi, input_size.x == %zi\n", input_size.y, input_size.x);
        throw std::runtime_error("Stopping everything so you can inspect the last printed values");
    }

    return;
}

const implementation_t::gpu_channels_t& implementation_t::get_gpu_channels() const
{
    return channels;
}

implementation_t::gpu_integral_channels_t&  implementation_t::get_gpu_integral_channels()
{
    return integral_channels;
}

void implementation_t::compute_v0()
{
    // v0 is mainly based on OpenCv's GpuMat

    // smooth the input image --
    compute_smoothed_image_v0();

    // compute the HOG channels --
    compute_hog_channels_v0();

    // compute the LUV channels --
    compute_luv_channels_v0();

    // resize and compute integral images for each channel --
    resize_and_integrate_channels_v0();

    return;
}


void implementation_t::compute_smoothed_image_v0()
{
    // smooth the input image --
    smoothed_input_gpu_mat.create(input_gpu_mat.size(), input_gpu_mat.type());
    pre_smoothing_filter_p->apply(input_gpu_mat, smoothed_input_gpu_mat);

    return;
}


void implementation_t::compute_hog_channels_v0()
{
    if(false)
    {
        printf("input_size.x == %zi, input_size.y == %zi\n",
               input_size.x, input_size.y);
        printf("channels.size == [%zi, %zi, %zi]\n",
               channels.size[0], channels.size[1], channels.size[2]);
        printf("channels.stride == [%zi, %zi, %zi]\n",
               channels.stride[0], channels.stride[1], channels.stride[2]);

        throw std::runtime_error("Stopped everything so you can inspect the printed vaules");
    }

    cv::gpu::cvtColor(smoothed_input_gpu_mat, hog_input_gpu_mat, CV_RGBA2GRAY);

    if(hog_input_gpu_mat.type() != CV_8UC1)
    {
        printf("compute_hog_channels(...) input_image.type() == %i\n", hog_input_gpu_mat.type());
        printf("CV_8UC1 == %i, CV_8UC3 == %i,  CV_16UC3 == %i,  CV_8UC4 == %i\n",
               CV_8UC1, CV_8UC3, CV_16UC3, CV_8UC4);
        throw std::invalid_argument("doppia::integral_channels::compute_hog_channels expects an input image of type CV_8UC1");
    }

    // compute the HOG channels  --
    doppia::integral_channels::compute_hog_channels(hog_input_gpu_mat, channels);

    return;
}


void implementation_t::compute_luv_channels_v0()
{
    // compute the LUV channels --

    const bool use_opencv = false;

    if(use_opencv)
    {

        // CV_RGB2HSV and CV_RGB2Luv seem to work fine even when the input is RGBA
        //cv::gpu::cvtColor(smoothed_input_gpu_mat, luv_gpu_mat, CV_RGB2Luv);

        // warning doing HSV until LUV is actually implemented
        cv::gpu::cvtColor(smoothed_input_gpu_mat, luv_gpu_mat, CV_RGB2HSV);

        // split the LUV image into the L,U and V channels
        std::vector<GpuMat> destinations(3);

        destinations[0] = get_slice(channels, 7);
        destinations[1] = get_slice(channels, 8);
        destinations[2] = get_slice(channels, 9);

        cv::gpu::split(luv_gpu_mat, destinations);

        if(false)
        {
            cv::Mat test_image;
            luv_gpu_mat.download(test_image);
            cv::imwrite("debug_image.png", test_image);
            throw std::runtime_error("Stopped everything so you can inspect debug_image.png");
        }

    }
    else
    {
        if(smoothed_input_gpu_mat.type() != CV_8UC4)
        {
            throw std::invalid_argument("doppia::integral_channels::compute_luv_channels expects an RGBA image as input");
        }

        doppia::integral_channels::compute_luv_channels(smoothed_input_gpu_mat, channels);
    }
    return;
}

void implementation_t::shrink_channel_v0(
        GpuMat &feature_channel, GpuMat &shrunk_channel,
        const int shrinking_factor, cv::gpu::Stream &channel_stream)
{
    // we assume that shrunk_channel is already properly allocated
    assert(shrunk_channel.empty() == false);

    if(shrinking_factor == 4)
    {
        cv::gpu::pyrDown(feature_channel, one_half, channel_stream);
        cv::gpu::pyrDown(one_half, shrunk_channel, channel_stream); // shrunk_channel == one_fourth
    }
    else if (shrinking_factor == 2)
    {
        cv::gpu::pyrDown(feature_channel, one_half, channel_stream);
        shrunk_channel = one_half;
    }
    else
    {
        shrunk_channel = feature_channel;
    }

    return;
}

void implementation_t::shrink_channel_v1(
        GpuMat &feature_channel, GpuMat &shrunk_channel,
        const int shrinking_factor, cv::gpu::Stream &channel_stream)
{

    // FIXME need to add tests to verify that _v1 and _v0 output the same results

    // lazy allocation
    //shrunk_channel.create(shrunk_channel_size.y, shrunk_channel_size.x, feature_channel.type());
    // we assume that shrunk_channel is already properly allocated
    //assert(shrunk_channel.empty() == false);

    shrink_channel_buffer_a.create(feature_channel.rows, feature_channel.cols, feature_channel.type());
    shrink_channel_buffer_b.create((feature_channel.rows + 1)/2, (feature_channel.cols+1)/2, feature_channel.type());

    static const float pyrDown_kernel_1d_values[5] = {0.0625, 0.25, 0.375, 0.25, 0.0625},
            pyrDown4_kernel_1d_values[9] = {
        0.01483945, 0.04981729, 0.11832251, 0.198829, 0.23638351, 0.198829, 0.11832251, 0.04981729, 0.01483945 };

    static const vector<float>
            t_vec(&pyrDown_kernel_1d_values[0], &pyrDown_kernel_1d_values[5]),
            t_vec4(&pyrDown4_kernel_1d_values[0], &pyrDown4_kernel_1d_values[9]);

    static const cv::Mat
            pyrDown_kernel_1d = cv::Mat(t_vec, true),
            pyrDown4_kernel_1d = cv::Mat(t_vec4, true);
    //pyrDown_kernel_2d =  pyrDown_kernel_1d * pyrDown_kernel_1d.t();
    static const cv::Rect dummy_roi = cv::Rect(0,0,-1,-1);
    static filter_shared_pointer_t pyrDown_smoothing_filter_p =
            cv::gpu::createSeparableLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown_kernel_1d, pyrDown_kernel_1d);
    //cv::gpu::createLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown_kernel_2d);

    static filter_shared_pointer_t pyrDown4_smoothing_filter_p =
            cv::gpu::createSeparableLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown4_kernel_1d, pyrDown4_kernel_1d);
    //cv::gpu::createLinearFilter_GPU(CV_8UC1, CV_8UC1, pyrDown4_kernel_2d);

    if(shrinking_factor == 4)
    {

        const bool use_one_filter = true;

        if(use_one_filter)
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown4_smoothing_filter_p->apply(feature_channel, shrink_channel_buffer_a,
                                               dummy_roi, channel_stream);
            cv::gpu::resize(shrink_channel_buffer_a, shrunk_channel,
                            cv::Size(shrunk_channel_size.x, shrunk_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        else
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown_smoothing_filter_p->apply(feature_channel, shrink_channel_buffer_a,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_channel_buffer_a, one_half,
                            shrink_channel_buffer_b.size(), 0, 0, cv::INTER_NEAREST, channel_stream);

            pyrDown_smoothing_filter_p->apply(one_half, shrink_channel_buffer_b,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_channel_buffer_b, shrunk_channel,
                            cv::Size(shrunk_channel_size.x, shrunk_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        // shrunk_channel == one_fourth
    }
    else
    {
        throw std::runtime_error("shrink_channel_v1 only supports shrinking factor == 4");
    }

    return;
}

void implementation_t::shrink_channel_v2(
        GpuMat &feature_channel, GpuMat &shrunk_channel,
        const int shrinking_factor, cv::gpu::Stream &channel_stream)
{

    // _v2 is based on the realization that for detection purposes we do not want to do a nice Gaussian filter + subsampling,
    // actually we want to do an average pooling, so simply averaging over a box filter + subsampling is both correct and faster


    shrink_channel_buffer_a.create(feature_channel.rows, feature_channel.cols, feature_channel.type());
    shrink_channel_buffer_b.create((feature_channel.rows + 1)/2, (feature_channel.cols+1)/2, feature_channel.type());

    static const cv::Rect dummy_roi = cv::Rect(0,0,-1,-1);
    static filter_shared_pointer_t pyrDown_smoothing_filter_p =
            cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(2,2));

    static filter_shared_pointer_t pyrDown4_smoothing_filter_p =
            cv::gpu::createBoxFilter_GPU(CV_8UC1, CV_8UC1, cv::Size(4, 4));

    if(shrinking_factor == 4)
    {

        // FIXME based on the Nvidia forum answer, using the smoothing filter here is useless
        // http://forums.nvidia.com/index.php?showtopic=210066

        const bool use_one_filter = false;
        if(use_one_filter)
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown4_smoothing_filter_p->apply(feature_channel, shrink_channel_buffer_a,
                                               dummy_roi, channel_stream);
            cv::gpu::resize(shrink_channel_buffer_a, shrunk_channel,
                            cv::Size(shrunk_channel_size.x, shrunk_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        else
        {
            // after filtering, no need to use linear interpolation, thus we use cv::INTER_NEAREST
            pyrDown_smoothing_filter_p->apply(feature_channel, shrink_channel_buffer_a,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_channel_buffer_a, one_half,
                            shrink_channel_buffer_b.size(), 0, 0, cv::INTER_NEAREST, channel_stream);

            pyrDown_smoothing_filter_p->apply(one_half, shrink_channel_buffer_b,
                                              dummy_roi, channel_stream);
            cv::gpu::resize(shrink_channel_buffer_b, shrunk_channel,
                            cv::Size(shrunk_channel_size.x, shrunk_channel_size.y), 0, 0,
                            cv::INTER_NEAREST, channel_stream);
        }
        // shrunk_channel == one_fourth
    }
    else
    {
        throw std::runtime_error("shrink_channel_v2 only supports shrinking factor == 4");
    }

    return;
}

/// Helper method to handle OpenCv's GPU data
template <typename DstType>
Cuda::DeviceMemoryReference2D<DstType> gpumat_to_device_reference_2d(cv::gpu::GpuMat &src)
{
    Cuda::Layout<DstType, 2> layout(Cuda::Size<2>(src.cols, src.rows));
    layout.setPitch(src.step); // step is in bytes
    return Cuda::DeviceMemoryReference2D<DstType>(layout, reinterpret_cast<DstType *>(src.data));
}


void implementation_t::shrink_channel_v3(
        GpuMat &feature_channel, GpuMat &shrunk_channel,
        const int shrinking_factor, cv::gpu::Stream &/*channel_stream*/)
{
    // the channel_stream argument is currently ignored

    assert(feature_channel.type() == CV_8UC1);
    assert(shrunk_channel.type() == CV_8UC1);

    // lazy allocation
    shrunk_channel.create(shrunk_channel_size.y, shrunk_channel_size.x, feature_channel.type());

    Cuda::DeviceMemoryReference2D<uint8_t>
            feature_channel_reference = gpumat_to_device_reference_2d<uint8_t>(feature_channel),
            shrunk_channel_reference = gpumat_to_device_reference_2d<uint8_t>(shrunk_channel);

    doppia::integral_channels::shrink_channel(feature_channel_reference, shrunk_channel_reference, shrinking_factor);

    return;
}

void implementation_t::resize_and_integrate_channels_v0()
{
    // using streams here seems to have zero effect on the computation speed (at Jabbah and the Europa laptop)
    // we keep "just in case" we run over a 2 GPUs system
    cv::gpu::Stream stream_a, stream_b;

    // shrink and compute integral over the channels ---
    for(size_t channel_index = 0; channel_index < channels.size[2]; channel_index +=1)
    {
        // we do not wait last channel completion to launch the new computation
        cv::gpu::Stream *channel_stream_p = &stream_a;
        cv::gpu::GpuMat *integral_channel_buffer_p = &integral_channel_buffer_a;
        if((channel_index % 2) == 1)
        {
            channel_stream_p = &stream_b;
            integral_channel_buffer_p = &integral_channel_buffer_b;
        }

        cv::gpu::Stream &channel_stream = *channel_stream_p;

        gpu::GpuMat feature_channel = get_slice(channels, channel_index);

        // v0 is about twice as fast as v1
        // v2 is significantly slower than v0 (1.7 Hz versus 2.3 Hz on detection rate)
        // v3 is brutaly faster, 4.5 Hz versus 3.8 Hz on Jabbah
        //shrink_channel_v0(feature_channel, shrunk_channel, parent.resizing_factor, channel_stream);
        //shrink_channel_v1(feature_channel, shrunk_channel, parent.resizing_factor, channel_stream);
        //shrink_channel_v2(feature_channel, shrunk_channel, parent.resizing_factor, channel_stream);
        shrink_channel_v3(feature_channel, shrunk_channel, parent.resizing_factor, channel_stream);


        const bool check_channels_sizes = false;
        if(check_channels_sizes)
        {
            if((shrunk_channel.rows != shrunk_channel_size.y)
               or (shrunk_channel.cols != shrunk_channel_size.x))
            {
                printf("shrunk_channel size == (%i, %i)\n",
                       shrunk_channel.cols, shrunk_channel.rows);
                printf("shrunk_channel size == (%zi, %zi) (expected value for shrunk_channel size)\n",
                       shrunk_channel_size.x, shrunk_channel_size.y);
                throw std::runtime_error("shrunk_channel size != expected shrunk channel size");
            }

            if((integral_channels.size[0] != static_cast<size_t>(shrunk_channel_size.x + 1))
               or (integral_channels.size[1] != static_cast<size_t>(shrunk_channel_size.y + 1)))
            {
                printf("integral channel size == (%zi, %zi)\n",
                       integral_channels.size[0], integral_channels.size[1]);
                printf("shrunk_channel size == (%zi, %zi) (expected value for shrunk_channel size)\n",
                       shrunk_channel_size.x, shrunk_channel_size.y);
                throw std::runtime_error("integral channel size != shrunk channel size + 1");
            }
        } // end of "if check channels sizes"


        // set mini test or not --
        const bool set_test_integral_image = false;
        if(set_test_integral_image)
        { // dummy test integral image, used for debugging only
            // mimics the equivalent cpu snippet in IntegralChannelsForPedestrians::compute_v0()
            cv::Mat channel_test_values(shrunk_channel.size(), shrunk_channel.type());

            if(shrunk_channel.type() != CV_8UC1)
            {
                throw std::runtime_error("shrunk_channel.type() has an unexpected value");
            }

            for(int row=0; row < channel_test_values.rows; row+=1)
            {
                for(int col=0; col < channel_test_values.cols; col+=1)
                {
                    const float row_scale = 100.0f/(channel_test_values.rows);
                    const float col_scale = 10.0f/(channel_test_values.cols);
                    channel_test_values.at<boost::uint8_t>(row,col) = \
                            static_cast<boost::uint8_t>(min(255.0f, row_scale*row + col_scale*col + channel_index));
                } // end of "for each col"
            } // end of "for each row"

            shrunk_channel.upload(channel_test_values);
        } // end of set_test_integral_image


        // compute integral images for shrunk_channel --
        gpu::GpuMat integral_channel = get_slice(integral_channels, channel_index);

        // sum will have CV_32S type, but will contain unsigned int values
        cv::gpu::integralBuffered(shrunk_channel, integral_channel, *integral_channel_buffer_p, channel_stream);

        cuda_safe_call( cudaGetLastError() );

        if(false and (channel_index == channels.size[2] - 1))
        {

            cv::Mat test_image;
            shrunk_channel.download(test_image);
            cv::imwrite("debug_image.png", test_image);

            cv::Mat integral_image, test_image2;

            if(true and channel_index > 0)
            {
                // we take the previous channel, to check if it was not erased
                gpu::GpuMat integral_channel2(get_slice(integral_channels, channel_index - 1));

                printf("integral_channel2 cols, rows, step, type, elemSize == %i, %i, %zi, %i, %zi\n",
                       integral_channel2.cols, integral_channel2.rows,
                       integral_channel2.step, integral_channel2.type(), integral_channel2.elemSize() );

                printf("integral_channel cols, rows, step, type, elemSize == %i, %i, %zi, %i, %zi\n",
                       integral_channel.cols, integral_channel.rows,
                       integral_channel.step, integral_channel.type(), integral_channel.elemSize() );

                printf("CV_8UC1 == %i, CV_32UC1 == NOT DEFINED,  CV_32SC1 == %i,  CV_32SC2 == %i, CV_USRTYPE1 == %i\n",
                       CV_8UC1, CV_32SC1, CV_32SC2, CV_USRTYPE1);

                integral_channel2.download(integral_image); // copy from GPU to CPU
            }
            else
            {
                integral_channel.download(integral_image); // copy from GPU to CPU
            }

            test_image2 = cv::Mat(shrunk_channel.size(), cv::DataType<float>::type);
            for(int y=0; y < test_image2.rows; y+=1)
            {
                for(int x=0; x < test_image2.cols; x+=1)
                {
                    const uint32_t
                            a = integral_image.at<boost::uint32_t>(y,x),
                            b = integral_image.at<boost::uint32_t>(y+0,x+1),
                            c = integral_image.at<boost::uint32_t>(y+1,x+1),
                            d = integral_image.at<boost::uint32_t>(y+1,x+0);
                    test_image2.at<float>(y,x) = a +c -b -d;
                } // end of "for each column"
            } // end of "for each row"

            cv::imwrite("debug_image2.png", test_image2);

            throw std::runtime_error("Stopped everything so you can inspect debug_image.png and debug_image2.png");
        }

    } // end of "for each channel"

    stream_a.waitForCompletion();
    stream_b.waitForCompletion();

    return;
}


void implementation_t::resize_and_integrate_channels_v1()
{    
    if(not boost::is_same<gpu_integral_channels_t, gpu_3d_integral_channels_t>::value)
    {
        throw std::runtime_error("resize_and_integrate_channels_v1 should only be used with gpu_3d_integral_channels_t, "
                                 "please check your code");
    }

    // first we shrink all the channels ---
    {
        doppia::integral_channels::shrink_channels(channels, shrunk_channels, parent.resizing_factor);
    }

    // second, we compute the integral of each channel ---
    {
        // using streams here seems to have zero effect on the computation speed (at Jabbah and the Europa laptop)
        // we keep "just in case" we run over a 2 GPUs system
        cv::gpu::Stream stream_a, stream_b;

        // shrink the channels ---
        for(size_t channel_index = 0; channel_index < shrunk_channels.size[2]; channel_index +=1)
        {
            // we do not wait last channel completion to launch the new computation
            cv::gpu::Stream *channel_stream_p = &stream_a;
            cv::gpu::GpuMat *integral_channel_buffer_p = &integral_channel_buffer_a;
            if((channel_index % 2) == 1)
            {
                channel_stream_p = &stream_b;
                integral_channel_buffer_p = &integral_channel_buffer_b;
            }

            cv::gpu::Stream &channel_stream = *channel_stream_p;

            gpu::GpuMat shrunk_channel = get_slice(shrunk_channels, channel_index);

            // compute integral images for shrunk_channel --
            gpu::GpuMat integral_channel = get_slice(integral_channels, channel_index);

            // sum will have CV_32S type, but will contain unsigned int values
            cv::gpu::integralBuffered(shrunk_channel, integral_channel, *integral_channel_buffer_p, channel_stream);

            cuda_safe_call( cudaGetLastError() );

        } // end of "for each channel"

        stream_a.waitForCompletion();
        stream_b.waitForCompletion();
    }

    return;
}


void implementation_t::resize_and_integrate_channels_v2()
{
    if(not boost::is_same<gpu_integral_channels_t, gpu_2d_integral_channels_t>::value)
    {
        throw std::runtime_error("resize_and_integrate_channels_v2 should only be used with gpu_2d_integral_channels_t, "
                                 "please check your code");
    }

    // first we shrink all the channels ---
    {
        doppia::integral_channels::shrink_channels(channels, shrunk_channels, parent.resizing_factor);
    }

    {
        // second, we compute the integral of all channels in one shot ---
        const size_t
                shrunk_channels_area =  shrunk_channels.size[0]*shrunk_channels.size[1]*shrunk_channels.size[2],
                max_sum = shrunk_channels_area * std::numeric_limits<gpu_channels_t::Type>::max();

        // images of 1024*1024 over
        if(max_sum > std::numeric_limits<boost::uint32_t>::max())
        {
            printf("max_sum/max_uint32 value =~= %.4f \n",
                   static_cast<float>(max_sum) / std::numeric_limits<boost::uint32_t>::max());
            throw std::runtime_error("Using resize_and_integrate_channels_v2 will create an overflow, "
                                     "use resize_and_integrate_channels_v1 for this image size");
        }

        // size0 == x/cols, size1 == y/rows, size2 == num_channels
        // GpuMat(rows, cols, type)
        const gpu::GpuMat shrunk_channels_gpu_mat(
                    shrunk_channels.size[1]*shrunk_channels.size[2], shrunk_channels.size[0],
                    cv::DataType<gpu_channels_t::Type>::type,
                    shrunk_channels.getBuffer(),
                    shrunk_channels.getPitch());

        gpu::GpuMat integral_channels_gpu_mat(
                    integral_channels.size[1], integral_channels.size[0],
                    cv::DataType<gpu_integral_channels_t::Type>::type,
                    integral_channels.getBuffer(),
                    integral_channels.getPitch());

        // sum will have CV_32S type, but will contain unsigned int values
        cv::gpu::integralBuffered(shrunk_channels_gpu_mat, integral_channels_gpu_mat, integral_channel_buffer_a);

        cuda_safe_call( cudaGetLastError() );
    }

    return;
}


void implementation_t::compute_v1()
{
    // v1 is mainly based on v0,
    // but merges a few steps into single "bigger kernels" calls

    // smooth the input image --
    compute_smoothed_image_v0();

    // compute the HOG and LUV channels --
    compute_hog_and_luv_channels_v1();

    // resize and compute integral images for each channel --
    // with v1 we obtain 4.65 Hz versus 4.55 Hz with v0
    //resize_and_integrate_channels_v0();
    if(boost::is_same<gpu_integral_channels_t, gpu_3d_integral_channels_t>::value)
    {
        resize_and_integrate_channels_v1();
    }
    else
    {
        resize_and_integrate_channels_v2();
    }

    return;
}


void implementation_t::compute_hog_and_luv_channels_v1()
{

    cv::gpu::cvtColor(smoothed_input_gpu_mat, hog_input_gpu_mat, CV_RGBA2GRAY);

    if(hog_input_gpu_mat.type() != CV_8UC1)
    {
        printf("compute_hog_and_luv_channels(...) input_gray_image.type() == %i\n", hog_input_gpu_mat.type());
        printf("CV_8UC1 == %i, CV_8UC3 == %i,  CV_16UC3 == %i,  CV_8UC4 == %i\n",
               CV_8UC1, CV_8UC3, CV_16UC3, CV_8UC4);
        throw std::invalid_argument("doppia::integral_channels::compute_hog_and_luv_channels expects an input gray image of type CV_8UC1");
    }

    if(smoothed_input_gpu_mat.type() != CV_8UC4)
    {
        throw std::invalid_argument("doppia::integral_channels::compute_hog_luv_channels expects to have an RGBA image as an input");
    }

    doppia::integral_channels::compute_hog_and_luv_channels(hog_input_gpu_mat, smoothed_input_gpu_mat,  channels);

    return;
}



} // end of namespace doppia


namespace doppia {

GpuIntegralChannelsForPedestrians::GpuIntegralChannelsForPedestrians()
    : resizing_factor(get_shrinking_factor()),
      self_p(new GpuIntegralChannelsForPedestriansImplementation(*this)),
      self(*self_p)
{
    // nothing to do here
    return;
}


GpuIntegralChannelsForPedestrians::~GpuIntegralChannelsForPedestrians()
{
    // nothing to do here
    return;
}


int GpuIntegralChannelsForPedestrians::get_shrinking_factor()
{
    return IntegralChannelsForPedestrians::get_shrinking_factor();
}


void GpuIntegralChannelsForPedestrians::set_image(const boost::gil::rgb8c_view_t &input_view)
{
    self.set_image(input_view);
    return;
}


void GpuIntegralChannelsForPedestrians::set_image(const cv::gpu::GpuMat &input_image)
{
    self.set_image(input_image);
    return;
}


void GpuIntegralChannelsForPedestrians::compute()
{
    //self.compute_v0();
    self.compute_v1();

    return;
}


GpuIntegralChannelsForPedestrians::gpu_integral_channels_t&
GpuIntegralChannelsForPedestrians::get_gpu_integral_channels() const
{
    return self.get_gpu_integral_channels();
}

/// helper function to access the computed channels on cpu
/// this is quite slow (large data transfer between GPU and CPU)
/// this method should be used for debugging only
const GpuIntegralChannelsForPedestrians::channels_t &GpuIntegralChannelsForPedestrians::get_channels()
{
    const gpu_channels_t& gpu_channels = self.get_gpu_channels();

    const Cuda::Size<3> &data_size = gpu_channels.getLayout().size;

    // resize the CPU memory storage --
    // Cuda::DeviceMemoryPitched3D store the size indices in reverse order with respect to boost::multi_array
    channels.resize(boost::extents[data_size[2]][data_size[1]][data_size[0]]);

    // create cudatemplates reference --
    Cuda::HostMemoryReference3D<channels_t::element>
            channels_memory_reference(data_size, channels.origin());

    // copy from GPU to CPU --
    Cuda::copy(channels_memory_reference, gpu_channels);

    return channels;
}


template <typename GpuIntegralChannelsType>
void integral_channels_gpu_to_cpu(
        const GpuIntegralChannelsType &gpu_integral_channels,
        GpuIntegralChannelsForPedestrians::integral_channels_t &integral_channels)
{ // default code is of the 3d case

    const Cuda::Size<3> &data_size = gpu_integral_channels.getLayout().size;

    // resize the CPU memory storage --
    // Cuda::DeviceMemoryPitched3D store the size indices in reverse order with respect to boost::multi_array
    integral_channels.resize(boost::extents[data_size[2]][data_size[1]][data_size[0]]);

    // create cudatemplates reference --
    Cuda::HostMemoryReference3D<GpuIntegralChannelsForPedestrians::integral_channels_t::element>
            integral_channels_host_reference(data_size, integral_channels.origin());

    // copy from GPU to CPU --
    Cuda::copy(integral_channels_host_reference, gpu_integral_channels);

    const bool print_sizes = false;
    if(print_sizes)
    {
        printf("gpu_integral_channels layout size == [%zi, %zi, %zi]\n",
               data_size[0], data_size[1], data_size[2]);

        const Cuda::Size<3> &data_stride = gpu_integral_channels.getLayout().stride;
        printf("gpu_integral_channels layout stride == [%zi, %zi, %zi]\n",
               data_stride[0], data_stride[1], data_stride[2]);

        printf("integral_channels shape == [%zi, %zi, %zi]\n",
               integral_channels.shape()[0],
               integral_channels.shape()[1],
               integral_channels.shape()[2]);

        printf("integral_channels strides == [%zi, %zi, %zi]\n",
               integral_channels.strides()[0],
               integral_channels.strides()[1],
               integral_channels.strides()[2]);

        throw std::runtime_error("Stopping everything so you can inspect the last printed values");
    }

    return;
}


template <>
void integral_channels_gpu_to_cpu <GpuIntegralChannelsForPedestrians::gpu_2d_integral_channels_t> (
        const GpuIntegralChannelsForPedestrians::gpu_2d_integral_channels_t &/*gpu_integral_channels*/,
        GpuIntegralChannelsForPedestrians::integral_channels_t &/*integral_channels*/)
{ // special code for the 2d case

    throw std::runtime_error("integral_channels_gpu_to_cpu is not yet implemented for the 2d case");
/*
    const Cuda::Size<2> &data_size = gpu_integral_channels.getLayout().size;

    // resize the CPU memory storage --
    // Cuda::DeviceMemoryPitched3D store the size indices in reverse order with respect to boost::multi_array
    integral_channels.resize(boost::extents[data_size[2]][data_size[1]][data_size[0]]);

    // create cudatemplates reference --
    Cuda::HostMemoryReference3D<integral_channels_t::element>
            integral_channels_host_reference(data_size, integral_channels.origin());

    // copy from GPU to CPU --
    Cuda::copy(integral_channels_host_reference, gpu_integral_channels);

    const bool print_sizes = false;
    if(print_sizes)
    {
        printf("gpu_integral_channels layout size == [%zi, %zi]\n",
               data_size[0], data_size[1]);

        const Cuda::Size<2> &data_stride = gpu_integral_channels.getLayout().stride;
        printf("gpu_integral_channels layout stride == [%zi, %zi, %zi]\n",
               data_stride[0], data_stride[1]);

        printf("integral_channels shape == [%zi, %zi, %zi]\n",
               integral_channels.shape()[0],
               integral_channels.shape()[1],
               integral_channels.shape()[2]);

        printf("integral_channels strides == [%zi, %zi, %zi]\n",
               integral_channels.strides()[0],
               integral_channels.strides()[1],
               integral_channels.strides()[2]);

        throw std::runtime_error("Stopping everything so you can inspect the last printed values");
    }
*/
    return;
}



/// helper function to access the integral channels on cpu
/// this is quite slow (large data transfer between GPU and CPU)
/// this method should be used for debugging only
const GpuIntegralChannelsForPedestrians::integral_channels_t &GpuIntegralChannelsForPedestrians::get_integral_channels()
{
    const gpu_integral_channels_t& gpu_integral_channels = self.get_gpu_integral_channels();
    integral_channels_gpu_to_cpu(gpu_integral_channels, integral_channels);
    return integral_channels;
}


} // end of namespace doppia
