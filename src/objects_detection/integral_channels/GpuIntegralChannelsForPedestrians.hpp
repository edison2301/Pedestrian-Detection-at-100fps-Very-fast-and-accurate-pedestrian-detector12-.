#ifndef BICLOP_GPUINTEGRALCHANNELSFORPEDESTRIANS_HPP
#define BICLOP_GPUINTEGRALCHANNELSFORPEDESTRIANS_HPP

#include "IntegralChannelsForPedestrians.hpp"
#include <boost/scoped_ptr.hpp>

#include <opencv2/core/version.hpp>
#if CV_MINOR_VERSION <= 3
#include <opencv2/gpu/gpu.hpp> // opencv 2.3
#else
#include <opencv2/core/gpumat.hpp> // opencv 2.4
#endif


#include <cudatemplates/devicememorypitched.hpp>

namespace doppia {

/// we keep the gory implementation details out of the header
class GpuIntegralChannelsForPedestriansImplementation;

/// This is the GPU mirror of the IntegralChannelsForPedestrians
/// IntegralChannelsForPedestrians and GpuIntegralChannelsForPedestrians do not have a common basis class, because
/// altought they provide similar functions, their are aimed to be used in different pipelines.
/// In particular GpuIntegralChannelsForPedestrians does not aim at provide CPU access to the integral channels,
/// but GPU access to a following up GPU detection algorithm. CPU transfer of the integral channels is only
/// provided for debugging purposes
class GpuIntegralChannelsForPedestrians
{

public:
    typedef IntegralChannelsForPedestrians::input_image_t input_image_t;
    typedef IntegralChannelsForPedestrians::input_image_view_t input_image_view_t;

    typedef boost::gil::rgb8_pixel_t pixel_t;

    // the computed channels are uint8, after shrinking they are uint12
    // on CPU we use uint16, however opencv's GPU code only supports int16,
    // which is compatible with uint12
    // using gpu_channels_t::element == uint8 means loosing 4 bits,
    // this however generates only a small loss in the detections performance
    //typedef Cuda::DeviceMemoryPitched3D<boost::uint16_t> gpu_channels_t;
    //typedef Cuda::DeviceMemoryPitched3D<boost::int16_t> gpu_channels_t;
    typedef Cuda::DeviceMemoryPitched3D<boost::uint8_t> gpu_channels_t;
    typedef Cuda::DeviceMemoryPitched3D<boost::uint32_t> gpu_3d_integral_channels_t;
    typedef Cuda::DeviceMemoryPitched2D<boost::uint32_t> gpu_2d_integral_channels_t;
    typedef gpu_3d_integral_channels_t gpu_integral_channels_t; // 818.6 Hz on Kochab
    //typedef gpu_2d_integral_channels_t gpu_integral_channels_t; // 1007.7 Hz on Kochab

    //typedef IntegralChannelsForPedestrians::channels_t channels_t;
    typedef boost::multi_array<gpu_channels_t::Type, 3> channels_t;
    typedef IntegralChannelsForPedestrians::integral_channels_t integral_channels_t;
    typedef IntegralChannelsForPedestrians::integral_channels_view_t integral_channels_view_t;
    typedef IntegralChannelsForPedestrians::integral_channels_const_view_t integral_channels_const_view_t;

public:
    GpuIntegralChannelsForPedestrians();
    ~GpuIntegralChannelsForPedestrians();

    /// how much we shrink the channel images ?
    static int get_shrinking_factor();

    /// transfer CPU image data into the GPU
    void set_image(const boost::gil::rgb8c_view_t &input_image);

    /// keep a reference to an existing GPU image
    /// we assume that the given gpu image will not change during the compute calls
    void set_image(const cv::gpu::GpuMat &input_image);

    void compute();

    /// returns a reference to the GPU 3d structure holding the integral channels
    /// returns a non-const reference because cuda structures do not play nice with const-ness
    gpu_integral_channels_t& get_gpu_integral_channels() const;

    /// helper function to access the compute channels on cpu
    /// this is quite slow (large data transfer between GPU and CPU)
    /// this method should be used for debugging only
    const channels_t &get_channels();

    /// helper function to access the integral channels on cpu
    /// this is quite slow (large data transfer between GPU and CPU)
    /// this method should be used for debugging only
    const integral_channels_t &get_integral_channels();

protected:

    /// how much we shrink the channel images ?
    const int resizing_factor;

    /// cpu copy of the computed channels
    channels_t channels;

    /// cpu copy of the integral channels
    integral_channels_t integral_channels;

    /// the implementation object goes together with "this"
    /// this is only a trick to avoid exposing too many implementation details in the header file
    friend class GpuIntegralChannelsForPedestriansImplementation;
    boost::scoped_ptr<GpuIntegralChannelsForPedestriansImplementation> self_p;
    GpuIntegralChannelsForPedestriansImplementation &self;

};

} // end of namespace doppia

#endif // BICLOP_GPUINTEGRALCHANNELSFORPEDESTRIANS_HPP
