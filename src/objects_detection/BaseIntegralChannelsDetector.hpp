#ifndef BICLOP_BASEINTEGRALCHANNELSDETECTOR_HPP
#define BICLOP_BASEINTEGRALCHANNELSDETECTOR_HPP

#include "BaseObjectsDetectorWithNonMaximalSuppression.hpp"

namespace doppia {

/// Helper structure to store data for each specific scale
struct ScaleData
{
    typedef geometry::point_xy<size_t> image_size_t;
    typedef geometry::point_xy<boost::uint16_t> stride_t;

    image_size_t scaled_input_image_size;
    DetectorSearchRange scaled_search_range;
    AbstractObjectsDetector::detection_window_size_t scaled_detection_window_size;
    stride_t stride; ///< scaled x/y stride
};

/// This base class fleshes what is common amongst IntegralChannelsDetector and GpuIntegralChannelsDetector
/// The code and members defined here are mainly related to handling of the multiple scales
/// @see IntegralChannelsDetector
/// @see GpuIntegralChannelsDetector
class BaseIntegralChannelsDetector: public BaseObjectsDetectorWithNonMaximalSuppression
{
public:

    typedef SoftCascadeOverIntegralChannelsModel::fast_stages_t cascade_stages_t;
    typedef SoftCascadeOverIntegralChannelsModel::fast_stage_t cascade_stage_t;

    typedef SoftCascadeOverIntegralChannelsModel::stump_stages_t stump_cascade_stages_t;
    typedef SoftCascadeOverIntegralChannelsModel::stump_stage_t stump_cascade_stage_t;

    typedef ScaleData::image_size_t image_size_t;
    typedef ScaleData::stride_t stride_t;

    static boost::program_options::options_description get_args_options();

protected:

    /// the constructor is protected because this base class is should not be instanciated directly
    BaseIntegralChannelsDetector(
            const boost::program_options::variables_map &options,
            const boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            const boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold, const int additional_border);
    ~BaseIntegralChannelsDetector();

    void set_stixels(const stixels_t &stixels);
    void set_ground_plane_corridor(const ground_plane_corridor_t &corridor);

protected:

    const float score_threshold;
    bool use_the_detector_model_cascade;

    boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p;

    detection_window_size_t scale_one_detection_window_size;

    /// for single scale detector, all the elements of this vector will be copies of
    /// scale_one_detection_window_size, but for multiscales detector, they will be different
    /// @see MultiScalesIntegralChannelsDetector
    std::vector<detection_window_size_t> detection_window_size_per_scale;

    /// for each entry inside AbstractObjectsDetector::search_ranges we
    /// store the corresponding detector cascade
    std::vector<cascade_stages_t> detection_cascade_per_scale;

    std::vector<stump_cascade_stages_t> detection_stump_cascade_per_scale;

    /// relative scale of the detection cascade (linked to detection_window_size per scale)
    /// (used for boundary checks only)
    std::vector<float> detector_cascade_relative_scale_per_scale;

    /// helper container for compute_extra_data_per_scale
    std::vector<float> original_detection_window_scales;

    /// additional data needed to compute detections are one specific scale
    std::vector<ScaleData> extra_data_per_scale;

    /// updates the values inside detection_cascade_per_scale and detection_window_size_per_scale
    virtual void compute_scaled_detection_cascades();

    virtual void compute_extra_data_per_scale(const size_t input_width, const size_t input_height);

    /// helper function that validates the internal consistency of the extra_data_per_scale
    void check_extra_data_per_scale();

    // store information related to ground plane and stixels
    ///@{
    const int additional_border;

    /// for each row in the image, assuming that is the object's bottom position,
    /// this vector stores the expected top object row
    /// top row value is -1 for rows above the horizon
    std::vector<int> estimated_ground_plane_corridor;

    stixels_t estimated_stixels;
    const int stixels_vertical_margin, stixels_scales_margin;
    ///@}

    /// this method must be implemented by the children classes
    virtual size_t get_input_width() const = 0;

    /// this method must be implemented by the children classes
    virtual size_t get_input_height() const = 0;

};


/// helper method used by IntegralChannelsDetector and GpuIntegralChannelsDetector
void add_detection(
        const boost::uint16_t detection_col, const boost::uint16_t detection_row, const float detection_score,
        const ScaleData &scale_data,
        AbstractObjectsDetector::detections_t &detections);

/// similar to add_detection, but keeps things in the integral channel reference frame instead of the input image
void add_detection_for_bootstrapping(
        const boost::uint16_t detection_col, const boost::uint16_t detection_row, const float detection_score,
        const AbstractObjectsDetector::detection_window_size_t &original_detection_window_size,
        AbstractObjectsDetector::detections_t &detections);


/// Helper class for sorting the search ranges by scale (indirectly)
class SearchRangeScaleComparator
{

public:
    SearchRangeScaleComparator(const detector_search_ranges_t &search_ranges);
    ~SearchRangeScaleComparator();

    bool operator()(const size_t a, const size_t b);

protected:
    const detector_search_ranges_t &search_ranges;

};

} // end of namespace doppia

#endif // BICLOP_BASEINTEGRALCHANNELSDETECTOR_HPP
