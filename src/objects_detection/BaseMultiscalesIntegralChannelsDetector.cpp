#include "BaseMultiscalesIntegralChannelsDetector.hpp"

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "ModelWindowToObjectWindowConverterFactory.hpp"

#include <boost/foreach.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include <cstdio>


namespace doppia {

typedef MultiScalesIntegralChannelsModel::detectors_t detectors_t;
typedef MultiScalesIntegralChannelsModel::detector_t detector_t;

typedef AbstractObjectsDetector::detection_window_size_t detection_window_size_t;
typedef AbstractObjectsDetector::detections_t detections_t;
typedef AbstractObjectsDetector::detection_t detection_t;

typedef BaseIntegralChannelsDetector::cascade_stages_t cascade_stages_t;

using boost::counting_iterator;



BaseMultiscalesIntegralChannelsDetector::BaseMultiscalesIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p_)
    :
      // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0),
      detector_model_p(detector_model_p_)
{
    // the MultiScalesIntegralChannelsModel constructor already validated the consistency of the data

    if(detector_model_p == false)
    {
        throw std::invalid_argument("BaseMultiscalesIntegralChannelsDetector requires "
                                    "a non-null MultiScalesIntegralChannelsModel");
    }


    bool found_scale_one = false;

    BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
    {
        // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
        if(detector.get_shrinking_factor() != IntegralChannelsForPedestrians::get_shrinking_factor())
        {
            printf("detector for scale %.3f has shrinking_factor == %i\n",
                   detector.get_scale(), detector.get_shrinking_factor());

            printf("(Gpu)IntegralChannelsForPedestrians::get_shrinking_factor() == %i\n",
                   IntegralChannelsForPedestrians::get_shrinking_factor());

            throw std::invalid_argument("One of the input models has a different shrinking factor than "
                                        "the currently used integral channels computer");
        }

        if(detector.get_scale() == 1.0f)
        {
            found_scale_one = true;

            // get the detection window size
            scale_one_detection_window_size = detector.get_model_window_size();

            // set the model to object window converter
            model_window_to_object_window_converter_p.reset(
                        ModelWindowToObjectWindowConverterFactory::new_instance(detector.get_model_window_size(),
                                                                                detector.get_object_window()));
        } // end of "if this is scale one"

    } // end of "for each detector"


    if(found_scale_one == false)
    {
        throw std::invalid_argument("Failed to construct MultiscalesIntegralChannelsDetector because "
                                    "the reiceved data does contain a detector for scale 1");
    }


    return;
}


BaseMultiscalesIntegralChannelsDetector::~BaseMultiscalesIntegralChannelsDetector()
{
    // nothing to do here
    return;
}

/// reordering search_ranges by scale and making sure
/// detection_{cascade, window_size}_per_scale is also in correct order
void reorder_by_search_range_scale(
        detector_search_ranges_t  &search_ranges,
        std::vector<cascade_stages_t>  &detection_cascade_per_scale,
        std::vector<detection_window_size_t> &detection_window_size_per_scale)
{

    // (sorting two arrays in C++ is a pain, see
    // http://www.stanford.edu/~dgleich/notebook/2006/03/sorting_two_arrays_simultaneou.html )

    std::vector<size_t> search_ranges_indices(counting_iterator<size_t>(0),
                                              counting_iterator<size_t>(search_ranges.size()));

    SearchRangeScaleComparator search_range_scale_comparator(search_ranges);
    std::sort(search_ranges_indices.begin(), search_ranges_indices.end(), search_range_scale_comparator);


    std::vector<cascade_stages_t> reordered_detection_cascade_per_scale;
    detector_search_ranges_t reordered_search_ranges;
    std::vector<detection_window_size_t> reordered_detection_window_size_per_scale;

    reordered_detection_cascade_per_scale.resize(detection_cascade_per_scale.size());
    reordered_search_ranges.resize(search_ranges.size());
    reordered_detection_window_size_per_scale.resize(search_ranges.size());

    assert(reordered_detection_cascade_per_scale.size() == reordered_search_ranges.size());

    for(size_t index=0; index < search_ranges_indices.size(); index +=1)
    {
        const size_t old_index = search_ranges_indices[index];
        reordered_search_ranges[index] = search_ranges[old_index];
        reordered_detection_cascade_per_scale[index] = detection_cascade_per_scale[old_index];
        reordered_detection_window_size_per_scale[index] = detection_window_size_per_scale[old_index];
    }

    detection_cascade_per_scale = reordered_detection_cascade_per_scale;
    search_ranges = reordered_search_ranges;
    detection_window_size_per_scale = reordered_detection_window_size_per_scale;

    return;
}

/// updates the values inside detection_cascade_per_scale
/// this variant will also update search_ranges,
/// (since we will be shifting the actual scales)
void BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades()
{
    static bool first_call = true;
    if(first_call)
    {
        printf("BaseMultiscalesIntegralChannelsDetector::compute_scaled_detection_cascades\n");
    }

    detection_cascade_per_scale.clear();
    detection_stump_cascade_per_scale.clear();
    detector_cascade_relative_scale_per_scale.clear();
    detection_window_size_per_scale.clear();
    original_detection_window_scales.clear();

    const size_t num_scales = search_ranges.size();
    detection_cascade_per_scale.reserve(num_scales);
    detection_stump_cascade_per_scale.reserve(num_scales);
    detector_cascade_relative_scale_per_scale.reserve(num_scales);
    detection_window_size_per_scale.reserve(num_scales);
    original_detection_window_scales.reserve(num_scales);


    for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
    {
        DetectorSearchRange &search_range = search_ranges[scale_index];

        if(search_range.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("MultiscalesIntegralChannelsDetector does not handle ratios != 1");
        }

        original_detection_window_scales.push_back(search_range.detection_window_scale);

        // search the nearest scale model ---
        const detector_t *nearest_detector_scale_p = NULL;
        float min_abs_log_scale = std::numeric_limits<float>::max();

        const float search_range_log_scale = log(search_range.detection_window_scale);
        BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
        {
            const float
                    log_detector_scale = log(detector.get_scale()),
                    abs_log_scale = std::abs<float>(search_range_log_scale - log_detector_scale);

            if(abs_log_scale < min_abs_log_scale)
            {
                min_abs_log_scale = abs_log_scale;
                nearest_detector_scale_p = &detector;
            }
        } // end of "for each detector"

        assert(nearest_detector_scale_p != NULL);

        if(first_call)
        {
            printf("Selected model scale %.3f for detection window scale %.3f\n",
                   nearest_detector_scale_p->get_scale(), search_range.detection_window_scale);
        }

        // update the search range scale --
        search_range.detection_window_scale /= nearest_detector_scale_p->get_scale();

        const float relative_scale = 1.0f; // we rescale the images, not the the features
        const cascade_stages_t cascade_stages = nearest_detector_scale_p->get_rescaled_fast_stages(relative_scale);
        detection_cascade_per_scale.push_back(cascade_stages);

        const stump_cascade_stages_t stump_cascade_stages = nearest_detector_scale_p->get_rescaled_stump_stages(relative_scale);
        detection_stump_cascade_per_scale.push_back(stump_cascade_stages);

        detector_cascade_relative_scale_per_scale.push_back(relative_scale);
        detection_window_size_per_scale.push_back(nearest_detector_scale_p->get_model_window_size());
    } // end of "for each search range"


    // reordering search_ranges by scale and making sure detection_cascade_per_scale is also in correct order
    reorder_by_search_range_scale(search_ranges, detection_cascade_per_scale, detection_window_size_per_scale);

    first_call = false;
    return;
}


} // end of namespace doppia
