#include "BaseVeryFastIntegralChannelsDetector.hpp"

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "helpers/objects_detection/create_json_for_mustache.hpp"
#include "helpers/get_option_value.hpp"

#include <boost/foreach.hpp>
#include <boost/math/special_functions/round.hpp>

#include <cstdio>


namespace doppia {

typedef AbstractObjectsDetector::detection_window_size_t detection_window_size_t;

typedef MultiScalesIntegralChannelsModel::detectors_t detectors_t;
typedef MultiScalesIntegralChannelsModel::detector_t detector_t;

typedef BaseIntegralChannelsDetector::cascade_stages_t cascade_stages_t;
typedef BaseVeryFastIntegralChannelsDetector::fractional_cascade_stages_t fractional_cascade_stages_t;

BaseVeryFastIntegralChannelsDetector::BaseVeryFastIntegralChannelsDetector(
        const boost::program_options::variables_map &options,
        const boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p)
    :
      // since the inheritance is virtual, and the constructor is protected,
      // this particular constructor parameters will be never passed,
      // but C++ still require to define them "just in case"
      BaseIntegralChannelsDetector(options,
                                   boost::shared_ptr<SoftCascadeOverIntegralChannelsModel>(),
                                   boost::shared_ptr<AbstractNonMaximalSuppression>(), 0, 0),
      // this constructor will be called, since the inheritance is non virtual
      BaseMultiscalesIntegralChannelsDetector(options, detector_model_p)
{
    // shuffling the scales makes the scales inhibition much more effective
    //should_shuffle_the_scales = true; // true by default, since it does not hurt the performance of other methods
    should_shuffle_the_scales = false; // FIXME just for testing


    if(options.count("use_stixels"))
    {
        const bool use_stixels = get_option_value<bool>(options, "use_stixels");
        if(use_stixels)
        {
            // scales inhibition in the stixels use a specific (alternating) access pattern to the scales,
            // which assumes ordered scales
            should_shuffle_the_scales = false;
        }
    }

    return;
}


BaseVeryFastIntegralChannelsDetector::~BaseVeryFastIntegralChannelsDetector()
{
    // nothing to do here
    return;
}

template<typename FeatureType>
void recenter_feature(FeatureType &feature, const float offset_x, const float offset_y)
{
    typename FeatureType::rectangle_t &box = feature.box;
    box.min_corner().x(box.min_corner().x() + offset_x);
    box.max_corner().x(box.max_corner().x() + offset_x);

    box.min_corner().y(box.min_corner().y() + offset_y);
    box.max_corner().y(box.max_corner().y() + offset_y);
    return;
}

template<typename CascadeStageType>
void recenter_cascade(std::vector<CascadeStageType> &stages, const float offset_x, const float offset_y)
{

    BOOST_FOREACH(CascadeStageType &stage, stages)
    {
        recenter_feature(stage.weak_classifier.level1_node.feature, offset_x, offset_y);
        recenter_feature(stage.weak_classifier.level2_true_node.feature, offset_x, offset_y);
        recenter_feature(stage.weak_classifier.level2_true_node.feature, offset_x, offset_y);
    } // end of "for each stage"

    return;
}

/// Helper method used in both CPU and GPU versions
/// for each detection windows, it shift the detection window such as the upper-left corner becomes the center
void recenter_detections(AbstractObjectsDetector::detections_t &detections)
{
    BOOST_FOREACH(Detection2d &detection, detections)
    {
        Detection2d::rectangle_t &box = detection.bounding_box;
        const float
                offset_x = (box.max_corner().x() - box.min_corner().x())/2.0f,
                offset_y = (box.max_corner().y() - box.min_corner().y())/2.0f;

        box.max_corner().x(box.max_corner().x() - offset_x);
        box.min_corner().x(box.min_corner().x() - offset_x);

        box.max_corner().y(box.max_corner().y() - offset_y);
        box.min_corner().y(box.min_corner().y() - offset_y);
    }

    return;
}


std::vector<size_t> get_shuffled_indices(const size_t size)
{
    std::vector<size_t> indices;

    if (size == 0)
    {
        return indices;
    }

    size_t step_size = size - 1;
    indices.push_back(0);

    while(indices.size() < size)
    {
        const std::vector<size_t> current_indices = indices; // copy
        BOOST_FOREACH(const size_t index, current_indices)
        {
            const size_t new_index = index + step_size;

            if (new_index < size)
            {
                const bool not_listed = (std::find(indices.begin(), indices.end(), new_index) == indices.end());
                if(not_listed)
                {
                    indices.push_back(new_index);
                }
            }

        } // end of "for each index in the current list"

        step_size = std::max<size_t>(1, step_size/2);

    } // end of "while not listed all the indices"

    return indices;
}

void BaseVeryFastIntegralChannelsDetector::compute_scaled_detection_cascades()
{
    static bool first_call = true;
    if(first_call)
    {
        printf("BaseVeryFastIntegralChannelsDetector::compute_scaled_detection_cascades\n");
    }

    detection_cascade_per_scale.clear();
    detector_cascade_relative_scale_per_scale.clear();
    fractional_detection_cascade_per_scale.clear();
    detection_window_size_per_scale.clear();
    detector_index_per_scale.clear();
    original_detection_window_scales.clear();

    const size_t num_scales = search_ranges.size();
    detection_cascade_per_scale.reserve(num_scales);
    detector_cascade_relative_scale_per_scale.reserve(num_scales);
    fractional_detection_cascade_per_scale.reserve(num_scales);
    detection_window_size_per_scale.reserve(num_scales);
    detector_index_per_scale.reserve(num_scales);
    original_detection_window_scales.reserve(num_scales);


    // shuffling the scales makes the scales inhibition much more effective
    if(should_shuffle_the_scales)
    {
        printf("BaseVeryFastIntegralChannelsDetector has the scales shuffling enabled\n");
        const std::vector<size_t> scale_indices = get_shuffled_indices(num_scales);

        detector_search_ranges_t reordered_search_ranges;
        for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
        {
            if(false)
            {
                printf("Index %zi is now %zi\n", scale_index, scale_indices[scale_index]);
            }
            reordered_search_ranges.push_back(search_ranges[scale_indices[scale_index]]);
        }

        search_ranges = reordered_search_ranges;
    }

    for(size_t scale_index=0; scale_index < num_scales; scale_index+=1)
    {
        DetectorSearchRange &search_range = search_ranges[scale_index];

        if(search_range.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("BaseVeryFastIntegralChannelsDetector does not handle ratios != 1");
        }

        original_detection_window_scales.push_back(search_range.detection_window_scale);

        // search the nearest scale model ---
        const detector_t *nearest_detector_scale_p = NULL;
        size_t nearest_detector_scale_index = 0;
        float min_abs_log_scale = std::numeric_limits<float>::max();

        const float search_range_log_scale = log(search_range.detection_window_scale);
        size_t detector_index = 0;
        BOOST_FOREACH(const detector_t &detector, detector_model_p->get_detectors())
        {
            const float
                    log_detector_scale = log(detector.get_scale()),
                    abs_log_scale = std::abs<float>(search_range_log_scale - log_detector_scale);

            if(abs_log_scale < min_abs_log_scale)
            {
                min_abs_log_scale = abs_log_scale;
                nearest_detector_scale_p = &detector;
                nearest_detector_scale_index = detector_index;
            }

            detector_index += 1;
        } // end of "for each detector"

        assert(nearest_detector_scale_p != NULL);

        if(first_call)
        {
            printf("Selected model scale %.3f for detection window scale %.3f\n",
                   nearest_detector_scale_p->get_scale(), search_range.detection_window_scale);
        }

        // update the search range scale --
        search_range.detection_window_scale /= nearest_detector_scale_p->get_scale();


        const SoftCascadeOverIntegralChannelsModel::model_window_size_t &model_window_size =
                nearest_detector_scale_p->get_model_window_size();

        detector_t nearest_detector_scale =  *nearest_detector_scale_p; // simple copy

        //const bool recenter_the_search_range = true;
        const bool recenter_the_search_range = false;

        if(recenter_the_search_range)
        {
            const float
                    offset_x = model_window_size.x()/2.0f,
                    offset_y = model_window_size.y()/2.0f;
            recenter_cascade(nearest_detector_scale.get_stages(), -offset_x, -offset_y);
        }
        else
        {
            nearest_detector_scale =  *nearest_detector_scale_p; // simple copy
        }


        // get the rescaled detection cascade --
        const float relative_scale = search_range.detection_window_scale;
        const cascade_stages_t
                cascade_stages = nearest_detector_scale.get_rescaled_fast_stages(relative_scale);
        const fractional_cascade_stages_t
                fractional_cascade_stages = nearest_detector_scale.get_rescaled_fast_fractional_stages(relative_scale);

        if(recenter_the_search_range)
        {
            //const float
            //        offset_x = (model_window_size.x() * relative_scale)/2.0f,
            //        offset_y = (model_window_size.y() * relative_scale)/2.0f;

            //search_range.min_x += offset_x; search_range.max_x += offset_x;
            //search_range.min_y += offset_y; search_range.max_y += offset_y;


            // FIXME just for testing, very conservative search range
            search_range.min_x += model_window_size.x(); search_range.max_x -= model_window_size.x();
            search_range.min_y += model_window_size.y();
            search_range.max_y = std::max<int>(0,  search_range.max_y - static_cast<int>(model_window_size.y()));
        }

        detection_cascade_per_scale.push_back(cascade_stages);
        fractional_detection_cascade_per_scale.push_back(fractional_cascade_stages);
        detector_cascade_relative_scale_per_scale.push_back(relative_scale);
        detection_window_size_per_scale.push_back(model_window_size);
        detector_index_per_scale.push_back(nearest_detector_scale_index);
    } // end of "for each search range"


    // In BaseMultiscalesIntegraChannelsDetector we want to re-order the search ranges to group them by similar
    // resized image size, in the very_fast case, all scales use the same image size; so no reordering is needed

    const bool call_create_json_for_mustache = false;
    if(call_create_json_for_mustache)
    {
        // this call will raise an exception and stop the execution
        create_json_for_mustache(detection_cascade_per_scale);
    }

    first_call = false;
    return;
}


void BaseVeryFastIntegralChannelsDetector::compute_extra_data_per_scale(
        const size_t input_width, const size_t input_height)
{
    static bool first_call = true;

    using boost::math::iround;

    extra_data_per_scale.clear();
    extra_data_per_scale.reserve(search_ranges.size());

    // IntegralChannelsForPedestrians::get_shrinking_factor() == GpuIntegralChannelsForPedestrians::get_shrinking_factor()
    const float channels_resizing_factor = 1.0f/IntegralChannelsForPedestrians::get_shrinking_factor();

    for(size_t scale_index=0; scale_index < search_ranges.size(); scale_index+=1)
    {
        const DetectorSearchRange &search_range = search_ranges[scale_index];

        if(search_range.detection_window_ratio != 1.0)
        {
            throw std::invalid_argument("BaseVeryFastIntegralChannelsDetector does not handle ratios != 1");
        }

        // set the extra data --
        ScaleData extra_data;


        // update the scaled input sizes
        {
            // no image resizing, at any scale, yes this is magic !
            extra_data.scaled_input_image_size = image_size_t(input_width, input_height);
        }


        // update the scaled search ranges and strides
        {
            const float
                    detection_window_scale = original_detection_window_scales[scale_index],
                    input_to_channel_scale = channels_resizing_factor,
                    stride_scaling = detection_window_scale*input_to_channel_scale;

            extra_data.stride = stride_t(
                                    std::max<stride_t::coordinate_t>(1, iround(x_stride*stride_scaling)),
                                    std::max<stride_t::coordinate_t>(1, iround(y_stride*stride_scaling)));
            if(first_call)
            {
                printf("Detection window scale %.3f has strides (x,y) == (%.3f, %.3f) [image pixels] =>\t(%i, %i) [channel pixels]\n",
                       detection_window_scale,
                       x_stride*stride_scaling, y_stride*stride_scaling,
                       extra_data.stride.x(),  extra_data.stride.y());
            }

            // from input dimensions to integral channel dimensions
            extra_data.scaled_search_range = search_range.get_rescaled(input_to_channel_scale);
        }

        // update the scaled detection window sizes
        {
            const detection_window_size_t &original_detection_window_size = detection_window_size_per_scale[scale_index];
            const float
                    original_window_scale = search_range.detection_window_scale,
                    original_window_ratio = search_range.detection_window_ratio,
                    original_window_scale_x = original_window_scale*original_window_ratio;

            const detection_window_size_t::coordinate_t
                    detection_width = iround(original_detection_window_size.x()*original_window_scale_x),
                    detection_height = iround(original_detection_window_size.y()*original_window_scale);

            extra_data.scaled_detection_window_size = detection_window_size_t(detection_width, detection_height);
        }


        extra_data_per_scale.push_back(extra_data);
    } // end of "for each search range"

    // FIXME we use centered detection, so we need a different sanity check
    // sanity check
    //check_extra_data_per_scale();

    first_call = false;
    return;
}


} // end of namespace doppia
