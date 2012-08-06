#include "SoftCascadeOverIntegralChannelsModel.hpp"

#include "SoftCascadeOverIntegralChannelsFastFractionalStage.hpp"

#include "detector_model.pb.h"

#include "helpers/Log.hpp"

#include <boost/foreach.hpp>
#include <boost/type_traits/is_same.hpp>
//#include <boost/geometry/geometry.hpp>
#include <boost/math/special_functions/round.hpp>

#include <ostream>
#include <vector>
#include <cmath>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::ostream & log_warning()
{
    return  logging::log(logging::WarningMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "SoftCascadeOverIntegralChannelsModel");
}

std::vector<float> upscaling_factors;
std::vector<float> downscaling_factors;

} // end of anonymous namespace


namespace doppia {

SoftCascadeOverIntegralChannelsModel::SoftCascadeOverIntegralChannelsModel(const doppia_protobuf::DetectorModel &model)
{

    if(model.has_detector_name())
    {
        log_info() << "Parsing model " << model.detector_name() << std::endl;
    }

    if(model.detector_type() != doppia_protobuf::DetectorModel::SoftCascadeOverIntegralChannels)
    {
        throw std::runtime_error("Received model is not of the expected type SoftCascadeOverIntegralChannels");
    }

    if(model.has_soft_cascade_model() == false)
    {
        throw std::runtime_error("The model content does not match the model type");
    }

    set_stages_from_model(model.soft_cascade_model());

    shrinking_factor = model.soft_cascade_model().shrinking_factor();

    if(model.has_scale())
    {
        scale = model.scale();
    }
    else
    {
        // default scale is 1 ("I am the canonical scale")
        scale = 1.0;
    }

    if(model.has_model_window_size())
    {
        model_window_size.x(model.model_window_size().x());
        model_window_size.y( model.model_window_size().y());
    }
    else
    {
        // default to INRIAPerson model size
        model_window_size.x(64);
        model_window_size.y(128);
    }

    if(model.has_object_window())
    {
        const doppia_protobuf::Box &the_object_window = model.object_window();
        object_window.min_corner().x(the_object_window.min_corner().x());
        object_window.min_corner().y(the_object_window.min_corner().y());
        object_window.max_corner().x(the_object_window.max_corner().x());
        object_window.max_corner().y(the_object_window.max_corner().y());
    }
    else
    {
        // default to INRIAPerson object size
        object_window.min_corner().x(8);
        object_window.min_corner().y(16);
        object_window.max_corner().x(8+48);
        object_window.max_corner().y(16+96);
    }

    return;
}



SoftCascadeOverIntegralChannelsModel::~SoftCascadeOverIntegralChannelsModel()
{
    // nothing to do here
    return;
}


template<typename DecisionStumpType>
void set_decision_stump_feature(const doppia_protobuf::IntegralChannelDecisionStump &stump_data,
                                DecisionStumpType &stump)
{

    stump.feature.channel_index = stump_data.feature().channel_index();

    const doppia_protobuf::Box &box_data = stump_data.feature().box();
    IntegralChannelsFeature::rectangle_t &box = stump.feature.box;
    box.min_corner().x(box_data.min_corner().x());
    box.min_corner().y(box_data.min_corner().y());
    box.max_corner().x(box_data.max_corner().x());
    box.max_corner().y(box_data.max_corner().y());

    stump.feature_threshold = stump_data.feature_threshold();

    //const float box_area = boost::geometry::area(box);
    const float box_area =
            (box.max_corner().x() - box.min_corner().x())*(box.max_corner().y() - box.min_corner().y());


    if(box_area == 0)
    {
        log_warning() << "feature.box min_x " << box.min_corner().x() << std::endl;
        log_warning() << "feature.box min_y " << box.min_corner().y() << std::endl;
        log_warning() << "feature.box max_x " << box.max_corner().x() << std::endl;
        log_warning() << "feature.box max_y " << box.max_corner().y() << std::endl;
        throw std::runtime_error("One of the input features has area == 0");
    }


    return;
}


void set_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump_data, DecisionStump &stump)
{
    set_decision_stump_feature(stump_data, stump);
    stump.larger_than_threshold = stump_data.larger_than_threshold();
    return;
}


void set_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump_data, SimpleDecisionStump &stump)
{
    set_decision_stump_feature(stump_data, stump);

    // nothing else todo here

    // no need to check this, handled at the set_weak_classifier level
    if(false and stump_data.larger_than_threshold() == false)
    {
        throw std::runtime_error("SimpleDecisionStump was set using stump_data.larger_than_threshold == false, "
                                 "we expected it to be true; code needs to be changed to handle this case.");
    }
    return;
}


void set_decision_stump(const doppia_protobuf::IntegralChannelDecisionStump &stump_data,
                        const float stage_weight,
                        DecisionStumpWithWeights &stump)
{
    set_decision_stump_feature(stump_data, stump);

    if(stump_data.has_true_leaf_weight() and stump_data.has_false_leaf_weight())
    {

        if(stump_data.larger_than_threshold())
        {
            stump.weight_true_leaf = stump_data.true_leaf_weight();
            stump.weight_false_leaf = stump_data.false_leaf_weight();
        }
        else
        {
            stump.weight_true_leaf = stump_data.false_leaf_weight();
            stump.weight_false_leaf = stump_data.true_leaf_weight();
        }

        if(stage_weight != 1.0)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsStage the leaves have true/false weights, "
                                     "but the stage weight is different than 1.0. "
                                     "This case is not currently handled.");
        }
    }
    else
    {
        if(stump_data.larger_than_threshold())
        {
            stump.weight_true_leaf = stage_weight;
            stump.weight_false_leaf = -stage_weight;
        }
        else
        {
            stump.weight_true_leaf = -stage_weight;
            stump.weight_false_leaf = stage_weight;
        }
    }

    return;
}


void set_weak_classifier(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data,
                         SoftCascadeOverIntegralChannelsModel::stump_stage_t &stage)
{

    const doppia_protobuf::IntegralChannelDecisionStump &stump_data = stage_data.decision_stump();
    const float stage_weight = stage_data.weight();

    set_decision_stump(stump_data, stage_weight, stage.weak_classifier);

    return;
}


void set_weak_classifier(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data,
                         SoftCascadeOverIntegralChannelsModel::stage_t &stage)
{

    throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::stage_t is not supported anymore. "
                             "Use ::fast_stage_t");

    if(boost::is_same<SoftCascadeOverIntegralChannelsStage::weak_classifier_t, Level2DecisionTree>::value)
    {
        if(stage_data.feature_type() != doppia_protobuf::SoftCascadeOverIntegralChannelsStage::Level2DecisionTree)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsStage contains a feature_type != Level2DecisionTree");
        }

        if(stage_data.has_level2_decision_tree() == false)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsStage feature_type == Level2DecisionTree but "
                                     "no level2_decision_tree has been set");
        }


        if(stage_data.level2_decision_tree().nodes().size() != 3)
        {
            log_error() << "stage_data.level2_decision_tree.nodes().size() == " << stage_data.level2_decision_tree().nodes().size() << ", not 3" << std::endl;
            throw std::runtime_error("SoftCascadeOverIntegralChannelsStage level2_decision_tree does not contain 3 nodes");
        }

        const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode
                *level_1_node_p = NULL, *level2_true_node_p = NULL, *level2_false_node_p = NULL;
        for(int i=0; i < stage_data.level2_decision_tree().nodes().size(); i+=1)
        {
            const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode &node = stage_data.level2_decision_tree().nodes(i);
            if(node.id() == node.parent_id())
            {
                level_1_node_p =  &node;
            }
            else if(node.has_parent_value() and node.parent_value() == true)
            {
                level2_true_node_p = &node;
            }
            else if(node.has_parent_value() and node.parent_value() == false)
            {
                level2_false_node_p = &node;
            }
            else
            {
                // we skip weird nodes
            }
        }

        if(level_1_node_p == NULL)
        {
            throw std::runtime_error("Could not find the parent of the decision tree");
        }

        if(level2_true_node_p == NULL or level2_false_node_p == NULL)
        {
            throw std::runtime_error("Could not find one of the children nodes of the decision tree");
        }

        set_decision_stump( level_1_node_p->decision_stump(), stage.weak_classifier.level1_node);
        set_decision_stump( level2_true_node_p->decision_stump(), stage.weak_classifier.level2_true_node);
        set_decision_stump( level2_false_node_p->decision_stump(), stage.weak_classifier.level2_false_node);
    }
    else
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsModel with ?? weak classifiers, not yet implemented");
    }

    return;
}


void set_weak_classifier(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data,
                         SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage)
{

    if(boost::is_same<SoftCascadeOverIntegralChannelsFastStage::weak_classifier_t, Level2DecisionTreeWithWeights>::value)
    {
        if(stage_data.feature_type() != doppia_protobuf::SoftCascadeOverIntegralChannelsStage::Level2DecisionTree)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage contains a feature_type != Level2DecisionTree");
        }

        if(stage_data.has_level2_decision_tree() == false)
        {
            throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage feature_type == Level2DecisionTree but "
                                     "no level2_decision_tree has been set");
        }

        if(stage_data.level2_decision_tree().nodes().size() != 3)
        {
            log_error() << "stage_data.level2_decision_tree.nodes().size() == "
                        << stage_data.level2_decision_tree().nodes().size() << ", not 3" << std::endl;
            throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage level2_decision_tree does not contain 3 nodes");
        }

        const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode
                *level_1_node_p = NULL, *level2_true_node_p = NULL, *level2_false_node_p = NULL;
        for(int i=0; i < stage_data.level2_decision_tree().nodes().size(); i+=1)
        {
            const doppia_protobuf::IntegralChannelBinaryDecisionTreeNode &node = stage_data.level2_decision_tree().nodes(i);
            if(node.id() == node.parent_id())
            {
                level_1_node_p =  &node;
            }
            else if(node.has_parent_value() and node.parent_value() == true)
            {
                level2_true_node_p = &node;
            }
            else if(node.has_parent_value() and node.parent_value() == false)
            {
                level2_false_node_p = &node;
            }
            else
            {
                // we skip weird nodes
            }
        }

        if(level_1_node_p == NULL)
        {
            throw std::runtime_error("Could not find the parent of the decision tree");
        }

        if(level2_true_node_p == NULL or level2_false_node_p == NULL)
        {
            throw std::runtime_error("Could not find one of the children nodes of the decision tree");
        }

        if(level_1_node_p->decision_stump().larger_than_threshold() == false)
        {
            std::swap(level2_true_node_p, level2_false_node_p);
        }

        const float stage_weight = stage_data.weight();
        set_decision_stump(level_1_node_p->decision_stump(), stage.weak_classifier.level1_node);

        set_decision_stump(level2_true_node_p->decision_stump(), stage_weight, stage.weak_classifier.level2_true_node);
        set_decision_stump(level2_false_node_p->decision_stump(), stage_weight, stage.weak_classifier.level2_false_node);
    }
    else
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsFastStage with ?? weak classifiers, not yet implemented");
    }

    stage.weak_classifier.compute_bounding_box();
    return;
}


void SoftCascadeOverIntegralChannelsModel::set_stages_from_model(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &model)
{
    typedef google::protobuf::RepeatedPtrField< doppia_protobuf::IntegralChannelsFeature > features_t;

    log_info() << "The soft cascade contains " << model.stages().size() << " stages" << std::endl;

    stump_stages.clear();
    stages.clear();
    fast_stages.clear();

    for(int c=0; c < model.stages().size(); c+=1)
    {
        const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data = model.stages(c);

        if(stage_data.has_decision_stump())
        {
            stump_stage_t stage;
            stage.cascade_threshold = stage_data.cascade_threshold();
            set_weak_classifier(stage_data, stage);

            stump_stages.push_back(stage);
        }
        else if(stage_data.has_level2_decision_tree())
        {
            fast_stage_t stage;

            //stage.weight = stage_data.weight();
            stage.cascade_threshold = stage_data.cascade_threshold();
            set_weak_classifier(stage_data, stage);

            //stages.push_back(stage); stage_t usage is now deprecated
            fast_stages.push_back(stage);

            if(true and ((c == 0) or (c == (model.stages().size() - 1))))
            {
                printf("Stage %i cascade_threshold == %.6f\n", c, stage.cascade_threshold);
            }
        }
        else if(stage_data.has_leveln_decision_tree())
        {
            throw std::invalid_argument("LevelN decisions tree are not yet supported in the code");
        }
        else
        {
            throw std::invalid_argument("SoftCascadeOverIntegralChannelsModel received an unknown stage type");
        }

    } // end of "for each stage in the cascade"


    //const bool print_stages = true;
    const bool print_stages = false;
    if(print_stages)
    {
        //print_detection_cascade_stages(log_info(), stages);
        //print_detection_cascade_stages(log_debug(), stages);
        print_detection_cascade_stages(log_debug(), fast_stages);
    }


    //if(true)
    if(false)
    {
        log_warning() << "Shuffling the cascade stages, this should only be used for debugging, BAD IDEA" << std::endl;
        srand(time(NULL));
        std::random_shuffle(stages.begin(), stages.end());
        std::random_shuffle(fast_stages.begin(), fast_stages.end());
    }

    return;
}



/// Helper method that gives the crucial information for the FPDW implementation
/// these numbers are obtained via
/// doppia/src/test/objects_detection/test_objects_detection + plot_channel_statistics.py
/// (this method is not speed critical)
float get_channel_scaling_factor(const boost::uint8_t channel_index,
                                 const float relative_scale)
{
    float channel_scaling = 1, up_a = 1, down_a = 1, up_b = 2, down_b = 2;

    // FIXME how to propagate here which method is used ?
    // should these values be part of the computing structure ?

    if(relative_scale == 1)
    { // when no rescaling there is no scaling factor
        return 1.0f;
    }

    const bool
            use_p_dollar_estimates = true,
            use_v0_estimates = false,
            use_no_estimate = false;

    if(use_p_dollar_estimates)
    {
        const float lambda = 1.099, a = 0.89;

        if(channel_index <= 6)
        { // Gradient histograms and gradient magnitude
            down_a = a; down_b = lambda / log(2);

            // upscaling case is roughly a linear growth
            // these are the ideal values
            up_a = 1; up_b = 0;
        }
        else if((channel_index >= 7) and (channel_index <= 9))
        { // LUV channels, quadratic growth
            // these are the ideal values
            down_a = 1; down_b = 2;
            up_a = 1; up_b = 2;
        }
        else
        {
            throw std::runtime_error("get_channel_scaling_factor use_p_dollar_estimates called with "
                                     "an unknown integral channel index");
        }

    }
    else if(use_v0_estimates)
    {
        // these values hold for IntegralChannelsForPedestrians::compute_v0
        // FIXME these are old values, need update

        // num_scales ==  12
        // r = a*(k**b); r: feature scaling factor; k: relative scale
        // HOG	for downscaling r = 0.989*(x**-1.022), for upscaling  r = 1.003*(x**1.372)
        // L	for downscaling r = 0.963*(x**-1.771), for upscaling  r = 0.956*(x**1.878)
        // UV	for downscaling r = 0.966*(x**-2.068), for upscaling  r = 0.962*(x**2.095)

        if(channel_index <= 6)
        { // Gradient histograms and gradient magnitude
            down_a = 1.0f/0.989; down_b = +1.022;
            // upscaling case is roughly a linear growth
            up_a = 1.003; up_b = 1.372;
        }
        else if(channel_index == 7)
        { // L channel, quadratic growth
            // for some strange reason test_objects_detection + plot_channel_statistics.py indicate
            // that the L channel behaves differently than UV channels
            down_a = 1.0f/0.963; down_b = +1.771;
            up_a = 0.956; up_b = 1.878;
        }
        else if(channel_index == 8 or channel_index ==9)
        { // UV channels, quadratic growth
            down_a = 1.0f/0.966; down_b = +2.068;
            up_a = 0.962; up_b = 2.095;
        }
        else
        {
            throw std::runtime_error("get_channel_scaling_factor use_v0_estimates called with "
                                     "an unknown integral channel index");
        }
    } // end of "IntegralChannlesForPedestrians::compute_v0"
    else if(use_no_estimate)
    {
        // we disregard the scaling and keep the same feature value
        up_a = 1; up_b = 0;
        down_a = 1; down_b = 0;
    }
    else
    {
        throw std::runtime_error("no estimate was selected for get_channel_scaling_factor");
    }


    {
        float a=1, b=2;
        if(relative_scale >= 1)
        { // upscaling case
            a = up_a;
            b = up_b;
        }
        else
        { // size_scaling < 1, downscaling case
            a = down_a;
            b = down_b;
        }

        channel_scaling = a*pow(relative_scale, b);

        const bool check_scaling = true;
        if(check_scaling)
        {
            if(relative_scale >= 1)
            { // upscaling
                if(channel_scaling < 1)
                {
                    throw std::runtime_error("get_channel_scaling_factor upscaling parameters are plain wrong");
                }
            }
            else
            { // downscaling
                if(channel_scaling > 1)
                {
                    throw std::runtime_error("get_channel_scaling_factor upscaling parameters are plain wrong");
                }
            }
        } // end of check_scaling
    }

    return channel_scaling;
}


void scale_the_box(IntegralChannelsFeature::rectangle_t &box, const float relative_scale)
{
    using boost::math::iround;
    box.min_corner().x( iround(box.min_corner().x() * relative_scale) );
    box.min_corner().y( iround(box.min_corner().y() * relative_scale) );
    box.max_corner().x( std::max(box.min_corner().x() + 1, iround(box.max_corner().x() * relative_scale)) );
    box.max_corner().y( std::max(box.min_corner().y() + 1, iround(box.max_corner().y() * relative_scale)) );

    assert(rectangle_area(box) >= 1);
    return;
}


void scale_the_box(IntegralChannelsFractionalFeature::rectangle_t &box, const float relative_scale)
{
    box.min_corner().x( box.min_corner().x() * relative_scale );
    box.min_corner().y( box.min_corner().y() * relative_scale );
    box.max_corner().x( box.max_corner().x() * relative_scale );
    box.max_corner().y( box.max_corner().y() * relative_scale );
    return;
}


/// we change the size of the rectangle and
/// adjust the threshold to take into the account the slight change in area
template<typename StumpType>
void scale_the_stump(StumpType &decision_stump,
                     const float relative_scale)
{

    const float channel_scaling_factor = get_channel_scaling_factor(
                                             decision_stump.feature.channel_index, relative_scale);

    typename StumpType::feature_t::rectangle_t &box = decision_stump.feature.box;
    const float original_area = rectangle_area(box);
    scale_the_box(box, relative_scale);
    const float new_area = rectangle_area(box);

    float area_approximation_scaling_factor = 1;
    if((new_area != 0) and (original_area != 0))
    {
        // integral_over_new_area * (original_area / new_area) =(approx)= integral_over_original_area
        //area_approximation_scaling_factor = original_area / new_area;
        const float expected_new_area = original_area*relative_scale*relative_scale;
        area_approximation_scaling_factor = expected_new_area / new_area;
        //printf("area_approximation_scaling_factor %.3f\n", area_approximation_scaling_factor);
    }

    decision_stump.feature_threshold /= area_approximation_scaling_factor; // FIXME this seems wrong !

    decision_stump.feature_threshold *= channel_scaling_factor;

    const bool print_channel_scaling_factor = false;
    if(print_channel_scaling_factor)
    {
        printf("relative_scale %.3f -> channel_scaling_factor %.3f\n", relative_scale, channel_scaling_factor);
    }

    return;
}


SoftCascadeOverIntegralChannelsModel::stages_t
SoftCascadeOverIntegralChannelsModel::get_rescaled_stages(const float relative_scale) const
{
    throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::stages_t is deprecated, "
                             "this get_rescaled_stages is deprecated too");
    stages_t rescaled_stages;
    rescaled_stages.reserve(stages.size());

    BOOST_FOREACH(const stage_t &stage, stages)
    {
        stage_t rescaled_stage = stage;

        Level2DecisionTree &weak_classifier = rescaled_stage.weak_classifier;
        scale_the_stump(weak_classifier.level1_node, relative_scale);
        scale_the_stump(weak_classifier.level2_true_node, relative_scale);
        scale_the_stump(weak_classifier.level2_false_node, relative_scale);

        //weak_classifier.compute_bounding_box(); // we update the bounding box of the weak classifier
        rescaled_stages.push_back(rescaled_stage);
    }

    if(false and (not stages.empty()))
    {
        printf("SoftCascadeOverIntegralChannelsModel::get_rescaled_stages "
               "Rescaled stage 0 cascade_threshold == %3.f\n",
               rescaled_stages[0].cascade_threshold);
    }

    return rescaled_stages;
}


SoftCascadeOverIntegralChannelsModel::fast_stages_t
SoftCascadeOverIntegralChannelsModel::get_rescaled_fast_stages(const float relative_scale) const
{
    fast_stages_t rescaled_stages;
    rescaled_stages.reserve(fast_stages.size());

    BOOST_FOREACH(const fast_stage_t &stage, fast_stages)
    {
        fast_stage_t rescaled_stage = stage;

        fast_stage_t::weak_classifier_t &weak_classifier = rescaled_stage.weak_classifier;
        scale_the_stump(weak_classifier.level1_node, relative_scale);
        scale_the_stump(weak_classifier.level2_true_node, relative_scale);
        scale_the_stump(weak_classifier.level2_false_node, relative_scale);

        weak_classifier.compute_bounding_box(); // we update the bounding box of the weak classifier
        rescaled_stages.push_back(rescaled_stage);
    }

    if(false and (not stages.empty()))
    {
        printf("SoftCascadeOverIntegralChannelsModel::get_rescaled_fast_stages "
               "Rescaled stage 0 cascade_threshold == %3.f\n",
               rescaled_stages[0].cascade_threshold);
    }

    return rescaled_stages;
}


SoftCascadeOverIntegralChannelsModel::fast_fractional_stages_t
SoftCascadeOverIntegralChannelsModel::get_rescaled_fast_fractional_stages(const float relative_scale) const
{
    // this is a copy and paste from SoftCascadeOverIntegralChannelsModel::get_rescaled_stages
    fast_fractional_stages_t rescaled_stages;
    rescaled_stages.reserve(stages.size());

    BOOST_FOREACH(const stage_t &stage, stages)
    {
        fast_fractional_stage_t rescaled_stage = stage;

        fast_fractional_stage_t::weak_classifier_t &weak_classifier = rescaled_stage.weak_classifier;
        scale_the_stump(weak_classifier.level1_node, relative_scale);
        scale_the_stump(weak_classifier.level2_true_node, relative_scale);
        scale_the_stump(weak_classifier.level2_false_node, relative_scale);

        //weak_classifier.compute_bounding_box(); // we update the bounding box of the weak classifier

        rescaled_stages.push_back(rescaled_stage);
    }

    if(false and (not stages.empty()))
    {
        printf("SoftCascadeOverIntegralChannelsModel::get_rescaled_fast_fractional_stages "
               "Rescaled stage 0 cascade_threshold == %3.f\n",
               rescaled_stages[0].cascade_threshold);
    }

    return rescaled_stages;
}


SoftCascadeOverIntegralChannelsModel::stump_stages_t
SoftCascadeOverIntegralChannelsModel::get_rescaled_stump_stages(const float relative_scale) const
{
    stump_stages_t rescaled_stages;
    rescaled_stages.reserve(stump_stages.size());

    BOOST_FOREACH(const stump_stage_t &stage, stump_stages)
    {
        stump_stage_t rescaled_stage = stage;

        DecisionStumpWithWeights &weak_classifier = rescaled_stage.weak_classifier;
        scale_the_stump(weak_classifier, relative_scale);

        //weak_classifier.compute_bounding_box(); // we update the bounding box of the weak classifier

        rescaled_stages.push_back(rescaled_stage);
    }

    if(false and (not stump_stages.empty()))
    {
        printf("SoftCascadeOverIntegralChannelsModel::get_rescaled_stages "
               "Rescaled stump stage 0 cascade_threshold == %3.f\n",
               rescaled_stages[0].cascade_threshold);
    }

    return rescaled_stages;
}


SoftCascadeOverIntegralChannelsModel::stages_t &SoftCascadeOverIntegralChannelsModel::get_stages()
{
    throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::get_stages is obsolete, use get_fast_stages instead.");
    return stages;
}

const SoftCascadeOverIntegralChannelsModel::stages_t &SoftCascadeOverIntegralChannelsModel::get_stages() const
{
    throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::get_stages is obsolete, use get_fast_stages instead.");
    return stages;
}

const SoftCascadeOverIntegralChannelsModel::fast_stages_t  &SoftCascadeOverIntegralChannelsModel::get_fast_stages() const
{
    return fast_stages;
}

const SoftCascadeOverIntegralChannelsModel::stump_stages_t &SoftCascadeOverIntegralChannelsModel::get_stump_stages() const
{
    return stump_stages;
}

int SoftCascadeOverIntegralChannelsModel::get_shrinking_factor() const
{
    return shrinking_factor;
}

float SoftCascadeOverIntegralChannelsModel::get_last_cascade_threshold() const
{

    if(stages.empty() == false)
    {
        const float last_cascade_threshold = stages.back().cascade_threshold;
        return last_cascade_threshold;
    }
    else if(fast_stages.empty() == false)
    {
        const float last_cascade_threshold = fast_stages.back().cascade_threshold;
        return last_cascade_threshold;
    }
    else if(stump_stages.empty() == false)
    {
        const float last_cascade_threshold = stump_stages.back().cascade_threshold;
        return last_cascade_threshold;
    }
    else
    {
        throw std::runtime_error("SoftCascadeOverIntegralChannelsModel::get_last_cascade_threshold " \
                                 "failed to find the stages");
    }
}

float SoftCascadeOverIntegralChannelsModel::get_scale() const
{
    return scale;
}

const SoftCascadeOverIntegralChannelsModel::model_window_size_t &SoftCascadeOverIntegralChannelsModel::get_model_window_size() const
{
    return model_window_size;
}

const SoftCascadeOverIntegralChannelsModel::object_window_t &SoftCascadeOverIntegralChannelsModel::get_object_window() const
{
    return object_window;
}

bool SoftCascadeOverIntegralChannelsModel::has_soft_cascade() const
{
    bool use_the_detector_model_cascade = false;
    const SoftCascadeOverIntegralChannelsModel::fast_stages_t &stages = get_fast_stages();
    if(stages.empty() == false)
    {
        // if the last weak learner has a "non infinity" cascade threshold,
        // then the model has a non trivial cascade, we should use it
        use_the_detector_model_cascade = (stages.back().cascade_threshold > -1E5);
    }

    if(use_the_detector_model_cascade)
    {
        log_info() << "The detector model seems to include a cascade thresholds" << std::endl;
    }
    else
    {
        log_info() << "Will ignore the trivial cascade thresholds of the detector model" << std::endl;
    }

    return use_the_detector_model_cascade;
}


void print_detection_cascade_stages(std::ostream &log, const SoftCascadeOverIntegralChannelsModel::stages_t &stages)
{
    int stage_index = 0;

    log << std::endl;

    BOOST_FOREACH(const SoftCascadeOverIntegralChannelsModel::stage_t &stage, stages)
    {
        log << "Cascade stage " << stage_index << std::endl;

        log << "stage.weight " << stage.weight << std::endl;
        log << "stage.cascade_threshold " << stage.cascade_threshold << std::endl;

        const SoftCascadeOverIntegralChannelsModel::stage_t::weak_classifier_t &c = stage.weak_classifier;

        log << "stage.weak_classifier.level1_node " << std::endl;
        log << "\tlevel1_node.feature_threshold " << c.level1_node.feature_threshold << std::endl;
        log << "\tlevel1_node.larger_than_threshold " << c.level1_node.larger_than_threshold << std::endl;
        log << "\t\tlevel1_node.feature.channel_index " << static_cast<int>(c.level1_node.feature.channel_index) << std::endl;
        log << "\t\tarea(level1_node.feature.box) " << rectangle_area(c.level1_node.feature.box) << std::endl;
        log << "\t\t\tlevel1_node.feature.box min_x " << c.level1_node.feature.box.min_corner().x() << std::endl;
        log << "\t\t\tlevel1_node.feature.box min_y " << c.level1_node.feature.box.min_corner().y() << std::endl;
        log << "\t\t\tlevel1_node.feature.box max_x " << c.level1_node.feature.box.max_corner().x() << std::endl;
        log << "\t\t\tlevel1_node.feature.box max_y " << c.level1_node.feature.box.max_corner().y() << std::endl;


        log << "stage.weak_classifier.level2_true_node " << std::endl;
        log << "\tlevel2_true_node.feature_threshold " << c.level2_true_node.feature_threshold << std::endl;
        log << "\tlevel2_true_node.larger_than_threshold " << c.level2_true_node.larger_than_threshold << std::endl;
        log << "\t\tlevel2_true_node.feature.channel_index " << static_cast<int>(c.level2_true_node.feature.channel_index) << std::endl;
        log << "\t\tarea(level2_true_node.feature.box) " << rectangle_area(c.level2_true_node.feature.box) << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box min_x " << c.level2_true_node.feature.box.min_corner().x() << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box min_y " << c.level2_true_node.feature.box.min_corner().y() << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box max_x " << c.level2_true_node.feature.box.max_corner().x() << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box max_y " << c.level2_true_node.feature.box.max_corner().y() << std::endl;

        log << "stage.weak_classifier.level2_false_node " << std::endl;
        log << "\tlevel2_false_node.feature_threshold " << c.level2_false_node.feature_threshold << std::endl;
        log << "\tlevel2_false_node.larger_than_threshold " << c.level2_false_node.larger_than_threshold << std::endl;
        log << "\t\tlevel2_false_node.feature.channel_index " << static_cast<int>(c.level2_false_node.feature.channel_index) << std::endl;
        log << "\t\tarea(level2_false_node.feature.box) " << rectangle_area(c.level2_false_node.feature.box) << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box min_x " << c.level2_false_node.feature.box.min_corner().x() << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box min_y " << c.level2_false_node.feature.box.min_corner().y() << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box max_x " << c.level2_false_node.feature.box.max_corner().x() << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box max_y " << c.level2_false_node.feature.box.max_corner().y() << std::endl;

        stage_index += 1;
    }

    return;
}

void print_detection_cascade_stages(std::ostream &log, const SoftCascadeOverIntegralChannelsModel::fast_stages_t &stages)
{
    int stage_index = 0;

    log << std::endl;

    BOOST_FOREACH(const SoftCascadeOverIntegralChannelsModel::fast_stage_t &stage, stages)
    {
        log << "Cascade stage " << stage_index << std::endl;

        log << "stage.cascade_threshold " << stage.cascade_threshold << std::endl;

        const SoftCascadeOverIntegralChannelsModel::fast_stage_t::weak_classifier_t &c = stage.weak_classifier;

        log << "stage.weak_classifier.level1_node " << std::endl;
        log << "\tlevel1_node.feature_threshold " << c.level1_node.feature_threshold << std::endl;
        log << "\t\tlevel1_node.feature.channel_index " << static_cast<int>(c.level1_node.feature.channel_index) << std::endl;
        log << "\t\tarea(level1_node.feature.box) " << rectangle_area(c.level1_node.feature.box) << std::endl;
        log << "\t\t\tlevel1_node.feature.box min_x " << c.level1_node.feature.box.min_corner().x() << std::endl;
        log << "\t\t\tlevel1_node.feature.box min_y " << c.level1_node.feature.box.min_corner().y() << std::endl;
        log << "\t\t\tlevel1_node.feature.box max_x " << c.level1_node.feature.box.max_corner().x() << std::endl;
        log << "\t\t\tlevel1_node.feature.box max_y " << c.level1_node.feature.box.max_corner().y() << std::endl;


        log << "stage.weak_classifier.level2_true_node " << std::endl;
        log << "\tlevel2_true_node.feature_threshold " << c.level2_true_node.feature_threshold << std::endl;
        log << "\tlevel2_true_node.weight_true_leaf " << c.level2_true_node.weight_true_leaf << std::endl;
        log << "\tlevel2_true_node.weight_false_leaf " << c.level2_true_node.weight_false_leaf << std::endl;
        log << "\t\tlevel2_true_node.feature.channel_index " << static_cast<int>(c.level2_true_node.feature.channel_index) << std::endl;
        log << "\t\tarea(level2_true_node.feature.box) " << rectangle_area(c.level2_true_node.feature.box) << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box min_x " << c.level2_true_node.feature.box.min_corner().x() << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box min_y " << c.level2_true_node.feature.box.min_corner().y() << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box max_x " << c.level2_true_node.feature.box.max_corner().x() << std::endl;
        log << "\t\t\tlevel2_true_node.feature.box max_y " << c.level2_true_node.feature.box.max_corner().y() << std::endl;

        log << "stage.weak_classifier.level2_false_node " << std::endl;
        log << "\tlevel2_false_node.feature_threshold " << c.level2_false_node.feature_threshold << std::endl;
        log << "\tlevel2_false_node.weight_true_leaf " << c.level2_false_node.weight_true_leaf << std::endl;
        log << "\tlevel2_false_node.weight_false_leaf " << c.level2_false_node.weight_false_leaf << std::endl;
        log << "\t\tlevel2_false_node.feature.channel_index " << static_cast<int>(c.level2_false_node.feature.channel_index) << std::endl;
        log << "\t\tarea(level2_false_node.feature.box) " << rectangle_area(c.level2_false_node.feature.box) << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box min_x " << c.level2_false_node.feature.box.min_corner().x() << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box min_y " << c.level2_false_node.feature.box.min_corner().y() << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box max_x " << c.level2_false_node.feature.box.max_corner().x() << std::endl;
        log << "\t\t\tlevel2_false_node.feature.box max_y " << c.level2_false_node.feature.box.max_corner().y() << std::endl;

        stage_index += 1;
    }

    return;
}


} // end of namespace doppia
