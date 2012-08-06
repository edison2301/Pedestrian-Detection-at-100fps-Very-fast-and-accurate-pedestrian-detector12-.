#ifndef _StrongClassifier_H
#define _StrongClassifier_H
#include "WeakDiscreteTreeLearner.hpp"
#include "LabeledData.hpp"
#include "TrainingData.hpp"

#include "bootstrapping_lib.hpp"

namespace boosted_learning {

class StrongClassifier
{
public:

    typedef bootstrapping::integral_channels_const_view_t integral_channels_const_view_t; 
    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;
    StrongClassifier(std::vector<WeakDiscreteTree> learners): _learners(learners)
    {
        if (_learners[_learners.size()-1]._cascadeThreshold == -std::numeric_limits<float>::max())
        {
            _learners[_learners.size()-1]._cascadeThreshold = 0;
        }

        if (_learners.back()._cascadeThreshold <= -std::numeric_limits<int>::max())
            //throw runtime_error("there is a bug in the cascade: last stage threshold too negative");
        {
            _learners.back()._cascadeThreshold = 0;
        }
    }
    StrongClassifier(const std::vector<WeakDiscreteTreeLearner> & learners);

    void convertToSoftCascade(const LabeledData::shared_ptr data, const double detection_rate);
    void writeClassifier(const std::string filename,
                         const std::string trainedDatasetName,
                         const point_t modelWindow, const rectangle_t objectWindow);
    int classify(const integral_channels_const_view_t &integralImage, const bool use_cascade=true) const;
    void classify(const LabeledData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade=true) const;

    void classify(const TrainingData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade=true) const;

    std::vector<WeakDiscreteTree> _learners;
    void classify_real(const LabeledData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade=true) const;
protected:

};

} // end of namespace boosted_learning

#endif// StrongClassifier_H
