#ifndef BOOSTED_LEARNING_TRAININGDATA_HPP
#define BOOSTED_LEARNING_TRAININGDATA_HPP

#include "Feature.hpp"

#include <boost/shared_ptr.hpp>

namespace boosted_learning {


/// This class stores the strict minimum information required to be able to do Adaboost training.
/// In particular, we do _not_ store the integral images, but only the features responses
class TrainingData
{
public:    

    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    typedef ImageData meta_datum_t;
    typedef std::vector<meta_datum_t> meta_data_t;

    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef bootstrapping::integral_channels_view_t integral_channels_view_t;
    typedef bootstrapping::integral_channels_const_view_t integral_channels_const_view_t;
    typedef std::vector<integral_channels_t> IntegralImages;

    typedef boost::shared_ptr<TrainingData> shared_ptr;
    typedef boost::shared_ptr<const TrainingData> ConstSharePointer;

    typedef bootstrapping::integral_channels_computer_t integral_channels_computer_t;

public:
    TrainingData(ConstFeaturesSharedPointer featuresConfigurations,
                 const std::vector<bool> &valid_features,
                 const size_t maxNumExamples,
                 const point_t modelWindow, const rectangle_t objectWindow);
    ~TrainingData();

    /// how many features in the pool ?
    size_t getFeaturesPoolSize() const;

    /// maximum number of examples that can be added ?
    size_t getMaxNumExamples() const;

    /// how many examples are currently here ?
    size_t getNumExamples() const;

    size_t getNumPositiveExamples() const;
    size_t getNumNegativeExamples() const;

    const rectangle_t & getObjectWindow() const;
    const point_t & getModelWindow() const;

    /// meta-data access method,
    /// @param index is the training example index
    int getClassLabel(const size_t Index) const;

    /// meta-data access method,
    /// @param index is the training example index
    const string &getFilename(const size_t Index) const;

    const ConstFeaturesSharedPointer getFeaturesConfigurations() const;


    /// the first index enumerates the features,
    /// the second index enumerates the training examples
    const FeaturesResponses &getFeatureResponses() const;

    const Feature &getFeature(const size_t featureIndex) const;
    bool getFeatureValidity(const size_t featureIndex) const;

    /// append new data to the training data
    void appendData(const LabeledData &labeledData);

    void setDatum(const size_t datumIndex,
                  const meta_datum_t &metaDatum, const LabeledData::integral_channels_t &integralImage);


    void addPositiveSamples(const std::vector<std::string> &filenamesPositives,
                            const point_t &modelWindowSize, const point_t &dataOffset);

    void addNegativeSamples(const std::vector<std::string> &filenamesBackground,
                            const point_t &modelWindowSize, const point_t &dataOffset,
                            const size_t numNegativeSamplesToAdd);

    void addBootstrappingSamples(const std::string classifierPath,
                                 const std::vector<std::string> &filenamesBackground,
                                 const point_t &modelWindowSize, const point_t &dataOffset,
                                 const size_t numNegativeSamplesToAdd, const int maxFalsePositivesPerImage);

protected:

    const int _backgroundClassLabel; ///< Label of the class for background images

    ConstFeaturesSharedPointer _featuresConfigurations;
public:
    std::vector<bool> _validFeatures;
protected:
    point_t _modelWindow;
    rectangle_t _objectWindow;

    FeaturesResponses _featureResponses;

    meta_data_t _metaData; ///< labels of the classes
    size_t _numPositivesExamples, _numNegativesExamples;

    integral_channels_computer_t _integralChannelsComputer;

};


} // namespace boosted_learning

#endif // BOOSTED_LEARNING_TRAININGDATA_HPP
