#ifndef __FEAT_CONFIG_H
#define __FEAT_CONFIG_H

#include "IntegralChannelsComputer.hpp"

#include "LabeledData.hpp"

#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <iosfwd>

namespace boosted_learning {

/// Simple structure of a feature
class Feature
{
public:
    /// The constructor that set everything to 0.
    Feature();

    /// The constructor that ask for the sizes.
    Feature(int x, int y, int width, int height, int channel);

    int x; ///< x position.
    int y; ///< y position.
    int width; ///< width or phi1
    int height; ///< height or phi2
    int channel;

    bool operator==(const Feature &other);

    void printConfigurationString(std::ostream &os) const;
    void readConfigurationString(std::istream &is);

    int getResponse(const bootstrapping::integral_channels_t &integralImage) const;
};

typedef std::vector<Feature> Features;
typedef boost::shared_ptr<Features> FeaturesSharedPointer;
typedef boost::shared_ptr<const Features> ConstFeaturesSharedPointer;

//typedef std::vector<int> FeaturesResponses;
typedef boost::multi_array<int, 2> FeaturesResponses;
typedef boost::shared_ptr<FeaturesResponses> FeaturesResponsesSharedPointer;
typedef boost::shared_ptr<const FeaturesResponses> ConstFeaturesResponsesSharedPointer;

typedef std::vector<int> MinOrMaxFeaturesResponses;
typedef boost::shared_ptr<MinOrMaxFeaturesResponses> MinOrMaxFeaturesResponsesSharedPointer;
typedef boost::shared_ptr<const MinOrMaxFeaturesResponses> ConstMinOrMaxFeaturesResponsesSharedPointer;


/// inline for speed reasons
inline int Feature::getResponse(const bootstrapping::integral_channels_t &integralImage) const
{
    const int
            a = integralImage[channel][y][x],
            b = integralImage[channel][y+0][x+width],
            c = integralImage[channel][y+height][x+width],
            d = integralImage[channel][y+height][x+0];
    return a + c - b - d;
}


/// this function gets most of its parameters from the Parameters::getParameter singleton
void computeRandomFeaturesConfigurations(const LabeledData::point_t &modelWindow, const int num_of_features, Features &featuresConfigurations);


} // end of namespace boosted_learning

#endif // __FEAT_CONFIG_H
