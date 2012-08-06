
#include "Feature.hpp"
#include "Parameters.hpp"

#include <boost/random.hpp>

#include <stdexcept>
#include <iostream>
#include <ctime>

namespace boosted_learning {


/// The constructor that set everything to 0.
Feature::Feature() : x(0), y(0), width(0), height(0), channel(0)
{
    // nothing to do here
    return;
}


/// The constructor that ask for the sizes.
Feature::Feature(int x_, int y_, int width_, int height_, int channel_)
    : x(x_), y(y_), width(width_), height(height_), channel(channel_)
{
    // nothing to do here
    return;
}

bool Feature::operator==(const Feature &other)
{
    return (x == other.x) and (y == other.y) and
            (width == other.width) and (height == other.height) and (channel == other.channel);
}


void Feature::printConfigurationString(std::ostream &os) const
{
    os << x << "\t" << y << "\t" << width << "\t" << height << "\t" << channel << "\t";
}

void Feature::readConfigurationString(std::istream &is)
{
    if (!(is >> x >> y >> width >> height >> channel))
    {
        throw std::runtime_error("Reading configuration failed");
    }
}


void computeRandomFeaturesConfigurations(const LabeledData::point_t &modelWindow, const int num_of_features,  Features &featuresConfigurations)
{

    //if (_verbose > 2)
    if(false)
    {
        std::cout << "Computing the features configurations" << std::endl;
    }


    //const double maxFeatureSizeRatio = Parameters::getParameter<double>("train.maxFeatureSizeRatio");

    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    const int
            minWidth = std::max(1, Parameters::getParameter<int>("train.minFeatWidth") / shrinking_factor),
            minHeight = std::max(1, Parameters::getParameter<int>("train.minFeatHeight") / shrinking_factor);
    //maxWidth = static_cast<int>(maxFeatureSizeRatio * modelWidth),
    //maxHeight = static_cast<int>(maxFeatureSizeRatio * modelHeight);

    int
            maxWidth = Parameters::getParameter<int>("train.maxFeatWidth"),
            maxHeight = Parameters::getParameter<int>("train.maxFeatHeight");

    if(maxWidth < 0)
    {
        maxWidth = modelWidth;
    }
    else
    {
        maxWidth = std::max(1, maxWidth/shrinking_factor);
    }

    if(maxHeight < 0)
    {
        maxHeight = modelHeight;
    }
    else
    {
        maxHeight = std::max(1, maxHeight/shrinking_factor);
    }

    if((minWidth >= maxWidth) or (minHeight >= maxHeight))
    {
        throw std::invalid_argument("min width/height should be smaller than max width/height (check your configuration file)");
    }

    const int numChannels = 10; // FIXME hardcoded value
    static int call_counter=0;
    boost::uint32_t random_seed = std::time(NULL);
    const int input_seed = Parameters::getParameter<boost::uint32_t>("train.featuresPoolRandomSeed");
    if(input_seed > 0)
    {
        random_seed = input_seed+call_counter;
        call_counter+=1;
        printf("computeRandomFeaturesConfigurations is using user provided seed == %i\n", random_seed);
    }
    else
    {
        printf("computeRandomFeaturesConfigurations is using random_seed == %i\n", random_seed);
    }

    boost::mt19937 random_generator(random_seed);

    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    boost::uniform_int<>
            x_distribution(0, (modelWidth - 1) - minWidth),
            y_distribution(0, (modelHeight -1) - minHeight),
            channel_distribution(0, numChannels - 1),
            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) or
            (y_distribution.max() <= 0) or
            (width_distribution.max() <= 0) or
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", modelWidth, modelHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),
            channel_generator(random_generator, channel_distribution),
            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    size_t total_num_of_features = featuresConfigurations.size() + num_of_features;
    featuresConfigurations.reserve(total_num_of_features);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?

    while(featuresConfigurations.size() < total_num_of_features)
    {
        const int
                x = x_generator(),
                y = y_generator(),
                c = channel_generator(),
                w = width_generator(),
                h = height_generator();
        //std::cout << x << " ";
        if(((x + w) < modelWidth) and ((y + h) < modelHeight))
        {
            Feature featureConfiguration(x, y, w, h, c);

            // we check if the feature already exists in the set or not
            const bool featureAlreadyInSet =
                    std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                              featureConfiguration) != featuresConfigurations.end();

            if(featureAlreadyInSet)
            {
                rejectionsInARow += 1;
                repetitionsCounter += 1;
                if(rejectionsInARow > maxRejectionsInARow)
                {
                    printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                           featuresConfigurations.size(), maxRejectionsInARow);
                    throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                }
                continue;
            }
            else
            {
                rejectionsInARow = 0;
                featuresConfigurations.push_back(featureConfiguration);
            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    if(true)
    {
        printf("When sampling %zi features, randomly found (and rejected) %i repetitions\n",
               num_of_features, repetitionsCounter);
    }

    return;
}


} // end of namespace boosted_learning
