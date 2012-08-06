#include "TrainingData.hpp"

#include "Parameters.hpp"
#include "bootstrapping_lib.hpp"

#include "video_input/ImagesFromDirectory.hpp" // for the open_image helper method
#include "integral_channels_helpers.hpp"

#include <boost/format.hpp>
#include <boost/progress.hpp>

#include <cstdio>

namespace boosted_learning {

using namespace boost;

TrainingData::TrainingData(ConstFeaturesSharedPointer featuresConfigurations,
                           const std::vector<bool> &valid_features,
                           const size_t maxNumExamples, const point_t modelWindow, const rectangle_t objectWindow)
    :
      _backgroundClassLabel(-1), // FIXME hardcoded value
      _featuresConfigurations(featuresConfigurations),
      _validFeatures(valid_features),
      _modelWindow(modelWindow),
      _objectWindow(objectWindow),
      _numPositivesExamples(0),
      _numNegativesExamples(0)

{
    // we allocated the full data memory at the begining
    _featureResponses.resize(boost::extents[_featuresConfigurations->size()][maxNumExamples]);
    _metaData.resize(maxNumExamples);
    printf("Allocated features responses for %zi features and a maximum of %zi samples\n",
           _featuresConfigurations->size(), maxNumExamples);

    if(boost::is_same<integral_channels_computer_t, doppia::IntegralChannelsForPedestrians>::value)
    {
        std::cout << "Using doppia::IntegralChannelsForPedestrians" << std::endl;
    }
    else if(boost::is_same<integral_channels_computer_t, doppia::GpuIntegralChannelsForPedestrians>::value)
    {
        std::cout << "Using doppia::GpuIntegralChannelsForPedestrians" << std::endl;
    }
    else
    {
        std::cout << "Using an unknown integral_channels_computer_t" << std::endl;
    }

    return;
}

TrainingData::~TrainingData()
{
    // nothing to do here
    return;
}


size_t TrainingData::getFeaturesPoolSize() const
{
   // size_t ret = 0;
   // for (size_t k =0; k< _validFeatures.size(); ++k)
   // {
   //     if (_validFeatures[k] == true)
   //         ret++;
   // }
   // return ret;

    return  _featureResponses.shape()[0];
}

size_t TrainingData::getMaxNumExamples() const
{
    return _featureResponses.shape()[1];
}

size_t TrainingData::getNumExamples() const
{
    return _numPositivesExamples + _numNegativesExamples;
}

size_t TrainingData::getNumPositiveExamples()const
{
    return _numPositivesExamples;
}

size_t TrainingData::getNumNegativeExamples() const
{
    return _numNegativesExamples;
}

const TrainingData::rectangle_t & TrainingData::getObjectWindow() const
{
    return _objectWindow;
}

const TrainingData::point_t & TrainingData::getModelWindow() const
{
    return _modelWindow;
}

int TrainingData::getClassLabel(const size_t Index) const
{
    return _metaData[Index].imageClass;
}

const string &TrainingData::getFilename(const size_t Index) const
{
    return _metaData[Index].filename;
}

const ConstFeaturesSharedPointer TrainingData::getFeaturesConfigurations() const
{
    return _featuresConfigurations;
}

const FeaturesResponses &TrainingData::getFeatureResponses() const
{
    return _featureResponses;
}

const Feature &TrainingData::getFeature(const size_t featureIndex) const
{
    return (*_featuresConfigurations)[featureIndex];
}

bool TrainingData::getFeatureValidity(const size_t featureIndex) const
{
        return _validFeatures[featureIndex];
}


void TrainingData::setDatum(
        const size_t datumIndex,
        const meta_datum_t &metaDatum,
        const LabeledData::integral_channels_t &integralImage)
{
    assert(datumIndex < getMaxNumExamples());

    for (size_t featuresIndex = 0; featuresIndex < _featuresConfigurations->size(); featuresIndex+=1)
    {
        const int featureResponse = (*_featuresConfigurations)[featuresIndex].getResponse(integralImage);
        _featureResponses[featuresIndex][datumIndex] = featureResponse;
    }

    _metaData[datumIndex] = metaDatum;

    if(metaDatum.imageClass == _backgroundClassLabel)
    {
        _numNegativesExamples += 1;
    }
    else
    {
        _numPositivesExamples += 1;
    }

    const bool save_integral_images = false;
    if(save_integral_images)
    {        
        static int false_positive_counter = 0;
        const int max_images_to_save = 100;
        //const size_t startDatumIndex = 5000 + false_positives;
        const size_t startDatumIndex = 637 - 1;
        if((datumIndex > startDatumIndex) and (false_positive_counter < max_images_to_save))
        {

            boost::format filename_pattern("false_positive_%i.png");
            const boost::filesystem::path
                    storage_path = "/tmp/",
                    file_path = storage_path / boost::str( filename_pattern % false_positive_counter);
            doppia::save_integral_channels_to_file(integralImage, file_path.string());

            false_positive_counter += 1;

            printf("Saved %i false positives integral channels images inside %s\n",
                   false_positive_counter, storage_path.string().c_str());
            //throw std::runtime_error("Stopping everything so you can look at the false positive integral channels images");
        }
    } // end of "if shoudl save integral images"

    return;
}


/// append new data to the training data
void TrainingData::appendData(const LabeledData &labeledData)
{

    assert((labeledData.getNumPosExamples() + labeledData.getNumNegExamples()) == labeledData.getNumExamples());

    const size_t
            initialNumberOfTrainingSamples = getNumExamples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + labeledData.getNumExamples();
    if(finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        throw std::runtime_error("TrainingData::appendData is trying to add more data than initially specified");
    }

    // compute and save the feature responses
#pragma omp parallel for default(none) ordered shared(labeledData)
    for (size_t labelDataIndex = 0; labelDataIndex < labeledData.getNumExamples(); ++labelDataIndex)
    {
        const meta_datum_t &metaDatum = labeledData.getMetaDatum(labelDataIndex);
        const LabeledData::integral_channels_t &integralImage = labeledData.getIntegralImage(labelDataIndex);
        setDatum(initialNumberOfTrainingSamples + labelDataIndex,
                 metaDatum, integralImage);
    } // end of "for each labeled datum"

    return;
}


void TrainingData::addPositiveSamples(const std::vector<std::string> &filenamesPositives,
                                      const point_t &modelWindowSize, const point_t &dataOffset)
{
    const size_t
            initialNumberOfTrainingSamples = getNumExamples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + filenamesPositives.size();
    if(finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        throw std::runtime_error("TrainingData::addPositiveSamples is trying to add more data than initially specified");
    }


    printf("\nCollecting %zi positive samples\n", filenamesPositives.size());
    boost::progress_display progress_indicator(filenamesPositives.size());


    meta_datum_t  metaDatum;
    integral_channels_t sampleIntegralChannels;

    // integralChannelsComputer is already multithreaded, so no benefit on paralelizing this for loop
    for (size_t filenameIndex = 0; filenameIndex < filenamesPositives.size(); filenameIndex +=1)
    {
        gil::rgb8_image_t image;
        gil::rgb8c_view_t image_view = doppia::open_image(filenamesPositives[filenameIndex].c_str(), image);

        _integralChannelsComputer.set_image(image_view);
        _integralChannelsComputer.compute();

        get_integral_channels(_integralChannelsComputer.get_integral_channels(),
                              modelWindowSize, dataOffset, _integralChannelsComputer.get_shrinking_factor(),
                              sampleIntegralChannels);

        metaDatum.filename = filenamesPositives[filenameIndex];
        metaDatum.imageClass = 1;//classes[k];
        metaDatum.x = dataOffset.x();
        metaDatum.y = dataOffset.y();

        setDatum(initialNumberOfTrainingSamples + filenameIndex,
                 metaDatum, sampleIntegralChannels);

        ++progress_indicator;
    } // end of "for each filename"

    return;
}

void TrainingData::addNegativeSamples(const std::vector<std::string> &filenamesBackground,
                                      const point_t &modelWindowSize, const point_t &dataOffset,
                                      const size_t numNegativeSamplesToAdd)
{

    const size_t
            initialNumberOfTrainingSamples = getNumExamples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + numNegativeSamplesToAdd;
    if(finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        throw std::runtime_error("TrainingData::addNegativeSamples is trying to add more data than initially specified");
    }

    printf("\nCollecting %zi random negative samples\n", numNegativeSamplesToAdd);
    boost::progress_display progress_indicator(numNegativeSamplesToAdd);

    meta_datum_t  metaDatum;
    integral_channels_t sampleIntegralChannels;

#if defined(DEBUG)
    srand(1);
#else
    srand(time(NULL));
#endif
    srand(1);

    const int samplesPerImage = std::max<int>(1, numNegativeSamplesToAdd / filenamesBackground.size());

    // FIXME no idea what the +1 does
    const int
            minWidth = (modelWindowSize.x()+1 + 2*dataOffset.x()),
            minHeight = (modelWindowSize.y()+1 + 2*dataOffset.y());

    const float maxSkippedFraction = 0.25;

    size_t numNegativesSamplesAdded = 0, numSkippedImages = 0, filenameIndex = 0;

    // integralChannelsComputer is already multithreaded, so no benefit on paralelizing this for loop
    while (numNegativesSamplesAdded < numNegativeSamplesToAdd)
    {
        if (filenameIndex >= filenamesBackground.size())
        {
            // force to loop until we have reached the desired number of samples
            filenameIndex = 0;
        }

        const string &filename = filenamesBackground[filenameIndex];
        filenameIndex +=1;

        gil::rgb8c_view_t imageView;
        gil::rgb8_image_t image;
        imageView = doppia::open_image(filename.c_str(), image);

        if ((imageView.width() < minWidth) or (imageView.height() < minHeight))
        {
            // if input image is too small, we skip it
            //printf("Skipping negative sample %s, because it is too small\n", filename.c_str());
            numSkippedImages += 1;

            const float skippedFraction = static_cast<float>(numSkippedImages) / filenamesBackground.size();
            if (skippedFraction > maxSkippedFraction)
            {
                printf("Skipped %i images (out of %zi, %.3f%%) because they where too small\n",
                       numSkippedImages, filenamesBackground.size(), skippedFraction*100);

                throw std::runtime_error("Too many negatives images where skipped. Dataset needs to be fixed");
            }
            continue;
        }

        const int
                maxRandomX = (imageView.width() - modelWindowSize.x()+1 - 2*dataOffset.x()),
                maxRandomY = (imageView.height() - modelWindowSize.y()+1 - 2*dataOffset.y());

        _integralChannelsComputer.set_image(imageView);
        _integralChannelsComputer.compute();

        metaDatum.filename = filename;
        metaDatum.imageClass = _backgroundClassLabel;

        size_t numSamplesForImage = std::min<size_t>(samplesPerImage,
                                                           (numNegativeSamplesToAdd - numNegativesSamplesAdded));
                numSamplesForImage = 1;
        for (size_t randomSampleIndex = 0; randomSampleIndex < numSamplesForImage; randomSampleIndex += 1)
        {
            //const point_t::coordinate_t
            size_t
			x = dataOffset.x() + rand() % maxRandomX, 
				  y = dataOffset.y() + rand() % maxRandomY;
            //printf("random x,y == %i, %i\n", x,y);
                        const point_t randomOffset(x,y);
            metaDatum.x = randomOffset.x(); metaDatum.y = randomOffset.y();
            get_integral_channels(_integralChannelsComputer.get_integral_channels(),
                                  modelWindowSize, randomOffset, _integralChannelsComputer.get_shrinking_factor(),
                                  sampleIntegralChannels);

            setDatum(initialNumberOfTrainingSamples + numNegativesSamplesAdded,
                     metaDatum, sampleIntegralChannels);

            numNegativesSamplesAdded += 1;
            ++progress_indicator;
        }

    } // end of "for each background image"



    if (numSkippedImages > 0)
    {
        const float skippedFraction = static_cast<float>(numSkippedImages) / filenamesBackground.size();
        printf("Skipped %zi images (out of %zi, %.3f%%) because they where too small\n",
               numSkippedImages, filenamesBackground.size(), skippedFraction*100);
    }


    return;
}

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

class AppendDatumFunctor
{
public:
    AppendDatumFunctor(TrainingData &trainingData);
    ~AppendDatumFunctor();

    void operator()(const TrainingData::meta_datum_t &metaDatum,
                    const TrainingData::integral_channels_t &integralImage);

protected:

    TrainingData &trainingData;
    size_t datumIndex;
};


AppendDatumFunctor::AppendDatumFunctor(TrainingData &trainingData_)
    : trainingData(trainingData_),
      datumIndex(trainingData_.getNumExamples())
{
    // nothing to do here
    return;
}

AppendDatumFunctor::~AppendDatumFunctor()
{
    // nothing to do here
    return;
}

void AppendDatumFunctor::operator()(const TrainingData::meta_datum_t &metaDatum,
                                    const TrainingData::integral_channels_t &integralImage)
{
    trainingData.setDatum(datumIndex, metaDatum, integralImage);
    datumIndex += 1;
    return;
}


// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

void TrainingData::addBootstrappingSamples(
        const std::string classifierPath,
        const std::vector<std::string> &filenamesBackground,
        const point_t &modelWindowSize, const point_t &dataOffset,
        const size_t numNegativeSamplesToAdd, const int maxFalsePositivesPerImage)
{

    const size_t
            initialNumberOfTrainingSamples = getNumExamples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + numNegativeSamplesToAdd;
    if (finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        printf("initialNumberOfTrainingSamples == %zi\n", initialNumberOfTrainingSamples);
        printf("numNegativeSamplesToAdd == %zi\n", numNegativeSamplesToAdd);
        printf("finalNumberOfTrainingSamples %zi > getMaxNumExamples() %zi \n",
               finalNumberOfTrainingSamples, getMaxNumExamples());
        throw std::runtime_error("TrainingData::addBootstrappingSamples is trying to add more data than initially specified");
    }

    printf("Searching for hard %zi negatives given the current model (%s), please wait...\n",
           numNegativeSamplesToAdd, classifierPath.c_str());
    //boost::progress_display progress_indicator(numNegativeSamplesToAdd);

    const int
            numScales = Parameters::getParameter<int>("bootstrapTrain.num_scales"),
            numRatios = Parameters::getParameter<int>("bootstrapTrain.num_ratios");

    const float
            minScale = Parameters::getParameter<float>("bootstrapTrain.min_scale"),
            maxScale = Parameters::getParameter<float>("bootstrapTrain.max_scale"),
            minRatio = Parameters::getParameter<float>("bootstrapTrain.min_ratio"),
            maxRatio = Parameters::getParameter<float>("bootstrapTrain.max_ratio");

    const bool use_less_memory = Parameters::getParameter<bool>("bootstrapTrain.frugalMemoryUsage");


    const size_t initialIntegralImagesSize = getNumExamples();
    bootstrapping::append_result_functor_t the_functor = AppendDatumFunctor(*this);
    bootstrapping::bootstrap(boost::filesystem::path(classifierPath), filenamesBackground,
                             numNegativeSamplesToAdd, maxFalsePositivesPerImage,
                             minScale, maxScale, numScales,
                             minRatio, maxRatio, numRatios,
                             use_less_memory,
                             the_functor);

    const size_t numFoundFalsePositives = getNumExamples() - initialIntegralImagesSize;
    if (numFoundFalsePositives < numNegativeSamplesToAdd)
    {
        const size_t numRandomNegativesToAdd = numNegativeSamplesToAdd - numFoundFalsePositives;
        addNegativeSamples(filenamesBackground, modelWindowSize, dataOffset, numRandomNegativesToAdd);
    }

    return;
}




} // namespace boosted_learning
