

#include "Parameters.hpp"
#include "LabeledData.hpp"
#include "AdaboostLearner.hpp"
#include "ModelIO.hpp"

#include "helpers/Log.hpp"
#include "helpers/geometry.hpp"

#include <boost/format.hpp>
#include <boost/array.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <boost/timer.hpp>

#include <string>
#include <stdexcept>

#include <sstream>
#include <ctime>

#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace boosted_learning {


int TreeNode::idCount = 0;


TrainingData::point_t getModelWindowSize()
{
    const std::string modelWindowString= Parameters::getParameter<std::string>("train.modelWindow");

    //get model window
    std::vector<std::string> modelWindowStrings;
    boost::split(modelWindowStrings, modelWindowString, boost::is_any_of(","));
    assert(modelWindowStrings.size() == 2);
    TrainingData::point_t modelWindow(boost::lexical_cast<int>(modelWindowStrings[0]),
                                      boost::lexical_cast<int>(modelWindowStrings[1]));

    return modelWindow;
}


TrainingData::point_t getTrainingDataOffset()
{
    const int
            offsetX = Parameters::getParameter<int>("train.offsetX"),
            offsetY = Parameters::getParameter<int>("train.offsetY");

    TrainingData::point_t testOffset(offsetX, offsetY);
    return testOffset;
}


TrainingData::point_t getTestingDataOffset()
{
    const int
            offsetX = Parameters::getParameter<int>("test.offsetX"),
            offsetY = Parameters::getParameter<int>("test.offsetY");

    TrainingData::point_t testOffset(offsetX, offsetY);
    return testOffset;
}


TrainingData::rectangle_t getObjectWindow()
{
    //read files for training
    const std::string objectWindowString = Parameters::getParameter<std::string>("train.objectWindow");

    //get object window
    std::vector<std::string> objectWindowStrings;
    boost::split(objectWindowStrings, objectWindowString, boost::is_any_of(","));
    assert(objectWindowStrings.size() == 4);
    doppia::geometry::point_xy<int> mincorner(boost::lexical_cast<int>(objectWindowStrings[0]),
                                              boost::lexical_cast<int>(objectWindowStrings[1]));
    doppia::geometry::point_xy<int> maxcorner(mincorner.x() + boost::lexical_cast<int>(objectWindowStrings[2]),
                                              mincorner.y() + boost::lexical_cast<int>(objectWindowStrings[3]));
    TrainingData::rectangle_t  objectWindow(mincorner, maxcorner);

    return objectWindow;
}




void retrieveFilesNames(const boost::filesystem::path directoryPath, std::vector<std::string> &filesNames)
{
    using namespace boost::filesystem;

    directory_iterator
            directoryIterator = directory_iterator(directoryPath),
            directoryEnd;
    while(directoryIterator != directoryEnd)
    {
        const string
        #if BOOST_VERSION <= 104400
                fileName = directoryIterator->path().filename();
#else
                fileName = directoryIterator->path().filename().string();
#endif

        // should we check the filePath extension ?
        const path filePath = directoryPath / fileName;
        filesNames.push_back(filePath.string());

        ++directoryIterator;
    } // end of "for each file in the directory"

    return;
}



void getImageFileNames(
        const string nameFile,
        const int backgroundClassLabel,
        std::vector<std::string> &positiveExamplesPaths,
        std::vector<std::string> &negativeExamplesPaths)
{
    using boost::filesystem::path;
    using boost::filesystem::is_directory;

    path inputPath(nameFile);
    if(is_directory(inputPath))
    {
        // handling the INRIAPerson case http://pascal.inrialpes.fr/data/human
        const path
                inriaPersonPositivesPath = inputPath / "pos",
                inriaPersonNegativesPath = inputPath / "neg";

        const bool isInriaPedestriansDirectory = is_directory(inriaPersonPositivesPath) and is_directory(inriaPersonNegativesPath);


        const string
                positivePathPrefix = "positives_octave_",
        #if BOOST_VERSION <= 104400
                inputPathFilename = inputPath.filename();
#else
                inputPathFilename = inputPath.filename().string();
#endif
        float octave_number = 0;
        if(boost::algorithm::starts_with(inputPathFilename, positivePathPrefix))
        {
            // inputPathFilename should be something like "positives_octave_-2.0"
            const string number_string =
                    inputPathFilename.substr(positivePathPrefix.size(),
                                             inputPathFilename.size() - positivePathPrefix.size());
            octave_number = boost::lexical_cast<float>(number_string);
        }

        const path
                multiScalesPositivesPath = inputPath,
                multiScalesNegativesPath = \
                inputPath.parent_path() / boost::str(boost::format("negatives_octave_%.1f") % octave_number);

        const bool isMultiScalesDirectory =
                is_directory(multiScalesPositivesPath) and
                boost::algorithm::starts_with(inputPathFilename, positivePathPrefix) and
                is_directory(multiScalesNegativesPath);

        if(true and (not isInriaPedestriansDirectory) and (not isMultiScalesDirectory))
        { // just for debugging
            printf("is_directory(multiScalesPositivesPath) == %i\n", is_directory(multiScalesPositivesPath));
            printf("is_directory(%s) == %i\n",
                   multiScalesNegativesPath.string().c_str(), is_directory(multiScalesNegativesPath));
#if BOOST_VERSION <= 104400
            printf("starts_with(%s, %s) == %i\n",
                   std::string(multiScalesPositivesPath.filename()).c_str(), positivePathPrefix.c_str(),
                   boost::algorithm::starts_with(multiScalesPositivesPath.filename(), positivePathPrefix));
#else
            printf("starts_with(%s, %s) == %i\n",
                   multiScalesPositivesPath.filename().string().c_str(), positivePathPrefix.c_str(),
                   boost::algorithm::starts_with(multiScalesPositivesPath.filename().string(), positivePathPrefix));
#endif
        }

        if(isInriaPedestriansDirectory)
        {
            retrieveFilesNames(inriaPersonPositivesPath, positiveExamplesPaths);
            retrieveFilesNames(inriaPersonNegativesPath, negativeExamplesPaths);
        }
        else if(isMultiScalesDirectory)
        {
            retrieveFilesNames(multiScalesPositivesPath, positiveExamplesPaths);
            retrieveFilesNames(multiScalesNegativesPath, negativeExamplesPaths);
        }
        else
        {
            printf("LabeledData::getImageFileNames is trying to read %s\n", nameFile.c_str());
            throw std::invalid_argument("LabeledData::getImageFileNames received a directory, but "
                                        "could not recognize it as an INRIAPerson Train/Test directory "
                                        "nor as multiscales_inria_person/data_set/positives_octave_* directory");
        }

    }
    else
    { // input path is a list file

        ifstream inFile(nameFile.c_str());

        if (!inFile.is_open())
        {
            printf("LabeledData::getImageFileNames is trying to read %s\n", nameFile.c_str());
            throw std::invalid_argument("LabeledData::getImageFileNames could not open the indicated file");
        }


        // just read the file
        while (!inFile.eof())
        {
            //get next file

            int  tmpClassNum;
            inFile >> tmpClassNum; // store class

            // additional check to avoid problems in the case of an empty line at the end
            // of the file
            if (inFile.eof())
            {
                break;
            }

            string filename;
            inFile >> filename;

            //if (_verbose > 2)
            if(false)
            {
                std::cout << filename << std::endl;
            }

            if (tmpClassNum != backgroundClassLabel)
            {
                positiveExamplesPaths.push_back(filename);
            }
            else
            {
                negativeExamplesPaths.push_back(filename);
            }

        } // end of "while reading file"

    } // end of "if nameFile is a directory or not"

    return;
} // end of getImageFileNames



int printModel()
{
    std::string outputModelFileName = Parameters::getParameter<string>("train.outputModelFileName");
    ModelIO modelReader(10);
    modelReader.readModel(outputModelFileName);
    StrongClassifier learner = modelReader.read();

    modelReader.print();
    return 0;

}


void test(const int verbose)
{

    //read files for training
    const std::string
            testSetPaths = Parameters::getParameter<std::string>("test.testSet"),
            classifierName = Parameters::getParameter<std::string>("test.classifierName");

    const int
            backgroundClassLabel = Parameters::getParameter<int>("backgroundClassLabel"),
            offsetX = Parameters::getParameter<int>("test.offsetX"),
            offsetY = Parameters::getParameter<int>("test.offsetY");


    // AdaboostLearner Learner(verbose,labeledData);


    ModelIO modelReader;
    modelReader.readModel(classifierName);
    StrongClassifier classifier = modelReader.read();
    //classifier._learners[classifier._learners.size()-1]._cascadeThreshold = 0;
    //classifier.writeClassifier("0803cascade5000.firstIter.proto.bin");

    const TrainingData::point_t modelWindow = getModelWindowSize();
    //const TrainingData::rectangle_t objectWindow = getObjectWindow();

    //get Data

    std::vector<std::string> filenamesPositives, filenamesBackground;
    getImageFileNames(testSetPaths, backgroundClassLabel, filenamesPositives, filenamesBackground);

    LabeledData labeledTestData(verbose, backgroundClassLabel);

    labeledTestData.createIntegralImages(filenamesPositives, filenamesBackground,
                                         modelWindow, offsetX, offsetY);


    boost::timer timer;

    //do classification
    int tp, fp, fn, tn;
    classifier.classify(labeledTestData, tp, fp, fn, tn);

    const float time_in_seconds = timer.elapsed();

    std::cout << "Time required for execution: " << time_in_seconds << " seconds." << "\n\n";
    std::cout << "FrameRate: " << labeledTestData.getNumExamples() / time_in_seconds << std::endl;


    std::cout << "Classification Results (TestData): " << std::endl;
    std::cout << "Detection Rate: " << double(tp + tn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Rate: " << double(fp + fn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Positives: " <<  double(fn) / (tp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Negatives: " <<  double(fp) / (tn + fp) * 100 << " %" <<  std::endl;
    std::cout << "\n";

    return;
}



// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
// FIXME move this to a separate file

class AppendDataFunctor
{

public:
    AppendDataFunctor(TrainingData::shared_ptr dataStore);
    ~AppendDataFunctor();

    void operator()(const TrainingData::meta_datum_t &meta_datum, const TrainingData::integral_channels_t &integral_image);

protected:
    size_t datumIndex;
    TrainingData::shared_ptr dataStore;

};


AppendDataFunctor::AppendDataFunctor(TrainingData::shared_ptr dataStore_)
    : datumIndex(dataStore_->getNumExamples()),
      dataStore(dataStore_)
{
    // nothing to do here
    return;
}

AppendDataFunctor::~AppendDataFunctor()
{
    // nothing to do here
    return;
}

void AppendDataFunctor::operator()(const TrainingData::meta_datum_t &metaDatum,
                                   const TrainingData::integral_channels_t &integralImage)
{
    if(datumIndex >= dataStore->getMaxNumExamples())
    {
        throw std::runtime_error("AppendDataFunctor::operator() has been called "
                                 "more times than the data store memory allocation allows");
    }

    dataStore->setDatum(datumIndex, metaDatum, integralImage);
    datumIndex += 1;
    return;
}

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

void bootstrapTrain(const int verbose, const bool doBootstrap = true)
{

    //read files for training
    const std::string
            trainSetPath = Parameters::getParameter<std::string>("train.trainSet"),
            testSetPath = Parameters::getParameter<std::string>("train.testSet");


    const int
            backgroundClassLabel = Parameters::getParameter<int>("backgroundClassLabel"),
            numBootstrappingSamples = Parameters::getParameter<int>("bootstrapTrain.numBootstrappingSamples");

    std::vector<int> stages, maxNumSamplesPerImage;

    if(doBootstrap)
    {
        stages = Parameters::getParameter<std::vector<int> >("bootstrapTrain.classifiersPerStage");
        maxNumSamplesPerImage = Parameters::getParameter<std::vector<int> >("bootstrapTrain.maxNumSamplesPerImage");
    }
    else
    {
        const int numIterations = Parameters::getParameter<int>("train.numIterations");
        stages.push_back(numIterations);
        maxNumSamplesPerImage.push_back(-1);
    }


    if (stages.size() != maxNumSamplesPerImage.size())
    {
        throw runtime_error("Size miss match between the vectors classifiersperStage and maxNumSamplesPerImage");
    }


    const TrainingData::point_t
            modelWindowSize = getModelWindowSize(),
            trainDataOffset = getTrainingDataOffset();

    const TrainingData::rectangle_t objectWindow = getObjectWindow();

    std::vector<std::string> filenamesPositives, filenamesBackground;
    getImageFileNames(trainSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground);

    const int trainNumNegativeSamples = Parameters::getParameter<int>("train.numNegativeSamples");

    size_t maxNumExamples = filenamesPositives.size() + trainNumNegativeSamples;
    if(stages.size() > 1)
    {
        maxNumExamples += numBootstrappingSamples*(stages.size() - 1);
    }

    const std::string initialBootstrapFileName = Parameters::getParameter<std::string>("train.bootStrapLearnerFile");
    if (!initialBootstrapFileName.empty())
    {
        maxNumExamples += numBootstrappingSamples;
    }


    // computed all feature configurations available for training.
    const size_t featuresPoolSize = Parameters::getParameter<int>("train.featuresPoolSize");
    FeaturesSharedPointer featuresConfigurations(new Features());
    //first basic feature pool
    computeRandomFeaturesConfigurations(modelWindowSize, featuresPoolSize,  *featuresConfigurations);

    //all features are valid for this setup
    std::vector<bool> valid_features(featuresConfigurations->size(),true);
    //fill (valid_features.begin(),valid_features.begin()+featuresPoolSize,true);



    TrainingData::shared_ptr trainingData(new TrainingData(featuresConfigurations, valid_features, maxNumExamples,
                                                           modelWindowSize, objectWindow));
    trainingData->addPositiveSamples(filenamesPositives, modelWindowSize, trainDataOffset);
    trainingData->addNegativeSamples(filenamesBackground, modelWindowSize, trainDataOffset, trainNumNegativeSamples);

    if (!initialBootstrapFileName.empty())
    {
        //for a weak model it should be avoided to sample all hard negatives from a single image
        const int maxFalsePositivesPerImage = 5;
        trainingData->addBootstrappingSamples(initialBootstrapFileName, filenamesBackground,
                                              modelWindowSize, trainDataOffset,
                                              numBootstrappingSamples, maxFalsePositivesPerImage);
    }

    const bool check_boostrapping = false; // for debugging only
    if(check_boostrapping)
    {
        ModelIO modelReader;
        modelReader.readModel(initialBootstrapFileName);
        StrongClassifier classifier = modelReader.read();

        int tp, fp, fn, tn;
        classifier.classify(*trainingData, tp, fp, fn, tn);
        std::cout << "Classification Results (TestData): " << std::endl;
        std::cout << "Detection Rate: " << double(tp + tn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
        std::cout << "Error Rate: " << double(fp + fn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
        std::cout << "Error Positives: " <<  double(fn) / (tp + fn) * 100 << " %" <<  std::endl;
        std::cout << "Error Negatives: " <<  double(fp) / (tn + fp) * 100 << " %" <<  std::endl;
        std::cout << std::endl;
        throw std::runtime_error("End of game, just doing a mini-test");
    }

    AdaboostLearner Learner(verbose, trainingData);

    if (not testSetPath.empty())
    {
        // FIXME test data should use TrainingData class instead of the deprecated LabeledData class
        const TrainingData::point_t testDataOffset = getTestingDataOffset();

        std::vector<std::string> filenamesPositives, filenamesBackground;
        getImageFileNames(testSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground);

        printf("\nCollecting test data...\n");
        LabeledData::shared_ptr labeledTestData(new LabeledData(verbose, backgroundClassLabel));
        //printf("testDataOffsets: %i, %i\n", testDataOffset.x(), testDataOffset.y() );
        labeledTestData->createIntegralImages(filenamesPositives, filenamesBackground,
                                              modelWindowSize, testDataOffset.x(), testDataOffset.y());
        Learner.setTestData(labeledTestData);
    }

    const std::string baseOuputModelFilename = Learner.getOuputModelFileName();

    for (size_t k = 0; k < stages.size(); ++k)
    {
        // bootstrap new negatives
        if (k != 0)
        {
            const std::string bootstrapFile =
                    boost::str(boost::format("%s.bootstrap%i") % baseOuputModelFilename % (k - 1));

            // sample new (hard) negatives using bootstrapping
            trainingData->addBootstrappingSamples(bootstrapFile, filenamesBackground,
                                                  modelWindowSize, trainDataOffset,
                                                  numBootstrappingSamples, maxNumSamplesPerImage[k]);
        }

        Learner.setNumIterations(stages[k]);
        Learner.setOutputModelFileName(boost::str(boost::format("%s.bootstrap%i") % baseOuputModelFilename % (k)));

        if (k == stages.size()-1)
            Learner.train(true);
        else
            Learner.train(false);
    } // end of "for each stage"

    boost::filesystem::copy_file(Learner.getOuputModelFileName(), baseOuputModelFilename);

    printf("Finished the %zi bootstrapping stages. Model was trained over %zi samples (%zi positives, %zi negatives).\n"
           "Final model saved at %s\n",
           stages.size(),
           trainingData->getNumExamples(),
           trainingData->getNumPositiveExamples(), trainingData->getNumNegativeExamples(),
           //Learner.getOuputModelFileName().c_str()
           baseOuputModelFilename.c_str());


    return;
}




void train(const int verbose)
{

    const bool doBootstrap = false;
    bootstrapTrain(verbose, doBootstrap);

    return;
}

/// print to console the adequate log level
void setup_logging(const int verboseLevel)
{
    logging::get_log().clear(); // we reset previously existing options

    logging::LogRuleSet logging_rules;

    if( verboseLevel > 0)
    {
        logging_rules.add_rule(logging::WarningMessage, "*");
    }
    else if( verboseLevel > 1)
    {
        logging_rules.add_rule(logging::InfoMessage, "*");
    }
    else if( verboseLevel > 2)
    {
        logging_rules.add_rule(logging::EveryMessage, "*");
    }

    logging::get_log().add(std::cout, logging_rules);

    return;
}


void boosted_learning_main(int argc, char *argv[])
{
    // instantiate parameter object, read all options
    Parameters::loadParameters(argc, argv);

    // get task
    const std::string task = Parameters::getParameter<std::string>("task");
    const int verboseLevel = Parameters::getParameter<int>("verbose");

    setup_logging(verboseLevel);

    std::cout << "Task: " << task << std::endl;

    if (task == "train")
    {
        train(verboseLevel);
    }
    else if (task == "test")
    {
        test(verboseLevel);
    }
    else if (task == "bootstrapTrain")
    {
        bootstrapTrain(verboseLevel);
    }
    else if (task == "printModel")
    {
        printModel();
    }
    else
    {
        throw std::invalid_argument("unknown task given");
    }

    printf("End of game. Have a nice day !\n");
    return;
}

} // end of namespace boosted_learning


int main(int argc, char *argv[])
{

    int ret = EXIT_SUCCESS;

    try
    {
        boosted_learning::boosted_learning_main(argc, argv);
    }
    // on linux re-throw the exception in order to get the information
    catch (std::exception & e)
    {
        std::cout << "\033[1;31mA std::exception was raised:\033[0m " << e.what () << std::endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }
    catch (...)
    {
        std::cout << "\033[1;31mAn unknown exception was raised\033[0m " << std::endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }

    return ret;
}
