
#include "Parameters.hpp"

#include <boost/program_options.hpp>

namespace boosted_learning {


namespace po = boost::program_options;

boost::shared_ptr<Parameters> Parameters::s_instance;

Parameters::Parameters()
{
    setOptionsDescriptions();
    return;
}

Parameters &Parameters::i()
{
    if (!s_instance)
    {
        s_instance.reset(new Parameters());
    }

    return *s_instance;
}

void Parameters::loadParameters(int argc, char **argv)
{
    Parameters &self = i();

    po::store(po::command_line_parser(argc, argv).options(self.options_descriptions).run(), self.variablesMap);

    if (self.variablesMap.count("help"))
    {
        showHelp();
        // end of game
        throw std::runtime_error("End of game. Have a nice day !");
    }

    if (self.variablesMap.count("conf"))
    {
        const std::string configurationFile = self.variablesMap["conf"].as<std::string>();
        self.load(configurationFile);
        std::cout << "Reading data from " << configurationFile << std::endl;
    }
    else
    {
        std::cout << "[Warning] No configuration file indicated" << std::endl;
    }

    po::notify(self.variablesMap);
    return;
}

void Parameters::load(const std::string fileName)
{

    std::ifstream inputFileStream(fileName.c_str());

    if (inputFileStream)
    {
        const bool allowUnregistered = false; //  dis-allow unregistered options
        po::store(po::parse_config_file(inputFileStream, options_descriptions, allowUnregistered), variablesMap);
    }
    else
    {
        std::cerr << "Configuration filename == " << fileName << std::endl;
        throw std::invalid_argument("Could not open the configuration file");
    }

    return;
}


Parameters::~Parameters()
{
    // nothing to do here
    return;
}


void Parameters::showHelp()
{
    Parameters &self = i();
    std::cout << self.options_descriptions << std::endl;
    return;
}



const boost::program_options::variables_map &Parameters::getVariablesMap()
{
    return variablesMap;
}


void Parameters::setOptionsDescriptions()
{
    namespace po = boost::program_options;
    using namespace boost::program_options;

    options_descriptions.add_options()
            ("verbose", po::value<int>()->default_value(1),
             "verbose level (the highest, the more verbose)")
            ("backgroundClassLabel", po::value<int>()->default_value(0),
             "label for the background class")
            ;

    {
        options_description commandline_options("Program options");
        commandline_options.add_options()
                ("help,h", "show this message")
                ("conf,c", po::value<std::string>(), "configuration .ini file to read options from")
		("task,t" , po::value<std::string>()->default_value("bootstrapTrain"), "the task to execute (you probably want to use bootstrapTrain)")
                ;

        options_descriptions.add(commandline_options);
    }

    {
        options_description training_options("Training task options");

        training_options.add_options()

                ("train.trainSetName", po::value<std::string>()->default_value("INRIAPerson"),
                 "String identifying the training set used. This for human consuption.")

                ("train.trainSet", po::value<std::string>(),
                 "File containing the Training set, each line <class> <image_path>")

                ("train.testSet", po::value<std::string>(),
                 "File containing the Testing set, each line <class> <image_path>")

                ("train.modelWindow", po::value<std::string>()->default_value("64,128"),
                 "Window size of the training images (w,h)")

                ("train.numIterations", po::value<int>(),
                 "number of iterations for training."
                 "For vanilla adaboost this defines the number of weak classifiers used in the strong classifier.")

                ("train.numNegativeSamples",
                 po::value<int>()->default_value(5000),
                 "Number of negative samples to be used for the training")

                ("train.objectWindow", po::value<std::string>()->default_value("8,16,48,96"),
                 "bounding box of the object in (x,y,w,h)")

                ("train.typeAdaboost", po::value<std::string>()->default_value("vanilla"),
                 "Type of the Adaboost used: vanilla or vanilla")
                // FIXME this option is not really used, should add checks in the code
                //  default, GENTLE_ADA, other=DISCRETE_ADA

                ("train.decisionTreeDepth", po::value<int>()->default_value(1),
                 "depth of the decision trees (0 equals decision stump)")

                ("train.offsetX", po::value<int>(),
                 "offset in X direction between the training image border and the model window")

                ("train.offsetY", po::value<int>(),
                 "offset in y direction between the training image border and the model window")

                ("train.bootStrapLearnerFile", po::value<std::string>()->default_value(std::string()),
                 "File with the learner used for bootstrapping")

                ("train.cascadeType", po::value<std::string>()->default_value("dbp"),
                 "Type of the soft cascade: none, dbp or mip."
                 "dbp stands for Direct Backward Prunning (see C. Zang and P. Viola 2007).")

                ("train.minFeatWidth", po::value<int>()->default_value(1),
                 "minimal width of the features to train on (in pixels of the original image)")

                ("train.minFeatHeight", po::value<int>()->default_value(1),
                 "minimal height of the features to train on (in pixels of the original image)")

                ("train.maxFeatWidth", po::value<int>()->default_value(-1),
                 "maximum width of the features to train on (in pixels of the original image). "
                 "If negative value, no limit is imposed.")

                ("train.maxFeatHeight", po::value<int>()->default_value(-1),
                 "maximum height of the features to train on (in pixels of the original image). "
                 "If negative value, no limit is imposed.")

                // FIXME is this ever used ?
                //("train.maxFeatureSizeRatio", po::value<double>(),
                // "defines the maximal size of a feature: 0.6 means max 60% of the training image size")

                ("train.featuresPoolSize", po::value<int>(),
                 "Size of the set of features used to build up weak classifiers at each iteration of the boosting algorithm")

                ("train.pushUpBias", po::value<double>()->default_value(0.0),
                 "biasing the features to be located in the upper half of the object window: 0.0 means not pushing up at all 1.0 results in a half (top) classifier")

                ("train.pushLeftBias", po::value<double>()->default_value(0.0),
                 "biasing the features to be located in the left half of the object window: 0.0 means not pushing left at all 1.0 results in a half (left) classifier")


                ("train.recursiveOcclusion", po::value<bool>()->default_value(true),
                 "use recursive option for training the occlusion classifier")

                ("train.resetBeta", po::value<bool>()->default_value(true),
                 "set true to reweight the features, after removing some")
                ("train.removeHurtingFeatures", po::value<bool>()->default_value(true),
                 "set true to remove as well features, that generate a training error > 0.5")

                ("train.extendFeaturePool", po::value<bool>()->default_value(true),
                 "extend feature pool for occlusion classifiers. Every classifier should have the same size of feature pool")




                ("train.enableFrankenClassifier", po::value<bool>()->default_value(false),
                 "enables the training of the franken classifier")
                ("train.frankenType", po::value<std::string>()->default_value("up"),
                 "set this value to \"up\" or \"left\" to eighter build the bottom occlusion franken classifier or the left occlusion one")

                ("train.featuresPoolRandomSeed", po::value<boost::uint32_t>()->default_value(0),
                 "random seed used to generate the features pool. If the value is 0 the current time is used as seed (recommended)."
                 "Fixing the value to a value > 0 allows to repeat trainings with the same set of features "
                 "(but negative samples still randomly sampled)")

                ("train.outputModelFileName", po::value<std::string>(),
                 "file to write the trained detector into")
                ("train.svmSaveProblemFile", po::value<std::string>()->default_value(""),
                 "if the filename is not equal to \"\", the svm-problem will be saved as asci file to run with liblinear")
                ("train.useSVM", po::value<bool>()->default_value(false),
                 "specifies to use SVM training to generate the franken-classifier")
                ;

        options_descriptions.add(training_options);
    }

    {
        options_description testing_options("Testing task options");

        testing_options.add_options()
                ("test.classifierName", po::value<std::string>(),
                 "filename of the classifier to test on")

                ("test.testSet", po::value<std::string>(),
                 "File containing the Testing set, each line <class> <image_path>")

                ("test.offsetX", po::value<int>()->default_value(3), // default value fits INRIAPerson
                 "offset in X direction between the testing image border and the model window")

                ("test.offsetY", po::value<int>()->default_value(3), // default value fits INRIAPerson
                 "offset in y direction between the testing image border and the model window")

                ;
        options_descriptions.add(testing_options);
    }

    {
        options_description boostrapping_options("bootstrapTrain task options");

        boostrapping_options.add_options()
                ("bootstrapTrain.classifiersPerStage", po::value<std::vector<int> >()->multitoken(),
                 "List of number of classifiers trained per stage.")

                ("bootstrapTrain.maxNumSamplesPerImage", po::value<std::vector<int> >()->multitoken(),
                 "List of number of samples per image allowed for bootstrapping")

                ("bootstrapTrain.numBootstrappingSamples",
                 po::value<int>()->default_value(5000),
                 "Number of samples to be searched by bootstrapping")

                ("bootstrapTrain.min_scale", value<float>()->default_value(0.3),
                 "minimum detection window scale explored for detections")

                ("bootstrapTrain.max_scale", value<float>()->default_value(5.0),
                 "maximum detection window scale explored for detections")

                ("bootstrapTrain.num_scales", value<int>()->default_value(10),
                 "number of scales to explore. (this is combined with num_ratios)")

                ("bootstrapTrain.min_ratio", value<float>()->default_value(1),
                 "minimum ratio (width/height) of the detection window used to explore detections")

                ("bootstrapTrain.max_ratio", value<float>()->default_value(1),
                 "max ratio (width/height) of the detection window used to explore detections")

                ("bootstrapTrain.num_ratios", value<int>()->default_value(1),
                 "number of ratios to explore. (this is combined with num_scales)")

                ("bootstrapTrain.frugalMemoryUsage", value<bool>()->default_value(false),
                 "By default we use as much GPU memory as useful for speeding things up. "
                 "If frugal memory usage is enabled, we will reduce the memory usage, "
                 "at the cost of longer computation time.")

                ;

        options_descriptions.add(boostrapping_options);
    }

    return;
}


} // end of namespace boosted_learning
