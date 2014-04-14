#include "AdaboostLearner.hpp"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#include "Parameters.hpp"
#include "Feature.hpp"
#include "ModelIO.hpp"
#include "IntegralChannelsComputer.hpp"

#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/multi_array.hpp>

#include "linear.h"

#include <sys/mman.h>
#include <omp.h>

#include <vector>
#include <algorithm>
#include <limits>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <limits>


namespace boosted_learning {

using namespace std;
void AdaboostLearner::getParameters()
{
    _typeAdaboost = Parameters::getParameter<string>("train.typeAdaboost");
    setNumIterations(Parameters::getParameter<int>("train.numIterations"));

    using namespace boost::posix_time;
    const ptime current_time(second_clock::local_time());

    const string outputModelFilename =
            boost::str( boost::format("%i_%02i_%02i_%i_%s")
                        % current_time.date().year()
                        % current_time.date().month().as_number()
                        % current_time.date().day()
                        % current_time.time_of_day().total_seconds()
                        % Parameters::getParameter<string>("train.outputModelFileName") );

    setOutputModelFileName(outputModelFilename);

    return;
}


AdaboostLearner::AdaboostLearner(int verbose, TrainingData::shared_ptr data):
    _verbose(verbose), _trainData(data)
{
    getParameters();
    return;
}


AdaboostLearner::~AdaboostLearner()
{
    // nothing to do here
    return;
}


void AdaboostLearner::setTestData(LabeledData::shared_ptr data)
{
    _testData = data;
    return;
}


void AdaboostLearner::setValidationData(LabeledData::shared_ptr data)
{
    _validationData = data;
    return;
}


void AdaboostLearner::setNumIterations(const int i)
{
    _numIterations = i;
    return;
}


void AdaboostLearner::setOutputModelFileName(const std::string update)
{
    _outputModelFileName = update;
    return;
}


std::string AdaboostLearner::getOuputModelFileName() const
{
    return _outputModelFileName;
}


double AdaboostLearner::classify(const std::vector<WeakDiscreteTree> &classifier)
{
    StrongClassifier strongClassifier(classifier);
    int tp, fp, fn, tn;
    //scf.classify(_data, tp, fp, fn,tn);
    //std::cout<< "Classification Results (TrainData): " << std::endl;
    //std::cout << "Detection Rate: " << double(tp+tn)/(tp+tn+fp+fn) * 100 << " %" <<  std::endl;
    //std::cout << "Error Rate: " << double(fp+fn)/ (tp+tn+fp+fn) * 100 << " %" <<  std::endl;
    //std::cout << "Error Positives: " <<  double(fn)/ (tp+fn) * 100 << " %" <<  std::endl;
    //std::cout << "Error Negatives: " <<  double(fp)/ (tn+fp) * 100 << " %" <<  std::endl;
    //std::cout << "\n";

    strongClassifier.classify(*(_testData), tp, fp, fn, tn);
    double errorrate = double(fp + fn) / (tp + tn + fp + fn);
    std::cout << "Classification Results (TestData): " << std::endl;
    std::cout << "Detection Rate: " << double(tp + tn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Rate: " << errorrate * 100 << " %" <<  std::endl;
    std::cout << "Error Positives: " <<  double(fn) / (tp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Negatives: " <<  double(fp) / (tn + fp) * 100 << " %" <<  std::endl;
    std::cout << std::endl;

    return errorrate;
}



const TrainingData::shared_ptr AdaboostLearner::getTrainData() const
{
    return _trainData;

}



void AdaboostLearner::train(bool last)
{
    const int decisionTreeDepth = Parameters::getParameter<int>("train.decisionTreeDepth");

    ModelIO modelWriter(_verbose);
    const std::string
            datasetName = Parameters::getParameter<std::string>("train.trainSetName"),
            trainedModelName = "Model created via boosted_learning";
    modelWriter.initWrite(datasetName,
                          doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                          trainedModelName,
                          _trainData->getModelWindow(), _trainData->getObjectWindow());


    printf("\nStarting training with %zi positive samples and %zi negative samples (%zi total)\n",
           _trainData->getNumPositiveExamples(), _trainData->getNumNegativeExamples(), _trainData->getNumExamples());

    //------------------------------------
    //initialized weights and get filenames of background data.

    // the weights of the algorithm Adaboost
    WeakDiscreteTreeLearner::weights_t weights(_trainData->getNumExamples());
    std::vector<int> classLabels(_trainData->getNumExamples());

    std::vector<std::string> filenamesBG;
    filenamesBG.reserve(_trainData->getNumExamples()*0.75); // rough heuristic

    for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
    {
        classLabels[trainingSampleIndex] = _trainData->getClassLabel(trainingSampleIndex);

        if (_trainData->getClassLabel(trainingSampleIndex) == -1)
        {
            weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumNegativeExamples());
            filenamesBG.push_back(_trainData->getFilename(trainingSampleIndex));
        }
        else
        {
            weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumPositiveExamples());
        }
    } // end of "for each training example"


    // minWeight is an heuristic to avoid "extreme cases"
    // (see entropy regularized LPBoost paper)
    //const double minWeight = 1.0 / (_trainData->getNumExamples() * 10); // this idea did not work, at all
    //const double minWeight = 0;

    MinOrMaxFeaturesResponsesSharedPointer
            maxvs(new MinOrMaxFeaturesResponses(_trainData->getFeaturesPoolSize())),
            minvs(new MinOrMaxFeaturesResponses(_trainData->getFeaturesPoolSize()));

    calcMinMaxFeatureResponses(_trainData, minvs, maxvs);

    std::vector<WeakDiscreteTree> classifier;
    std::vector<double> scores(_trainData->getNumExamples(), 0);

    bool previousErrorRateIsZero = false;


    double start_wall_time = omp_get_wtime();

    // numIterations define the number of weak classifier in the boosted strong classifier
    for(int iterationsCounter = 0; iterationsCounter < _numIterations; iterationsCounter +=1)
    {
        if(previousErrorRateIsZero == false)
        {
            std::cout << "Training stage: " << iterationsCounter << flush << std::endl;
        }
        else
        {
            // no need to print the same data again and again...
        }


        WeakDiscreteTreeLearner weakLearner(_verbose, decisionTreeDepth, -1,
                                            _trainData, classLabels, minvs, maxvs);
        const double error = weakLearner.buildBalancedTree(weights);
        double normalizeFactor = 0;

        if (error >= 0.5)
        {
            throw runtime_error("TRAINING STOP: Not possible to reduce error anymore");
        }

        // classify the data save misclassification indices
        double errorCalc = 0;

        for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
        {
            // hypothesis_label is either 1 or -1
            const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
            scores[trainingSampleIndex] += hypothesisLabel * weakLearner.getBeta();

            double &sampleWeight = weights[trainingSampleIndex];
            if (hypothesisLabel != classLabels[trainingSampleIndex])
            {
                // error increases when predicted label does not match the data label
                errorCalc += sampleWeight;
                // update weights
                sampleWeight *= exp(weakLearner.getBeta());
            }
            else
            {
                // update weights
                sampleWeight *= exp( -weakLearner.getBeta());
            }

            // FIXME is this really a good idea ?
            // sampleWeight = std::max(sampleWeight, minWeight); // not a good idea

            normalizeFactor += sampleWeight;
        }

        //normalize weights
        for (size_t i = 0; i < _trainData->getNumExamples(); ++i)
        {
            weights[i] /= normalizeFactor;
        }

        //classification ================================:
        int truePositives = 0, falsePositives = 0, falseNegatives = 0, trueNegatives = 0;

        for (size_t i = 0; i < _trainData->getNumExamples(); ++i)
        {
            if (scores[i] >= 0)
            {
                if (_trainData->getClassLabel(i) == 1)
                {
                    truePositives++;
                }
                else
                {
                    falsePositives++;
                }
            }
            else
            {
                if (_trainData->getClassLabel(i) == 1)
                {
                    falseNegatives++;
                }
                else
                {
                    trueNegatives++;
                }
            }
        }

        const double errorRate = double(falsePositives + falseNegatives) / _trainData->getNumExamples() * 100;
        const bool errorRateIsZero = (errorRate == 0);
        if((previousErrorRateIsZero == false) and (errorRateIsZero == true))
        {
            std::cout << std::endl;
        }

        if((previousErrorRateIsZero == false) or (errorRateIsZero == false))
        {
            const double detectionRate = double(truePositives + trueNegatives) / _trainData->getNumExamples() * 100;
            std::cout << "Classification Results (Trainset): " << std::endl;
            std::cout << "Detection Rate: " << detectionRate << " %" <<  std::endl;
            std::cout << "Error Rate: " << errorRate << " %" <<  std::endl;
            std::cout << "Error Positives: " <<  double(falseNegatives) / (truePositives + falseNegatives) * 100 << " %" <<  std::endl;
            std::cout << "Error Negatives: " <<  double(falsePositives) / (trueNegatives + falsePositives) * 100 << " %" <<  std::endl;
            std::cout << std::endl;
        }
        else
        {
            // no need to print the same data again and again...
            // overwrite the previous output line
            std::cout << "\rError rate stable at zero until " << iterationsCounter << std::flush;
        }

        // update in case the error rate fluctuated
        previousErrorRateIsZero = errorRateIsZero;


        // store the new learned weak classifier --
        classifier.push_back(weakLearner); // converted from WeakDiscreteTreeLearner to WeakDiscreteTree
        modelWriter.addStage(weakLearner);

        // save a temporary model
        //if ((iterationsCounter % 200) == 10) // == 10 to avoid saving an empty model
        if ((iterationsCounter % 1000) == 10)
        {
            const string temporary_filename = _outputModelFileName + ".tmp";
            modelWriter.write(temporary_filename);
            std::cout << std::endl << "Created " << temporary_filename << std::endl;
            //classify(classifier);
        }

    } // end of "for all the iterations"

    std::cout << std::endl;

    if(true)
    {
        printf("Total time for the %i iterations is %.3f [seconds]\n",
               _numIterations,  (omp_get_wtime() - start_wall_time) );
    }

    modelWriter.write(_outputModelFileName);



    // evaluate with test data
    if (_testData)
        classify(classifier);

    return;
}




void calcMinMaxFeatureResponses(TrainingData::shared_ptr trainData, MinOrMaxFeaturesResponsesSharedPointer minvs,
                                MinOrMaxFeaturesResponsesSharedPointer maxvs){
    const FeaturesResponses &featuresResponses = trainData->getFeatureResponses();

    for (size_t featureIndex = 0; featureIndex < trainData->getFeaturesPoolSize(); ++featureIndex)
    {
        if (trainData->_validFeatures[featureIndex] == false)
            continue;
        int minv = std::numeric_limits<int>::max();
        int maxv = -std::numeric_limits<int>::max();

        for (size_t exampleIndex = 0; exampleIndex < trainData->getNumExamples(); ++exampleIndex)
        {
            const int val = featuresResponses[featureIndex][exampleIndex];
            minv = std::min(val, minv);
            maxv = std::max(val, maxv);
        } // end of "for each example"

        (*maxvs)[featureIndex] = maxv;
        (*minvs)[featureIndex] = minv;

    } // end of "for each feature"

}





} // end of namespace boosted_learning
