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

double AdaboostLearner::classify_real(const std::vector<WeakDiscreteTree> &classifier)
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

    strongClassifier.classify_real(*(_testData), tp, fp, fn, tn);
    double errorrate = double(fp + fn) / (tp + tn + fp + fn);
    std::cout << "Classification Results (TestData): " << std::endl;
    std::cout << "Detection Rate: " << double(tp + tn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Rate: " << errorrate * 100 << " %" <<  std::endl;
    std::cout << "Error Positives: " <<  double(fn) / (tp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Negatives: " <<  double(fp) / (tn + fp) * 100 << " %" <<  std::endl;
    std::cout << std::endl;

    return errorrate;
}


//***************************************************************************************************

void AdaboostLearner::toSoftCascadeDBP(const LabeledData::shared_ptr data,
                                       const std::string inputModelFileName,
                                       const std::string softCascadeFileName,
                                       const TrainingData::point_t &modelWindow,
                                       const TrainingData::rectangle_t &objectWindow)
{

    printf("Starting computation to create a DBP (discrete backward prunning (?)) soft cascade. Please be patient...\n");
    ModelIO modelReader;
    modelReader.readModel(inputModelFileName);
    StrongClassifier learner = modelReader.read();

    // modelReader.print();
    const double detectionRate = 1.0;
    //const double detectionRate = 0.92; // This value corresponds to the FPDW and LatentSVM responses at 1 FPPI on INRIA
    learner.convertToSoftCascade(data, detectionRate);
    learner.writeClassifier(softCascadeFileName,
                            modelReader.getModelTrainingDatasetName(),
                            modelWindow, objectWindow);

    return;
}

const TrainingData::shared_ptr AdaboostLearner::getTrainData() const
{
    return _trainData;


}
#if 0
void AdaboostLearner::doBootStrapping(std::vector<std::string> &filenamesBG,
                                      std::vector<double> & scores, std::vector<WeakDiscreteTreeLearner> &classifier,
                                      std::vector<double> & weights, std::vector<int> & cl,
                                      std::vector<int> &maxvs, std::vector<int>  &minvs)
{

    StrongClassifier scf(classifier);
    int maxNr = min(_data.getNumNegExamples() / 2, 3000);
    std::cout << "no of elements to bootstrap: " << maxNr << std::endl;
    scf.convertToSoftCascade(*_validationData, 1.0);
    //scf.convertToSoftCascade(_data, 1.0);
    _data.bootstrap(scf, maxNr, filenamesBG);

    scores.clear();
    weights.clear();
    cl.clear();
    classifier.clear();
    //set new sizes
    scores.resize(_data.getNumExamples(), 0);
    weights.resize(_data.getNumExamples());
    cl.resize(_data.getNumExamples());

    //reset weights
    for (int hh = 0; hh < _data.getNumExamples(); hh++)
    {
        cl[hh] = _data.getClassLabel(hh);

        if (_data.getClassLabel(hh) == -1)
        {
            weights[hh] = 1.0 / (2.0 * _data.getNumNegExamples());
        }
        else
        {
            weights[hh] = 1.0 / (2.0 * _data.getNumPosExamples());
        }
    }

    //update Feature response
    int numOfElements = _numOfFeatToUse * _data.getNumExamples();
    _featureResp->close_mmap();
    jhb::MemMappedFile<int>::shared_ptr tmp(new jhb::MemMappedFile<int>);
    //its a shared pointer, so this should work
    _featureResp = tmp;
    _featureResp->open_mmap(_memFile, numOfElements, jhb::MemMappedFile<int>::WRITE);

#pragma omp parallel for default(none) ordered

    for (int i = 0; i < _data.getNumExamples(); ++i)
    {
        const integral_channels_t &integralImage = _data.getIntImage(i);

        for (size_t j = 0; j < _precomputedConfigs.size(); ++j)
        {
            int ff = getFeatResponse(integralImage, _precomputedConfigs[j]);
            (*_featureResp)[i+j*_data.getNumExamples()] = ff ;
        }
    }

    //get min and max values
    for (size_t j = 0; j < _precomputedConfigs.size(); ++j)
    {
        int minv = std::numeric_limits<int>::max();
        int maxv = -std::numeric_limits<int>::max();

        for (int i = 0; i < _data.getNumExamples(); ++i)
        {
            int val = (*_featureResp)[i+j*_data.getNumExamples()];
            minv = std::min(val, minv);
            maxv = std::max(val, maxv);
        }

        maxvs[j] = maxv;
        minvs[j] = minv;
    }

    return;
}
#endif

#if false
void AdaboostLearner::recalculateWeights(const LabeledData &data,
                                         const std::vector<WeakDiscreteTree> & learner,
                                         std::vector<double> & weights,
                                         std::vector<double> & scores)
{

    //reset weights
    weights.clear();
    scores.clear();
    weights.resize(data.getNumExamples());
    scores.resize(data.getNumExamples(), 0);
    std::vector<int>cl(data.getNumExamples());

    for (size_t hh = 0; hh < data.getNumExamples(); hh++)
    {
        cl[hh] = data.getClassLabel(hh);

        if (data.getClassLabel(hh) == -1)
        {
            weights[hh] = 1.0 / (2.0 * data.getNumNegExamples());
        }
        else
        {
            weights[hh] = 1.0 / (2.0 * data.getNumPosExamples());
        }

    }

    for (size_t j = 0; j < learner.size(); ++j)
    {
        const WeakDiscreteTree &weakLearner = learner[j];
        double normalizeFactor = 0;

        for (size_t i = 0; i < data.getNumExamples(); ++i)
        {
            const integral_channels_t &integralImage = data.getIntegralImage(i);
            int ht = weakLearner.classify(integralImage);
            scores[i] += ht * weakLearner.getBeta();

            if (ht != cl[i])
            {
                //updateweights
                weights[i] = weights[i] * exp(weakLearner.getBeta());
            }
            else
            {
                //updateweights
                weights[i] = weights[i] * exp(- weakLearner.getBeta());
            }

            normalizeFactor += weights[i];
        }

        for (size_t i = 0; i < data.getNumExamples(); ++i)
        {
            weights[i] /= normalizeFactor;
        }
    }

    return;
}
#endif

void AdaboostLearner::trainONGivenClassifier(std::vector<WeakDiscreteTree> & classifier){
    WeakDiscreteTreeLearner::weights_t weights(_trainData->getNumExamples());
    std::vector<int> classLabels(_trainData->getNumExamples());
    std::vector<bool> usedClassifiers(classifier.size(), false);

    for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
    {
        classLabels[trainingSampleIndex] = _trainData->getClassLabel(trainingSampleIndex);

        if (_trainData->getClassLabel(trainingSampleIndex) == -1)
        {
            weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumNegativeExamples());
        }
        else
        {
            weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumPositiveExamples());
        }
    } // end of "for each training example"

    while(true){
        double minerror = 5;
        int minClassifierIndex = -1;
        double beta = 0;
        //find lowest error
        for (size_t classifierIndex = 0; classifierIndex < classifier.size(); ++ classifierIndex){
            if (usedClassifiers[classifierIndex] == true)
                continue;

            WeakDiscreteTree & weakLearner = classifier[classifierIndex];



            double incorrectweight=0;
            for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
            {
                // hypothesis_label is either 1 or -1
                const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
                const int realLabel = _trainData->getClassLabel(trainingSampleIndex);
                if (hypothesisLabel != realLabel)
                {
                    incorrectweight += weights[trainingSampleIndex];
                }
            }
            double error = incorrectweight;
            assert(error <=1);
            if (error<minerror){
                minerror = error;
                minClassifierIndex = classifierIndex;
            }
        }
        if (minerror > 0.5)
            break;

        if (minerror < 1e-12)
        {
            beta = 0;
        }
        else
        {
            beta = 0.5 * log((1.0 - minerror) / minerror);
        }
        WeakDiscreteTree & weakLearner = classifier[minClassifierIndex];
        usedClassifiers[minClassifierIndex] = true;

        weakLearner.setBeta(beta);


        //now reweight
        double normalizeFactor = 0;
        for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
        {
            // hypothesis_label is either 1 or -1
            const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);

            double &sampleWeight = weights[trainingSampleIndex];
            if (hypothesisLabel != classLabels[trainingSampleIndex])
            {
                // error increases when predicted label does not match the data label
                //errorCalc += sampleWeight;
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
        for (size_t l = 0; l < _trainData->getNumExamples(); ++l)
        {
            weights[l] /= normalizeFactor;
        }
    }
    for (size_t classifierIndex = classifier.size()-1; classifierIndex == 0; -- classifierIndex){
        if (usedClassifiers[classifierIndex] == false)
            classifier.erase(classifier.begin()+classifierIndex);
    }

}

//***************************************************************************************************
void AdaboostLearner::trainOcclusionClassifier(std::vector<WeakDiscreteTree> & classifier, double C_value, std::vector<double> old_weights)
{

    const int shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor();
    const int featHeight = (_trainData->getModelWindow().y()/shrinking_factor) -1;
    const int featWidth  = (_trainData->getModelWindow().x()/shrinking_factor) -1;
    std::vector<WeakDiscreteTree> classifier_copy;
    bool recursive =  Parameters::getParameter<bool>("train.recursiveOcclusion");
    bool resetBeta =  Parameters::getParameter<bool>("train.resetBeta");
    bool removeHurtingFeatures = Parameters::getParameter<bool>("train.removeHurtingFeatures");
    if (!recursive){
        classifier_copy = classifier;

    }

    const string saveProblemFile = Parameters::getParameter<string>("train.svmSaveProblemFile");
    const string frankenType = Parameters::getParameter<string>("train.frankenType");
    const int desired_featurePool_size = Parameters::getParameter<int> ("train.featuresPoolSize");
    size_t no_of_models =-1;

    if (frankenType == "up"){
        no_of_models = (_trainData->getModelWindow().y()/2)/shrinking_factor;
    }
    else
    {
        no_of_models = (_trainData->getModelWindow().x()/2)/shrinking_factor;
    }

    for (int i = 1 ; i< (1 + (no_of_models)); ++i){
        if (!recursive){
            classifier = classifier_copy;
        }
        std::vector<double> scores(_trainData->getNumExamples(), 0);
        TrainingData::point_t this_model_window(0,0);
        if (frankenType == "up"){
            TrainingData::point_t tmp_model_window(
                        _trainData->getModelWindow().x(),
                        _trainData->getModelWindow().y()-(i*shrinking_factor));
            this_model_window = tmp_model_window;

        }else{
            TrainingData::point_t tmp_model_window(
                        _trainData->getModelWindow().x()-(i*shrinking_factor),
                        _trainData->getModelWindow().y());
            this_model_window = tmp_model_window;

        }

        filterFeatures(*_trainData->getFeaturesConfigurations(), _trainData->_validFeatures,this_model_window,desired_featurePool_size);
        std::cout << "Crop: " << i*shrinking_factor << flush << std::endl;
        const int decisionTreeDepth = Parameters::getParameter<int>("train.decisionTreeDepth");
        MinOrMaxFeaturesResponsesSharedPointer
                maxvs(new MinOrMaxFeaturesResponses(_trainData->getFeaturesPoolSize())),
                minvs(new MinOrMaxFeaturesResponses(_trainData->getFeaturesPoolSize()));

        calcMinMaxFeatureResponses(_trainData, minvs, maxvs);

        ModelIO modelWriter(_verbose);
        const std::string
                datasetName = Parameters::getParameter<std::string>("train.trainSetName"),
                trainedModelName = "Model created via boosted_learning";
        modelWriter.initWrite(datasetName,
                              doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                              trainedModelName,
                              _trainData->getModelWindow(), _trainData->getObjectWindow());

        //remove features from classifier that are in the current occlusion area
        std::vector<size_t> deleteFeatures;
        if (frankenType == "up"){
            for (size_t k = 0; k< classifier.size(); ++k){
                WeakDiscreteTree & treenode = classifier[k];
                const TreeNodePtr root = treenode.getRoot();
                const TreeNodePtr left = root->left;
                const TreeNodePtr right = root->right;
                if (((root->_feature.y + root->_feature.height) > (featHeight - i) )||
                        ((left->_feature.y + left->_feature.height) > (featHeight - i)) ||
                        ((right->_feature.y + right->_feature.height) > (featHeight - i) ))
                    deleteFeatures.push_back(k);
            }

        }else{
            for (size_t k = 0; k< classifier.size(); ++k){
                WeakDiscreteTree & treenode = classifier[k];
                const TreeNodePtr root = treenode.getRoot();
                const TreeNodePtr left = root->left;
                const TreeNodePtr right = root->right;
                if (((root->_feature.x + root->_feature.width) > (featWidth - i) )||
                        ((left->_feature.x + left->_feature.width) > (featWidth - i)) ||
                        ((right->_feature.x + right->_feature.width) > (featWidth - i) ))
                    deleteFeatures.push_back(k);
            }


        }
        sort (deleteFeatures.begin(), deleteFeatures.end());

        //delete features starting from the back not to mess up the vector
        for(int k=deleteFeatures.size() - 1; k >= 0; k--){
            classifier.erase(classifier.begin() + deleteFeatures[k]);
        }
        std::cout << "remaining Features for cropping: " << i*shrinking_factor << " is " << classifier.size() << "\n";
        size_t iterationsToGo = _numIterations - classifier.size();
        std::cout << "newFeatures to train: " << iterationsToGo << std::endl;
        std::cout << "now retrain  classifier\n";
        //trainONGivenClassifier(classifier);
        //iterationsToGo = _numIterations - classifier.size();
        //std::cout << "newFeatures to train after removing hurting features: " << iterationsToGo << std::endl;

        //-------------------------reweight classifier using svm--------------
        const bool useSVM = Parameters::getParameter<bool>("train.useSVM");
        if (useSVM){
            string saveThisProblemFile = saveProblemFile;
            if (saveProblemFile != "")
                saveThisProblemFile = saveProblemFile + "crop_"+boost::lexical_cast<std::string>(i*4);
            learn_weights_via_svm(classifier, C_value, saveThisProblemFile);
        }

        //---------------------------------------------------------------------
        //-------------------------------INITIALIZE WEIGHTS---------------------
        WeakDiscreteTreeLearner::weights_t weights(_trainData->getNumExamples());
        std::vector<int> classLabels(_trainData->getNumExamples());

        for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
        {
            classLabels[trainingSampleIndex] = _trainData->getClassLabel(trainingSampleIndex);

            if (_trainData->getClassLabel(trainingSampleIndex) == -1)
            {
                weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumNegativeExamples());
            }
            else
            {
                weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumPositiveExamples());
            }
        } // end of "for each training example"

        //----------------------------------REWEIGHT DATA-----------------------------------
        std::vector<int> invalidClassifiers;
        for (size_t classifierIndex = 0; classifierIndex < classifier.size(); ++ classifierIndex){
            WeakDiscreteTree & weakLearner = classifier[classifierIndex];

            double normalizeFactor = 0;
            //first find new beta

            if (resetBeta&& !useSVM){

                double incorrectweight=0;
                for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
                {
                    // hypothesis_label is either 1 or -1
                    const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
                    const int realLabel = _trainData->getClassLabel(trainingSampleIndex);
                    if (hypothesisLabel != realLabel)
                    {
                        incorrectweight += weights[trainingSampleIndex];
                    }
                }
                double error = incorrectweight;
                double beta =0;
                if (error > 0.5 && removeHurtingFeatures){
                    invalidClassifiers.push_back(classifierIndex);
                    continue;
                   }
                //update beta
                //
                if (error < 1e-12)
                {
                    beta = 0;
                }
                else
                {
                    beta = 0.5 * log((1.0 - error) / error);
                }
                //compare betas
                //std::cout << "old beta: " << weakLearner.getBeta() << " new beta: " << beta << std::endl;
                weakLearner.setBeta(beta);
            }
            modelWriter.addStage(weakLearner);
            //now reweight
            for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
            {
                // hypothesis_label is either 1 or -1
                const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
                scores[trainingSampleIndex] += hypothesisLabel * weakLearner.getBeta();

                double &sampleWeight = weights[trainingSampleIndex];
                if (hypothesisLabel != classLabels[trainingSampleIndex])
                {
                    // error increases when predicted label does not match the data label
                    //errorCalc += sampleWeight;
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
            for (size_t l = 0; l < _trainData->getNumExamples(); ++l)
            {
                weights[l] /= normalizeFactor;
         //       std::cout << "old weight: " << old_weights[l] << " and now: " << weights[l] << std::endl;
            }
        }
        /*(for (size_t l = 0; l < _trainData->getNumExamples(); ++l)
        {
            if (fabs(old_weights[l]- weights[l]) > std::numeric_limits<double>::epsilon())
                std::cout << "old weight: " << old_weights[l] << " and now: " << weights[l] << std::endl;
        }

        throw runtime_error("debugging");
        */
        // delete invalidClassifiers
        sort (invalidClassifiers.begin(), invalidClassifiers.end());

        //delete features starting from the back not to mess up the vector
        for(int k=invalidClassifiers.size() - 1; k >= 0; k--){
            classifier.erase(classifier.begin() + invalidClassifiers[k]);
        }
        std::cout << "remaining Features for after removing bad ones: " << i*shrinking_factor << " is " << classifier.size() << "\n";
        iterationsToGo = _numIterations - classifier.size();
        std::cout << "new Features to train now: " << iterationsToGo << std::endl;

        //---------------------------------ADABOOST MAINLOOP---------------------------
        // numIterations define the number of weak classifier in the boosted strong classifier
        for(size_t iterationsCounter = 0; iterationsCounter < iterationsToGo; iterationsCounter +=1)
        {

            WeakDiscreteTreeLearner weakLearner(_verbose, decisionTreeDepth, -1,
                                                _trainData, classLabels, minvs, maxvs);
            //HACK remove true
            // just done to train the compound classifier without tau
            if (!recursive or true){
                weakLearner.setPushBias(0);
            }
            //HACK remove : recursi
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

            ////classification ================================:
            //int truePositives = 0, falsePositives = 0, falseNegatives = 0, trueNegatives = 0;
            //
            //for (size_t i = 0; i < _trainData->getNumExamples(); ++i)
            //{
            //	if (scores[i] >= 0)
            //	{
            //		if (_trainData->getClassLabel(i) == 1)
            //		{
            //			truePositives++;
            //		}
            //		else
            //		{
            //			falsePositives++;
            //		}
            //	}
            //	else
            //	{
            //		if (_trainData->getClassLabel(i) == 1)
            //		{
            //			falseNegatives++;
            //		}
            //		else
            //		{
            //			trueNegatives++;
            //		}
            //	}
            //}
            classifier.push_back(weakLearner); // converted from WeakDiscreteTreeLearner to WeakDiscreteTree
            modelWriter.addStage(weakLearner);

        }

        const string temporary_filename = _outputModelFileName + "crop_"+boost::lexical_cast<std::string>(i*4);
        modelWriter.write(temporary_filename);
    }
    return;
}


//***************************************************************************************************
void AdaboostLearner::trainOcclusionClassifier_fill_in(std::vector<WeakDiscreteTree> & classifier, double C_value, std::vector<double> old_weights)
{

    const int shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor();
    const int featHeight = (_trainData->getModelWindow().y()/shrinking_factor) -1;
    const int featWidth  = (_trainData->getModelWindow().x()/shrinking_factor) -1;
    std::vector<WeakDiscreteTree> classifier_copy;
    bool recursive =  Parameters::getParameter<bool>("train.recursiveOcclusion");
    bool resetBeta =  Parameters::getParameter<bool>("train.resetBeta");
    bool removeHurtingFeatures = Parameters::getParameter<bool>("train.removeHurtingFeatures");
    if (!recursive){
        classifier_copy = classifier;

    }

    const string saveProblemFile = Parameters::getParameter<string>("train.svmSaveProblemFile");
    const string frankenType = Parameters::getParameter<string>("train.frankenType");
    const int desired_featurePool_size = Parameters::getParameter<int> ("train.featuresPoolSize");
    size_t no_of_models =-1;

    if (frankenType == "up"){
        no_of_models = (_trainData->getModelWindow().y()/2)/shrinking_factor;
    }
    else
    {
        no_of_models = (_trainData->getModelWindow().x()/2)/shrinking_factor;
    }

    for (int i = 1 ; i< (1 + (no_of_models)); ++i){
        if (!recursive){
            classifier = classifier_copy;
        }
        std::vector<double> scores(_trainData->getNumExamples(), 0);
        TrainingData::point_t this_model_window(0,0);
        if (frankenType == "up"){
            TrainingData::point_t tmp_model_window(
                        _trainData->getModelWindow().x(),
                        _trainData->getModelWindow().y()-(i*shrinking_factor));
            this_model_window = tmp_model_window;

        }else{
            TrainingData::point_t tmp_model_window(
                        _trainData->getModelWindow().x()-(i*shrinking_factor),
                        _trainData->getModelWindow().y());
            this_model_window = tmp_model_window;

        }

        filterFeatures(*_trainData->getFeaturesConfigurations(), _trainData->_validFeatures,this_model_window,desired_featurePool_size);
        std::cout << "Crop: " << i*shrinking_factor << flush << std::endl;
        const int decisionTreeDepth = Parameters::getParameter<int>("train.decisionTreeDepth");
        MinOrMaxFeaturesResponsesSharedPointer
                maxvs(new MinOrMaxFeaturesResponses(_trainData->getFeaturesPoolSize())),
                minvs(new MinOrMaxFeaturesResponses(_trainData->getFeaturesPoolSize()));

        calcMinMaxFeatureResponses(_trainData, minvs, maxvs);

        ModelIO modelWriter(_verbose);
        const std::string
                datasetName = Parameters::getParameter<std::string>("train.trainSetName"),
                trainedModelName = "Model created via boosted_learning";
        modelWriter.initWrite(datasetName,
                              doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                              trainedModelName,
                              _trainData->getModelWindow(), _trainData->getObjectWindow());

        //remove features from classifier that are in the current occlusion area
        std::vector<size_t> deleteFeatures;
        if (frankenType == "up"){
            for (size_t k = 0; k< classifier.size(); ++k){
                WeakDiscreteTree & treenode = classifier[k];
                const TreeNodePtr root = treenode.getRoot();
                const TreeNodePtr left = root->left;
                const TreeNodePtr right = root->right;
                if (((root->_feature.y + root->_feature.height) > (featHeight - i) )||
                        ((left->_feature.y + left->_feature.height) > (featHeight - i)) ||
                        ((right->_feature.y + right->_feature.height) > (featHeight - i) ))
                    deleteFeatures.push_back(k);
            }

        }else{
            for (size_t k = 0; k< classifier.size(); ++k){
                WeakDiscreteTree & treenode = classifier[k];
                const TreeNodePtr root = treenode.getRoot();
                const TreeNodePtr left = root->left;
                const TreeNodePtr right = root->right;
                if (((root->_feature.x + root->_feature.width) > (featWidth - i) )||
                        ((left->_feature.x + left->_feature.width) > (featWidth - i)) ||
                        ((right->_feature.x + right->_feature.width) > (featWidth - i) ))
                    deleteFeatures.push_back(k);
            }


        }
        sort (deleteFeatures.begin(), deleteFeatures.end());

        //delete features starting from the back not to mess up the vector
        //for(int k=deleteFeatures.size() - 1; k >= 0; k--){
        //    classifier.erase(classifier.begin() + deleteFeatures[k]);
        //}
        std::cout << "remaining Features for cropping: " << i*shrinking_factor << " is " << classifier.size()-deleteFeatures.size()<< "\n";
        size_t iterationsToGo = deleteFeatures.size();
        std::cout << "newFeatures to train: " << iterationsToGo << std::endl;
        std::cout << "now retrain  classifier\n";
        //trainONGivenClassifier(classifier);
        //iterationsToGo = _numIterations - classifier.size();
        //std::cout << "newFeatures to train after removing hurting features: " << iterationsToGo << std::endl;

        //-------------------------reweight classifier using svm--------------
        const bool useSVM = Parameters::getParameter<bool>("train.useSVM");
        if (useSVM){
            string saveThisProblemFile = saveProblemFile;
            if (saveProblemFile != "")
                saveThisProblemFile = saveProblemFile + "crop_"+boost::lexical_cast<std::string>(i*4);
            learn_weights_via_svm(classifier, C_value, saveThisProblemFile);
        }

        //---------------------------------------------------------------------
        //-------------------------------INITIALIZE WEIGHTS---------------------
        WeakDiscreteTreeLearner::weights_t weights(_trainData->getNumExamples());
        std::vector<int> classLabels(_trainData->getNumExamples());

        for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
        {
            classLabels[trainingSampleIndex] = _trainData->getClassLabel(trainingSampleIndex);

            if (_trainData->getClassLabel(trainingSampleIndex) == -1)
            {
                weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumNegativeExamples());
            }
            else
            {
                weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumPositiveExamples());
            }
        } // end of "for each training example"

        //----------------------------------REWEIGHT DATA and Train-----------------------------------
        std::vector<int> invalidClassifiers;
        int numinvalidFeatures=0;
        for (size_t classifierIndex = 0; classifierIndex < classifier.size(); ++ classifierIndex){
            if (find(deleteFeatures.begin(), deleteFeatures.end(), classifierIndex) != deleteFeatures.end()){
                //retrain this feature
                WeakDiscreteTreeLearner new_weakLearner(_verbose, decisionTreeDepth, -1,
                                                    _trainData, classLabels, minvs, maxvs);
                if (!recursive){
                    new_weakLearner.setPushBias(0);
                }
                const double error = new_weakLearner.buildBalancedTree(weights);

                if (error >= 0.5)
                {
                    throw runtime_error("TRAINING STOP: Not possible to reduce error anymore");
                }

                classifier[classifierIndex] = new_weakLearner;

            }

            WeakDiscreteTree & weakLearner = classifier[classifierIndex];

            double normalizeFactor = 0;
            //first find new beta

            if (resetBeta&& !useSVM){

                double incorrectweight=0;
                for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
                {
                    // hypothesis_label is either 1 or -1
                    const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
                    const int realLabel = _trainData->getClassLabel(trainingSampleIndex);
                    if (hypothesisLabel != realLabel)
                    {
                        incorrectweight += weights[trainingSampleIndex];
                    }
                }
                double error = incorrectweight;
                double beta =0;
                if (error > 0.5 && removeHurtingFeatures){
                    numinvalidFeatures +=1;
                    WeakDiscreteTreeLearner new_weakLearner(_verbose, decisionTreeDepth, -1,
                                                        _trainData, classLabels, minvs, maxvs);
                    if (!recursive){
                        new_weakLearner.setPushBias(0);
                    }
                    const double error = new_weakLearner.buildBalancedTree(weights);

                    if (error >= 0.5)
                    {
                        throw runtime_error("TRAINING STOP: Not possible to reduce error anymore");
                    }

                    classifier[classifierIndex] = new_weakLearner;
                   }
                //update beta
                //
                if (error < 1e-12)
                {
                    beta = 0;
                }
                else
                {
                    beta = 0.5 * log((1.0 - error) / error);
                }
                //compare betas
                //std::cout << "old beta: " << weakLearner.getBeta() << " new beta: " << beta << std::endl;
                weakLearner.setBeta(beta);
            }
            modelWriter.addStage(weakLearner);
            //now reweight
            for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
            {
                // hypothesis_label is either 1 or -1
                const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
                scores[trainingSampleIndex] += hypothesisLabel * weakLearner.getBeta();

                double &sampleWeight = weights[trainingSampleIndex];
                if (hypothesisLabel != classLabels[trainingSampleIndex])
                {
                    // error increases when predicted label does not match the data label
                    //errorCalc += sampleWeight;
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
            for (size_t l = 0; l < _trainData->getNumExamples(); ++l)
            {
                weights[l] /= normalizeFactor;
         //       std::cout << "old weight: " << old_weights[l] << " and now: " << weights[l] << std::endl;
            }

        }
        std::cout << "number of features with training error > 0.5: " << numinvalidFeatures << std::endl;

        const string temporary_filename = _outputModelFileName + "fill_in_crop_"+boost::lexical_cast<std::string>(i*4);
        modelWriter.write(temporary_filename);
    }
    return;
}




//***************************************************************************************************
void AdaboostLearner::train_naive_OcclusionClassifier(std::vector<WeakDiscreteTree> & classifier)
{

    const int shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor();
    const int featHeight = (_trainData->getModelWindow().y()/shrinking_factor) -1;
    const int featWidth  = (_trainData->getModelWindow().x()/shrinking_factor) -1;

    bool removeHurtingFeatures = Parameters::getParameter<bool>("train.removeHurtingFeatures");
    bool resetBeta =  Parameters::getParameter<bool>("train.resetBeta");

    const string frankenType = Parameters::getParameter<string>("train.frankenType");


    size_t no_of_models =-1;

    if (frankenType == "up"){
        no_of_models = (_trainData->getModelWindow().y()/2)/shrinking_factor;
    }
    else
    {
        no_of_models = (_trainData->getModelWindow().x()/2)/shrinking_factor;
    }

    for (int i = 1 ; i< (1 + (no_of_models)); ++i){


        TrainingData::point_t this_model_window(0,0);
        if (frankenType == "up"){
            TrainingData::point_t tmp_model_window(
                        _trainData->getModelWindow().x(),
                        _trainData->getModelWindow().y()-(i*shrinking_factor));
            this_model_window = tmp_model_window;

        }else{
            TrainingData::point_t tmp_model_window(
                        _trainData->getModelWindow().x()-(i*shrinking_factor),
                        _trainData->getModelWindow().y());
            this_model_window = tmp_model_window;

        }


        std::cout << "Crop: " << i*shrinking_factor << flush << std::endl;


        ModelIO modelWriter(_verbose);
        const std::string
                datasetName = Parameters::getParameter<std::string>("train.trainSetName"),
                trainedModelName = "Model created via boosted_learning";
        modelWriter.initWrite(datasetName,
                              doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                              trainedModelName,
                              _trainData->getModelWindow(), _trainData->getObjectWindow());

        //remove features from classifier that are in the current occlusion area
        std::vector<size_t> deleteFeatures;
        if (frankenType == "up"){
            for (size_t k = 0; k< classifier.size(); ++k){
                WeakDiscreteTree & treenode = classifier[k];
                const TreeNodePtr root = treenode.getRoot();
                const TreeNodePtr left = root->left;
                const TreeNodePtr right = root->right;
                if (((root->_feature.y + root->_feature.height) > (featHeight - i) )||
                        ((left->_feature.y + left->_feature.height) > (featHeight - i)) ||
                        ((right->_feature.y + right->_feature.height) > (featHeight - i) ))
                    deleteFeatures.push_back(k);
            }

        }else{
            for (size_t k = 0; k< classifier.size(); ++k){
                WeakDiscreteTree & treenode = classifier[k];
                const TreeNodePtr root = treenode.getRoot();
                const TreeNodePtr left = root->left;
                const TreeNodePtr right = root->right;
                if (((root->_feature.x + root->_feature.width) > (featWidth - i) )||
                        ((left->_feature.x + left->_feature.width) > (featWidth - i)) ||
                        ((right->_feature.x + right->_feature.width) > (featWidth - i) ))
                    deleteFeatures.push_back(k);
            }


        }
        sort (deleteFeatures.begin(), deleteFeatures.end());

        //delete features starting from the back not to mess up the vector
        for(int k=deleteFeatures.size() - 1; k >= 0; k--){
            classifier.erase(classifier.begin() + deleteFeatures[k]);
        }
        std::cout << "remaining Features for cropping: " << i*shrinking_factor << " is " << classifier.size() << "\n";
        size_t iterationsToGo = _numIterations - classifier.size();
        std::cout << "newFeatures to train: " << iterationsToGo << std::endl;
        std::cout << "now retrain  classifier\n";




        //---------------------------------------------------------------------
        //-------------------------------INITIALIZE WEIGHTS---------------------
        WeakDiscreteTreeLearner::weights_t weights(_trainData->getNumExamples());
        std::vector<int> classLabels(_trainData->getNumExamples());

        for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
        {
            classLabels[trainingSampleIndex] = _trainData->getClassLabel(trainingSampleIndex);

            if (_trainData->getClassLabel(trainingSampleIndex) == -1)
            {
                weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumNegativeExamples());
            }
            else
            {
                weights[trainingSampleIndex] = 1.0 / (2.0 * _trainData->getNumPositiveExamples());
            }
        } // end of "for each training example"

        //----------------------------------REWEIGHT DATA-----------------------------------
        std::vector<int> invalidClassifiers;
        for (size_t classifierIndex = 0; classifierIndex < classifier.size(); ++ classifierIndex){
            WeakDiscreteTree & weakLearner = classifier[classifierIndex];

            double normalizeFactor = 0;
            //first find new beta

            double incorrectweight=0;
            for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
            {
                // hypothesis_label is either 1 or -1
                const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);
                const int realLabel = _trainData->getClassLabel(trainingSampleIndex);
                if (hypothesisLabel != realLabel)
                {
                    incorrectweight += weights[trainingSampleIndex];
                }
            }
            double error = incorrectweight;
            double beta =0;
            if (error > 0.5 && removeHurtingFeatures){
                invalidClassifiers.push_back(classifierIndex);
                continue;
            }
            //update beta
            //
            if (error < 1e-12)
            {
                beta = 0;
            }
            else
            {
                beta = 0.5 * log((1.0 - error) / error);
            }
            //compare betas
            //std::cout << "old beta: " << weakLearner.getBeta() << " new beta: " << beta << std::endl;
            if (resetBeta)
                weakLearner.setBeta(beta);

            modelWriter.addStage(weakLearner);
            //now reweight
            for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
            {
                // hypothesis_label is either 1 or -1
                const int hypothesisLabel = weakLearner.classify(_trainData->getFeatureResponses(), trainingSampleIndex);


                double &sampleWeight = weights[trainingSampleIndex];
                if (hypothesisLabel != classLabels[trainingSampleIndex])
                {
                    // error increases when predicted label does not match the data label
                    //errorCalc += sampleWeight;
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
            for (size_t l = 0; l < _trainData->getNumExamples(); ++l)
            {
                weights[l] /= normalizeFactor;
         //       std::cout << "old weight: " << old_weights[l] << " and now: " << weights[l] << std::endl;
            }
        }

        sort (invalidClassifiers.begin(), invalidClassifiers.end());

        //delete features starting from the back not to mess up the vector
        for(int k=invalidClassifiers.size() - 1; k >= 0; k--){
            classifier.erase(classifier.begin() + invalidClassifiers[k]);
        }
        std::cout << "remaining Features for after removing bad ones: " << i*shrinking_factor << " is " << classifier.size() << "\n";
        iterationsToGo = _numIterations - classifier.size();
        std::cout << "new Features to train now: " << iterationsToGo << std::endl;


        const string temporary_filename = _outputModelFileName + "_reweight_remove_crop_"+boost::lexical_cast<std::string>(i*4);
        modelWriter.write(temporary_filename);
    }
    return;
}


void AdaboostLearner::train(bool last)
{

#if 0
    // test
    std::cout << "read protocol buffer ------------------------------------------------\n";
    ModelIO modelReader(_verbose);
    modelReader.initRead(_outputModelFileName);
    std::vector<WeakDiscreteTreeLearner> learner = modelReader.read();
    classify(learner);

    return;
#endif

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
#if 0
        if (iterationsCounter <= 1000){
            _trainData->filterFeatures(0,64,0,0);
            std::cout << "Training stage: " << iterationsCounter <<" bottom Crop: " << 64 << flush << std::endl;
        }
        else
        {
            int kk= (iterationsCounter - 1000) /15;
            int xx = std::max(64 - kk,0);
            std::cout << "Training stage: " << iterationsCounter <<" bottom Crop: " << xx << flush << std::endl;
            _trainData->filterFeatures(0,xx, 0, 0);
        }

#endif

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

    //-----------------------------find best C--
    //double C = 10000;
    double bestC = 1;
    //double smallestError = 1.0;
    //for (int i = 1; i < 13; ++i){
    //    C = C*0.1;
    //    learn_weights_via_svm(classifier, C);
    //    double this_error = classify(classifier);
    //    if (smallestError > this_error){
    //        bestC = C;
    //        smallestError = this_error;
    //        std::cout << "best value for C so far: " << bestC << std::endl;
    //    }
    //}

    //HACK--------FIX ME33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    bestC = 0.001;


    //-----------------------------end find best C












    // evaluate with test data
    if (_testData)
        classify(classifier);



    const bool enableFranken = Parameters::getParameter<bool>("train.enableFrankenClassifier");
    const bool fill_in = false;

    if (last && enableFranken)
    {
        if (fill_in)
            trainOcclusionClassifier_fill_in(classifier, bestC, weights);
        else
            trainOcclusionClassifier(classifier, bestC, weights);
    }

    const bool useSVM = Parameters::getParameter<bool>("train.useSVM");
    if (useSVM){
        ModelIO svm_modelWriter(_verbose);
        svm_modelWriter.initWrite(datasetName,
                                  doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                                  trainedModelName,
                                  _trainData->getModelWindow(), _trainData->getObjectWindow());
        const string saveProblemFile = Parameters::getParameter<string>("train.svmSaveProblemFile");
        learn_weights_via_svm(classifier, bestC, saveProblemFile);
        for (size_t classifier_index = 0; classifier_index < classifier.size();++ classifier_index){
            svm_modelWriter.addStage(classifier[classifier_index]);
        }
        svm_modelWriter.write(_outputModelFileName + ".svm_weights");
    }
    //train_naive_OcclusionClassifier(classifier);

    return;
}

void AdaboostLearner::train_real(bool last)
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
        //HACK: error set to 0
        const double error = weakLearner.buildBalancedTree_real(weights);

        double normalizeFactor = 0;

        //FIXME: howto check that now?
        //if (error >= 0.5)
        //{
        //    throw runtime_error("TRAINING STOP: Not possible to reduce error anymore");
        //}


        for (size_t trainingSampleIndex = 0; trainingSampleIndex < _trainData->getNumExamples(); ++trainingSampleIndex)
        {
            // hypothesis_label is either 1 or -1
            const double beta = weakLearner.classify_real(_trainData->getFeatureResponses(), trainingSampleIndex);

            scores[trainingSampleIndex] += beta;

            double &sampleWeight = weights[trainingSampleIndex];
            int label =0;
            if (classLabels[trainingSampleIndex] ==1)
                label = 1;
            else
                label =-1;
            // update weights
            sampleWeight *= exp(-label * beta);

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
        modelWriter.addStage_real(weakLearner);

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

    //-----------------------------find best C--
    //double C = 10000;
    double bestC = 1;
    //double smallestError = 1.0;
    //for (int i = 1; i < 13; ++i){
    //    C = C*0.1;
    //    learn_weights_via_svm(classifier, C);
    //    double this_error = classify(classifier);
    //    if (smallestError > this_error){
    //        bestC = C;
    //        smallestError = this_error;
    //        std::cout << "best value for C so far: " << bestC << std::endl;
    //    }
    //}

    //HACK--------FIX ME33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
    bestC = 0.001;

    for(size_t i = 0; i < classifier.size(); ++i){
        std::cout << "Cascade stage " << i << std::endl;

        std::cout << "stage.weight " << 1 << std::endl;
        std::cout << "stage.cascade_threshold " << -1 << std::endl;
        std::cout << "stage.weak_classifier.level1_node " << std::endl;


        const WeakDiscreteTree &tree = classifier[i];
        TreeNode_v2 node = tree._rootd2[0];
        node.print("\tlevel1_node");

        std::cout << "stage.weak_classifier.level2_true_node " << std::endl;

        node = tree._rootd2[1];
        std::cout << "left node:" << std::endl;
        node.print("\tlevel2_true_node");

        std::cout << "stage.weak_classifier.level2_false_node " << std::endl;

        node = tree._rootd2[2];
        std::cout << "right node:" << std::endl;
        node.print("\tlevel2_false_node");

    }
    //-----------------------------end find best C












    // evaluate with test data
    if (_testData)
        classify_real(classifier);



    const bool enableFranken = Parameters::getParameter<bool>("train.enableFrankenClassifier");
    const bool fill_in = false;

    if (last && enableFranken)
    {
        if (fill_in)
            trainOcclusionClassifier_fill_in(classifier, bestC, weights);
        else
            trainOcclusionClassifier(classifier, bestC, weights);
    }

    const bool useSVM = Parameters::getParameter<bool>("train.useSVM");
    if (useSVM){
        ModelIO svm_modelWriter(_verbose);
        svm_modelWriter.initWrite(datasetName,
                                  doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels,
                                  trainedModelName,
                                  _trainData->getModelWindow(), _trainData->getObjectWindow());
        const string saveProblemFile = Parameters::getParameter<string>("train.svmSaveProblemFile");
        learn_weights_via_svm(classifier, bestC, saveProblemFile);
        for (size_t classifier_index = 0; classifier_index < classifier.size();++ classifier_index){
            svm_modelWriter.addStage(classifier[classifier_index]);
        }
        svm_modelWriter.write(_outputModelFileName + ".svm_weights");
    }
    //train_naive_OcclusionClassifier(classifier);

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

void AdaboostLearner::learn_weights_via_svm(std::vector<WeakDiscreteTree> &classifier, double C_value, string saveName)
{

    printf("Preparing SVM training data\n");
    const size_t num_examples = _trainData->getNumExamples();
    const size_t num_neg_example = _trainData->getNumNegativeExamples();
    const size_t num_pos_example = _trainData->getNumPositiveExamples();
    const size_t num_weak_classifiers = classifier.size();
    problem svm_problem;
    svm_problem.bias = -1; // no bias, since 0 should be the cutting value
    svm_problem.n = num_weak_classifiers;
    svm_problem.l = num_examples;

    std::ofstream svm_problem_file;
    if (saveName != ""){
        svm_problem_file.open(saveName.c_str());
    }

    int positives_repetition = 1;
    if (num_neg_example > num_pos_example){
        positives_repetition = int(round(num_neg_example/ float(num_pos_example)));
    }
    std::cout << "positives samples are repeated by a factor of " << positives_repetition << std::endl;
    const size_t svm_num_examples = num_neg_example + num_pos_example*positives_repetition;




    struct feature_node *x_space;
    svm_problem.x = Malloc(struct feature_node *,svm_num_examples);
    svm_problem.y = Malloc(int,svm_num_examples);
    x_space = Malloc(struct feature_node,svm_num_examples*num_weak_classifiers+svm_num_examples);



    size_t j =0;
    size_t svm_sample_index = 0;
    for (size_t example_index=0; example_index< num_examples; ++example_index)
    {
        const int example_label = _trainData->getClassLabel(example_index);
        int repeat_sample =1;
        if (example_label ==1)
            repeat_sample = positives_repetition;
        for (int i = 0; i< repeat_sample; ++i){
            svm_problem.y[svm_sample_index] = example_label;
            svm_problem.x[svm_sample_index] = &x_space[j];

            if (saveName != ""){
                svm_problem_file << example_label << " ";
            }


            for (size_t weak_classifier_index = 0; weak_classifier_index< num_weak_classifiers; ++weak_classifier_index)
            {
                //std::cout << "weak_classifier_index: " << weak_classifier_index<< std::endl;
                //std::cout << "example_index: " << example_index<< std::endl;
                //std::cout << "no examples: " << num_examples << std::endl;
                const float response = classifier[weak_classifier_index].classify(_trainData->getFeatureResponses(),example_index);


                x_space[j].index = weak_classifier_index+1; //index is 1 based
                x_space[j].value = response;
                j++;
                if (saveName != ""){
                    svm_problem_file << weak_classifier_index+1 << ":" << response << " ";
                }
            } // end of "for each weak classifier"

            x_space[j].index = -1;
            j++;
            svm_sample_index++;
            if (saveName != ""){
                svm_problem_file << "\n";
            }
        }// end of repeat_sample



    } // end of "for each training example"



    std::cout<< "memorysize: "<< svm_num_examples*num_weak_classifiers+svm_num_examples << " data written into it: " << j << std::endl;


    parameter training_parameters;
    training_parameters.C = C_value;
    training_parameters.solver_type = L2R_L2LOSS_SVC_DUAL;
    training_parameters.nr_weight = 0;
    training_parameters.weight = NULL;
    training_parameters.weight_label = NULL;
    training_parameters.eps = 0.001;




    printf("Launching SVM training\n");

    model *model_p= NULL;
    model_p = ::train(&svm_problem, &training_parameters);

    {
        // we set the weak classifier weights based on the learned model
        for (size_t weak_classifier_index = 0; weak_classifier_index< num_weak_classifiers; ++weak_classifier_index)
        {
            const float beta = model_p->w[weak_classifier_index];
            classifier[weak_classifier_index].setBeta(beta);
        } // end of "for each weak classifier"


    }

    destroy_param(&training_parameters);
    free(x_space);
    free(svm_problem.x);
    free(svm_problem.y);
    free_and_destroy_model(&model_p);



} // end of AdaboostLearner::learn_weight_via_svm(std::vector<WeakDiscreteTree> &classifier)

size_t filterFeatures(const Features &featuresConfigurations, std::vector<bool> &validFeatures,const TrainingData::point_t &modelWindow, size_t maxFeatures)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor ,
            modelHeight = modelWindow.y() / shrinking_factor;
    //set the filter according to the shrinking factor
    size_t unused_features = 0;
    size_t used_features = 0;


    validFeatures.clear();
    size_t no_featureconf = featuresConfigurations.size();
    validFeatures.resize(no_featureconf, false);
    unused_features = 0;// no_featureconf;
    //std::cout << validFeatures.size() << std::endl;
    //std::cout << featuresConfigurations.size() << std::endl;


    for (size_t i =0; i< featuresConfigurations.size(); ++i)
    {
        const Feature &ff = featuresConfigurations[i];

        //bottom or right
        if (((ff.y + ff.height) >= modelHeight)
                or ((ff.x + ff.width)>= modelWidth ))
        {
            validFeatures[i] = false;
            unused_features +=1;
        }
        else{

            validFeatures[i] = true;
            used_features +=1;
            if (maxFeatures != 0 && used_features == maxFeatures){
                unused_features = featuresConfigurations.size()-used_features;
                break;
            }
        }
    }
    // std::cout << "features that are set to valid: " << used_features << std::endl;

    return unused_features;
}


#if 0
void AdaboostLearner::learn_weights_via_svm(std::vector<WeakDiscreteTree> &classifier, double C_value)
{
    printf("Preparing SVM training data\n");
    const size_t num_examples = _trainData->getNumExamples();
    const size_t num_neg_example = _trainData->getNumNegativeExamples();
    const size_t num_pos_example = _trainData->getNumPositiveExamples();
    const size_t num_weak_classifiers = classifier.size();
    problem svm_problem;
    svm_problem.bias = -1; // no bias, since 0 should be the cutting value
    svm_problem.n = num_weak_classifiers;
    svm_problem.l = num_examples;

    int positives_repetition = 1;
    if (num_neg_example > num_pos_example){
        positives_repetition = int(round(num_neg_example/ float(num_neg_example)));


    }
    const size_t svm_num_examples = num_neg_example + num_pos_example*positives_repetition;




    struct feature_node *x_space;
    svm_problem.x = Malloc(struct feature_node *,svm_num_examples);
    svm_problem.y = Malloc(int,num_examples);
    x_space = Malloc(struct feature_node,svm_num_examples*num_weak_classifiers+svm_num_examples);



    size_t j =0;
    size_t svm_sample_index = 0;
    for (size_t example_index=0; example_index< num_examples; ++example_index)
    {
        const int example_label = _trainData->getClassLabel(example_index);
        int repeat_sample =1;
        if (example_label ==1)
            repeat_sample = positives_repetition;
        for (int i = 0; i< repeat_sample; ++i){
            svm_problem.y[svm_sample_index] = example_label;
            svm_problem.x[svm_sample_index] = &x_space[j];



            for (size_t weak_classifier_index = 0; weak_classifier_index< num_weak_classifiers; ++weak_classifier_index)
            {
                const float response = classifier[weak_classifier_index].classify(_trainData->getFeatureResponses(),example_index);


                x_space[j].index = weak_classifier_index+1; //index is 1 based
                x_space[j].value = response;
                j++;
            } // end of "for each weak classifier"

            x_space[j].index = -1;
            j++;
            svm_sample_index++;
        }// end of repeat_sample



    } // end of "for each training example"



    std::cout<< "memorysize: "<< svm_num_examples*num_weak_classifiers+svm_num_examples << " data written into it: " << j << std::endl;


    parameter training_parameters;
    training_parameters.C = C_value;
    training_parameters.solver_type = L2R_L2LOSS_SVC_DUAL;
    training_parameters.nr_weight = 0;
    training_parameters.weight = NULL;
    training_parameters.weight_label = NULL;

    if(training_parameters.solver_type == L2R_LR || training_parameters.solver_type == L2R_L2LOSS_SVC)
    {
        training_parameters.eps = 0.01;
    }
    else if(training_parameters.solver_type == L2R_L2LOSS_SVC_DUAL
            || training_parameters.solver_type == L2R_L1LOSS_SVC_DUAL
            || training_parameters.solver_type == MCSVM_CS
            || training_parameters.solver_type == L2R_LR_DUAL)
    {
        training_parameters.eps = 0.1;
    }
    else if(training_parameters.solver_type == L1R_L2LOSS_SVC || training_parameters.solver_type == L1R_LR)
    {
        training_parameters.eps = 0.01;
    }


    printf("Launching SVM training\n");

    model *model_p= NULL;
    model_p = ::train(&svm_problem, &training_parameters);

    {
        // we set the weak classifier weights based on the learned model
        for (size_t weak_classifier_index = 0; weak_classifier_index< num_weak_classifiers; ++weak_classifier_index)
        {
            const float beta = model_p->w[weak_classifier_index];
            classifier[weak_classifier_index].setBeta(beta);
        } // end of "for each weak classifier"


    }

    destroy_param(&training_parameters);
    free(x_space);
    free(svm_problem.x);
    free(svm_problem.y);
    free_and_destroy_model(&model_p);



} // end of AdaboostLearner::learn_weight_via_svm(std::vector<WeakDiscreteTree> &classifier)
#endif


} // end of namespace boosted_learning
