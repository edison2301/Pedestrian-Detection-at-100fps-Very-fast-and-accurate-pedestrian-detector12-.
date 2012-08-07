#ifndef __AdaboostLearner_H
#define __AdaboostLearner_H


#include "TrainingData.hpp"
#include "Feature.hpp"
#include "WeakDiscreteTreeLearner.hpp"

#include "bootstrapping_lib.hpp"

#include <iostream>
#include <vector>
#include <string>

namespace boosted_learning {

using namespace std;

//-----------------------------------
//#define FULL_MODE	1000 //Compute all available features
//#define RANDOM_MODE 1001 //This will select random reatures during X time.
//#define DISCRETE_ADA	1
//#define GENTLE_ADA		2
//-----------------------------------

//-----------------------------------
struct configTraining
{
    int typeAdaboost; // default, GENTLE_ADA, other=DISCRETE_ADA
    int minimumSizeW;  //minimum size of features (width).
    int minimumSizeH;  //minimum size of features (height).
    int percentOfFeat;	//This is only for RANDOM_MODE seconds looking for a specific feature.

    string outputModelFileName;//name of file.

};
//-----------------------------------


class AdaboostLearner
{

public:
    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef Eigen::VectorXf feature_vector_t;
    //------------------------------------------------------

    AdaboostLearner(int verbose, TrainingData::shared_ptr data);
    ~AdaboostLearner();
    void setTestData(LabeledData::shared_ptr data);
    void setValidationData(LabeledData::shared_ptr data);

    //------------------------------------------------------
    double classify(const std::vector<WeakDiscreteTree> & classifier);

    void train(bool last=false); ///< Call the training of N features (Iterations)

    int _verbose;

    void setNumIterations(const int i);

    void setOutputModelFileName(const std::string update);
    const TrainingData::shared_ptr getTrainData() const;

    std::string getOuputModelFileName() const;

protected:

    int getFeatResponse(const integral_channels_t &integralImage, const Feature &feat);

    void doBootStrapping(std::vector<std::string> &filenamesBG,  std::vector<double> & scores,
                         std::vector<WeakDiscreteTreeLearner> &classifier, std::vector<double> & weights,
                         std::vector<int> & cl, std::vector<int> &maxvs, std::vector<int>  &minvs);

    void getParameters(); ///< get all predefined parameters

    int _numIterations;
    string _typeAdaboost; ///<  default, GENTLE_ADA, other=DISCRETE_ADA
    string _outputModelFileName; ///< name of file.

    TrainingData::shared_ptr _trainData; ///< this variable is used to hold all the information of the training set.
    LabeledData::shared_ptr _testData; ///< this variable is used to hold all the information of the testing set.
    LabeledData::shared_ptr _validationData; ///< this variable is used to hold all the information of the training set.

    void recalculateWeights(const LabeledData &data,
                            const std::vector<WeakDiscreteTree> & learner,
                            std::vector<double> & weights, std::vector<double> & scores);


};
void calcMinMaxFeatureResponses(TrainingData::shared_ptr trainData, MinOrMaxFeaturesResponsesSharedPointer minvs,
                                                                 MinOrMaxFeaturesResponsesSharedPointer maxvs);

} // end of namespace boosted_learning

#endif // __AdaboostLearner_H

