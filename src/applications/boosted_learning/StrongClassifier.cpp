#include "StrongClassifier.hpp"

#include "ModelIO.hpp"
#include "Parameters.hpp"

#include "integral_channels_helpers.hpp"
#include "video_input/ImagesFromDirectory.hpp" // for the open_image helper method


#include <boost/filesystem.hpp>
#include <boost/array.hpp>
#include <boost/progress.hpp>

#include <cassert>

namespace boosted_learning {


namespace gil = boost::gil;
using boost::shared_ptr;

StrongClassifier::StrongClassifier(const std::vector<WeakDiscreteTreeLearner> & learners)
{
    //copy data

    std::copy(learners.begin(), learners.end(), std::back_inserter(_learners));

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


void StrongClassifier::classify(const LabeledData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade) const
{
    int tpp = 0;
    int fpp = 0;
    int fnn = 0;
    int tnn = 0;
#pragma omp parallel for reduction(+:tpp, fpp, fnn, tnn) default(none) shared(data, std::cout)

    for (size_t i = 0; i < data.getNumExamples(); ++i)
    {
        double res = 0;
        bool goingthrough = true;

        for (size_t l = 0; l < _learners.size(); ++l)
        {

            const int h = _learners[l].classify(data.getIntegralImage(i));
            res += h * _learners[l].getBeta();

            if (use_cascade and res < _learners[l]._cascadeThreshold)
            {
                goingthrough = false;
                break;
            }

        }

        //res = classify(data.getIntImage(i));
        if (goingthrough)
        {
            if (data.getClassLabel(i) == 1)
            {
                tpp ++;
            }
            else
            {
                fpp ++;
                //#pragma omp critical
                //{
                //    std::cout << "fp: " << data.getFilename(i) << " pos: (" << data.getX(i) << "," << data.getY(i) << ")" <<std::endl;
                //}
            }
        }
        else
        {
            if (data.getClassLabel(i) == -1)
            {
                tnn++;
            }
            else
            {
                fnn++;
                //#pragma omp critical
                //{
                //    std::cout << "fn: " << data.getFilename(i) << " pos: (" << data.getX(i) << "," << data.getY(i) << ")" <<std::endl;
                //}
            }
        }

    }

    tn = tnn;
    tp = tpp;
    fp = fpp;
    fn = fnn;

    return;
}

void StrongClassifier::classify_real(const LabeledData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade) const
{
    int tpp = 0;
    int fpp = 0;
    int fnn = 0;
    int tnn = 0;
#pragma omp parallel for reduction(+:tpp, fpp, fnn, tnn) default(none) shared(data, std::cout)

    for (size_t i = 0; i < data.getNumExamples(); ++i)
    {
        double res = 0;
        bool goingthrough = true;

        for (size_t l = 0; l < _learners.size(); ++l)
        {

            const double h = _learners[l].classify_real(data.getIntegralImage(i));
            res += h;

            if (use_cascade and res < _learners[l]._cascadeThreshold)
            {
                goingthrough = false;
                break;
            }

        }

        //res = classify(data.getIntImage(i));
        if (goingthrough)
        {
            if (data.getClassLabel(i) == 1)
            {
                tpp ++;
            }
            else
            {
                fpp ++;
                //#pragma omp critical
                //{
                //    std::cout << "fp: " << data.getFilename(i) << " pos: (" << data.getX(i) << "," << data.getY(i) << ")" <<std::endl;
                //}
            }
        }
        else
        {
            if (data.getClassLabel(i) == -1)
            {
                tnn++;
            }
            else
            {
                fnn++;
                //#pragma omp critical
                //{
                //    std::cout << "fn: " << data.getFilename(i) << " pos: (" << data.getX(i) << "," << data.getY(i) << ")" <<std::endl;
                //}
            }
        }

    }

    tn = tnn;
    tp = tpp;
    fp = fpp;
    fn = fnn;

    return;
}

void StrongClassifier::classify(const TrainingData &data, int &tp, int &fp, int &fn, int &tn, const bool use_cascade) const
{

    int tpp = 0;
    int fpp = 0;
    int fnn = 0;
    int tnn = 0;
#pragma omp parallel for reduction(+:tpp, fpp, fnn, tnn) default(none) shared(data, std::cout)

    for (size_t i = 0; i < data.getNumExamples(); ++i)
    {
        double res = 0;
        bool goingthrough = true;

        for (size_t l = 0; l < _learners.size(); ++l)
        {
            // h is the response of the weak classifier
            const int h = _learners[l].classify(data.getFeatureResponses(), i);
            res += h * _learners[l].getBeta();

            if (use_cascade and res < _learners[l]._cascadeThreshold)
            {
                goingthrough = false;
                break;
            }

        }

        //res = classify(data.getIntImage(i));
        if (goingthrough)
        {
            if (data.getClassLabel(i) == 1)
            {
                tpp ++;
            }
            else
            {
                fpp ++;
                //#pragma omp critical
                //{
                //    std::cout << "fp: " << data.getFilename(i) << " pos: (" << data.getX(i) << "," << data.getY(i) << ")" <<std::endl;
                //}
            }
        }
        else
        {
            if (data.getClassLabel(i) == -1)
            {
                tnn++;
            }
            else
            {
                fnn++;
                //#pragma omp critical
                //{
                //    std::cout << "fn: " << data.getFilename(i) << " pos: (" << data.getX(i) << "," << data.getY(i) << ")" <<std::endl;
                //}
            }
        }

    }

    tn = tnn;
    tp = tpp;
    fp = fpp;
    fn = fnn;
}



int StrongClassifier::classify(const integral_channels_const_view_t &integralImage, const bool use_cascade) const
{
    double res = 0;

    for (size_t l = 0; l < _learners.size(); ++l)
    {

        int ht = _learners[l].classify(integralImage, false);
        res += ht * _learners[l].getBeta();

        if (use_cascade and res < _learners[l]._cascadeThreshold)
        {
            std::cout << "classification result: " << res << std::endl;
            return -1;
        }

    }

    return 1;
}

// enable re-ordering of features or not
#if 1
// no features re-ordering
void StrongClassifier::convertToSoftCascade(const LabeledData::shared_ptr data, const double detectionRate)
{
    const int numExamples = data->getNumExamples();

    // stores the score and the sample index
    std::vector<std::pair<double, int> > scores;
    std::vector<bool> valid(numExamples, true);

    const bool useModelSoftCascade = false;
    boost::progress_display softCascadeProgress(data->getNumPosExamples() + _learners.size());

#pragma omp parallel for default(none) shared(scores, softCascadeProgress)
    for (int i = 0 ; i < numExamples; ++i)
    {
        double res = 0;

        if (data->getClassLabel(i) != -1) //data.getbackgroundClassLabel()){
        {
            //get result of every weak learner
            for (size_t l = 0; l < _learners.size(); ++l)
            {
                const int ht = _learners[l].classify(data->getIntegralImage(i), useModelSoftCascade);
                res += ht * _learners[l].getBeta();
            }

#pragma omp critical
            {
                scores.push_back(std::make_pair<double, int>(res, i));
                ++softCascadeProgress;
            }
        }
    }

    std::sort(scores.begin(), scores.end());
    //const int pos = int(scores.size() * (1 - detectionRate) + 0.5);

    //this is the cascasde threshold for the detection rate
    //const double theta = scores[pos].first - 1e-6;
    const double theta = 5; // FIXME hardcoded value
    std::cout << "Theta calculated: " << theta << flush << std::endl;

    for (int i = 0 ; i < (int)scores.size(); ++i)
    {
        if (scores[i].first < theta)
        {
            valid[scores[i].second] = false;
        }
    }

    std::vector<double> res(numExamples, 0);



    //find a valid positive sample with least score per stage
    for (size_t l = 0; l < _learners.size(); ++l)
    {
        double stage_threshold = std::numeric_limits<double>::max();

        //#pragma omp parallel for default(none) shared(no, data, thisthr, valid, l, res)
        for (int i = 0 ; i < numExamples; ++i)
        {
            if (valid[i] and data->getClassLabel(i) != -1) //data.getbackgroundClassLabel()){
            {
                //std::string fn = data.getFilename(i);
                int ht = _learners[l].classify(data->getIntegralImage(i), useModelSoftCascade);
                res[i] += ht * _learners[l].getBeta();
                stage_threshold = min(stage_threshold, res[i]);
            }
        }

        std::cout << "stage_threshold: " << stage_threshold << std::endl;


        // remove a small value to make sure that < and <= work properly
        const float epsilon = 1e-6;
        _learners[l]._cascadeThreshold = stage_threshold - epsilon;

        ++softCascadeProgress;
    }

    return;
}
#else
// will do features re-ordering
void StrongClassifier::convertToSoftCascade(const LabeledData::shared_ptr data, const double detectionRate)
{
    const int numExamples = data->getNumExamples();
    std::vector<std::pair<double, int> > scores;
    std::vector<bool> valid(numExamples, true);

    const bool useModelSoftCascade = false;
    boost::progress_display softCascadeProgress(data->getNumPosExamples() + _learners.size());

#pragma omp parallel for default(none) shared(scores, softCascadeProgress)
    for (int i = 0 ; i < numExamples; ++i)
    {
        double res = 0;

        if (data->getClassLabel(i) != -1) //data.getbackgroundClassLabel()){
        {
            //get result of every weak learner
            for (size_t l = 0; l < _learners.size(); ++l)
            {
                int ht = _learners[l].classify(data->getIntegralImage(i), useModelSoftCascade);
                res += ht * _learners[l].getBeta();
            }

#pragma omp critical
            {
                scores.push_back(std::make_pair<double, int>(res, i));
                ++softCascadeProgress;
            }
        }
    }

    std::sort(scores.begin(), scores.end());
    int position = int(scores.size() * (1 - detectionRate) + 0.5);

    //this is the cascasde threshold for the detection rate dr
    double theta = scores[position].first - 1e-6;
    //std::cout << "Theta calculated: " << theta << flush << std::endl;

    for (size_t i = 0 ; i < scores.size(); ++i)
    {
        if (scores[i].first < theta)
        {
            valid[scores[i].second] = false;
        }
    }


    // Start doing the reordering of the features (more discriminative first) --

    std::vector<double> scoreResults(numExamples, 0);
    std::vector<bool> learnersAlreadyUsed(_learners.size(), false);
    //find a valid positive sample with least score per stage
    std::vector<int> newIndices;
    for (size_t l = 0; l < _learners.size(); ++l)
    {

        double stage_threshold = std::numeric_limits<double>::max();
        double steepest_stage_threshold = -std::numeric_limits<double>::max();
        int steepest_learner_index = -1;
        for (size_t k = 0; k < _learners.size(); ++k)
        {
            if (learnersAlreadyUsed[k] == false)
            {
                std::vector<double> tmpRes(scoreResults);
                double minimum_stage_threshold = std::numeric_limits<double>::max();

                for (int i = 0; i < numExamples; ++i)
                {
                    if (valid[i] and (data->getClassLabel(i) != -1)) //data.getbackgroundClassLabel()){
                    {
                        //std::string fn = data.getFilename(i);
                        int ht = _learners[k].classify(data->getIntegralImage(i), useModelSoftCascade);
                        tmpRes[i] += ht * _learners[k].getBeta();
                        if (tmpRes[i] < minimum_stage_threshold){
                            minimum_stage_threshold = tmpRes[i];
                        }
                    }
                } // end of "for each example"

                //now search the biggest one
                if (minimum_stage_threshold > steepest_stage_threshold)
                {
                    steepest_stage_threshold = minimum_stage_threshold;
                    steepest_learner_index = k;
                }
            }
        } // end of "for each learner"

        for (int i = 0 ; i < numExamples; ++i)
        {
            if (valid[i] and data->getClassLabel(i) != -1) //data.getbackgroundClassLabel()){
            {
                //std::string fn = data.getFilename(i);
                int ht = _learners[steepest_learner_index].classify(data->getIntegralImage(i), useModelSoftCascade);
                scoreResults[i] += ht * _learners[steepest_learner_index].getBeta();
            }
        }

        learnersAlreadyUsed[steepest_learner_index] = true;
        stage_threshold = steepest_stage_threshold;
        std::cout << "stage_threshold: " << stage_threshold << std::endl;
        std::cout << "stage:  " << l << " steepest  idx: " << steepest_learner_index << std::endl;

        const float epsilon = 1e-6;
        _learners[steepest_learner_index]._cascadeThreshold = stage_threshold - epsilon;
        newIndices.push_back(steepest_learner_index);
        ++softCascadeProgress;

    } // end of "for each learner"

    for(size_t index=0; index < _learners.size(); index+=1)
    {
        if(std::count(newIndices.begin(), newIndices.end(), index) != 1)
        {
            throw std::runtime_error("StrongClassifier::convertToSoftCascade newIndices is flawed. "
                                     "Something went terribly wrong");
        }
    }

    //sort learners
    std::vector<WeakDiscreteTree> reorderedLearners;
    for (size_t i = 0; i< newIndices.size(); ++i)
    {
        reorderedLearners.push_back(_learners[newIndices[i]]);
    }

    _learners = reorderedLearners;

    return;
}
#endif


void StrongClassifier::writeClassifier(
        const std::string filename,
        const std::string trainedDatasetName,
        const point_t modelWindow, const rectangle_t objectWindow)
{
    ModelIO modelWriter;

    const std::string trainedModelName = "Soft cascade Model created via boosted_learning";

    modelWriter.initWrite(trainedDatasetName,
                          doppia_protobuf::DetectorModel::SoftCascadeOverIntegralChannels,
                          trainedModelName,
                          modelWindow, objectWindow);

    for (size_t i = 0; i < _learners.size(); ++i)
    {
        modelWriter.addStage(_learners[i]);
    }

    modelWriter.write(filename);
    google::protobuf::ShutdownProtobufLibrary();

    return;
}

} // end of namespace boosted_learning

