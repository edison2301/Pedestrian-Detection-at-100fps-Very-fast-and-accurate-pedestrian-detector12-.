#include "WeakDiscreteTreeLearner.hpp"

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <limits>
#include <cmath>

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <thrust/system/omp/vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include "Parameters.hpp"

#include <omp.h>

namespace boosted_learning {

using boost::counting_iterator;

WeakDiscreteTreeLearner::WeakDiscreteTreeLearner()
    :_negativeClass(-1)
{
    // nothing to do here
    return;
}

WeakDiscreteTreeLearner::WeakDiscreteTreeLearner(
        const int verbose, const int depth, const int negClass,
        TrainingData::ConstSharePointer trainingData,
        const std::vector<int> & classes,
        ConstMinOrMaxFeaturesResponsesSharedPointer mins,
        ConstMinOrMaxFeaturesResponsesSharedPointer maxs)
    : WeakDiscreteTree(verbose, depth),
      _negativeClass(negClass),
      _trainingData(trainingData),
      _mins(mins) , _maxs(maxs),
      _classes(classes)
{


    return;
}

struct comparator
{
    comparator(const FeaturesResponses &featuresResponses, const size_t featureIndex):
        feature_responses(featuresResponses[featureIndex])
    {
        // nothing to do here
        return;
    }

    bool operator()(const size_t trainingSampleIndexA, const size_t trainingSampleIndexB) const
    {
        return feature_responses[trainingSampleIndexA] < feature_responses[trainingSampleIndexB];
    }

    /// vector containing the responses of the the feature of interest
    FeaturesResponses::const_reference feature_responses;
};

void WeakDiscreteTreeLearner::sortIndexesBasedOnDataPositions(indices_t &positions,
                                                              const size_t featureIndex) const
{

    const FeaturesResponses &featuresResponses = _trainingData->getFeatureResponses();
    std::sort(positions.begin(), positions.end(), comparator(featuresResponses, featureIndex));

    return;
}



double WeakDiscreteTreeLearner::buildBalancedTree(
        const weights_t &weights)
{
    //build root --
    indices_t indices;
    indices.reserve(_trainingData->getNumExamples()); // does the reserve improve speed or not at all ?
    indices.assign( counting_iterator<indices_t::value_type>(0),
                    counting_iterator<indices_t::value_type>(_trainingData->getNumExamples()) );

    double errorSum = 0;

    const bool isLeft = true;

    int ret = createNode(weights,
                         indices, 0, _trainingData->getNumExamples(),
                         _root, errorSum, isLeft);
    if (ret==-1)
        throw std::runtime_error("ERROR: root node could not be constructed in WeakDiscreteTreeLearner::buildBalancedTree");

    if (errorSum >= 1e-12)
    {
        _beta = 0.5 * log((1.0 - errorSum) / errorSum);

    }
    else
    {
        // 13.8 = 0.5 * log((1.0 - 1e-12) / 1e-12);
       _beta = 14;
       return errorSum;

    }
    int root_node_bottom_value = _root->_feature.y + _root->_feature.height;
    int root_node_left_value = _root->_feature.x + _root->_feature.width;

    double minError = errorSum;

    // now build the children
    for (int d = 0; d < _depth; ++d)
    {
        TreeNode *t_node; // temporary node
        //TreeNode::shared_ptr bestLeaf;
        errorSum = 0;

        while (_root->getLeaf(d, t_node) == 1)
        {
            double errorLeft, errorRight;
            TreeNode::shared_ptr left;
            TreeNode::shared_ptr right;

            bool isLeft = true;
            const int leftResult = createNode(weights, t_node->_indices, 0, t_node->_splitIndex,
                                              left, errorLeft, isLeft, root_node_bottom_value, root_node_left_value);

            isLeft = false;
            const int rightResult = createNode(weights, t_node->_indices, t_node->_splitIndex, t_node->_indices.size(),
                                               right, errorRight, isLeft, root_node_bottom_value, root_node_left_value);

            // no Leafs found -> do not look for that node again
            if (leftResult == -1 && rightResult == -1)
            {
                t_node->setInvalid();
            }

            if (leftResult == 0)
            {
                errorSum += errorLeft ;
                t_node->left = left;
            }

            if (rightResult == 0)
            {
                t_node->right = right;
                errorSum += errorRight;
            }
        }

        if (errorSum != 0)
        {
            minError = errorSum;

            if (errorSum < 1e-12)
            {
                _beta = 14;
            }
            else
            {
                _beta = 0.5 * log((1.0 - errorSum) / errorSum);
            }
        }
    }

    if (_verbose > 3)
    {
        _root->print();
    }

    return minError;
}



inline
int WeakDiscreteTreeLearner::getErrorEstimate(
        const weights_t &weights,
        const indices_t& indices, const size_t featureIndex,
        int num_bins , const int minv, const int maxv, double &error) const
{
    if ((maxv - minv) < num_bins)
    {
        num_bins = maxv - minv;
    }

    //std::cout << "binsize: " << binsize << std::endl;
    std::vector<double> bin_pos(num_bins + 1, 0), bin_neg(num_bins + 1, 0);

    double cumNeg = 0;
    double cumPos = 0;
    error = std::numeric_limits<double>::max();

    const FeaturesResponses &featuresResponses = _trainingData->getFeatureResponses();

    for (size_t i = 0; i < indices.size(); ++i)
    {
        const size_t trainingSampleIndex = indices[i];
        //FIXME CHECK this
        //if (weights[trainingSampleIndex]< 1E-12)
        //    continue;
        const int featureResponse = featuresResponses[featureIndex][trainingSampleIndex];
        const int bin = int(num_bins / double(maxv - minv) * (featureResponse - minv));
        //int splitpos = minv + (bin/(double)binsize) *(maxv-minv);
        //int bin2= binsize/maxv*((*_featureResp)[pos+ind[i]] - minv);
        //assert(bin==bin2);
        const int the_class = _classes[indices[i]];

        if (the_class == _negativeClass)
        {
            bin_neg[bin] += weights[indices[i]];
            cumNeg += weights[indices[i]];

        }
        else
        {
            bin_pos[bin] += weights[indices[i]];
            cumPos += weights[indices[i]];

        }
    }

    //run test by setting this to return 0 with error 0
    if (cumPos == 0 || cumNeg == 0)
    {
        return -1;
    }

    double
            //positives left
            positivesLeftError = cumPos,
            negativesLeftError = 0,
            //positives right
            negativesRightError = cumNeg,
            positivesRightError = 0;
    //int minbin = -1;

    for (int i = 0; i < num_bins; ++i)
    {
        positivesLeftError -= bin_pos[i];
        negativesLeftError += bin_neg[i];

        positivesRightError += bin_pos[i];
        negativesRightError -= bin_neg[i];

        double binError = 0;

        if (positivesLeftError + negativesLeftError < positivesRightError + negativesRightError)
        {
            binError = positivesLeftError + negativesLeftError;
        }
        else
        {
            binError = positivesRightError + negativesRightError;
        }

        // we keep the min error
        if (binError < error)
        {
            //minbin = i;
            error = binError;
        }
    }

    return 0;
}

struct sort_pair {
    bool operator ()(std::pair<double, size_t> const& a, std::pair<double, size_t> const& b) {
        return a.first < b.first;
    }
};



int WeakDiscreteTreeLearner::createNode(
        const weights_t &weights,
        const indices_t &indices, const size_t start, const size_t end,
        TreeNode::shared_ptr &node, double &minError,
        const bool isLeft, const int root_node_bottom_height, const int root_node_left_width) const
{
    TrainingData::point_t modelWindow = _trainingData->getModelWindow();
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor ,
            modelHeight = modelWindow.y() / shrinking_factor;


    //for all features get responses on every image
    //find feature with lowest error
    const size_t numFeatures = _trainingData->getFeaturesPoolSize();
    std::vector<std::pair<double, size_t> >
            minErrorsForSearch(numFeatures, std::make_pair(std::numeric_limits<double>::max(), 0));

    indices_t indicesCrop;
    const int binsize = 1000;
    if(start > end)
    {
        throw std::invalid_argument("WeakDiscreteTreeLearner::createNode received start > end but expected start <= end");
    }

    indicesCrop.resize(end - start);
    std::copy(indices.begin() + start, indices.begin() + end, indicesCrop.begin());

    if (indicesCrop.size() == 0)
    {
        return -1;
    }


    int return_value = 0;
    //find max valid feature index
    size_t max_valid_feature_index= 0;
    for (size_t featureIndex = numFeatures-1; featureIndex >=0; --featureIndex)
    {
        if (_trainingData->getFeatureValidity(featureIndex))
        {
            max_valid_feature_index = featureIndex+1;
            break;
        }


    }
    //biasing features not yet supported
    double _pushBias = 0;
#pragma omp parallel for reduction(+:return_value) schedule(guided)
    for (size_t featureIndex = 0; featureIndex < max_valid_feature_index; ++featureIndex)
    {
        if (_trainingData->getFeatureValidity(featureIndex)){
            const int minv = (*_mins)[featureIndex], maxv = (*_maxs)[featureIndex];
            double error = std::numeric_limits<double>::max();
            return_value += getErrorEstimate(weights, indicesCrop, featureIndex, binsize, minv, maxv, error);
            minErrorsForSearch[featureIndex] = std::make_pair(error, featureIndex);
        }
    } // end of "for each feature"
    //if ( *(std::min_element(minErrorsForSearch.begin(),minErrorsForSearch.end(), sort_pair())) >2)//== mm)
    if (return_value < 0)
    {
        //create dummy node and return;
        //node = TreeNode::shared_ptr(new TreeNode(minthr, alphamin, _features[minFeat], minFeat, indicesCrop, splitIndexMin ));
        return -1;
    }
    const int deltaH=0;
    //get kth minimal elements
    //std::sort(minErrorsForSearch.begin(), minErrorsForSearch.end(), sort_pair());
    //thrust::sort(thrust::reinterpret_tag<thrust::omp::tag>(minErrorsForSearch.begin()),
    //             thrust::reinterpret_tag<thrust::omp::tag>(minErrorsForSearch.end()),
    //             sort_pair());

    std::sort(minErrorsForSearch.begin(),
                minErrorsForSearch.end(),
                 sort_pair());


    //search the feature that is highest up in the image
    int miny = std::numeric_limits<int>::max();
    int minx = std::numeric_limits<int>::max();
    int minFeatureIndex = -1;
    double errorth;

    if (minErrorsForSearch.size() > 0){
        errorth= std::min(0.49999999999, minErrorsForSearch[0].first* (1.0+_pushBias));
    }
    else{
        throw std::runtime_error("minErrorsForSearch has size 0: no features in pool?");
    }
    for (size_t i = 0; i< minErrorsForSearch.size(); ++i){
        const int this_feat_idx = minErrorsForSearch[i].second;
        double error= minErrorsForSearch[i].first;
        if (_pushBias ==0){
            minFeatureIndex = this_feat_idx;

            break;
        }


        if (error > errorth)
            break;


        const Feature this_feat = _trainingData->getFeature(this_feat_idx);
        int y = this_feat.y + this_feat.height;
        int x = this_feat.x + this_feat.width;


    }
    if (minFeatureIndex == -1)
    {
        //create dummy node and return;
        //node = TreeNode::shared_ptr(new TreeNode(minthr, alphamin, _features[minFeat], minFeat, indicesCrop, splitIndexMin ));
        return -1;
    }






    sortIndexesBasedOnDataPositions(indicesCrop, minFeatureIndex);
    int splitIndexMin = -1;
    int minThreshold = -1;
    int alphaMin = -1;

    if (findThreshold(weights, indicesCrop, minFeatureIndex, minError, minThreshold,
                      alphaMin, splitIndexMin) == -1)
    {
        return -1;
    }

    if(splitIndexMin < 0)
    {
        throw std::runtime_error("WeakDiscreteTreeLearner::createNode, findThreshold failed to find an adequate split index");
    }

    // set the shared_ptr to point towards a new node instance
    node.reset(new TreeNode(minThreshold, alphaMin,
                            _trainingData->getFeature(minFeatureIndex),
                            minFeatureIndex, indicesCrop, splitIndexMin, isLeft));
    return 0;
}





int WeakDiscreteTreeLearner::findThreshold(
        const weights_t &weights,
        const indices_t & sortedIndices,
        const size_t featureIndex,
        double &errorMin, int &thresholdMin, int &alphaMin, int &splitIndex) const
{

    const FeaturesResponses &featuresResponses = _trainingData->getFeatureResponses();
    const FeaturesResponses::const_reference featureResponses = featuresResponses[featureIndex];

    //getSum of data
    double sumPos = 0;
    double sumNeg = 0;
    errorMin = std::numeric_limits<int>::max();

    for (size_t i = 0; i < sortedIndices.size(); ++i)
    {
        const int index = sortedIndices[i];

        if (_classes[index] == _negativeClass)
        {
            sumNeg += weights[index];
        }
        else
        {
            sumPos += weights[index];
        }

    }

    if (_verbose > 3)
    {
        std::cout << "sumneg: " << sumNeg << " sumpos: " << sumPos << " both: " << sumPos + sumNeg << "\n";
    }

    if (sumNeg == 0)
    {
        thresholdMin = featureResponses[sortedIndices[0]] - 1;
        alphaMin = -1;
        errorMin = 0;
        return -1;
    }

    if (sumPos == 0)
    {
        thresholdMin = featureResponses[sortedIndices[0]] - 1;
        alphaMin =  1;
        errorMin = 0;
        return -1;
    }


    //positives left
    double positiveLeftError = sumPos;
    double negativeLeftError = 0;


    //positives right
    double negativeRightError = sumNeg;
    double positiveRightError = 0;
    int minIndex = -1;


    // go over all sorted data elements
    for (size_t i = 0; i < sortedIndices.size(); ++i)
    {
        const int index = sortedIndices[i];
        const int threshold = featureResponses[index];

        if (_classes[index] == _negativeClass)
        {
            negativeLeftError += weights[index];
            negativeRightError -= weights[index];
        }
        else
        {
            positiveLeftError -= weights[index];
            positiveRightError += weights[index];
        }


        double error = 0;
        int alpha = 0;

        if (positiveLeftError + negativeLeftError < positiveRightError + negativeRightError)
        {
            alpha = 1;
            error = positiveLeftError + negativeLeftError;
        }
        else
        {
            alpha = -1;
            error = positiveRightError + negativeRightError;

        }

        bool cond = false;

        if (i < sortedIndices.size() - 1)
        {
            cond = (error < errorMin && threshold != featureResponses[sortedIndices[i+1]]);
        }
        else
        {
            cond = (error < errorMin)	;
        }

        if (cond)
        {
            errorMin = error;
            thresholdMin = threshold;
            alphaMin = alpha;
            minIndex = i;
            splitIndex = i + 1;
        }

    }// end of "for each sorted data element"

    //set the threshold between suceeding values, except the last one(+1)
    if (minIndex ==  int(sortedIndices.size() - 1))
    {
        thresholdMin = thresholdMin + 1;
    }
    else if (thresholdMin + 1 == featureResponses[sortedIndices[minIndex+1]])
    {
        thresholdMin = thresholdMin + 1;
    }
    else
    {
        thresholdMin = int((featureResponses[sortedIndices[minIndex]] + featureResponses[sortedIndices[minIndex+1]]) / 2.0);
    }

    if (_verbose > 3)
    {
        std::cout << "sorted data:\n";

        for (size_t i = 0; i < sortedIndices.size(); ++i)
        {
            const int index = sortedIndices[i];
            const int threshold = featureResponses[index];
            std::cout << std::setw(6) << threshold << " ";
        }

        std::cout << "sorted classes:\n";

        for (size_t i = 0; i < sortedIndices.size(); ++i)
        {
            int ind = sortedIndices[i];
            std::cout << std::setprecision(2) << std::setw(6) << _classes[ind]*weights[ind] << " ";
        }

        std::cout << "threshold: " << thresholdMin << " alpha: " << alphaMin << " error: " << errorMin;
    }

    return 0;
}
}


