#ifndef WeakDiscreteTreeLEARNER_H
#define WeakDiscreteTreeLEARNER_H

#include "TrainingData.hpp"
#include "Feature.hpp"
//#include "MemMappedFile.hpp"
#include "IntegralChannelsForPedestrians.hpp"

#include "TreeNode.hpp"
#include "WeakDiscreteTree.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>

namespace boosted_learning {

class WeakDiscreteTreeLearner : public WeakDiscreteTree
{

public:

    typedef TreeNode::indices_t indices_t;
    typedef std::vector<double> weights_t;

    WeakDiscreteTreeLearner();

    WeakDiscreteTreeLearner(const int verbose, const int depth, const int negClass,
                            TrainingData::ConstSharePointer trainingData,
                            const std::vector<int> & classes,
                            ConstMinOrMaxFeaturesResponsesSharedPointer mins,
                            ConstMinOrMaxFeaturesResponsesSharedPointer maxs);

    double buildBalancedTree(const weights_t &weights);

protected:

    int findThreshold(
            const weights_t &weights,
            const indices_t& sortedIndices, const size_t featureIndex,
            double &errormin, int &thrmin, int &alphamin, int &splitIndex) const;



    int getErrorEstimate(
            const weights_t &weights,
            const indices_t& indices, const size_t featureIndex,
            int binsize , const int minv, const int maxv, double &error) const;


    int createNode(
            const weights_t &weights,
            const indices_t& indices, const size_t start, const size_t end,
            TreeNode::shared_ptr &node, double &minerror,
            const bool isLeft, const int root_node_bottom_height = 0, const int root_node_left_width=0) const;


    //double calcError(jhb::MemMappedFile<int>::shared_ptr featureResp, const std::vector<double> & weights);
    //std::vector<int> getSortedIndexVector(jhb::MemMappedFile<int> & data, int start, int size);

    void sortIndexesBasedOnDataPositions(indices_t &positions, const size_t start) const;

    const int _negativeClass;

    TrainingData::ConstSharePointer _trainingData;
    /// minumim and maximum value that a single feature has along all images
    ConstMinOrMaxFeaturesResponsesSharedPointer _mins, _maxs;

    const std::vector<int> _classes;

};

} // end of namespace boosted_learning

#endif // WeakDiscreteTreeLEARNER_H
