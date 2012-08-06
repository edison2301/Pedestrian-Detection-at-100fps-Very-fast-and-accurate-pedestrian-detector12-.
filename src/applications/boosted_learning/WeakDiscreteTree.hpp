#ifndef WEAKDISCRETETREE_HPP
#define WEAKDISCRETETREE_HPP

#include "IntegralChannelsComputer.hpp"
#include "TreeNode.hpp"

#include <boost/cstdint.hpp>
#include <boost/array.hpp>

#include <iostream>

namespace boosted_learning {

// forward declarations
class TreeNode;
typedef boost::shared_ptr<TreeNode> TreeNodePtr;
typedef boost::shared_ptr<const TreeNode> ConstTreeNodePtr;

class WeakDiscreteTreeLearner;

class WeakDiscreteTree
{
public:

    typedef bootstrapping::integral_channels_t integral_channels_t;

    WeakDiscreteTree();
    WeakDiscreteTree(const int verbose, const int depth);

    /// convertion constructor
    WeakDiscreteTree(const WeakDiscreteTreeLearner &other);

    void setBeta(double b);
    void setRoot(TreeNodePtr root);
    void setDepth(int depth);

    double getBeta() const;
    double getDepth() const;
    const TreeNodePtr getRoot() const;

    int classify(const integral_channels_t &integralImage, const bool output = false) const;
    double classify_real(const integral_channels_t &integralImage, const bool output = false) const;


    int classify(const FeaturesResponses &featuresResponses,
                 const size_t trainingSampleIndex,
                 const bool output = false) const;

    double classify_real(const FeaturesResponses &featuresResponses,
                 const size_t trainingSampleIndex, const bool output=false) const;

    void convertDepthTwo();
    int getFeatResponse(const integral_channels_t &integralImage, const Feature &feat) const;

//protected:
public:

    int _verbose;
    double _beta;
    TreeNodePtr _root;
    std::vector<TreeNode_v2> _rootd2;
    int _depth;
    double _cascadeThreshold;

    /// only for used when calling convertDepthTwo
    boost::array<boost::int8_t, 8> _lut;
    boost::array<Feature, 3> _features;
    boost::array<double, 3> _thresholds;

};


// inlined for performance reasons
inline
int WeakDiscreteTree::getFeatResponse(const integral_channels_t &integralImage, const Feature &feature) const
{
    const Feature &f = feature;
    const int
            a = integralImage[f.channel][f.y][f.x],
            b = integralImage[f.channel][f.y+0][f.x+f.width],
            c = integralImage[f.channel][f.y+f.height][f.x+f.width],
            d = integralImage[f.channel][f.y+f.height][f.x+0];
    return a + c - b - d;
}

// FIXME does this really help speed ? (do benchmarks)
// inlined for performance reasons
inline
int WeakDiscreteTree::classify(const integral_channels_t &integralImage, const bool output) const
{

    //start at the root
    assert(_root);
    TreeNode::shared_ptr current = _root, next;

    for (;;)
    {
        //    while ( !current->isLeaf()){
        const int response = getFeatResponse(integralImage, current->_feature);

        if (output)
        {
            std::cout << "feat Resp: " << response << std::endl;
        }

        if (response < current->_threshold)
        {
            next = current->left;
        }
        else
        {
            next = current->right;
        }

        if (!next)
        {
            if (current->_alpha > 0)
            {
                if (response < current->_threshold)
                {
                    return 1;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (response >= current->_threshold)
                {
                    return 1;
                }
                else
                {
                    return -1;
                }
            }

            //bool ret = resp < current->_thr;
            //return (current->_alpha>0) ? ret : !ret;
        }

        current = next;
    } // end of "forever"

    return 0; // should never reach this, and return 1 or -1 instead
}

inline
double WeakDiscreteTree::classify_real(const WeakDiscreteTree::integral_channels_t &integralImage, const bool output) const
{
    //start at the root

    const int response = getFeatResponse(integralImage, _rootd2[0]._feature);

    if (response < _rootd2[0]._threshold)
    {
        const int response = getFeatResponse(integralImage, _rootd2[1]._feature);
        //response = featuresResponses[_rootd2[1]._featureIndex][trainingSampleIndex];
        if (response < _rootd2[1]._threshold)
            return _rootd2[1]._betaTrue;
        else
            return _rootd2[1]._betaFalse;
    }
    else{
        const int response = getFeatResponse(integralImage, _rootd2[2]._feature);
        //response = featuresResponses[_rootd2[2]._featureIndex][trainingSampleIndex];
        if (response < _rootd2[2]._threshold)
            return _rootd2[2]._betaTrue;
        else
            return _rootd2[2]._betaFalse;
    }


    return 0; // should never reach this
}

// FIXME does this really help speed ? (do benchmarks)
// inlined for performance reasons
inline
int WeakDiscreteTree::classify(const FeaturesResponses &featuresResponses,
                               const size_t trainingSampleIndex,
                               const bool output) const
{
    //start at the root
    assert(_root);
    TreeNode::shared_ptr current = _root, next;

    for (;;)
    {
        //    while ( !current->isLeaf()){
        const int response = featuresResponses[current->_featureIndex][trainingSampleIndex];

        if (output)
        {
            std::cout << "Feature response: " << response << std::endl;
        }

        if (response < current->_threshold)
        {
            next = current->left;
        }
        else
        {
            next = current->right;
        }

        if (!next)
        {
            if (current->_alpha > 0)
            {
                if (response < current->_threshold)
                {
                    return 1;
                }
                else
                {
                    return -1;
                }
            }
            else
            {
                if (response >= current->_threshold)
                {
                    return 1;
                }
                else
                {
                    return -1;
                }
            }

            //bool ret = resp < current->_thr;
            //return (current->_alpha>0) ? ret : !ret;
        }

        current = next;
    } // end of "forever"

    return 0; // should never reach this, and return 1 or -1 instead
}


inline
double WeakDiscreteTree::classify_real(const FeaturesResponses &featuresResponses,
                               const size_t trainingSampleIndex, const bool output) const
{
    //start at the root
    int response = featuresResponses[_rootd2[0]._featureIndex][trainingSampleIndex];

    if (response < _rootd2[0]._threshold)
    {
        response = featuresResponses[_rootd2[1]._featureIndex][trainingSampleIndex];
        if (response < _rootd2[1]._threshold)
            return _rootd2[1]._betaTrue;
        else
            return _rootd2[1]._betaFalse;
    }
    else{
        response = featuresResponses[_rootd2[2]._featureIndex][trainingSampleIndex];
        if (response < _rootd2[2]._threshold)
            return _rootd2[2]._betaTrue;
        else
            return _rootd2[2]._betaFalse;
    }


    return 0; // should never reach this
}

} // end of namespace boosted_learning

#endif // WEAKDISCRETETREE_HPP
