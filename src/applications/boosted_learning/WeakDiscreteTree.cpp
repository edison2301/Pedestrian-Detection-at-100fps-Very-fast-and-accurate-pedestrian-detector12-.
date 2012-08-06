#include "WeakDiscreteTree.hpp"

#include "WeakDiscreteTreeLearner.hpp"

namespace boosted_learning {

// all methods are implemented in the header file


WeakDiscreteTree::WeakDiscreteTree()
    :  _verbose(2), _beta(-1), _depth(-1), _cascadeThreshold(-std::numeric_limits<float>::max())
{
    // nothing to do here
    return;
}

WeakDiscreteTree::WeakDiscreteTree(const int verbose, const int depth)
    :  _verbose(verbose), _beta(-1), _depth(depth), _cascadeThreshold(-std::numeric_limits<float>::max())
{
    // nothing to do here
    return;
}

WeakDiscreteTree::WeakDiscreteTree(const WeakDiscreteTreeLearner &other)
{
    // FIXME is this constructor really needed ?
    _verbose = other._verbose;
    _beta = other._beta;
    _root = other._root;
    _depth = other._depth;

    _cascadeThreshold = other._cascadeThreshold;
    _lut = other._lut;
    _features = other._features;
    _thresholds = other._thresholds;

    // we ignore all other fields from WeakDiscreteTreeLearner
    return;
}

void WeakDiscreteTree::setBeta(double b)
{
    _beta = b;
}

void WeakDiscreteTree::setRoot(TreeNodePtr root)
{
    _root = root;
}

void WeakDiscreteTree::setDepth(int depth)
{
    _depth = depth;
}

double WeakDiscreteTree::getBeta() const
{
    return _beta;
}
double WeakDiscreteTree::getDepth() const
{
    return _depth;
}
const TreeNodePtr WeakDiscreteTree::getRoot() const
{
    return _root;
}


void WeakDiscreteTree::convertDepthTwo()
{
    TreeNode::shared_ptr left = _root->left;
    TreeNode::shared_ptr right = _root->right;

    for (int i = 0; i < 8; ++i)
    {
        _lut[i] = -1;
    }

    unsigned resp;
    resp = (1 << 0) | ((left->_alpha > 0)  << 1) | (0 << 2);
    _lut[resp] = 1;
    resp = (1 << 0) | ((left->_alpha > 0)  << 1) | (1 << 2);
    _lut[resp] = 1;
    resp = (0 << 0) | (1 << 1)        | ((right->_alpha > 0)  << 2);
    _lut[resp] = 1;
    resp = (0 << 0) | (0 << 1)        | ((right->_alpha > 0)  << 2);
    _lut[resp] = 1;
    _features[0] = _root->_feature;
    _features[1] = _root->left->_feature;
    _features[2] = _root->right->_feature;

    _thresholds[0] = _root->_threshold;
    _thresholds[1] = _root->left->_threshold;
    _thresholds[2] = _root->right->_threshold;

    return;
}



#if 0
inline int WeakDiscreteTree::classify(const integral_channels_t &integralImage, bool output = false) const
{
    const size_t resp = ((getFeatResponse(integralImage, _features[0]) < _thresholds[0]) << 0)
                        | ((getFeatResponse(integralImage, _features[1]) < _thresholds[1]) << 1)
                        | ((getFeatResponse(integralImage, _features[2]) < _thresholds[2]) << 2);

    return _lut[resp];

}
#endif



} // end of namespace boosted_learning
