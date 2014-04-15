#include "TreeNode.hpp"

#include <iostream>

namespace boosted_learning {

TreeNode::TreeNode(const int thr, const int alpha, const Feature &feature, const size_t featureIndex):
    _threshold(thr), _alpha(alpha), _feature(feature), _featureIndex(featureIndex),
    _valid(true), _id(-1), _parentId(-1), _isLeft(true)
{
    // nothing to do here
    return;
}

TreeNode::TreeNode(const int thr, const int alpha, const Feature &feature, const size_t featureIndex,
         const indices_t& indices, const int splitIndex , const bool isLeft):
    _threshold(thr), _alpha(alpha), _feature(feature), _featureIndex(featureIndex),
    _indices(indices),
    _splitIndex(splitIndex), _valid(true), _id(-1), _parentId(-1), _isLeft(isLeft)
{
    // nothing to do here
    return;
}

void TreeNode::setParentId(int id)
{
    _parentId = id;
    return;
}


void TreeNode::setId(int id)
{

    _id = id;
    return;
}


void TreeNode::setAlpha(int alpha)
{

    _alpha = alpha;
    return;
}


int TreeNode::getAlpha() const
{

    return _alpha;
}


void TreeNode::setLeftChild(TreeNode::shared_ptr l)
{
    left = l;
    return;
}


void TreeNode::setRightChild(TreeNode::shared_ptr r)
{
    right = r;
    return;
}


int TreeNode::getId() const
{
    return _id;
}


int TreeNode::getParentId() const
{
    return _parentId;
}

void TreeNode::setInvalid()
{
    _valid = false;
    return;
}

void TreeNode::setIsLeft(bool isleft)
{
    _isLeft = isleft;
    return;
}


bool TreeNode::isLeft() const
{
    return _isLeft;
}


bool TreeNode::isLeaf() const
{
    return (left.get() == NULL) and (right.get() == NULL);
}


int TreeNode::getLeaf(int maxDepth, TreeNode * &node)
{
    if (!this->_valid)
    {
        return 0;
    }

    if (this->isLeaf())
    {
        node = this;
        return 1;
    }

    if (maxDepth == 0)
    {
        return 0;
    }

    if (left.get() != NULL)
    {
        if (left->getLeaf(maxDepth - 1, node) == 1)
        {
            return 1;
        }
        else
        {
            if (right.get() != NULL)
            {
                return right->getLeaf(maxDepth - 1, node);
            }
        }
    }

    return 0;
}


int TreeNode::getNewID()
{
    idCount++;
    return idCount;
}


void TreeNode::print(const int depth)
{
    std::cout << std::endl;
    std::cout << " --------------level" << depth << std::endl;
    std::cout << _feature.x << "\t" << _feature.y << "\t" << _feature.width << "\t" << _feature.height << "\t" << _feature.channel << "\n";
    std::cout << _threshold << std::endl;
    std::cout << _alpha << std::endl;
    std::cout << " ----------------------------" <<  std::endl;
    std::cout << std::endl;

    if (left.get() != NULL)
    {
        left->print(depth + 1);
    }

    if (right.get() != NULL)
    {
        right->print(depth + 1);
    }
    return;
}


int TreeNode::getNodes(const int depth, TreeNode::shared_ptr node, std::vector<TreeNode::shared_ptr> & nodes,
                       int id, int aktDepth)
{

    node->_id = id;

    if (aktDepth == 0)
    {
        node->_parentId = id;
    }

    //int newid = std::pow(2,aktDepth) + id;
    if (depth == 0)
    {
        //only root
        node->_parentId = id;
        nodes.push_back(node);
        return 0;
    }

    if (depth == 1)
    {
        if (node->left.get() != NULL)
        {
            node->left->_id = getNewID();//newid+1;
            node->left->_parentId = id;
            nodes.push_back(node->left);
        }

        if (node->right.get() != NULL)
        {
            node->right->_parentId = id;
            node->right->_id = getNewID();//newid+2;
            nodes.push_back(node->right);
        }

        return 0;
    }
    else
    {
        aktDepth ++;

        if (node->isLeaf())
        {
            return -1;
        }

        if (node->left.get() != NULL)
        {
            node->left->_parentId = id;
            getNodes(depth - 1, node->left, nodes, getNewID()/*newid+1*/, aktDepth);
        }

        if (node->right.get() != NULL)
        {
            node->right->_parentId = id;
            getNodes(depth - 1, node->right, nodes, getNewID()/*newid+2*/, aktDepth);
        }
    }

    return -1;
}


TreeNode_v2::TreeNode_v2():
    _threshold(0), _betaTrue(-1), _betaFalse(-1), _feature(Feature(-1,-1,-1,-1,-1)), _featureIndex(-1)
{
    // nothing to do here
    return;
}

TreeNode_v2::TreeNode_v2(const int thr, const double betaTrue, const double betaFalse, const Feature &feature,const size_t featureIndex):
    _threshold(thr), _betaTrue(betaTrue), _betaFalse(betaFalse), _feature(feature), _featureIndex(featureIndex)

{
    // nothing to do here
    return;
}

TreeNode_v2::TreeNode_v2(const int thr, const double betaTrue, const double betaFalse, const Feature &feature,const size_t featureIndex, const TreeNode_v2::indices_t &indices, const int splitIndex):
    _threshold(thr), _betaTrue(betaTrue), _betaFalse(betaFalse), _feature(feature), _featureIndex(featureIndex),_indices(indices),
    _splitIndex(splitIndex)
{
    // nothing to do here
    return;
}

void TreeNode_v2::print(std::string prefix) const
{



    std::cout << prefix.c_str() << ".feature_threshold " << _threshold << std::endl;
    std::cout << prefix.c_str() << ".larger_than_threshold " << false << std::endl;
    std::cout << prefix.c_str() << ".weight_true_leaf " << _betaTrue << std::endl;
    std::cout << prefix.c_str() << ".weight_false_leaf " << _betaFalse << std::endl;
    std::cout << prefix.c_str() << ".feature.channel_index " << _feature.channel << std::endl;
    std::cout << prefix.c_str() << ".feature.box) " << _feature.width * _feature.height << std::endl;
    std::cout << prefix.c_str() <<  ".feature.box min_x " <<  _feature.x  << std::endl;
    std::cout << prefix.c_str() << ".feature.box min_y " <<  _feature.y << std::endl;
    std::cout << prefix.c_str() << ".feature.box max_x " <<  _feature.x +  _feature.width << std::endl;
    std::cout << prefix.c_str() << ".feature.box max_y " <<  _feature.y +  _feature.height   << std::endl;


    return;
}



} // end of namespace boosted_learning
