#ifndef TREENODE_HPP
#define TREENODE_HPP

#include "Feature.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>

namespace boosted_learning {

class TreeNode
{
public:

    typedef boost::shared_ptr<TreeNode> shared_ptr;
    typedef std::vector<size_t> indices_t;

    TreeNode(const int thr, const int alpha, const Feature &feature, const size_t featureIndex);

    TreeNode(const int thr, const int alpha, const Feature &feature, const size_t featureIndex,
             const indices_t& indices, const int splitIndex , const bool isLeft);

    int _threshold;
    int _alpha;
    const Feature _feature;
    const size_t _featureIndex;

    indices_t _indices;
    size_t _splitIndex;
    bool _valid;
    TreeNode::shared_ptr left;
    TreeNode::shared_ptr right;
    int _id;
    int _parentId;
    bool _isLeft;
    static int idCount;

    // setters
    void setParentId(const int id);
    void setId(const int id);
    void setAlpha(const int alpha);
    void setLeftChild(TreeNode::shared_ptr l);
    void setRightChild(TreeNode::shared_ptr r);
    void setInvalid();
    void setIsLeft(bool isleft);

    // getters
    int getId() const;
    int getParentId() const;
    static int getNewID();
    int getAlpha() const;

    static int getNodes(const int depth,
                        TreeNode::shared_ptr node,
                        std::vector<TreeNode::shared_ptr> & nodes,
                        int id = 0, int aktDepth = 0);
    int getLeaf(int maxDepth, TreeNode * &node);

    bool isLeft() const;
    bool isLeaf() const;

    void print(const int depth = 0);

};


class TreeNode_v2
{
public:

    typedef std::vector<size_t> indices_t;

    TreeNode_v2();
    TreeNode_v2(const int thr, const double betaTrue, const double betaFalse, const Feature &feature,const size_t featureIndex );

    TreeNode_v2(const int thr, const double betaTrue, const double betaFalse, const Feature &feature, const size_t featureIndex, const indices_t& indices, const int splitIndex);

    int _threshold;


    double _betaTrue;
    double _betaFalse;
    Feature _feature;
    size_t _featureIndex;

    indices_t _indices;
    size_t _splitIndex;
    void print(string prefix) const;

};



} // end of namespace boosted_learning


#endif // TREENODE_HPP
