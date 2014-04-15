#ifndef ModelIO_HPP
#define ModelIO_HPP

#include "WeakDiscreteTreeLearner.hpp"
#include "StrongClassifier.hpp"

#include "detector_model.pb.h"

#include <string>

namespace boosted_learning {

class ModelIO
{
public:
    //typedef doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels softCascade_type;
    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    ModelIO(const int verbose = 0);
    ~ModelIO();
    void initWrite(const std::string datasetName,
                   const doppia_protobuf::DetectorModel::DetectorTypes type,
                   const std::string detectorName,
                   const point_t modelWindow,
                   const rectangle_t objectWindow);
    void write(const std::string fileName);
    void readModel(const std::string filename);
    void addStage(const WeakDiscreteTree &wl);
    void setStump(doppia_protobuf::IntegralChannelDecisionStump *stump, const TreeNode::shared_ptr);
    void setL2Tree(doppia_protobuf::IntegralChannelBinaryDecisionTree *tree, const TreeNode::shared_ptr);
    void setLNTree(doppia_protobuf::IntegralChannelBinaryDecisionTree *tree, const TreeNode::shared_ptr, int level);
    void print();

    //functions for reading

    TreeNode::shared_ptr readStump(const doppia_protobuf::IntegralChannelDecisionStump &stump);
    void readDecisionTree(const doppia_protobuf::IntegralChannelBinaryDecisionTree &tree, WeakDiscreteTree &wl);
    void readStage(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage, WeakDiscreteTree &wl);
    void readSoftCascade(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &sc, std::vector<WeakDiscreteTree> & sl);
    StrongClassifier read();


    const std::string getModelTrainingDatasetName();
    const TrainingData::point_t getModelWindowSize();
    const TrainingData::rectangle_t getObjectWindow();


    void addStage_real(const WeakDiscreteTree &wl);
    void setL2Tree_real(doppia_protobuf::IntegralChannelBinaryDecisionTree *tree, const vector<TreeNode_v2> &treeNodes);
    void setStump_real(doppia_protobuf::IntegralChannelDecisionStump *stump, const TreeNode_v2 &node);
protected:
    doppia_protobuf::DetectorModel _model;
    int _verbose;
};

} // end of namespace boosted_learning

#endif // ModelIO_HPP
