#include "ModelIO.hpp"

#include "IntegralChannelsComputer.hpp"

#include "detections.pb.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

namespace boosted_learning {

using namespace doppia_protobuf;


ModelIO::ModelIO(const int verbose): _verbose(verbose)
{

    // set the shrinking factor in case we will write this model
    SoftCascadeOverIntegralChannelsModel *model_p =  _model.mutable_soft_cascade_model();
    model_p->set_shrinking_factor(bootstrapping::integral_channels_computer_t::get_shrinking_factor());


    return;
}

ModelIO::~ModelIO()
{
    //google::protobuf::ShutdownProtobufLibrary();
    return;
}


void ModelIO::readModel(const std::string filename)
{
    fstream input(filename.c_str(), ios::in | std::ios::binary);

    if( input.is_open() == false)
    {
        printf("Failed to open model file %s\n", filename.c_str());
        throw std::runtime_error("Failed to open model file");
    }

    if (!_model.ParseFromIstream(&input))
    {
        throw std::runtime_error("Failed to read Detector File.");
    }

    return;
}

TreeNode::shared_ptr ModelIO::readStump(const doppia_protobuf::IntegralChannelDecisionStump &stump)
{
    double thr = (double) stump.feature_threshold();
    int alpha;

    if (stump.larger_than_threshold() == true)
    {
        alpha = -1;
    }
    else
    {
        alpha = 1;
    }

    int ch = stump.feature().channel_index();
    int x = stump.feature().box().min_corner().x();
    int y = stump.feature().box().min_corner().y();
    int w = stump.feature().box().max_corner().x() - stump.feature().box().min_corner().x();
    int h = stump.feature().box().max_corner().y() - stump.feature().box().min_corner().y();
    Feature c(x, y, w, h, ch);
    TreeNode::shared_ptr node(new TreeNode(thr, alpha, c, -1));

    return node;
}


void ModelIO::readDecisionTree(const doppia_protobuf::IntegralChannelBinaryDecisionTree &tree, WeakDiscreteTree &wl)
{

    std::vector<TreeNode::shared_ptr> nodes;
    TreeNode::shared_ptr root;

    int noNodes = tree.nodes_size();

    for (int i = 0; i < noNodes; ++i)
    {
        //read node
        TreeNode::shared_ptr node = readStump(tree.nodes(i).decision_stump());
        node->setId(tree.nodes(i).id());
        int parentid = tree.nodes(i).parent_id();
        //node->setIsLeft(!tree.nodes(i).parent_value());
        node->setIsLeft(tree.nodes(i).parent_value());
        node->setParentId(parentid);

        if (tree.nodes(i).id() == tree.nodes(i).parent_id())
        {
            root = node;
        }
        else
        {
            nodes.push_back(node);
        }
    }


    //build tree
    std::vector< TreeNode::shared_ptr > investigateNode;
    investigateNode.push_back(root);

    while (investigateNode.size() > 0)
    {
        //search
        TreeNode::shared_ptr n = investigateNode.back();
        int parentid = n->getId();
        investigateNode.pop_back();
        bool childFound = false;

        for (size_t i = 0; i < nodes.size(); ++i)
        {
            TreeNode::shared_ptr &potential_child = nodes[i];

            if (potential_child->getParentId() == parentid)
            {
                if (nodes[i]->isLeft())
                {
                    childFound = true;
                    n->setLeftChild(nodes[i]);
                    investigateNode.push_back(nodes[i]);
                }
                else
                {
                    childFound = true;
                    n->setRightChild(nodes[i]);
                    investigateNode.push_back(nodes[i]);
                }
            }
        }

        if (childFound)
        {
            n->setAlpha(1);
        }

    } // end of "while nodes to inve

    wl.setRoot(root);
    return;
}


void ModelIO::print()
{
    _model.PrintDebugString();
    return;
}


void ModelIO::readStage(const doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage, WeakDiscreteTree &wl)
{
    wl.setBeta(stage.weight());
    wl._cascadeThreshold = (stage.cascade_threshold());

    //TODO: Cascade threshold ignored for now
    if (stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_Stumps)
    {
        readStump(stage.decision_stump());
    }

    if (stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_Level2DecisionTree)
    {
        wl.setDepth(1);
        readDecisionTree(stage.level2_decision_tree(), wl);
    }

    if (stage.feature_type() == doppia_protobuf::SoftCascadeOverIntegralChannelsStage_FeatureTypes_LevelNDecisionTree)
    {
        readDecisionTree(stage.leveln_decision_tree(), wl);
    }


    //TODO check
    //wl.convertDepthTwo();

    return;
}


void ModelIO::readSoftCascade(const doppia_protobuf::SoftCascadeOverIntegralChannelsModel &sc, std::vector<WeakDiscreteTree> & sl)
{
    //typedef doppia_protobuf::Sof
    sl.resize(sc.stages_size());

    if (_verbose > 1)
    {
        std::cout << "Cascade contains " << sc.stages_size() << " stages\n";
    }

    for (int i = 0; i < sc.stages_size(); ++i)
    {
        readStage(sc.stages(i), sl[i]);
    }

    return;
}


StrongClassifier ModelIO::read()
{
    std::vector<WeakDiscreteTree> out;

    if (_verbose > 1)
    {
        std::cout << " Reading Detector: " << _model.detector_name() << std::endl;
        std::cout << " Trained on: " << _model.training_dataset_name() << std::endl;
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_LinearSvm)
    {
        std::cout << " Type of the Detector: Linear SVM\n";
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels)
    {
        std::cout << " Type of the Detector: SoftCascade over Integral Channels\n";
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_HoughForest)
    {
        std::cout << " Type of the Detector: HoughForest\n";
    }

    if (_model.detector_type() == doppia_protobuf::DetectorModel_DetectorTypes_SoftCascadeOverIntegralChannels)
    {
        readSoftCascade(_model.soft_cascade_model(), out);
    }

    else
    {
        throw runtime_error("This type of detector has not been implemented yet");
    }

    StrongClassifier ret(out);

    return ret;
}


const std::string ModelIO::getModelTrainingDatasetName()
{
    if(_model.has_training_dataset_name() == false)
    {
        throw std::runtime_error("_model.has_training_dataset_name() == false");
    }
    return _model.training_dataset_name();
}

const TrainingData::point_t ModelIO::getModelWindowSize()
{
    if(_model.has_model_window_size() == false)
    {
        throw std::runtime_error("_model.has_model_window_size() == false");
    }

    return TrainingData::point_t(_model.model_window_size().x(),
                                 _model.model_window_size().y());
}

const TrainingData::rectangle_t ModelIO::getObjectWindow()
{
    if(_model.has_object_window() == false)
    {
        throw std::runtime_error("_model.has_object_window() == false");
    }

    TrainingData::rectangle_t object_window;
    object_window.min_corner().x( _model.object_window().min_corner().x() );
    object_window.min_corner().y( _model.object_window().min_corner().y() );
    object_window.max_corner().x( _model.object_window().max_corner().x() );
    object_window.max_corner().y( _model.object_window().max_corner().y() );

    return object_window;
}

void ModelIO::initWrite(const std::string datasetName,
                        const DetectorModel::DetectorTypes type,
                        const std::string detectorName,
                        const point_t modelWindow,
                        const rectangle_t objectWindow)
{

    doppia_protobuf::Point2d *model_window = _model.mutable_model_window_size();
    model_window->set_x(modelWindow.x());
    model_window->set_y(modelWindow.y());

    doppia_protobuf::Box *b = _model.mutable_object_window();
    b->mutable_min_corner()->set_x(objectWindow.min_corner().x());
    b->mutable_min_corner()->set_y(objectWindow.min_corner().y());
    b->mutable_max_corner()->set_x(objectWindow.max_corner().x());
    b->mutable_max_corner()->set_y(objectWindow.max_corner().y());

    _model.set_training_dataset_name(datasetName.c_str());
    _model.set_detector_type(type);
    _model.set_detector_name(detectorName);

    return;
}


void ModelIO::write(const std::string fileName)
{

    if(fileName.empty())
    {
        throw std::invalid_argument("ModelIO::write required an non empty fileName");
    }

    fstream output(fileName.c_str(), ios::out | std::ios::binary);

    if(output.is_open() == false)
    {
        printf("ModelIO failed to open file %s for writting\n", fileName.c_str());
        throw std::invalid_argument("ModelIO::write failed to create the output file.");
    }

    if (!_model.SerializeToOstream(&output))
    {
        throw std::runtime_error("Failed to write Detector File.");
    }

    output.close();
    return;
}


void ModelIO::addStage(const WeakDiscreteTree &wl)
{

    TreeNode::idCount = 0;

    SoftCascadeOverIntegralChannelsModel *model_p =  _model.mutable_soft_cascade_model();
    doppia_protobuf::SoftCascadeOverIntegralChannelsStage *stage = model_p->add_stages();
    stage->set_weight(wl.getBeta());

    if (wl.getDepth() == 0)
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_Stumps);
        setStump(stage->mutable_decision_stump(), wl.getRoot());
    }
    else if (wl.getDepth() == 1)
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_Level2DecisionTree);
        setL2Tree(stage->mutable_level2_decision_tree(), wl.getRoot());
    }
    else
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_LevelNDecisionTree);
        setL2Tree(stage->mutable_leveln_decision_tree(), wl.getRoot());
    }

    stage->set_cascade_threshold(wl._cascadeThreshold);

    if (_verbose >= 2)
    {
        stage->PrintDebugString();
    }

    return;
}

void ModelIO::addStage_real(const WeakDiscreteTree &wl)
{

    TreeNode::idCount = 0;

    SoftCascadeOverIntegralChannelsModel *model_p =  _model.mutable_soft_cascade_model();
    doppia_protobuf::SoftCascadeOverIntegralChannelsStage *stage = model_p->add_stages();
    stage->set_weight(1);

    if (wl.getDepth() == 0)
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_Stumps);
        setStump_real(stage->mutable_decision_stump(), wl._rootd2[0]);
    }
    else if (wl.getDepth() == 1)
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_Level2DecisionTree);
        setL2Tree_real(stage->mutable_level2_decision_tree(), wl._rootd2);
    }
    else
    {
        stage->set_feature_type(SoftCascadeOverIntegralChannelsStage_FeatureTypes_LevelNDecisionTree);
        setL2Tree_real(stage->mutable_leveln_decision_tree(), wl._rootd2);
    }

    stage->set_cascade_threshold(wl._cascadeThreshold);

    if (_verbose >= 2)
    {
        stage->PrintDebugString();
    }

    return;
}


void ModelIO::setStump(IntegralChannelDecisionStump *stump, TreeNode::shared_ptr node)
{
    stump->set_feature_threshold(node->_threshold);

    if (!node->isLeaf())
    {
        stump->set_larger_than_threshold(false);
    }
    else
    {
        if (node->_alpha == -1)
        {
            stump->set_larger_than_threshold(true);
        }

        if (node->_alpha == 1)
        {
            stump->set_larger_than_threshold(false);
        }
    }

    IntegralChannelsFeature *feat = stump->mutable_feature();
    feat->set_channel_index(node->_feature.channel);
    Box *b = feat->mutable_box();
    b->mutable_min_corner()->set_x(node->_feature.x);
    b->mutable_min_corner()->set_y(node->_feature.y);

    b->mutable_max_corner()->set_x(node->_feature.x + node->_feature.width);
    b->mutable_max_corner()->set_y(node->_feature.y + node->_feature.height);

    return;
}

void ModelIO::setStump_real(IntegralChannelDecisionStump *stump, const TreeNode_v2 & node)
{


    stump->set_feature_threshold(node._threshold);
    stump->set_true_leaf_weight(node._betaTrue);
    stump->set_false_leaf_weight(node._betaFalse);
    stump->set_larger_than_threshold(false);

    IntegralChannelsFeature *feat = stump->mutable_feature();
    feat->set_channel_index(node._feature.channel);
    Box *b = feat->mutable_box();
    b->mutable_min_corner()->set_x(node._feature.x);
    b->mutable_min_corner()->set_y(node._feature.y);

    b->mutable_max_corner()->set_x(node._feature.x + node._feature.width);
    b->mutable_max_corner()->set_y(node._feature.y + node._feature.height);

    return;
}


void ModelIO::setL2Tree(IntegralChannelBinaryDecisionTree *tree, TreeNode::shared_ptr node)
{
    std::vector<int> parentIndex;

    // depth one indicates that we expect 2 levels in the tree
    for (int d = 0; d <= 1; ++d)
    {
        //TreeNode::shared_ptr bestLeaf;
        std::vector<TreeNode::shared_ptr> nodes;
        std::vector<TreeNode::shared_ptr> nodes_tmp;
        TreeNode::getNodes(d, node, nodes);

        if (d ==1 and nodes.size() != 2){
            for(size_t i=nodes.size(); i < 2; i+=1)
            {
                TreeNode::getNodes(0, node, nodes_tmp);
                TreeNode::shared_ptr new_child( new TreeNode(*(nodes_tmp[0])));
                new_child->_parentId = new_child->_id;
                new_child->_id = TreeNode::getNewID();
                new_child->left.reset();
                new_child->right.reset();
                nodes.push_back(new_child);
            }

        }

        for (size_t k = 0; k < nodes.size(); ++k)
        {
            //maxdepth = 9
            IntegralChannelBinaryDecisionTreeNode *n = tree->add_nodes();
            n->set_id(nodes[k]->_id);
            n->set_parent_id(nodes[k]->_parentId);
            //if (nodes[k]->isLeaf()){
            //    //find parent
            //    for (size_t l = 0; l< nodes.size(); ++l){
            //        if (nodes[k]->parentId = nodes[l]->id){
            //            if (nodes[l]->getAlpha() ==1)
            //                n->set_parent_value(!isleft(nodes[k]->isLeft()));
            //            else
            //                n->set_parent_value(isleft(nodes[k]->isLeft()));
            //            break;
            //        }
            //
            //
            //    }
            //}else
            n->set_parent_value(nodes[k]->isLeft());
            setStump(n->mutable_decision_stump(), nodes[k]);
        }
    }
    return;
}

void ModelIO::setL2Tree_real(IntegralChannelBinaryDecisionTree *tree, const std::vector<TreeNode_v2> & treeNodes)
{

    IntegralChannelBinaryDecisionTreeNode *n = tree->add_nodes();
    n->set_id(0);
    n->set_parent_id(0);
    int trueNode =1;
    int falseNode =2;
    if (false && treeNodes[0]._betaTrue < 0.5){
            trueNode = 2;
            falseNode = 1;
    }


    n->set_parent_value(true);
    setStump_real(n->mutable_decision_stump(), treeNodes[0]);

    n = tree->add_nodes();
    n->set_id(1);
    n->set_parent_id(0);

    n->set_parent_value(true);
    setStump_real(n->mutable_decision_stump(), treeNodes[trueNode]);
   // n->mutable_decision_stump()->set_larger_than_threshold(trueNode==1);

    n = tree->add_nodes();
    n->set_id(2);
    n->set_parent_id(0);

    n->set_parent_value(false);
    setStump_real(n->mutable_decision_stump(), treeNodes[falseNode]);
   // n->mutable_decision_stump()->set_larger_than_threshold(trueNode==1);


}


void ModelIO::setLNTree(IntegralChannelBinaryDecisionTree *tree, TreeNode::shared_ptr node, int level)
{

    for (int d = 0; d <= level; ++d)
    {
        //TreeNode::shared_ptr bestLeaf;
        std::vector<TreeNode::shared_ptr> nodes;
        TreeNode::getNodes(d, node, nodes);

        for (size_t k = 0; k < nodes.size(); ++k)
        {
            //maxdepth = 9
            IntegralChannelBinaryDecisionTreeNode *n = tree->add_nodes();
            n->set_id(nodes[k]->_id);
            n->set_parent_id(nodes[k]->_parentId);
            //if (nodes[k]->isLeaf()){
            //    //find parent
            //    for (size_t l = 0; l< nodes.size(); ++l){
            //        if (nodes[k]->parentId = nodes[l]->id){
            //            if (nodes[l]->getAlpha() ==1)
            //                n->set_parent_value(!isleft(nodes[k]->isLeft()));
            //            else
            //                n->set_parent_value(isleft(nodes[k]->isLeft()));
            //            break;
            //        }
            //
            //
            //    }
            //}else
            n->set_parent_value(!nodes[k]->isLeft());

            setStump(n->mutable_decision_stump(), nodes[k]);

        }
    }
    return;
}

} // end of namespace boosted_learning
