#ifndef BOOSTED_LEARNING_IMAGEDATA_HPP
#define BOOSTED_LEARNING_IMAGEDATA_HPP

#include <string>

namespace boosted_learning {

class ImageData
{
public:
    std::string filename;
    int imageClass;
    int x;
    int y;
    double scale;
};


} // namespace boosted_learning

#endif // BOOSTED_LEARNING_IMAGEDATA_HPP
