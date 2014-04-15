#ifndef BICLOP_CREATE_JSON_FOR_MUSTACHE_HPP
#define BICLOP_CREATE_JSON_FOR_MUSTACHE_HPP

#include "objects_detection/BaseIntegralChannelsDetector.hpp"

#include <vector>

namespace doppia {

void create_json_for_mustache(std::vector<BaseIntegralChannelsDetector::cascade_stages_t> &detection_cascade_per_scale);

} // end namespace doppia

#endif // BICLOP_CREATE_JSON_FOR_MUSTACHE_HPP
