#ifndef BICLOP_OBJECTSTRACKERFACTORY_HPP
#define BICLOP_OBJECTSTRACKERFACTORY_HPP

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {

// forward declaration
class AbstractObjectsTracker;

class ObjectsTrackerFactory
{
public:
    static boost::program_options::options_description get_args_options();
    static AbstractObjectsTracker* new_instance(const boost::program_options::variables_map &options);
};

} // end of namespace doppia


#endif // BICLOP_OBJECTSTRACKERFACTORY_HPP
