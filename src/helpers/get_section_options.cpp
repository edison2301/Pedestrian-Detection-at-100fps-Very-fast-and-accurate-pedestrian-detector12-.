#include "get_section_options.hpp"


#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

using boost::shared_ptr;
using namespace boost::program_options;
using namespace std;

options_description get_section_options(const string section_name, const string group_caption, const options_description &original_options_descriptions)
{

    const std::vector< shared_ptr<option_description> >&original_options = original_options_descriptions.options();

    options_description new_options(group_caption.c_str());

    std::vector< shared_ptr<option_description> >::const_iterator options_it;
    for(options_it = original_options.begin(); options_it != original_options.end(); ++options_it)
    {

        const option_description &t_option = *(*options_it);
        string new_name(section_name + ".");
        new_name += t_option.long_name();


        // IMPORTANT: in order for this line not to crash the original_options_descriptions needs to be a static declaration
        // otherwise t_option.semantic().get() will point to an invalid memory direction and
        // the application will raise a segmentation fault
        new_options.add_options()
                (new_name.c_str(), t_option.semantic().get(), t_option.description().c_str());

    }

    return new_options;
}


