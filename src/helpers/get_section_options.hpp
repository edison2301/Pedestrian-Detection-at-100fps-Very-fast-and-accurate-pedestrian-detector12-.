#ifndef GET_SECTION_OPTIONS_HPP
#define GET_SECTION_OPTIONS_HPP


#include<string>
#include<boost/program_options.hpp>


boost::program_options::options_description get_section_options(const std::string section_name, const std::string group_caption,
                                                                const boost::program_options::options_description original_options_descriptions);

#endif // GET_SECTION_OPTIONS_HPP
