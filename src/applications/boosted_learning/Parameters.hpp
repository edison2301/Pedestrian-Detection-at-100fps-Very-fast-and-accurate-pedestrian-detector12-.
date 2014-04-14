#ifndef PARAMETERS_HPP
#define	PARAMETERS_HPP

#include <iostream>
#include <iomanip>
#include <fstream>

#include <string>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <stdexcept>

namespace boosted_learning {

class Parameters
{
public:

    template <typename T>
    static T getParameter(const std::string param);

    ~Parameters();

    static void loadParameters(int argc, char **argv);
    static void showHelp();
    static Parameters &i();

    const boost::program_options::variables_map &getVariablesMap();

protected:

    Parameters();

    boost::program_options::variables_map variablesMap;
    boost::program_options::options_description options_descriptions;

    void load(const std::string fileName);
    void setOptionsDescriptions();

    static boost::shared_ptr<Parameters> s_instance;
};


template <class T>
T Parameters::getParameter(const std::string parameter)
{
    T ret;
    try
    {
        ret = i().variablesMap[parameter].as<T>();
    }
    catch (boost::bad_any_cast e)
    {
        throw std::runtime_error("parameter: " + parameter + " not found");
    }

    return ret;
}

} // end of namespace boosted_learning

#endif	/* PARAMETERS_HPP */

