#include "any_to_string.hpp"

#include <boost/lexical_cast.hpp>

template<typename T>
bool try_type_to_string(const boost::any &object, std::string &the_string)
{
    if(boost::any_cast<T>(&object) != NULL)
    {
        the_string = boost::lexical_cast<std::string>(boost::any_cast<T>(object));
        return true;
    }

    return false;
}

std::string any_to_string(const boost::any &object)
{
    std::string the_string;

    bool success = false;
    if(success == false)
    {
       success = try_type_to_string<int>(object, the_string);
    }

    if(success == false)
    {
        success = try_type_to_string<bool>(object, the_string);
    }

    if (success == false)
    {
        success = try_type_to_string<float>(object, the_string);
    }

    if (success == false)
    {
        success = try_type_to_string<double>(object, the_string);
    }

    if (success == false)
    {
        success = try_type_to_string<short>(object, the_string);
    }

    if (success == false)
    {
        success = try_type_to_string<char>(object, the_string);
    }

    if (success == false)
    {
        success = try_type_to_string<std::string>(object, the_string);
    }

    if (success == false)
    {
        // could not guess the type, then we indicate the type
        the_string = object.type().name();
    }

    return the_string;
}

