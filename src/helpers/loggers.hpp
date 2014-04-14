#ifndef LOGGERS_HPP
#define LOGGERS_HPP

/**
  Helper definitions for loggers in multiple namespaces
  */

#include <ostream>

namespace bp {

    std::ostream &log_debug();

}

namespace common {

    std::ostream &log_debug();

}

namespace cuda {

    std::ostream &log_debug();

}

namespace depthmap {

    std::ostream &log_info();
    std::ostream &log_debug();

}


namespace video_input {

    std::ostream &log_info();
    std::ostream &log_warning();
    std::ostream &log_error();
    std::ostream &log_debug();

}


namespace detector {

    std::ostream &log_info();
    std::ostream &log_error();
    std::ostream &log_debug();

}

namespace geometry {

    std::ostream &log_debug();

}

namespace gpu {

    std::ostream &log_error();
    std::ostream &log_debug();
}

namespace graphics {

    std::ostream &log_info();
    std::ostream &log_error();
    std::ostream &log_debug();

}


namespace sfm {

    std::ostream &log_info();
    std::ostream &log_warning();
    std::ostream &log_error();
    std::ostream &log_debug();

}

namespace tracker {

    std::ostream &log_info();
    std::ostream &log_error();
    std::ostream &log_debug();

}

namespace utils {

    std::ostream &log_error();
    std::ostream &log_debug();

}


namespace vision {

    std::ostream &log_debug();
}


#endif // LOGGERS_HPP
