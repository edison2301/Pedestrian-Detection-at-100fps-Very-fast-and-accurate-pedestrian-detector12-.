#include "loggers.hpp"

#include "Log.hpp"

using namespace logging;



namespace bp {

    std::ostream &log_debug()
    {

        return log(DebugMessage, "bp");
    }

}


namespace common {

    std::ostream &log_debug()
    {

        return log(DebugMessage, "common");
    }

}

namespace cuda {

    std::ostream &log_debug()
    {

        return log(DebugMessage, "cuda");
    }

}

namespace depthmap {

    std::ostream &log_info()
    {
        return log(InfoMessage, "depthmap");
    }

    std::ostream &log_error()
    {

        return log(ErrorMessage, "depthmap");
    }


    std::ostream &log_debug()
    {

        return log(DebugMessage, "depthmap");
    }

}


namespace video_input {

    std::ostream &log_info()
    {
        return log(InfoMessage, "video_input");
    }

    std::ostream &log_warning()
    {
        return log(WarningMessage, "video_input");
    }


    std::ostream &log_error()
    {

        return log(ErrorMessage, "video_input");
    }

    std::ostream &log_debug()
    {

        return log(DebugMessage, "video_input");
    }

}


namespace detector {

    std::ostream &log_info()
    {
        return log(InfoMessage, "detector");
    }

    std::ostream &log_error()
    {

        return log(ErrorMessage, "detector");
    }

    std::ostream &log_debug()
    {

        return log(DebugMessage, "detector");
    }

}

namespace geometry {

    std::ostream &log_debug()
    {

        return log(DebugMessage, "geometry");
    }

}

namespace gpu {

    std::ostream &log_error()
    {

        return log(ErrorMessage, "gpu");
    }

    std::ostream &log_debug()
    {

        return log(DebugMessage, "gpu");
    }

}

namespace graphics {


    std::ostream &log_info()
    {
	

        return log(InfoMessage, "graphics");
    }


    std::ostream &log_error()
    {

        return log(ErrorMessage, "graphics");
    }


    std::ostream &log_debug()
    {

        return log(DebugMessage, "graphics");
    }

}


namespace sfm {

    std::ostream &log_info()
    {
        return log(InfoMessage, "sfm");
    }

    std::ostream &log_warning()
    {
        return log(WarningMessage, "sfm");
    }

    std::ostream &log_error()
    {
        return log(ErrorMessage, "sfm");
    }

    std::ostream &log_debug()
    {
        return log(DebugMessage, "sfm");
    }

}

namespace tracker {

    std::ostream &log_info()
    {
        return log(InfoMessage, "tracker");
    }

    std::ostream &log_error()
    {
        return log(ErrorMessage, "tracker");
    }

    std::ostream &log_debug()
    {
        return log(DebugMessage, "tracker");
    }

}

namespace utils {

    std::ostream &log_error()
    {

        return log(ErrorMessage, "utils");
    }


    std::ostream &log_debug()
    {

        return log(DebugMessage, "utils");
    }

}


namespace vision {

    std::ostream &log_debug()
    {

        return log(DebugMessage, "vision");
    }
}
