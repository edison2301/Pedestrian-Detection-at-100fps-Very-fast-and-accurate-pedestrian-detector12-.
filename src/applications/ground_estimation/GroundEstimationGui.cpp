#include "GroundEstimationGui.hpp"

#include "SDL/SDL.h"

#include "GroundEstimationApplication.hpp"
#include "video_input/AbstractVideoInput.hpp"

#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"

#include "video_input/MetricStereoCamera.hpp"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/for_each.hpp"
#include "helpers/xyz_indices.hpp"

#include <boost/format.hpp>

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <opencv2/core/core.hpp>
#include "boost/gil/extension/opencv/ipl_image_wrapper.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/draw_matrix.hpp"

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "GroundEstimationGui");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "GroundEstimationGui");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "GroundEstimationGui");
}

} // end of anonymous namespace


namespace doppia
{

using boost::array;
using namespace boost::gil;

program_options::options_description GroundEstimationGui::get_args_options(void)
{
    program_options::options_description desc("GroundEstimationGui options");

    // specific options --
    desc.add_options()

            /*("gui.colorize_disparity",
                                                     program_options::value<bool>()->default_value(true),
                                                     "colorize the disparity map or draw it using grayscale")*/

            ;

    // add base options --
    BaseSdlGui::add_args_options(desc);

    return desc;
}


GroundEstimationGui::GroundEstimationGui(GroundEstimationApplication &_application,
                                         const program_options::variables_map &options)
    :BaseSdlGui(_application, options), application(_application)
{

    if(application.video_input_p == false)
    {
        throw std::runtime_error("GroundEstimationGui constructor expects that video_input is already initialized");
    }

    if((!application.ground_plane_estimator_p) and
       (!application.fast_ground_plane_estimator_p))
    {
        throw std::runtime_error("GroundEstimationGui constructor expects that ground_plane_estimator_p or  and fast_ground_plane_estimator_p is already initialized");
    }


    // retrieve program options --
    max_disparity = get_option_value<int>(options, "max_disparity");

    // create the application window --
    {
        const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
        const int input_width = left_input_view.width();
        const int input_height = left_input_view.height();

        BaseSdlGui::init_gui(application.get_application_title(), input_width, input_height);
    }

    // populate the views map --
    views_map[SDLK_1] = view_t(boost::bind(&GroundEstimationGui::draw_video_input, this), "draw_video_input");
    views_map[SDLK_2] = view_t(boost::bind(&GroundEstimationGui::draw_ground_plane_estimation, this), "draw_ground_plane_estimation");
    //views_map[SDLK_3] = view_t(boost::bind(&GroundEstimationGui::draw_stixel_world, this), "draw_stixel_world");

    // draw the first image --
    draw_video_input();

    // set the initial view mode --
    current_view = views_map[SDLK_2];

    return;
}


GroundEstimationGui::~GroundEstimationGui()
{
    // nothing to do here
    return;
}


void GroundEstimationGui::draw_video_input()
{
    const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
    const AbstractVideoInput::input_image_view_t &right_input_view = application.video_input_p->get_right_image();

    copy_and_convert_pixels(left_input_view, screen_left_view);
    copy_and_convert_pixels(right_input_view, screen_right_view);

    return;
}


/// Draws the pedestrians bottom and top planes
void draw_the_ground_corridor(boost::gil::rgb8_view_t &view,
                              const MetricCamera& camera,
                              const GroundPlane &ground_plane)
{

    //const float min_z = 10, max_z = 50, z_step = 10;
    const float min_z = 2, max_z = 20, z_step = 1;
    const float far_left = -5.5; // -2.5 // [meters]
    const float far_right=  5.5; // 5.5  // [meters]

    const float average_person_height = 1.8; // [meters]

    for (float z = min_z; z <= max_z; z+= z_step)
    {
        draw_ground_line(view, camera, ground_plane, rgb8_colors::blue,
                         far_left, z, far_right, z, 0.0);
        draw_ground_line(view, camera, ground_plane, rgb8_colors::red,
                         far_left, z, far_right, z, average_person_height);
    }

    for (float x = far_left; x <= far_right; x += 1) {
        draw_ground_line(view, camera, ground_plane, rgb8_colors::blue,
                         x, 2, x, 10, 0.0);
        draw_ground_line(view, camera, ground_plane, rgb8_colors::red,
                         x, 2, x, 10, average_person_height);
    }

    draw_ground_line(view, camera, ground_plane, rgb8_colors::dark_blue,
                     far_left, 5, far_right, 5, 0.0);
    draw_ground_line(view, camera, ground_plane, rgb8_colors::dark_red,
                     far_left, 5, far_right, 5, average_person_height);

    // draw horizon line --
    draw_horizon_line(view, camera, ground_plane, rgb8_colors::dark_green);

    return;
}


void GroundEstimationGui::draw_ground_plane_estimation()
{

    const BaseGroundPlaneEstimator *ground_plane_estimator_p = NULL;
    if(application.ground_plane_estimator_p)
    {
        ground_plane_estimator_p = application.ground_plane_estimator_p.get();
    }
    else
    {
        ground_plane_estimator_p = application.fast_ground_plane_estimator_p.get();
    }

    // Left screen --
    {
        // copy left screen image ---
        const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
        copy_and_convert_pixels(left_input_view, screen_left_view);

        if(application.fast_ground_plane_estimator_p and
           application.fast_ground_plane_estimator_p->is_computing_residual_image())
        {
            using namespace boost::gil;

            rgb8_view_t screen_subview = subimage_view(screen_left_view,
                                                       0, screen_left_view.height()/2,
                                                       screen_left_view.width(),
                                                       screen_left_view.height()/2);

            const rgb8c_view_t &left_half_view = application.fast_ground_plane_estimator_p->get_left_half_view();
            copy_pixels(left_half_view, screen_subview);
        }

        // add the ground bottom and top corridor ---
        const GroundPlane &ground_plane = ground_plane_estimator_p->get_ground_plane();
        const MetricCamera &camera = application.video_input_p->get_metric_camera().get_left_camera();
        draw_the_ground_corridor(screen_left_view, camera, ground_plane);

        // add our prior on the ground area ---
        const bool draw_prior_on_ground_area = true;
        if(draw_prior_on_ground_area)
        {
            const std::vector<int> &ground_object_boundary_prior =
                    ground_plane_estimator_p->get_ground_area_prior();
            const std::vector<int> &boundary = ground_object_boundary_prior;
            for(std::size_t u=0; u < boundary.size(); u+=1)
            {
                const int &disparity = boundary[u];
                screen_left_view(u, disparity) = rgb8_colors::yellow;
            }
        }

    } // end of left screen -

    // Right screen --
    {
        if(application.ground_plane_estimator_p)
        {
            // will draw on the right screen
            draw_ground_plane_estimator(*(application.ground_plane_estimator_p));
        }
        else
        {
            // will draw on the right screen
            draw_ground_plane_estimator(*(application.fast_ground_plane_estimator_p));
        }
    }


    return;
} // end of GroundEstimationGui::draw_ground_plane_estimation


// small helper function
template<typename ViewType, typename PixelType>
void draw_v_disparity_line(ViewType &view, PixelType &color, const GroundPlaneEstimator::line_t &v_disparity_line)
{
    const int x1 = 0, y1 = v_disparity_line.origin()(0);
    const int x2 = view.width(),
            y2 = v_disparity_line.origin()(0) + x2*v_disparity_line.direction()(0);

    draw_line(view, color, x1, y1, x2, y2);

    // printf("draw_v_disparity_line (origin, direction) == (%.2f, %.2f) -> (x2, y2) == (%i, %i)\n",
    //        v_disparity_line.origin()(0), v_disparity_line.direction()(0),
    //        x2, y2);
    return;
}


void draw_v_disparity_lines(const BaseGroundPlaneEstimator &ground_plane_estimator,
                            const StereoCameraCalibration &stereo_calibration,
                            const boost::gil::rgb8_view_t &screen_subview)
{


    // draw image center horizontal line --
    {
        const int image_center_y =
                stereo_calibration.get_left_camera_calibration().get_image_center_y();
        draw_line(screen_subview, rgb8_colors::gray,
                  0, image_center_y, screen_subview.width() -1, image_center_y);
    }


    // draw bounds on expected ground plane --
    {
        // plus line -
        draw_v_disparity_line(screen_subview, rgb8_colors::pink,
                              ground_plane_estimator.get_prior_max_v_disparity_line());

        // minus line -
        draw_v_disparity_line(screen_subview, rgb8_colors::pink,
                              ground_plane_estimator.get_prior_min_v_disparity_line());

        { // prior line -
            const GroundPlane &ground_plane_prior = ground_plane_estimator.get_ground_plane_prior();
            const GroundPlaneEstimator::line_t prior_v_disparity_line =
                    ground_plane_estimator.ground_plane_to_v_disparity_line(ground_plane_prior);
            draw_v_disparity_line(screen_subview, rgb8_colors::violet, prior_v_disparity_line);
        }

        if(false)
        {
            const GroundPlane &ground_plane_prior = ground_plane_estimator.get_ground_plane_prior();
            log_debug() << "draw_ground_plane_estimator ground_plane_prior height == "
                        << ground_plane_prior.offset() << " [meters]" << std::endl;
            const float theta = -std::asin(ground_plane_prior.normal()(i_z));
            log_debug() << "draw_ground_plane_estimator ground_plane_prior theta == "
                        << theta << " [radians] == " << (180/M_PI)*theta << " [degrees]" << std::endl;
        }
    }

    // draw estimated ground --
    {
        const GroundPlane &ground_plane_estimate = ground_plane_estimator.get_ground_plane();

        const GroundPlaneEstimator::line_t v_disparity_line =
                ground_plane_estimator.ground_plane_to_v_disparity_line(ground_plane_estimate);

        draw_v_disparity_line(screen_subview, rgb8_colors::yellow, v_disparity_line);

        if(false)
        {
            log_debug() << "ground plane estimate v_disparity_line has " <<
                           "origin == " << v_disparity_line.origin()(0) << " [pixels]"
                           " and direction == " << v_disparity_line.direction()(0) << " [-]" <<
                           std::endl;
        }

    }

    return;
}


void GroundEstimationGui::draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator)
{

    // copy right screen image ---
    const AbstractVideoInput::input_image_view_t &right_input_view = application.video_input_p->get_right_image();
    copy_and_convert_pixels(right_input_view, screen_right_view);

    // draw v-disparity image in the right screen image --
    GroundPlaneEstimator::v_disparity_const_view_t raw_v_disparity_view =
            ground_plane_estimator.get_raw_v_disparity_view();

    boost::gil::rgb8_view_t screen_subview = boost::gil::subimage_view(screen_right_view,
                                                                       0,0,
                                                                       raw_v_disparity_view.width(),
                                                                       raw_v_disparity_view.height());
    copy_and_convert_pixels(raw_v_disparity_view, screen_subview);

    // copy v disparity into the blue channel -
    GroundPlaneEstimator::v_disparity_const_view_t v_disparity_view =
            ground_plane_estimator.get_v_disparity_view();
    copy_pixels(v_disparity_view, boost::gil::kth_channel_view<0>(screen_subview));

    draw_v_disparity_lines(ground_plane_estimator,
                           application.video_input_p->get_stereo_calibration(),
                           screen_subview);

    return;
} // end of GroundEstimationGui::draw_ground_plane_estimator(const GroundPlaneEstimator &)



void v_disparity_data_to_matrix(const FastGroundPlaneEstimator::v_disparity_data_t &image_data,
                                Eigen::MatrixXf &image_matrix)
{
    typedef FastGroundPlaneEstimator::v_disparity_data_t::const_reference row_slice_t;

    const int rows = image_data.shape()[0], cols = image_data.shape()[1];
    image_matrix.setZero(rows, cols);
    for(int row=0; row < rows; row +=1)
    {
        row_slice_t row_slice = image_data[row];
        row_slice_t::const_iterator data_it = row_slice.begin();
        for(int col=0; col < cols; ++data_it, col+=1)
        {
            image_matrix(row, col) = *data_it;
        } // end "for each column"

    } // end of "for each row"

    return;
}


void normalize_each_row(Eigen::MatrixXf &matrix)
{
    //const float row_max_value = 255.0f;
    const float row_max_value = 1.0f;

    for(int row=0; row < matrix.rows(); row += 1)
    {
        const float t_min = matrix.row(row).minCoeff();
        const float t_max = matrix.row(row).maxCoeff();
        matrix.row(row).array() -= t_min;
        matrix.row(row) *= row_max_value/ (t_max - t_min);
    } // end of "for each row"

    return;
}


void GroundEstimationGui::draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator)
{

    // copy right screen image ---
    const AbstractVideoInput::input_image_view_t &right_input_view = application.video_input_p->get_right_image();
    copy_and_convert_pixels(right_input_view, screen_right_view);

    // draw v-disparity image in the right screen image --
    FastGroundPlaneEstimator::v_disparity_const_view_t raw_v_disparity_view =
            ground_plane_estimator.get_v_disparity_view();

    boost::gil::rgb8_view_t screen_subview = boost::gil::subimage_view(screen_right_view,
                                                                       0,0,
                                                                       raw_v_disparity_view.width(),
                                                                       screen_right_view.height());
    fill_pixels(screen_subview, rgb8_pixel_t());

    boost::gil::rgb8_view_t screen_subsubview = boost::gil::subimage_view(screen_subview,
                                                                          0, raw_v_disparity_view.height(),
                                                                          raw_v_disparity_view.width(),
                                                                          raw_v_disparity_view.height());
    Eigen::MatrixXf v_disparity_data;
    v_disparity_data_to_matrix(ground_plane_estimator.get_v_disparity(),
                               v_disparity_data);
    normalize_each_row(v_disparity_data);
    draw_matrix(v_disparity_data, screen_subsubview);
    //copy_and_convert_pixels(raw_v_disparity_view, screen_subsubview);

    // copy v disparity into the blue channel -
    //FastGroundPlaneEstimator::v_disparity_const_view_t v_disparity_view =
    //        ground_plane_estimator.get_v_disparity_view();
    //copy_pixels(v_disparity_view, boost::gil::kth_channel_view<0>(screen_subview));

    draw_v_disparity_lines(ground_plane_estimator,
                           application.video_input_p->get_stereo_calibration(),
                           screen_subview);

    // draw the points used to estimate the objects
    typedef std::pair<int, int> point_t;
    const FastGroundPlaneEstimator::points_t &points = ground_plane_estimator.get_points();
    BOOST_FOREACH(const point_t &point, points)
    {
        *screen_subsubview.at(point.first, point.second) = rgb8_colors::orange;
    }

    return;
} // end of GroundEstimationGui::draw_ground_plane_estimator(const FastGroundPlaneEstimator &)


void draw_the_stixels(boost::gil::rgb8_view_t &view, const stixels_t &the_stixels)
{
    BOOST_FOREACH(const Stixel &t_stixel, the_stixels)
    {
        const int &x = t_stixel.x;
        if(t_stixel.type == Stixel::Occluded)
        {
            view(x, t_stixel.top_y) = rgb8_colors::dark_violet;
            view(x, t_stixel.bottom_y) = rgb8_colors::dark_violet;
        }
        else
        {
            view(x, t_stixel.top_y) = rgb8_colors::orange;
            view(x, t_stixel.bottom_y) = rgb8_colors::yellow;
        }
    } // end of "for each stixel in stixels"

    return;
}



} // end of namespace doppia




