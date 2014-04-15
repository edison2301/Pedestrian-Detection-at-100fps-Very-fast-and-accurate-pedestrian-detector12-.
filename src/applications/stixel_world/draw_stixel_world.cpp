#include "draw_stixel_world.hpp"

#include "drawing/gil/colors.hpp"
#include "drawing/gil/hsv_to_rgb.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/line.hpp"
#include "drawing/gil/draw_matrix.hpp"

#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"

#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "stereo_matching/stixels/ImagePlaneStixelsEstimator.hpp"

#include "video_input/calibration/StereoCameraCalibration.hpp"

#include "helpers/xyz_indices.hpp"
#include "helpers/Log.hpp"

#include <boost/foreach.hpp>

#include <boost/gil/extension/io/png_io.hpp>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "draw_stixel_world");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "draw_stixel_world");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "draw_stixel_world");
}

} // end of anonymous namespace


namespace doppia {

using namespace std;

void draw_the_stixels(boost::gil::rgb8_view_t &view, const stixels_t &the_stixels)
{

    const bool do_no_draw_stixels = false;
    //const bool do_no_draw_stixels = true; // only used for explanatory video creation
    if(do_no_draw_stixels)
    {
        return;
    }

    //const bool color_encoded_depth = true;
    const bool color_encoded_depth = true;

    const float saturation  = 0.7, value = 0.9;

    // Burnt orange (204, 85, 0)
    // according to http://en.wikipedia.org/wiki/Orange_(colour)
    const boost::gil::rgb8c_pixel_t burnt_orange =
            boost::gil::rgb8_view_t::value_type(204, 85, 0);


    const int line_tickness = 8;

    const int height = view.height();
    BOOST_FOREACH(const Stixel &t_stixel, the_stixels)
    {

        float hue = t_stixel.disparity / 128.0f;
        hue = 1 - (hue/2 + 0.5);
        const boost::gil::rgb8c_pixel_t depth_color = hsv_to_rgb(hue, saturation, value);

        const int &x = t_stixel.x;
        if(t_stixel.type == Stixel::Occluded)
        {
            view(x, t_stixel.top_y) = rgb8_colors::dark_violet;
            view(x, t_stixel.bottom_y) = rgb8_colors::dark_violet;
        }
        else
        {
            const bool draw_top = true;
            //const bool draw_top = false; // only used for explanatory video creation
            if(draw_top)
            {
                if(t_stixel.default_height_value == false)
                {
                    for(int y=max<int>(0, t_stixel.top_y-line_tickness); y < t_stixel.top_y; y+=1)
                    {
                        boost::gil::rgb8_pixel_t t_color = rgb8_colors::orange;
                        if(color_encoded_depth)
                        {
                            t_color = depth_color;
                        }

                        view(x, y) = t_color;
                    }
                    //view(x, t_stixel.top_y) = rgb8_colors::orange;
                }
                else
                {
                    boost::gil::rgb8_pixel_t t_color = burnt_orange;
                    if(color_encoded_depth)
                    {
                        t_color = depth_color;
                    }

                    for(int y=max<int>(0, t_stixel.top_y-line_tickness); y < t_stixel.top_y; y+=1)
                    {
                        view(x, y) = t_color;
                    }
                    //view(x, t_stixel.top_y) = t_color;
                }
            } // end of "if draw top"

            for(int y= t_stixel.bottom_y; y < min<int>(height, t_stixel.bottom_y+line_tickness); y+=1)
            {

                boost::gil::rgb8_pixel_t t_color = rgb8_colors::yellow;

                if(color_encoded_depth)
                {
                    t_color = depth_color;
                }

                view(x, y) = t_color;
            }
            //view(x, t_stixel.bottom_y) = rgb8_colors::yellow;
        }
    } // end of "for each stixel in stixels"

    return;
} // end of draw_the_stixels(...)


/// Draws the pedestrians bottom and top planes
void draw_the_ground_corridor(boost::gil::rgb8_view_t &view,
                              const MetricCamera& camera,
                              const GroundPlane &ground_plane)
{

    //const float min_z = 10, max_z = 50, z_step = 10;
    const float min_z = 2, max_z = 20, z_step = 1;
    const float far_left = -5.5; // -2.5 // [meters]
    const float far_right= 5.5; // 5.5  // [meters]

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




void draw_ground_plane_estimator(const GroundPlaneEstimator &ground_plane_estimator,
                                 const AbstractVideoInput::input_image_view_t &input_view,
                                 const StereoCameraCalibration &stereo_calibration,
                                 boost::gil::rgb8_view_t &screen_view)
{

    // copy right screen image ---
    copy_and_convert_pixels(input_view, screen_view);

    // draw v-disparity image in the right screen image --
    GroundPlaneEstimator::v_disparity_const_view_t raw_v_disparity_const_view =
            ground_plane_estimator.get_raw_v_disparity_view();

    boost::gil::rgb8_view_t screen_subview = boost::gil::subimage_view(screen_view,
                                                                       0,0,
                                                                       raw_v_disparity_const_view.width(),
                                                                       raw_v_disparity_const_view.height());
    copy_and_convert_pixels(raw_v_disparity_const_view, screen_subview);

    // copy v disparity into the blue channel -
    GroundPlaneEstimator::v_disparity_const_view_t v_disparity_const_view =
            ground_plane_estimator.get_v_disparity_view();
    copy_pixels(v_disparity_const_view, boost::gil::kth_channel_view<0>(screen_subview));


    draw_v_disparity_lines(ground_plane_estimator,
                           stereo_calibration,
                           screen_subview);

    return;
} // end of StixelWorldGui::draw_ground_plane_estimator(...)



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
            //printf("%.f\n", *data_it);
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


void draw_ground_plane_estimator(const FastGroundPlaneEstimator &ground_plane_estimator,
                                 const AbstractVideoInput::input_image_view_t &input_view,
                                 const StereoCameraCalibration &stereo_calibration,
                                 boost::gil::rgb8_view_t &screen_view)
{

    // copy right screen image ---
    copy_and_convert_pixels(input_view, screen_view);

    // draw v-disparity image in the right screen image --
    FastGroundPlaneEstimator::v_disparity_const_view_t raw_v_disparity_view =
            ground_plane_estimator.get_v_disparity_view();

    boost::gil::rgb8_view_t screen_subview = boost::gil::subimage_view(screen_view,
                                                                       0,0,
                                                                       raw_v_disparity_view.width(),
                                                                       screen_view.height());
    fill_pixels(screen_subview, boost::gil::rgb8_pixel_t());

    boost::gil::rgb8_view_t screen_subsubview = boost::gil::subimage_view(screen_subview,
                                                                          0, raw_v_disparity_view.height(),
                                                                          raw_v_disparity_view.width(),
                                                                          raw_v_disparity_view.height());
    Eigen::MatrixXf v_disparity_data;
    v_disparity_data_to_matrix(ground_plane_estimator.get_v_disparity(),
                               v_disparity_data);
    normalize_each_row(v_disparity_data);
    draw_matrix(v_disparity_data, screen_subsubview);

    if(false)
    {
        log_debug() << "(over)Writing ground_v_disparity_data.png" << std::endl;
        boost::gil::png_write_view("ground_v_disparity_data.png", screen_subsubview);
    }

    const bool draw_lines_on_top = true;
    if(draw_lines_on_top)
    {
        draw_v_disparity_lines(ground_plane_estimator,
                               stereo_calibration,
                               screen_subview);

        // draw the points used to estimate the objects
        typedef std::pair<int, int> point_t;
        const FastGroundPlaneEstimator::points_t &points = ground_plane_estimator.get_points();
        BOOST_FOREACH(const point_t &point, points)
        {
            *screen_subsubview.at(point.first, point.second) = rgb8_colors::orange;
        }
    }
    return;
} // end of StixelWorldGui::draw_ground_plane_estimator(const FastGroundPlaneEstimator &)




void draw_stixels_estimation(const StixelsEstimator &stixels_estimator,
                             const AbstractVideoInput::input_image_view_t &left_input_view,
                             boost::gil::rgb8_view_t &screen_left_view,
                             boost::gil::rgb8_view_t &screen_right_view)
{

    // draw left screen ---
    {
        copy_and_convert_pixels(left_input_view, screen_left_view);

        // cost on top -
        StixelsEstimator::u_disparity_cost_t u_disparity_cost =
                stixels_estimator.get_u_disparity_cost();

        // use M cost instead of u_disparity_cost
        //StixelsEstimator::u_disparity_cost_t u_disparity_cost =
        //        stixels_estimator.get_M_cost();

        const bool dirty_normalization_test = false; // FIXME just for debugging
        if(dirty_normalization_test)
        {
            // column wise normalization --
            for(int c=0; c < u_disparity_cost.cols(); c+=1)
            {
                const float col_max_value = u_disparity_cost.col(c).maxCoeff();
                const float col_min_value = u_disparity_cost.col(c).minCoeff();
                u_disparity_cost.col(c).array() -= col_min_value;

                if(col_max_value > col_min_value)
                {
                    const float scaling = 1.0f / (col_max_value - col_min_value);
                    u_disparity_cost.col(c) *= scaling;
                }
            }

            // log scaling --
            u_disparity_cost =
                    (u_disparity_cost.array() + 1).log();
        }


        boost::gil::rgb8_view_t left_top_sub_view =
                boost::gil::subimage_view(screen_left_view,
                                          0, 0,
                                          u_disparity_cost.cols(), u_disparity_cost.rows());

        draw_matrix(u_disparity_cost, left_top_sub_view);


        // draw estimated boundary -
        {
            const std::vector<int> &boundary =
                    stixels_estimator.get_u_disparity_ground_obstacle_boundary();
            for(std::size_t u=0; u < boundary.size(); u+=1)
            {
                const int &disparity = boundary[u];
                left_top_sub_view(u, disparity) = rgb8_colors::pink; // rgb8_colors::yellow;
            }
        }

        // draw the stixels -
        {
            draw_the_stixels(screen_left_view,
                             stixels_estimator.get_stixels());
        }
    } // end of draw left screen -

    // draw right screen ---
    {
        // since all matrices are computed to the left image,
        // it is more useful to see them compared to the left image
        //copy_and_convert_pixels(right_input_view, screen_right_view);
        copy_and_convert_pixels(left_input_view, screen_right_view);


        // objects cost on top --
        const StixelsEstimator::u_disparity_cost_t &objects_cost =
                stixels_estimator.get_object_u_disparity_cost();

        boost::gil::rgb8_view_t right_top_sub_view =
                boost::gil::subimage_view(screen_right_view,
                                          0, 0,
                                          objects_cost.cols(), objects_cost.rows());

        draw_matrix(objects_cost, right_top_sub_view);


        // ground cost on the bottom --
        const StixelsEstimator::u_disparity_cost_t &ground_cost =
                stixels_estimator.get_ground_u_disparity_cost();

        const int x_min = 0, y_min = std::max<int>(0, screen_left_view.height() - ground_cost.rows() );

        boost::gil::rgb8_view_t right_bottom_sub_view =
                boost::gil::subimage_view(screen_right_view,
                                          x_min, y_min,
                                          ground_cost.cols(), ground_cost.rows());

        draw_matrix(ground_cost, right_bottom_sub_view);
    }
    return;
}


void fill_cost_to_draw(const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost,
                       ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost_to_draw,
                       const int desired_height, const int stixel_width)
{

    typedef ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t cost_t;

    cost_t cost_without_max_value = cost;

    // we fix the lower left corner visualization problem
    {
        const float max_value = std::numeric_limits<float>::max();
        for(int row=0; row < cost.rows(); row +=1)
        {
            for(int column=0; column < cost.cols(); column +=1)
            {
                if(cost(row, column) == max_value)
                {
                    cost_without_max_value(row, column) = 0; // the lowest expected cost
                }
            }
        }

        const float true_max_value = cost_without_max_value.maxCoeff();
        for(int row=0; row < cost.rows(); row +=1)
        {
            for(int column=0; column < cost.cols(); column +=1)
            {
                if(cost(row, column) == max_value)
                {
                    cost_without_max_value(row, column) = true_max_value;
                }
            }
        }
    }

    const int pixels_per_row = desired_height / cost.cols();

    cost_to_draw = Eigen::MatrixXf::Zero(cost.cols()*pixels_per_row, cost.rows()*stixel_width);

    for(int row=0; row < cost_to_draw.rows(); row +=1)
    {
        for(int column=0; column < cost_to_draw.cols(); column +=1)
        {
            const int
                    row_step_index = row / pixels_per_row,
                    stixel_index = column / stixel_width;
            cost_to_draw(row, column) = cost_without_max_value(stixel_index, row_step_index);

        } // end of "for each row"
    } // end of "for each column"

    return;
}


void draw_stixels_estimation(const ImagePlaneStixelsEstimator &stixels_estimator,
                             const AbstractVideoInput::input_image_view_t &left_input_view,
                             boost::gil::rgb8_view_t &screen_left_view,
                             boost::gil::rgb8_view_t &screen_right_view)
{

    const int desired_cost_height = (left_input_view.height()*0.4);

    // draw left screen ---
    {
        copy_and_convert_pixels(left_input_view, screen_left_view);

        // draw the stixels --
        if(true)
        {
            draw_the_stixels(screen_left_view,
                             stixels_estimator.get_stixels());
        }

        // draw bottom candidates --
        {
            const ImagePlaneStixelsEstimator::row_given_stixel_and_row_step_t &candidate_bottom = \
                    stixels_estimator.row_given_stixel_and_row_step;
            for(size_t stixel_index=0; stixel_index < candidate_bottom.shape()[0]; stixel_index +=1)
            {

                const boost::uint16_t column = stixel_index * stixels_estimator.get_stixel_width();
                for(size_t row_step_index=0; row_step_index < candidate_bottom.shape()[1]; row_step_index +=1)
                {
                    const boost::uint16_t row = candidate_bottom[stixel_index][row_step_index];
                    const boost::uint16_t line_width = 2;
                    for(int t_row = row; (t_row < (row + line_width)) and (t_row < screen_left_view.height()); t_row +=1)
                    {
                        screen_left_view(column, t_row) = rgb8_colors::white;
                    }

                } // end of "for each row step"

            } // end of "for each stixel"
        }


        // cost on top --
        if(true)
        {
            const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost =
                    stixels_estimator.get_cost_per_stixel_and_row_step();

            ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t cost_to_draw;

            fill_cost_to_draw(cost, cost_to_draw, desired_cost_height, stixels_estimator.stixel_width);

            boost::gil::rgb8_view_t left_top_sub_view =
                    boost::gil::subimage_view(screen_left_view,
                                              0, 0,
                                              cost_to_draw.cols(), cost_to_draw.rows());

            draw_matrix(cost_to_draw, left_top_sub_view);


            // draw estimated boundary -
            {
                const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &cost =
                        stixels_estimator.get_cost_per_stixel_and_row_step();
                const int pixels_per_row = desired_cost_height / cost.cols();

                const std::vector<int> &boundary = stixels_estimator.get_stixel_and_row_step_ground_obstacle_boundary();

                for(std::size_t stixel_index=0; stixel_index < boundary.size(); stixel_index+=1)
                {
                    const int &row_step = boundary[stixel_index];
                    const int v =  pixels_per_row*row_step + (pixels_per_row/2);
                    const int begin_u = stixel_index*stixels_estimator.stixel_width,
                            end_u = std::min<int>((stixel_index+1)*stixels_estimator.stixel_width, left_top_sub_view.width());
                    for(int u=begin_u; u < end_u; u+=1)
                    {
                        left_top_sub_view(u, v) = rgb8_colors::pink;
                    }
                }
            }
        }

    } // end of draw left screen -

    // draw right screen ---
    {
        // since all matrices are computed to the left image,
        // it is more useful to see them compared to the left image
        //copy_and_convert_pixels(right_input_view, screen_right_view);
        copy_and_convert_pixels(left_input_view, screen_right_view);


        // objects cost on top --
        const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &object_cost =
                stixels_estimator.get_object_cost_per_stixel_and_row_step();

        ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t object_cost_to_draw;
        fill_cost_to_draw(object_cost, object_cost_to_draw, desired_cost_height, stixels_estimator.stixel_width);

        boost::gil::rgb8_view_t right_top_sub_view =
                boost::gil::subimage_view(screen_right_view,
                                          0, 0,
                                          object_cost_to_draw.cols(), object_cost_to_draw.rows());

        draw_matrix(object_cost_to_draw, right_top_sub_view);


        // ground cost on the bottom --
        const ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t &ground_cost =
                stixels_estimator.get_ground_cost_per_stixel_and_row_step();

        ImagePlaneStixelsEstimator::cost_per_stixel_and_row_step_t ground_cost_to_draw;
        fill_cost_to_draw(ground_cost, ground_cost_to_draw, desired_cost_height, stixels_estimator.stixel_width);


        const int x_min = 0, y_min = std::max<int>(0, screen_left_view.height() - ground_cost_to_draw.rows() );

        boost::gil::rgb8_view_t right_bottom_sub_view =
                boost::gil::subimage_view(screen_right_view,
                                          x_min, y_min,
                                          ground_cost_to_draw.cols(), ground_cost_to_draw.rows());

        draw_matrix(ground_cost_to_draw, right_bottom_sub_view);
    }
    return;
}



void draw_stixel_world(const stixels_t &the_stixels,
                       const AbstractVideoInput::input_image_view_t &left_input_view,
                       const AbstractVideoInput::input_image_view_t &right_input_view,
                       boost::gil::rgb8_view_t &screen_left_view,
                       boost::gil::rgb8_view_t &screen_right_view)
{

    // draw left input as background --
    copy_and_convert_pixels(left_input_view, screen_left_view);

    // draw the ground plane estimate ---
    //if(false)
    //{
    //    const GroundPlane &ground_plane = the_stixel_world_estimator_p->get_ground_plane();
    //    const MetricCamera &camera = video_input_p->get_metric_camera().get_left_camera();
    //    draw_the_ground_corridor(screen_left_view, camera, ground_plane);
    //}

    // draw the stixels ----
    {
        draw_the_stixels(screen_left_view, the_stixels);
    }

    // draw the right screen  ---
    {
        // copy right screen image ---
        copy_and_convert_pixels(right_input_view, screen_right_view);
    }

    return;
}


void draw_stixel_world(const stixels_t &the_stixels,
                       const Eigen::MatrixXf &depth_map,
                       const AbstractVideoInput::input_image_view_t &left_input_view,
                       boost::gil::rgb8_view_t &screen_left_view,
                       boost::gil::rgb8_view_t &screen_right_view)
{
    // draw left input as background --
    copy_and_convert_pixels(left_input_view, screen_left_view);

    // draw the ground plane estimate ---
    //if(false)
    //{
    //    const GroundPlane &ground_plane = the_stixel_world_estimator_p->get_ground_plane();
    //    const MetricCamera &camera = video_input_p->get_metric_camera().get_left_camera();
    //    draw_the_ground_corridor(screen_left_view, camera, ground_plane);
    //}

    // draw the stixels ----
    {
        draw_the_stixels(screen_left_view, the_stixels);
    }

    // draw the right screen  ---
    {
        // draw stixel_height_cost
        draw_matrix(depth_map, screen_right_view);
    }

    return;
}




void draw_stixel_match_lines( boost::gil::rgb8_view_t& left_screen_view,
                              boost::gil::rgb8_view_t& right_screen_view,
                              const stixels_t& left_screen_stixels,
                              const stixels_t& right_screen_stixels,
                              const std::vector< int >& stixel_matches )
{
    using rgb8_colors::jet_color_map;

    // Coefficients are in RBG order
    // The last entries are offset values
    const float  y_coeffs[ 4 ] = {  0.299,  0.587,   0.114,   0 };
    const float cb_coeffs[ 4 ] = { -0.169, -0.332,   0.500, 128 };
    const float cr_coeffs[ 4 ] = {  0.500, -0.419, -0.0813, 128 };

    // The coefficients are in YCbCr order
    // The last 3 elements in each array are the offset values in YCbCr order
    const double b_coeffs[ 6 ] = { 1.0,  1.7790,     0.0, 0.0,	-128.0,	    0.0 };
    const double g_coeffs[ 6 ] = { 1.0, -0.3455, -0.7169, 0.0,	-128.0,	 -128.0 };
    const double r_coeffs[ 6 ] = { 1.0,     0.0,  1.4075, 0.0,	   0.0,	 -128.0 };

    const size_t number_of_stixels = stixel_matches.size();

    const unsigned int stixel_sampling_width = 20; // Take the first of each 20
    const unsigned int color_stixel_sampling_width_ratio = 5;

    for(size_t i = 1; i < number_of_stixels; i+=1 )
    {
        if( (i % stixel_sampling_width) == 0 )
        {
            const int stixel_match = stixel_matches[ i ];

            if( stixel_match >= 0 )
            {
                const Stixel& right_stixel = right_screen_stixels[ i ];
                const Stixel& left_stixel = left_screen_stixels[ stixel_match ];

                if( right_stixel.type != Stixel::Occluded && left_stixel.type != Stixel::Occluded )
                {
                    const unsigned int color_index = ( i * color_stixel_sampling_width_ratio ) % number_of_stixels;

                    const boost::gil::rgb8c_pixel_t t_color = boost::gil::rgb8_view_t::value_type( jet_color_map[ color_index ][ 0 ],
                                                                                                   jet_color_map[ color_index ][ 1 ],
                                                                                                   jet_color_map[ color_index ][ 2 ] );

                    // Colorize stixels
                    const float color_cb = cb_coeffs[ 0 ] * t_color[ 0 ] +
                                           cb_coeffs[ 1 ] * t_color[ 1 ] +
                                           cb_coeffs[ 2 ] * t_color[ 2 ] +
                                           cb_coeffs[ 3 ];

                    const float color_cr = cr_coeffs[ 0 ] * t_color[ 0 ] +
                                           cr_coeffs[ 1 ] * t_color[ 1 ] +
                                           cr_coeffs[ 2 ] * t_color[ 2 ] +
                                           cr_coeffs[ 3 ];

                    // Colorize left stixel
                    for(int y = left_stixel.top_y; y < left_stixel.bottom_y; y+=1 )
                    {
                        const boost::gil::rgb8c_pixel_t t_pixel = left_screen_view( left_stixel.x, y );

                        const float color_y = y_coeffs[ 0 ] * t_pixel[ 0 ] +
                                              y_coeffs[ 1 ] * t_pixel[ 1 ] +
                                              y_coeffs[ 2 ] * t_pixel[ 2 ] +
                                              y_coeffs[ 3 ];

                        float color_r = r_coeffs[ 0 ] * ( color_y + r_coeffs[ 3 ] ) + r_coeffs[ 1 ] * ( color_cb + r_coeffs[ 4 ] ) + r_coeffs[ 2 ] * ( color_cr + r_coeffs[ 5 ] );
                        float color_g = g_coeffs[ 0 ] * ( color_y + g_coeffs[ 3 ] ) + g_coeffs[ 1 ] * ( color_cb + g_coeffs[ 4 ] ) + g_coeffs[ 2 ] * ( color_cr + g_coeffs[ 5 ] );
                        float color_b = b_coeffs[ 0 ] * ( color_y + b_coeffs[ 3 ] ) + b_coeffs[ 1 ] * ( color_cb + b_coeffs[ 4 ] ) + b_coeffs[ 2 ] * ( color_cr + b_coeffs[ 5 ] );

                        color_r = std::max< float >( 0, color_r ) ;
                        color_g = std::max< float >( 0, color_g ) ;
                        color_b = std::max< float >( 0, color_b ) ;

                        color_r = std::min< float >( 255, color_r ) ;
                        color_g = std::min< float >( 255, color_g ) ;
                        color_b = std::min< float >( 255, color_b ) ;

                        const int int_color_r = int( color_r + 0.5 );
                        const int int_color_g = int( color_g + 0.5 );
                        const int int_color_b = int( color_b + 0.5 );

                        const boost::gil::rgb8c_pixel_t t_new_color = boost::gil::rgb8_view_t::value_type( int_color_r, int_color_g, int_color_b );

                        left_screen_view( left_stixel.x, y ) = t_new_color;

                    }

                    // Colorize right stixel
                    for(int y = right_stixel.top_y; y < right_stixel.bottom_y; y+=1 )
                    {
                        const boost::gil::rgb8c_pixel_t t_pixel = right_screen_view( right_stixel.x, y );

                        const float color_y = y_coeffs[ 0 ] * t_pixel[ 0 ] +
                                              y_coeffs[ 1 ] * t_pixel[ 1 ] +
                                              y_coeffs[ 2 ] * t_pixel[ 2 ] +
                                              y_coeffs[ 3 ];

                        float color_r = r_coeffs[ 0 ] * ( color_y + r_coeffs[ 3 ] ) + r_coeffs[ 1 ] * ( color_cb + r_coeffs[ 4 ] ) + r_coeffs[ 2 ] * ( color_cr + r_coeffs[ 5 ] );
                        float color_g = g_coeffs[ 0 ] * ( color_y + g_coeffs[ 3 ] ) + g_coeffs[ 1 ] * ( color_cb + g_coeffs[ 4 ] ) + g_coeffs[ 2 ] * ( color_cr + g_coeffs[ 5 ] );
                        float color_b = b_coeffs[ 0 ] * ( color_y + b_coeffs[ 3 ] ) + b_coeffs[ 1 ] * ( color_cb + b_coeffs[ 4 ] ) + b_coeffs[ 2 ] * ( color_cr + b_coeffs[ 5 ] );

                        color_r = std::max< float >( 0, color_r ) ;
                        color_g = std::max< float >( 0, color_g ) ;
                        color_b = std::max< float >( 0, color_b ) ;

                        color_r = std::min< float >( 255, color_r ) ;
                        color_g = std::min< float >( 255, color_g ) ;
                        color_b = std::min< float >( 255, color_b ) ;

                        const int int_color_r = int( color_r + 0.5 );
                        const int int_color_g = int( color_g + 0.5 );
                        const int int_color_b = int( color_b + 0.5 );

                        const boost::gil::rgb8c_pixel_t t_new_color = boost::gil::rgb8_view_t::value_type( int_color_r, int_color_g, int_color_b );

                        right_screen_view( right_stixel.x, y ) = t_new_color;
                    }

                    // Draw top and bottom lines between stixels
                    int delta_top_y = right_stixel.top_y - left_stixel.top_y;
                    int delta_bottom_y = right_stixel.bottom_y - left_stixel.bottom_y;
                    int delta_x = left_screen_view.width() + right_stixel.x - left_stixel.x;

                    float slope_top_line = ( float( delta_top_y ) ) / delta_x;
                    float slope_bottom_line = ( float( delta_bottom_y ) ) / delta_x;

                    for( int x = left_stixel.x; x < left_screen_view.width(); ++x )
                    {
                        //                    left_screen_view( x, comon_bottom_y ) = t_color;
                        //                    left_screen_view( x, comon_top_y ) = t_color;

                        float y_top = slope_top_line * ( x - left_stixel.x ) + left_stixel.top_y;
                        float y_bottom = slope_bottom_line * ( x - left_stixel.x ) + left_stixel.bottom_y;

                        left_screen_view( x, int( y_top + 0.5 ) ) = t_color;
                        left_screen_view( x, int( y_bottom + 0.5 ) ) = t_color;
                    }

                    for( int x = right_stixel.x; x >= 0; --x )
                    {
                        //                    right_screen_view( x, comon_bottom_y ) = t_color;
                        //                    right_screen_view( x, comon_top_y ) = t_color;

                        float y_top = slope_top_line * ( x + left_screen_view.width() - left_stixel.x ) + left_stixel.top_y;
                        float y_bottom = slope_bottom_line * ( x + left_screen_view.width() - left_stixel.x ) + left_stixel.bottom_y;

                        right_screen_view( x, int( y_top + 0.5 ) ) = t_color;
                        right_screen_view( x, int( y_bottom + 0.5 ) ) = t_color;
                    }

                } // End of if( right_stixel is NOT Occluded && left_stixel is NOT Occluded )

            } // End of if( stixel_match >= 0 )
        }

    } // End of "for each stixel"

    return;
}





} // end of namespace doppia
