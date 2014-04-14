#include "draw_the_tracks.hpp"

#include "video_input/MetricCamera.hpp"
#include "video_input/calibration/CameraCalibration.hpp"

#include "stereo_matching/ground_plane/GroundPlane.hpp"

#include "objects_tracking/tracked_detections/AbstractTrackedDetection3d.hpp"
#include "objects_tracking/FrameData.hpp"
#include "objects_tracking/motion_models/AbstractMotionModel.hpp"
#include "objects_tracking/CameraPose.hpp"

#include "line.hpp"
#include "hsv_to_rgb.hpp"

#include <Eigen/Dense>

#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <boost/foreach.hpp>

#include <cstdio>


namespace doppia {

using namespace std;
using namespace boost;

namespace {
/// random_generator should be accessed only by one thread
boost::mt19937 random_generator;
} // end of anonymous namespace


gil::rgb8_pixel_t get_track_color(const int track_id,
                                  const float normalized_score,
                                  std::map<int, float> &track_id_to_hue)
{
    assert((normalized_score >= 0) and (normalized_score <= 1));


    //color = rgb8_colors::white;
    //color = gil::rgb8_pixel_t(normalized_score*255, 0, 0); // red box
    const float
            //value = 0.9,
            value = normalized_score,
            saturation = 0.8;


    if(track_id_to_hue.size() > 1000)
    {
        // to avoid memory leacks we reset the colors once in a while
        track_id_to_hue.clear();
    }


    if(track_id_to_hue.count(track_id) == 0)
    {
        boost::uniform_real<float> random_hue;
        // random generator should be accessed only by one thread
        track_id_to_hue[track_id] = random_hue(random_generator);
    }

    const float hue = track_id_to_hue[track_id];
    //printf("track_id_to_hue[%i] == %.3f\n", id, hue);
    //printf("track_id %i value == %.3f\n", id, value);
    return hsv_to_rgb(hue, saturation, value);
}


void draw_track(const DummyObjectsTracker::track_t &track,
                const int additional_border,
                const boost::gil::rgb8_pixel_t &color,
                const boost::gil::rgb8_view_t &view)
{

    // draw current bounding box --
    {
        DummyObjectsTracker::track_t::rectangle_t box = track.get_current_bounding_box();

        box.min_corner().x(box.min_corner().x() - additional_border);
        box.min_corner().y(box.min_corner().y() - additional_border);
        box.max_corner().x(box.max_corner().x() - additional_border);
        box.max_corner().y(box.max_corner().y() - additional_border);

        draw_rectangle(view, color, box, 4);
    }

    // draw tails
    {
        const AbstractObjectsTracker::detections_t &detections_in_time = track.get_detections_in_time();

        const AbstractObjectsTracker::detection_t::rectangle_t &bbox = detections_in_time.front().bounding_box;
        int
                previous_middle_low_point_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2,
                previous_middle_low_point_y = bbox.max_corner().y();

        BOOST_FOREACH(const AbstractObjectsTracker::detection_t &detection, detections_in_time)
        {
            const AbstractObjectsTracker::detection_t::rectangle_t &bbox = detection.bounding_box;

            const int
                    middle_low_point_x = (bbox.max_corner().x() + bbox.min_corner().x()) / 2,
                    middle_low_point_y = bbox.max_corner().y();

            draw_line(view, color, middle_low_point_x, middle_low_point_y,
                      previous_middle_low_point_x, previous_middle_low_point_y);

            previous_middle_low_point_x = middle_low_point_x;
            previous_middle_low_point_y = middle_low_point_y;
        } // end of "for each detection in time"

    }

    return;
}


void draw_the_tracks(
        const DummyObjectsTracker::tracks_t &tracks,
        float &max_detection_score,
        const int additional_border,
        std::map<int, float> &track_id_to_hue,
        const boost::gil::rgb8_view_t &view)
{

    typedef DummyObjectsTracker::track_t track_t;
    typedef DummyObjectsTracker::detection_t detection_t;
    typedef DummyObjectsTracker::detections_t detections_t;

    const float min_score = 0; // we will saturate at negative scores
    //float min_score = std::numeric_limits<float>::max();

    BOOST_FOREACH(const track_t &track, tracks)
    {
        //const float score = abs(track.get_current_detection().score);
        const float score = track.get_current_detection().score;
        //min_score = std::min(min_score, detection.score);
        max_detection_score = std::max(max_detection_score, score);
    }

    //printf("max_detection_score == %.3f\n", max_detection_score);

    const float scaling = 1.0 / (max_detection_score - min_score);

    BOOST_FOREACH(const track_t &track, tracks)
    {
        gil::rgb8_pixel_t color;

        // get track color
        {
            const int id = track.get_id();
            //const float score = abs(track.get_current_detection().score);
            const float score = track.get_current_detection().score;
            //printf("track_id %i score == %.3f\n", id, score);
            const float normalized_score = std::max(0.0f, (score - min_score)*scaling);

            color = get_track_color(id, normalized_score, track_id_to_hue);
        }

        draw_track(track, additional_border, color, view);

    } // end of "for each track"

    return;
}


void draw_hypothesis_bounding_box_2d(
        const Hypothesis& hypothesis,
        const Eigen::Vector3f& smoothed_point,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        const GroundPlane& ground_plane)
{
    using namespace Eigen;

    // Draw 2D bounding box
    const Vector3f
            ground_normal = ground_plane.normal(),
            hypothesis_height = ground_normal * (hypothesis.height + 0.2);

    Vector2f
            base = camera.project_3d_point(smoothed_point),
            top = camera.project_3d_point(smoothed_point + hypothesis_height);
    top(0) = base(0); // x axis

    const float
            object_height = (top - base).norm(),
            object_width = (object_height/96)*25; // FIXME hardcoded ratio

    const int line_width = 3;
    for(int c=0; c < line_width; c+=1)
    {
        draw_line(view, color,
                  top(0) - object_width-c, top(1)-c,
                  top(0) + object_width+c, top(1)+c);

        draw_line(view, color,
                  top(0) - object_width-c, top(1)-c,
                  top(0) - object_width+c, base(1)+c);

        draw_line(view, color,
                  top(0) + object_width-c, top(1)-c,
                  top(0) + object_width+c, base(1)+c);

        draw_line(view, color,
                  top(0) - object_width-c, base(1)-c,
                  top(0) + object_width + 1+c, base(1)+c);
    }

    return;
} // end of draw_hypothesis_bounding_box_2d


void draw_hypothesis_bounding_box_3d(
        const Hypothesis& hypothesis,
        const Eigen::Vector3f& smoothed_point,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        const GroundPlane& ground_plane,
        const Eigen::Matrix3f& homography)
{
    using namespace Eigen;

    const bool print_debug_information = false;
    if(print_debug_information)
    {

        std::stringstream camera_pose_stream;
        camera_pose_stream << camera.get_calibration().get_pose().translation.transpose();
        printf("Camera at pose %s\n", camera_pose_stream.str().c_str());

        std::stringstream point_stream;
        point_stream << smoothed_point.transpose();
        printf("Hypothesis %i, at position [%s]\n", hypothesis.id, point_stream.str().c_str());

        std::stringstream ground_plane_stream;
        ground_plane_stream << ground_plane.coeffs().transpose();
        printf("Ground plane %s\n", ground_plane_stream.str().c_str());

    } // end of "if print debug information"

    // Draw 3D bounding box
    const Vector3f ground_normal = ground_plane.normal();
    Vector3f direction, orthogonal;
    direction = ground_normal.cross(hypothesis.direction.cross(ground_normal));
    orthogonal = hypothesis.direction.cross(ground_normal);
    orthogonal = ground_normal.cross(orthogonal.cross(ground_normal));

    const float
            half_object_width = hypothesis.get_bounding_box_3d_size()(0),
            half_object_length = hypothesis.get_bounding_box_3d_size()(1);

    orthogonal *= half_object_width;
    direction *= half_object_length;

    Vector3f bounding_box[4];
    bounding_box[0] = -direction - orthogonal;
    bounding_box[1] = -direction + orthogonal;
    bounding_box[2] = direction + orthogonal;
    bounding_box[3] = direction - orthogonal;
    const Vector3f height_vector = ground_normal * hypothesis.height;

    for (size_t i = 0; i < 4; i+=1)
    {
        const Vector3f
                pp1 = smoothed_point + bounding_box[i],
                pp2 = smoothed_point + bounding_box[(i + 1)%4],
                pp3 = smoothed_point + bounding_box[i] + height_vector,
                pp4 = smoothed_point + bounding_box[(i + 1)%4] + height_vector;

        Vector3f // homogeneous 2d points
                p1 = homography * camera.project_3d_point(pp1).homogeneous(),
                p2 = homography * camera.project_3d_point(pp2).homogeneous(),
                p3 = homography * camera.project_3d_point(pp3).homogeneous(),
                p4 = homography * camera.project_3d_point(pp4).homogeneous();

        p1 /= p1(2); p2 /= p2(2); p3 /= p3(2); p4 /= p4(2);

        const int line_width = 3;
        for(int c=0; c < line_width; c+=1)
        {
            draw_line(view, color, p1(0)+c, p1(1), p2(0)+c, p2(1));
            draw_line(view, color, p1(0)+c, p1(1), p3(0)+c, p3(1));
            draw_line(view, color, p3(0)+c, p3(1), p4(0)+c, p4(1));
            draw_line(view, color, p4(0)+c, p4(1), p2(0)+c, p2(1));
        }
    } // end of "0,1,2,3"

    // Draw an arrow
    const bool is_moving_fast_enough = hypothesis.moving and (hypothesis.speed > 0.25);
    if (is_moving_fast_enough or true)
    {
        const Vector3f arrow_direction = direction * 0.4;

        const Vector3f
                pp1 = smoothed_point + height_vector*0.5,
                pp2 = smoothed_point + arrow_direction*0.8 + height_vector*0.5,
                pp3 = smoothed_point + arrow_direction*0.6 + height_vector*0.45,
                pp4 = smoothed_point + arrow_direction*0.6 + height_vector*0.55;

        Vector3f // homogenenous 2d points
                p1 = homography * camera.project_3d_point(pp1).homogeneous(),
                p2 = homography * camera.project_3d_point(pp2).homogeneous(),
                p3 = homography * camera.project_3d_point(pp3).homogeneous(),
                p4 = homography * camera.project_3d_point(pp4).homogeneous();

        p1 /= p1(2); p2 /= p2(2); p3 /= p3(2); p4 /= p4(2);

        const int line_width = 1;
        for(int c=0; c < line_width; c+=1)
        {
            draw_line(view, color, p1(0)+c, p1(1), p2(0)+c, p2(1));
            draw_line(view, color, p3(0)+c, p3(1), p2(0)+c, p2(1));
            draw_line(view, color, p4(0)+c, p4(1), p2(0)+c, p2(1));
        }
    } // end of "if is moving fast enough"


    return;
} // end of draw_hypothesis_bounding_box_3d


/// draw the trace of an hypothesis
/// @returns last smoothed point of the trajectory
Eigen::Vector3f
draw_hypothesis_trace(
        const Hypothesis& hypothesis,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        //const GroundPlane& ground_plane,
        const MetricCamera::Plane3d &camera_principal_plane,
        const Eigen::Matrix3f& homography,
        const int current_time_index)
{
    using namespace Eigen;

    const int line_width = 2;


    // Find index in trajectory that corresponds to this time-step
    const int max_delta_time = current_time_index - hypothesis.get_minimum_detection_time_index();

    // Smooth the trajectory points for display
    vector< Vector3f > smooth_trajectory(hypothesis.trajectory_points.size());
    const int smoothing_window = 5;
    for (int i = 0; i < (int)hypothesis.trajectory_points.size(); i+=1)
    {
        const int window_size = min(min(i, smoothing_window), (int)hypothesis.trajectory_points.size() - i - 1);
        Vector3f smoothed_point(0, 0, 0);
        for (int j = i - window_size; j <= i + window_size; j+=1)
        {
            smoothed_point += hypothesis.trajectory_points[j];
        }

        smooth_trajectory[i] = smoothed_point * (1.f / (2 * window_size + 1));
    } // end of "for each point in the trajectory"

    // Draw trajectory trace
    {
        Vector3f previous_point = Vector3f::Zero(); // 2d point in homogeneous coordinates
        for (int i = 0; i <= max_delta_time; i+=1)
        {
            const bool is_in_front_of_camera = camera_principal_plane.signedDistance(smooth_trajectory[i]) >  0.500;
            if (not is_in_front_of_camera)
            {
                // we skip points behind camera
                continue;
            }

            Vector3f point_2d =  homography * camera.project_3d_point(smooth_trajectory[i]).homogeneous();
            point_2d /= point_2d(2); // normalize the homogeneous point

            if (not hypothesis.moving)
            {
                // if it does not move, no need to draw the other overlapping points
                break;
            }

            if (i > 0)
            { // to make sure previous point is set

                for(int c=0; c < line_width; c+=1)
                {
                    draw_line(view, color,
                              point_2d(0)+c, point_2d(1),
                              previous_point(0)+c, previous_point(1));
                }
            }
            previous_point = point_2d;
        } // end of "for each delta time index"
    }

    return smooth_trajectory[max_delta_time];
}


/// Render a single hypothesis in the current image
/// @param h hypothesis to draw
/// @param color color to use for rendering
/// @param im image to render into
/// @param cam camera of current image
/// @param gp ground-plane (world frame)
/// @param H optional homography for 2D transformation (left to right image ?)
void draw_hypothesis(
        const Hypothesis& hypothesis,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        const GroundPlane& ground_plane,
        const Eigen::Matrix3f& homography,
        const int current_time_index)
{
    using namespace Eigen;

    const MetricCamera::Plane3d camera_principal_plane = camera.get_principal_plane();

    // we draw the "tail" of the hypothesis
    const Eigen::Vector3f point_to_draw =
            draw_hypothesis_trace(hypothesis, color, view,
                                  camera, /*ground_plane,*/ camera_principal_plane,
                                  homography, current_time_index);
    //const Eigen::Vector3f &point_to_draw = smooth_trajectory[max_delta_time];
    //const Eigen::Vector3f &point_to_draw = hypothesis.trajectory_points.back();

    // For terminated hypotheses, show only the trace
    // Don't draw the box if it's behind the camera (which, due to exit zones, shouldn't happen)
    const bool should_draw_the_bounding_box = \
            (not hypothesis.terminated)
            and (camera_principal_plane.signedDistance(point_to_draw) > 0);

    if (should_draw_the_bounding_box)
    {
        const bool draw_2d_bounding_box = false;
        if(draw_2d_bounding_box)
        {
            draw_hypothesis_bounding_box_2d(hypothesis, point_to_draw,
                                            color, view,
                                            camera, ground_plane);
        }
        else
        {
            draw_hypothesis_bounding_box_3d(hypothesis, point_to_draw,
                                            color, view,
                                            camera, ground_plane, homography);

        } // end of "draw 2d or 3d bounding box"
    } // end of "should draw the bounding box"

    // write trajectory ID
    /*{
        int id = hypothesis.id;
        if (hypothesis.getMaxDetTStamp() != current_time_index)
        {
            id = -id;
        }
        graphics::writeNumber(im, point(0) - 1, point(1) + 6, id, graphics::rgb8_t(0, 0, 0));
        graphics::writeNumber(im, point(0) + 1, point(1) + 6, id, graphics::rgb8_t(0, 0, 0));
        graphics::writeNumber(im, point(0), point(1) + 5, id, graphics::rgb8_t(0, 0, 0));
        graphics::writeNumber(im, point(0), point(1) + 7, id, graphics::rgb8_t(0, 0, 0));
        graphics::writeNumber(im, point(0), point(1) + 6, id, color);
    }*/

    /*
    // Check how many supporting detections from left/right camera
    int left_support = 0, right_support = 0;
    for (size_t i = 0; i < hypothesis.detections.size(); i+=1)
    {
        if (hypothesis.detections[i]->parent->getCameraIdx() == 0)
        {
            left_support++;
        }
        else
        {
            right_support++;
        }
    }*/

    //graphics::writeNumber(im, p(0) - 4, p(1) + 12, sl, graphics::rgb8_t(255, 0,0));
    //graphics::writeNumber(im, p(0) + 4, p(1) + 12, sr, graphics::rgb8_t(0, 255,0));

    // graphics::writeNumber(im, p(0) + 6, p(1) + 12, h.scoreMDL, color);
    // graphics::writeNumber(im, p(0), p(1) + 12, exp(h.score / (t - h.minDetTStamp())), color, 2);
    // graphics::writeNumber(im, p(0), p(1) + 18, h.occluded, color, 2);

    return;
} // end of function draw_hypothesis


void draw_the_tracks(const hypotheses_t &tracks,
                     float &max_detection_score,
                     const int additional_border,
                     std::map<int, float> &track_id_to_hue,
                     const MetricCamera &camera,
                     const boost::gil::rgb8_view_t &view)
{


    /*typedef DummyObjectsTracker::track_t track_t;
    typedef DummyObjectsTracker::detection_t detection_t;
    typedef DummyObjectsTracker::detections_t detections_t;
*/


    const float min_score = 0; // we will saturate at negative scores

    BOOST_FOREACH(const Hypothesis &hypothesis, tracks)
    {
        const Hypothesis::detection_t &detection = *(hypothesis.detections.back());
        max_detection_score = std::max(max_detection_score, detection.score);
    }

    //printf("max_detection_score == %.3f\n", max_detection_score);

    const float scaling = 1.0 / (max_detection_score - min_score);

    const Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();

    //log_debug() << "Drawing maximum of " << hypotheses.size() << " hypotheses..." << endl;
    //int num_hypotheses_drawed = 0;

    BOOST_FOREACH(const Hypothesis &hypothesis, tracks)
    {
        gil::rgb8_pixel_t color;

        // get track color
        {
            const Hypothesis::detection_t &detection = *(hypothesis.detections.back());
            const float score = detection.score;
            //printf("track_id %i score == %.3f\n", id, score);
            const float normalized_score = std::max(0.0f, (score - min_score)*scaling);

            color = get_track_color(hypothesis.id, normalized_score, track_id_to_hue);
        }

        //hypothesis.lastDrawn = true;

        //if ((occ_d > 0 and occ_d != hypos[i].id) or occ_s > 0) {
        if (hypothesis.probability_is_occluded == 1.0f)
        {
            // FIXME why is this commented ?
            // holor.R = max(0, (int)hcolor.R - 80);
            // color.G = max(0, (int)hcolor.G - 80);
            // color.B = max(0, (int)hcolor.B - 80);
        }


        // FIXME is this the good detection to consider ?
        const AbstractTrackedDetection3d &detection = *(hypothesis.detections.back());
        const FrameData &frame_data = *(detection.frame_data_p);

        draw_hypothesis(hypothesis, color, view,
                        camera,
                        frame_data.ground_plane,
                        homography,
                        frame_data.time_index);
        //draw_hypothesis(hypothesis, hcolor, image_right,
        //         frame_right.getCamera(), frame_right.getGroundplane(), H, frame_left.getTStamp());
        //num_hypotheses_drawed+=1;

    } // end of "for each track"


    //log_debug() << "Drawed (" << num_hypotheses_drawed << " hypotheses)." << std::endl;

    return;
}



/// draw the trace of a track
void draw_track_trace(
        const Dummy3dObjectsTracker::track_t &track,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        const GroundPlane& ground_plane,
        const MetricCamera::Plane3d &camera_principal_plane)
{
    using namespace Eigen;

    const int line_width = 2;

    typedef Dummy3dObjectsTracker::track_t::ground_position_in_time_t ground_position_in_time_t;
    const ground_position_in_time_t &ground_position_in_time = track.get_ground_position_in_time();

    // Smooth the trajectory points for display
    ground_position_in_time_t smooth_trajectory( ground_position_in_time.size() );
    const int
            smoothing_window = 5,
            num_points = static_cast<int>(ground_position_in_time.size());
    for (int i = 0; i < num_points; i+=1)
    {
        Vector2f &smoothed_point = smooth_trajectory[i];

        smoothed_point = Vector2f::Zero();

        const int window_size = min(min(i, smoothing_window), num_points - i - 1);
        int c = 0;
        for (int j = i - window_size, c=0; j <= i + window_size; j+=1, c+=1)
        {
            smoothed_point += ground_position_in_time.at(j);
        }

        if(c > 0)
        {
            smoothed_point /= c;
        }
        else
        {
            smoothed_point = ground_position_in_time[i];
        }
    } // end of "for each point in the trajectory"

    // Draw trajectory trace
    {
        Vector2f previous_point = Vector2f::Zero(); // 2d point on the ground
        for (size_t i = 0; i < smooth_trajectory.size(); i+=1)
        {
            const Eigen::Vector2f point_to_draw_xy = smooth_trajectory[i];

            // FIXME why is that minus sign needed ?
            const Eigen::Vector3f point_to_draw = \
                    camera.from_ground_to_3d(ground_plane, -point_to_draw_xy(0), point_to_draw_xy(1) );


            const bool is_in_front_of_camera = camera_principal_plane.signedDistance(point_to_draw) >  0.500;
            if (not is_in_front_of_camera)
            {
                // we skip points behind camera
                continue;
            }

            const Vector2f point_2d = camera.project_3d_point(point_to_draw);

            if (i > 0)
            { // to make sure previous point is set

                for(int c=0; c < line_width; c+=1)
                {
                    draw_line(view, color,
                              point_2d(0)+c, point_2d(1),
                              previous_point(0)+c, previous_point(1));
                }
            }
            previous_point = point_2d;
        } // end of "for each delta time index"
    }

    return;
}


void draw_track_bounding_box_3d(
        const Dummy3dObjectsTracker::track_t &track,
        const Eigen::Vector3f& point,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        const GroundPlane& ground_plane)
{
    using namespace Eigen;

    const Eigen::Vector3f object_direction = track.get_object_orientation();

    // Draw 3D bounding box
    const Vector3f ground_normal = ground_plane.normal();
    Vector3f direction, orthogonal;
    direction = ground_normal.cross(object_direction.cross(ground_normal));
    orthogonal = object_direction.cross(ground_normal);
    orthogonal = ground_normal.cross(orthogonal.cross(ground_normal));

    const int line_width = 2;

    float half_object_width = 0.5, half_object_length = 0.5; // default values, in meters

    if(track.object_class == Detection2d::Pedestrian)
    {
        // in meters
        half_object_width = 0.35;
        half_object_length = 0.35;
    }
    else if(track.object_class == Detection2d::Car)
    {
        // in meters
        half_object_width = 0.9;
        half_object_length = 2.0;
    }

    orthogonal *= half_object_width;
    direction *= half_object_length;

    Vector3f bounding_box[4];
    bounding_box[0] = -direction - orthogonal;
    bounding_box[1] = -direction + orthogonal;
    bounding_box[2] = direction + orthogonal;
    bounding_box[3] = direction - orthogonal;
    const Vector3f height_vector = ground_normal *  track.get_object_height();

    for (size_t i = 0; i < 4; i+=1)
    {
        const Vector3f
                pp1 = point + bounding_box[i],
                pp2 = point + bounding_box[(i + 1)%4],
                pp3 = point + bounding_box[i] + height_vector,
                pp4 = point + bounding_box[(i + 1)%4] + height_vector;

        const Vector2f // projected 2d points
                p1 = camera.project_3d_point(pp1),
                p2 = camera.project_3d_point(pp2),
                p3 = camera.project_3d_point(pp3),
                p4 = camera.project_3d_point(pp4);

        for(int c=0; c < line_width; c+=1)
        {
            draw_line(view, color, p1(0)+c, p1(1), p2(0)+c, p2(1));
            draw_line(view, color, p1(0)+c, p1(1), p3(0)+c, p3(1));
            draw_line(view, color, p3(0)+c, p3(1), p4(0)+c, p4(1));
            draw_line(view, color, p4(0)+c, p4(1), p2(0)+c, p2(1));
        }
    } // end of "0,1,2,3"


    // Draw an arrow
    const bool draw_arrow = true;
    if (draw_arrow)
    {
        const int arrow_line_width = 1;

        const Vector3f arrow_direction = direction * 0.4;

        const Vector3f
                pp1 = point + height_vector*0.5,
                pp2 = point + arrow_direction*0.8 + height_vector*0.5,
                pp3 = point + arrow_direction*0.6 + height_vector*0.45,
                pp4 = point + arrow_direction*0.6 + height_vector*0.55;

        const Vector2f // projected 2d points
                p1 = camera.project_3d_point(pp1),
                p2 = camera.project_3d_point(pp2),
                p3 = camera.project_3d_point(pp3),
                p4 = camera.project_3d_point(pp4);

        for(int c=0; c < arrow_line_width; c+=1)
        {
            draw_line(view, color, p1(0)+c, p1(1), p2(0)+c, p2(1));
            draw_line(view, color, p3(0)+c, p3(1), p2(0)+c, p2(1));
            draw_line(view, color, p4(0)+c, p4(1), p2(0)+c, p2(1));
        }
    } // end of "if is moving fast enough"


    return;
} // end of draw_hypothesis_bounding_box_3d


/// Render a single track in the current image
/// @param track to draw
/// @param color color to use for rendering
/// @param im image to render into
/// @param cam camera of current image
/// @param gp ground-plane (world frame)
void draw_track(
        const Dummy3dObjectsTracker::track_t &track,
        const boost::gil::rgb8_pixel_t &color,
        const boost::gil::rgb8_view_t &view,
        const MetricCamera& camera,
        const GroundPlane& ground_plane)
{
    using namespace Eigen;

    //const CameraPose camera_pose = camera.get_calibration().get_pose();
    const MetricCamera::Plane3d camera_principal_plane = camera.get_principal_plane();

    // Draw trace --
    draw_track_trace(track, color, view,camera, ground_plane, camera_principal_plane);



    // Draw 3d bounding box --
    const Eigen::Vector2f point_to_draw_xy = track.get_position_on_the_ground();
    // FIXME why is that minus sign needed ?
    const Eigen::Vector3f point_to_draw = \
            camera.from_ground_to_3d(ground_plane, -point_to_draw_xy(0), point_to_draw_xy(1) );
    //const Eigen::Vector3f point_to_draw(point_to_draw_xy(0), -point_to_draw_xy(2), point_to_draw_xy(1));
    //const Eigen::Vector3f point_to_draw = camera_pose.from_world_to_camera_coordinates(point_to_draw_xy);


    const float distance_to_plane = camera_principal_plane.signedDistance(point_to_draw);

    // For terminated hypotheses, show only the trace
    // Don't draw the box if it's behind the camera (which, due to exit zones, shouldn't happen)
    const bool in_front_of_camera = distance_to_plane > 0;

    if (in_front_of_camera)
    {
        draw_track_bounding_box_3d(track,
                                   point_to_draw,
                                   color, view,
                                   camera, ground_plane);

    } // end of "should draw the bounding box"



    const bool print_debug_information = false;
    if(print_debug_information)
    {
        std::stringstream camera_pose_stream;
        camera_pose_stream << camera.get_calibration().get_pose().translation.transpose();
        std::stringstream camera_plane_stream;
        camera_plane_stream << camera_principal_plane.coeffs().transpose();
        printf("Camera at pose %s, camera plane %s\n",
               camera_pose_stream.str().c_str(), camera_plane_stream.str().c_str());

        std::stringstream point_stream;
        point_stream << point_to_draw.transpose();
        printf("Track %i, at position [%s], has a distance to the plane == %.3f\n",
               track.get_id(), point_stream.str().c_str(), distance_to_plane);

        std::stringstream ground_plane_stream;
        ground_plane_stream << ground_plane.coeffs().transpose();
        printf("Ground plane %s\n",
               ground_plane_stream.str().c_str());

    } // end of "if print debug information"


    return;
} // end of function draw_track for Dummy3dObjectsTracker::track_t


void draw_the_tracks(
        const Dummy3dObjectsTracker::tracks_t &tracks,
        float &max_detection_score,
        const int /*additional_border*/,
        std::map<int, float> &track_id_to_hue,
        const GroundPlane &ground_plane,
        const MetricCamera &camera,
        const boost::gil::rgb8_view_t &view)
{

    const float min_score = 0; // we will saturate at negative scores


    BOOST_FOREACH(const Dummy3dObjectsTracker::track_t &track, tracks)
    {
        const float score = track.get_current_detection().score;
        max_detection_score = std::max(max_detection_score, score);
    }

    //printf("max_detection_score == %.3f\n", max_detection_score);

    const float scaling = 1.0 / (max_detection_score - min_score);


    BOOST_FOREACH(const Dummy3dObjectsTracker::track_t &track, tracks)
    {
        gil::rgb8_pixel_t color;

        // get track color
        {
            const float score = track.get_current_detection().score;

            //printf("track_id %i score == %.3f\n", track.get_id(), score);
            const float normalized_score = std::max(0.0f, (score - min_score)*scaling);

            color = get_track_color(track.get_id(), normalized_score, track_id_to_hue);
        }

        draw_track(track, color, view, camera, ground_plane);

    } // end of "for each track"


    return;
}


} // end of namespace doppia
