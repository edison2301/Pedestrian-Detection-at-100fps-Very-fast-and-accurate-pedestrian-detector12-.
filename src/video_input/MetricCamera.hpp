#ifndef METRICCAMERA_HPP
#define METRICCAMERA_HPP

#include <Eigen/Core>

namespace doppia {

// forward declarations
class CameraCalibration;
class GroundPlane;

/// Utility object that allows to do multiple kinds of geometric
/// transformations based on the camera calibration
class MetricCamera
{
public:

    typedef Eigen::Matrix<float, 3,3> RotationMatrix;

    MetricCamera(const CameraCalibration &calibration);
    ~MetricCamera();

    const CameraCalibration &get_calibration() const;

    /// Project a 3d point to the image plane
    /// @note the input point should be in the same reference frame than the
    /// camera.get_pose() vector, <em>not in the camera reference frame</em>.
    /// This is specially important when using the right camera of a stereo setup
    Eigen::Vector2f project_3d_point(const Eigen::Vector3f &point) const;

    /// Project a 3d point on a ground plane into the corresponding 2d image in the image plane
    /// The x,y input coordinates are defined over the plane and
    /// the z axis moves along the plane normal.
    /// y moves toward the front of the camera.
    /// In that sense the z axis corresponds to the height over the plane
    /// x,y and height are in [meters]
    Eigen::Vector2f project_ground_plane_point(const GroundPlane &ground_plane,
                                               const float x, const float y, const float height) const;

    /// x,y and height are in [meters]
    Eigen::Vector2f project_ground_plane_point(const GroundPlane &ground_plane,
                                               const Eigen::Vector3f &point_on_ground_coordinates) const;

    /// We assume z == 0
    /// x and y are in [meters]
    Eigen::Vector2f project_ground_plane_point(const GroundPlane &ground_plane,
                                               const float x, const float y) const;

    ///
    ///
    Eigen::Vector3f back_project_2d_point_to_3d( const Eigen::Vector2f& point_2d, const float depth ) const;


protected:

    const CameraCalibration &calibration;

    Eigen::Vector3f up_axis, forward_axis, left_axis;
    RotationMatrix transposed_R;
    Eigen::Vector3f camera_focal_point;

    /// Pre-multiplied rotation matrix
    RotationMatrix KR;
    RotationMatrix KR_inverse;

    /// Pre-multiplied translation vector
    Eigen::Vector3f Kt;
};

} // end of namespace doppia

#endif // METRICCAMERA_HPP
