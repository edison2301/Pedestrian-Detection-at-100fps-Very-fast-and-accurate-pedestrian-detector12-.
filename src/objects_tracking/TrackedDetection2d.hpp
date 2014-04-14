#ifndef BICLOP_TRACKEDDETECTION2D_HPP
#define BICLOP_TRACKEDDETECTION2D_HPP

#include "AbstractObjectsTracker.hpp"

namespace doppia {


/// Helper class for DummyObjectsTracker
/// Represents a track in the 2d image plane (only)
class TrackedDetection2d {

public:
    typedef Detection2d::ObjectClasses class_t;

    typedef Detection2d::rectangle_t rectangle_t;
    typedef AbstractObjectsTracker::detection_t detection_t;
    typedef AbstractObjectsTracker::detections_t detections_t;


public:
    TrackedDetection2d(const int id, const detection_t &detection, const int max_extrapolation_length_);
    ~TrackedDetection2d();

    /// found a match, we move forward in time
    void add_matched_detection(const detection_t &detection);

    /// we did not found a match, we move forward in time
    void skip_one_detection();

    int get_max_extrapolation_length() const;
    int get_extrapolation_length() const;
    size_t get_length() const;

    const detection_t &get_current_detection() const;
    const rectangle_t &get_current_bounding_box() const;

    const detections_t &get_detections_in_time() const;
    const int get_id() const;

    rectangle_t compute_extrapolated_bounding_box();

    void set_current_bounding_box_as_occluded();

public:
    class_t object_class;


protected:

    int track_id;
    rectangle_t current_bounding_box;
    detections_t detections_in_time;
    float max_detection_score;


    const int max_extrapolation_length;
    int num_extrapolated_detections, num_true_detections_in_time;
    int num_consecutive_detections, max_consecutive_detections;
};

} // end of namespace doppia

#endif // BICLOP_TRACKEDDETECTION2D_HPP
