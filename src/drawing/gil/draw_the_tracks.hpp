#ifndef BICLOP_DRAW_THE_TRACKS_HPP
#define BICLOP_DRAW_THE_TRACKS_HPP

#include "objects_tracking/DummyObjectsTracker.hpp"
#include "objects_tracking/Dummy3dObjectsTracker.hpp"
#include "objects_tracking/Hypothesis.hpp"

#include <map>

namespace doppia {

// forward declaration
class MetricCamera;
class GroundPlane;


void draw_the_tracks(
        const DummyObjectsTracker::tracks_t &tracks,
        float &max_detection_score,
        const int additional_border,
        std::map<int, float> &track_id_to_hue,
        const boost::gil::rgb8_view_t &view);


void draw_the_tracks(
        const hypotheses_t &tracks,
        float &max_detection_score,
        const int additional_border,
        std::map<int, float> &track_id_to_hue,
        const MetricCamera &camera,
        const boost::gil::rgb8_view_t &view);

/// @param ground_plane in the world coordinates reference frame
void draw_the_tracks(
        const Dummy3dObjectsTracker::tracks_t &tracks,
        float &max_detection_score,
        const int additional_border,
        std::map<int, float> &track_id_to_hue,
        const GroundPlane &ground_plane,
        const MetricCamera &camera,
        const boost::gil::rgb8_view_t &view);

} // end of namespace doppia

#endif // BICLOP_DRAW_THE_TRACKS_HPP
