package posture

import (
	"errors"
	"sort"
)

// Baseline is the personal reference posture captured during calibration.
// Deviations against it, rather than absolute angles, drive the checks once
// a baseline is set — this cancels out any fixed camera tilt/perspective.
type Baseline struct {
	NeckForward  float64 `json:"neck_forward"`
	TorsoForward float64 `json:"torso_forward"`
	HeadTilt     float64 `json:"head_tilt"`
}

const (
	// minCalibrationFrames is the minimum number of valid frames required.
	minCalibrationFrames = 8
	// minTorsoFrames is the minimum hip-visible frames for a torso baseline.
	minTorsoFrames = 5
)

var (
	// ErrTooFewCalibrationFrames is returned when not enough valid frames
	// were captured to derive a stable baseline.
	ErrTooFewCalibrationFrames = errors.New("calibration needs at least 8 valid pose frames")
)

// CalibratePoses derives a baseline from a batch of pose frames by taking
// the median of each metric across the frames where a person is present.
//
// Frames without hip landmarks only contribute to the neck/head baseline;
// if too few frames show hips, the torso baseline stays 0 (torso checks are
// skipped anyway whenever hips are out of frame).
func CalibratePoses(poses []Pose, minVisibility float64) (Baseline, error) {
	var neckFwd, torsoFwd, tilt []float64
	for _, p := range poses {
		if !p.personPresent(minVisibility) {
			continue
		}
		m := computeMetrics(p, minVisibility)
		neckFwd = append(neckFwd, m.NeckForward)
		tilt = append(tilt, m.HeadTilt)
		if m.TorsoAvailable {
			torsoFwd = append(torsoFwd, m.TorsoForward)
		}
	}
	if len(neckFwd) < minCalibrationFrames {
		return Baseline{}, ErrTooFewCalibrationFrames
	}

	b := Baseline{
		NeckForward: median(neckFwd),
		HeadTilt:    median(tilt),
	}
	if len(torsoFwd) >= minTorsoFrames {
		b.TorsoForward = median(torsoFwd)
	}
	return b, nil
}

// median returns the middle value (average of the two middles for even n).
// An empty slice yields 0.
func median(v []float64) float64 {
	n := len(v)
	if n == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
