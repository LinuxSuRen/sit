// Package posture implements a rule-based sitting posture evaluation engine.
//
// It consumes 33 human pose landmarks (MediaPipe Pose Landmarker layout,
// normalized coordinates) and reports whether the sitting posture is
// acceptable. All geometry is pure Go so the engine is fully unit-testable
// and independent of where the landmarks come from (browser, ONNX model, ...).
package posture

import "time"

// Landmark is a single normalized 3D keypoint as produced by MediaPipe.
// X, Y are normalized to [0,1] relative to the image; Z is a relative depth
// value on roughly the same scale as X (smaller/negative means closer to the
// camera). Visibility is MediaPipe's confidence that the landmark is present.
type Landmark struct {
	X           float64 `json:"x"`
	Y           float64 `json:"y"`
	Z           float64 `json:"z"`
	Visibility  float64 `json:"visibility"`
}

// MediaPipe Pose Landmarker indices (subset used by the engine).
const (
	LandmarkNose = iota
	LandmarkLeftEyeInner
	LandmarkLeftEye
	LandmarkLeftEyeOuter
	LandmarkRightEyeInner
	LandmarkRightEye
	LandmarkRightEyeOuter
	LandmarkLeftEar
	LandmarkRightEar
	LandmarkLeftMouth
	LandmarkRightMouth
	LandmarkLeftShoulder
	LandmarkRightShoulder
	LandmarkLeftElbow
	LandmarkRightElbow
	LandmarkLeftWrist
	LandmarkRightWrist
	LandmarkLeftPinky
	LandmarkRightPinky
	LandmarkLeftIndex
	LandmarkRightIndex
	LandmarkLeftThumb
	LandmarkRightThumb
	LandmarkLeftHip
	LandmarkRightHip
)

// landmarkCount is the exact number of landmarks the engine expects.
const landmarkCount = 33

// Pose is one frame of body landmarks plus its capture timestamp.
type Pose struct {
	Landmarks []Landmark `json:"landmarks"`
	// View labels the camera angle the frame comes from: "front", "side" or
	// empty/"default" for single-camera mode (see the View* constants).
	View string `json:"view,omitempty"`
	Timestamp time.Time  `json:"-"`
	// TimestampMs is the epoch milliseconds of the frame (JSON transport).
	TimestampMs int64 `json:"timestamp_ms"`
}

// Time returns the capture time of the pose frame.
func (p Pose) Time() time.Time {
	if !p.Timestamp.IsZero() {
		return p.Timestamp
	}
	return time.UnixMilli(p.TimestampMs)
}

// valid reports whether the landmark at index i is usable:
// the slice must be complete and the visibility high enough.
func (p Pose) valid(i int, minVisibility float64) bool {
	if len(p.Landmarks) != landmarkCount || i < 0 || i >= landmarkCount {
		return false
	}
	return p.Landmarks[i].Visibility >= minVisibility
}

// midpoint returns the midpoint of two valid landmarks.
func (p Pose) midpoint(a, b int, minVisibility float64) (Landmark, bool) {
	if !p.valid(a, minVisibility) || !p.valid(b, minVisibility) {
		return Landmark{}, false
	}
	la, lb := p.Landmarks[a], p.Landmarks[b]
	return Landmark{
		X:          (la.X + lb.X) / 2,
		Y:          (la.Y + lb.Y) / 2,
		Z:          (la.Z + lb.Z) / 2,
		Visibility: (la.Visibility + lb.Visibility) / 2,
	}, true
}

// personPresent reports whether the landmarks required for the core (neck and
// head) metrics are reliable. Hips are optional: they may be out of frame
// when the camera only sees the upper body, in which case torso metrics are
// skipped instead of failing the whole frame.
func (p Pose) personPresent(minVisibility float64) bool {
	mandatory := []int{
		LandmarkLeftEar, LandmarkRightEar,
		LandmarkLeftShoulder, LandmarkRightShoulder,
	}
	for _, i := range mandatory {
		if !p.valid(i, minVisibility) {
			return false
		}
	}
	return true
}

// hipsVisible reports whether both hip landmarks are reliable enough to
// compute torso metrics.
func (p Pose) hipsVisible(minVisibility float64) bool {
	return p.valid(LandmarkLeftHip, minVisibility) &&
		p.valid(LandmarkRightHip, minVisibility)
}
