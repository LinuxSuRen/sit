package posture

import "math"

// Metrics holds the geometric measurements (degrees) the engine derives from
// one pose frame. Zero means perfectly upright for every angle.
type Metrics struct {
	// NeckForward is the sagittal-plane angle between the shoulder-midpoint →
	// ear-midpoint vector and the vertical axis. Positive means the head is
	// protruding forward (forward head posture), negative means pulled back.
	NeckForward float64 `json:"neck_forward"`
	// NeckLateral is the frontal-plane tilt of the neck line: positive means
	// the head leans towards the positive X side of the image.
	NeckLateral float64 `json:"neck_lateral"`
	// TorsoForward is the sagittal-plane angle of the hip-midpoint →
	// shoulder-midpoint vector. Positive means slouching/leaning forward,
	// negative means leaning back.
	TorsoForward float64 `json:"torso_forward"`
	// TorsoLateral is the frontal-plane lean of the torso line.
	TorsoLateral float64 `json:"torso_lateral"`
	// HeadTilt is the angle of the ear line against the horizontal axis.
	// Positive means the right ear (in mirrored, user-perspective
	// coordinates) is lower.
	HeadTilt float64 `json:"head_tilt"`
	// TorsoAvailable is false when the hips were not visible, in which case
	// the torso angles are zero-valued and must not be judged.
	TorsoAvailable bool `json:"torso_available"`
}

// Image-space axes convention (MediaPipe normalized coordinates):
//
//	x: towards the right in the image
//	y: towards the bottom of the image
//	z: towards the camera is NEGATIVE (relative depth, roughly x scale)
//
// "Up" is therefore (0, -1, 0) and "towards camera" is (0, 0, -1).

// sagittalForward returns the forward-lean angle in degrees of vector v
// relative to the vertical axis, projected onto the y-z plane. Positive means
// leaning towards the camera.
func sagittalForward(v Landmark) float64 {
	return deg(math.Atan2(-v.Z, -v.Y))
}

// frontalLateral returns the sideways tilt in degrees of vector v relative to
// the vertical axis, projected onto the x-y plane. Positive means tilting
// towards image-right.
func frontalLateral(v Landmark) float64 {
	return deg(math.Atan2(v.X, -v.Y))
}

// lineTilt returns the tilt in degrees of the line from a to b against the
// horizontal axis, normalized to [-90, 90].
func lineTilt(a, b Landmark) float64 {
	return deg(math.Atan2(b.Y-a.Y, b.X-a.X))
}

func deg(rad float64) float64 { return rad * 180 / math.Pi }

// computeMetrics derives all metrics from a pose frame. The caller must have
// verified that ears and shoulders are present; hips are optional.
func computeMetrics(p Pose, minVisibility float64) Metrics {
	earMid, _ := p.midpoint(LandmarkLeftEar, LandmarkRightEar, minVisibility)
	shoulderMid, _ := p.midpoint(LandmarkLeftShoulder, LandmarkRightShoulder, minVisibility)

	sub := func(a, b Landmark) Landmark {
		return Landmark{X: a.X - b.X, Y: a.Y - b.Y, Z: a.Z - b.Z}
	}

	neck := sub(earMid, shoulderMid) // shoulder → ear
	m := Metrics{
		NeckForward: sagittalForward(neck),
		NeckLateral: frontalLateral(neck),
		HeadTilt:    lineTilt(p.Landmarks[LandmarkLeftEar], p.Landmarks[LandmarkRightEar]),
	}
	if hipMid, ok := p.midpoint(LandmarkLeftHip, LandmarkRightHip, minVisibility); ok {
		torso := sub(shoulderMid, hipMid) // hip → shoulder
		m.TorsoForward = sagittalForward(torso)
		m.TorsoLateral = frontalLateral(torso)
		m.TorsoAvailable = true
	}
	return m
}
