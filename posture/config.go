package posture

// Config carries every tunable threshold of the engine. All angle fields are
// in degrees; JSON field names are snake_case and shared with the HTTP API.
type Config struct {
	// MinVisibility is the landmark visibility below which a keypoint is
	// treated as missing. Range (0, 1].
	MinVisibility float64 `json:"min_visibility"`

	// NeckForwardMildDeg / NeckForwardSevereDeg grade forward head posture.
	NeckForwardMildDeg   float64 `json:"neck_forward_mild_deg"`
	NeckForwardSevereDeg float64 `json:"neck_forward_severe_deg"`

	// TorsoForwardMildDeg / TorsoBackMildDeg grade slouching forward and
	// leaning back. Severe grade reuses TorsoSevereDeg. TorsoLateral is
	// reported as a display metric only, not judged.
	TorsoForwardMildDeg  float64 `json:"torso_forward_mild_deg"`
	TorsoBackMildDeg     float64 `json:"torso_back_mild_deg"`
	TorsoSevereDeg       float64 `json:"torso_severe_deg"`

	// HeadTiltMildDeg / HeadTiltSevereDeg grade a tilted head.
	HeadTiltMildDeg   float64 `json:"head_tilt_mild_deg"`
	HeadTiltSevereDeg float64 `json:"head_tilt_severe_deg"`

	// LongSittingMinutes triggers a long-sitting reminder once the person has
	// been seated continuously for this long.
	LongSittingMinutes int `json:"long_sitting_minutes"`

	// HoldFrames is how many consecutive offending frames are required before
	// an issue is raised; ClearFrames is the consecutive clean-frame count
	// required to clear it. Higher values debounce noisy detections.
	HoldFrames  int `json:"hold_frames"`
	ClearFrames int `json:"clear_frames"`

	// AbsenceResetSeconds: after the person is absent for this long the
	// long-sitting timer resets (a short absence only pauses it).
	AbsenceResetSeconds int `json:"absence_reset_seconds"`
}

// DefaultConfig returns the built-in thresholds. They follow the angles used
// by common posture-detection projects (ear-shoulder line vs. vertical axis,
// torso line vs. vertical axis) and can be tuned at runtime via the API.
func DefaultConfig() Config {
	return Config{
		MinVisibility:       0.5,
		NeckForwardMildDeg:  20,
		NeckForwardSevereDeg: 35,
		TorsoForwardMildDeg: 20,
		TorsoBackMildDeg:    20,
		TorsoSevereDeg:      35,
		HeadTiltMildDeg:     12,
		HeadTiltSevereDeg:   25,
		LongSittingMinutes:  45,
		HoldFrames:          8,
		ClearFrames:         15,
		AbsenceResetSeconds: 120,
	}
}

// ErrInvalidConfig lists every constraint violated by cfg.
type ErrInvalidConfig struct{ Problems []string }

func (e *ErrInvalidConfig) Error() string {
	if len(e.Problems) == 0 {
		return "invalid config"
	}
	msg := "invalid config: " + e.Problems[0]
	for _, p := range e.Problems[1:] {
		msg += "; " + p
	}
	return msg
}

// Validate checks that cfg is self-consistent. It returns an
// *ErrInvalidConfig listing every problem found.
func (cfg Config) Validate() error {
	var problems []string
	positive := func(name string, v float64) {
		if v <= 0 {
			problems = append(problems, name+" must be > 0")
		}
	}
	if cfg.MinVisibility <= 0 || cfg.MinVisibility > 1 {
		problems = append(problems, "min_visibility must be in (0, 1]")
	}
	positive("neck_forward_mild_deg", cfg.NeckForwardMildDeg)
	positive("torso_forward_mild_deg", cfg.TorsoForwardMildDeg)
	positive("torso_back_mild_deg", cfg.TorsoBackMildDeg)
	positive("torso_severe_deg", cfg.TorsoSevereDeg)
	positive("head_tilt_mild_deg", cfg.HeadTiltMildDeg)
	positive("head_tilt_severe_deg", cfg.HeadTiltSevereDeg)
	if cfg.NeckForwardMildDeg >= cfg.NeckForwardSevereDeg {
		problems = append(problems, "neck_forward_mild_deg must be < neck_forward_severe_deg")
	}
	if cfg.TorsoForwardMildDeg >= cfg.TorsoSevereDeg {
		problems = append(problems, "torso_forward_mild_deg must be < torso_severe_deg")
	}
	if cfg.TorsoBackMildDeg >= cfg.TorsoSevereDeg {
		problems = append(problems, "torso_back_mild_deg must be < torso_severe_deg")
	}
	if cfg.HeadTiltMildDeg >= cfg.HeadTiltSevereDeg {
		problems = append(problems, "head_tilt_mild_deg must be < head_tilt_severe_deg")
	}
	if cfg.LongSittingMinutes <= 0 {
		problems = append(problems, "long_sitting_minutes must be > 0")
	}
	if cfg.HoldFrames <= 0 {
		problems = append(problems, "hold_frames must be > 0")
	}
	if cfg.ClearFrames <= 0 {
		problems = append(problems, "clear_frames must be > 0")
	}
	if cfg.AbsenceResetSeconds <= 0 {
		problems = append(problems, "absence_reset_seconds must be > 0")
	}
	if len(problems) > 0 {
		return &ErrInvalidConfig{Problems: problems}
	}
	return nil
}
