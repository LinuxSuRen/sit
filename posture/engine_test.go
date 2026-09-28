package posture

import (
	"math"
	"reflect"
	"testing"
	"time"
)

var base = time.Date(2025, 1, 1, 9, 0, 0, 0, time.UTC)

// testPose builds a 33-landmark pose from a handful of knobs:
//
//   - earZ/shoulderZ/hipZ shift depth (negative = closer to camera)
//   - rightEarDrop lowers (+) or raises (-) the user's right ear in the
//     mirrored, user-perspective coordinates the engine expects
//   - hipVis controls hip visibility (0.1 hides the hips)
func testPose(earZ, shoulderZ, hipZ, rightEarDrop, hipVis float64, ts time.Time) Pose {
	lm := make([]Landmark, landmarkCount)
	for i := range lm {
		lm[i] = Landmark{Visibility: 1}
	}
	set := func(i int, x, y, z float64) { lm[i] = Landmark{X: x, Y: y, Z: z, Visibility: 1} }

	set(LandmarkNose, 0.5, 0.18, 0)
	set(LandmarkLeftEye, 0.48, 0.22, 0)
	set(LandmarkRightEye, 0.52, 0.22, 0)
	set(LandmarkLeftEar, 0.47, 0.25-rightEarDrop/2, earZ)
	set(LandmarkRightEar, 0.53, 0.25+rightEarDrop/2, earZ)
	set(LandmarkLeftShoulder, 0.45, 0.40, shoulderZ)
	set(LandmarkRightShoulder, 0.55, 0.40, shoulderZ)
	set(LandmarkLeftHip, 0.46, 0.75, hipZ)
	set(LandmarkRightHip, 0.54, 0.75, hipZ)
	if hipVis < 1 {
		lm[LandmarkLeftHip].Visibility = hipVis
		lm[LandmarkRightHip].Visibility = hipVis
	}
	return Pose{Landmarks: lm, Timestamp: ts}
}

func degClose(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func TestComputeMetricsUpright(t *testing.T) {
	m := computeMetrics(testPose(0, 0, 0, 0, 1, base), 0.5)
	if !m.TorsoAvailable {
		t.Fatal("expected torso available with visible hips")
	}
	for name, v := range map[string]float64{
		"neck_forward": m.NeckForward, "neck_lateral": m.NeckLateral,
		"torso_forward": m.TorsoForward, "torso_lateral": m.TorsoLateral,
		"head_tilt": m.HeadTilt,
	} {
		if !degClose(v, 0, 0.01) {
			t.Errorf("%s = %.2f°, want ~0", name, v)
		}
	}
}

func TestComputeMetricsForwardHead(t *testing.T) {
	m := computeMetrics(testPose(-0.15, 0, 0, 0, 1, base), 0.5) // ears closer to camera
	want := 45.0 // atan2(0.15, 0.15)
	if !degClose(m.NeckForward, want, 0.5) {
		t.Errorf("neck_forward = %.2f°, want ~%.0f°", m.NeckForward, want)
	}
}

func TestComputeMetricsTorsoAngles(t *testing.T) {
	fwd := computeMetrics(testPose(0, -0.15, 0, 0, 1, base), 0.5)
	if want := 23.2; !degClose(fwd.TorsoForward, want, 0.5) {
		t.Errorf("torso_forward = %.2f°, want ~%.1f°", fwd.TorsoForward, want)
	}
	back := computeMetrics(testPose(0, 0.15, 0, 0, 1, base), 0.5)
	if want := -23.2; !degClose(back.TorsoForward, want, 0.5) {
		t.Errorf("torso_forward = %.2f°, want ~%.1f°", back.TorsoForward, want)
	}
}

func TestComputeMetricsHeadTiltDirections(t *testing.T) {
	right := computeMetrics(testPose(0, 0, 0, 0.03, 1, base), 0.5) // right ear lower
	if want := 26.57; !degClose(right.HeadTilt, want, 0.5) {
		t.Errorf("head_tilt = %.2f°, want ~%.2f° (right)", right.HeadTilt, want)
	}
	left := computeMetrics(testPose(0, 0, 0, -0.03, 1, base), 0.5)
	if want := -26.57; !degClose(left.HeadTilt, want, 0.5) {
		t.Errorf("head_tilt = %.2f°, want ~%.2f° (left)", left.HeadTilt, want)
	}
}

func TestComputeMetricsHipsHidden(t *testing.T) {
	m := computeMetrics(testPose(0, 0, 0, 0, 0.1, base), 0.5)
	if m.TorsoAvailable {
		t.Error("torso should be unavailable when hips have low visibility")
	}
	if !degClose(m.NeckForward, 0, 0.01) {
		t.Error("neck metric should still be computed")
	}
}

func TestEngineGoodPostureStaysGood(t *testing.T) {
	e := newCalibratedEngine(DefaultConfig())
	for i := 0; i < 30; i++ {
		r := e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(time.Duration(i)*time.Second)))
		if r.Status != StatusGood {
			t.Fatalf("frame %d: status = %s, want good", i, r.Status)
		}
		if len(r.Issues) != 0 {
			t.Fatalf("frame %d: unexpected issues %v", i, r.Issues)
		}
		if r.SeatedSeconds < float64(i)-1 || r.SeatedSeconds > float64(i)+1 {
			t.Fatalf("frame %d: seated = %.1fs, want ~%d", i, r.SeatedSeconds, i)
		}
	}
}

// engineCfg returns a fast-debouncing config for rule tests.
func engineCfg() Config {
	c := DefaultConfig()
	c.HoldFrames = 3
	c.ClearFrames = 2
	c.LongSittingMinutes = 1000 // keep out of the way
	return c
}

// newCalibratedEngine returns an engine with an all-zero baseline so checks
// judge absolute angles (the calibration reference is "upright = 0°").
func newCalibratedEngine(cfg Config) *Engine {
	e, err := NewEngine(cfg)
	if err != nil {
		panic(err)
	}
	e.SetBaseline(Baseline{})
	return e
}

func TestEngineForwardHeadDebounce(t *testing.T) {
	e := newCalibratedEngine(engineCfg())
	bad := testPose(-0.15, 0, 0, 0, 1, base) // 45° forward → severe

	// frames 1..2: hit but not yet active
	for i := 1; i <= 2; i++ {
		r := e.Evaluate(bad.WithTime(base.Add(time.Duration(i) * time.Second)))
		if r.Status != StatusGood || len(r.Issues) != 0 {
			t.Fatalf("frame %d: issue raised too early: %+v", i, r.Issues)
		}
	}
	// frame 3: HoldFrames reached → severe forward_head
	r := e.Evaluate(bad.WithTime(base.Add(3 * time.Second)))
	if r.Status != StatusBad {
		t.Fatalf("status = %s, want bad", r.Status)
	}
	if len(r.Issues) != 1 || r.Issues[0].Code != IssueForwardHead || r.Issues[0].Severity != SeveritySevere {
		t.Fatalf("issues = %+v, want single severe forward_head", r.Issues)
	}

	// one clean frame is not enough (ClearFrames = 2)
	r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(4*time.Second)))
	if len(r.Issues) != 1 {
		t.Fatalf("issue cleared after 1 clean frame: %+v", r.Issues)
	}
	// second clean frame clears it
	r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(5*time.Second)))
	if r.Status != StatusGood || len(r.Issues) != 0 {
		t.Fatalf("issue not cleared after 2 clean frames: status=%s %+v", r.Status, r.Issues)
	}
}

func TestEngineTorsoLeanDirections(t *testing.T) {
	cfg := engineCfg()

	fwd := newCalibratedEngine(cfg)
	r := fwd.Evaluate(testPose(0, -0.15, 0, 0, 1, base.Add(time.Second)))
	r = fwd.Evaluate(testPose(0, -0.15, 0, 0, 1, base.Add(2*time.Second)))
	r = fwd.Evaluate(testPose(0, -0.15, 0, 0, 1, base.Add(3*time.Second)))
	wantCodes := []string{IssueTorsoLeanFwd}
	got := issueCodes(r.Issues)
	if !reflect.DeepEqual(got, wantCodes) {
		t.Fatalf("forward lean issues = %v, want %v", got, wantCodes)
	}

	back := newCalibratedEngine(cfg)
	for i := 1; i <= cfg.HoldFrames; i++ {
		// head moves back with the torso, so the neck itself stays neutral
		r = back.Evaluate(testPose(0.15, 0.15, 0, 0, 1, base.Add(time.Duration(i)*time.Second)))
	}
	got = issueCodes(r.Issues)
	if !reflect.DeepEqual(got, []string{IssueTorsoLeanBack}) {
		t.Fatalf("back lean issues = %v, want [torso_lean_back]", got)
	}
}

func TestEngineHeadTiltDirections(t *testing.T) {
	cfg := engineCfg()

	right := newCalibratedEngine(cfg)
	var r Result
	for i := 1; i <= cfg.HoldFrames; i++ {
		r = right.Evaluate(testPose(0, 0, 0, 0.03, 1, base.Add(time.Duration(i)*time.Second)))
	}
	if got := issueCodes(r.Issues); !reflect.DeepEqual(got, []string{IssueHeadTiltRight}) {
		t.Fatalf("tilt issues = %v, want [head_tilt_right]", got)
	}

	left := newCalibratedEngine(cfg)
	for i := 1; i <= cfg.HoldFrames; i++ {
		r = left.Evaluate(testPose(0, 0, 0, -0.03, 1, base.Add(time.Duration(i)*time.Second)))
	}
	if got := issueCodes(r.Issues); !reflect.DeepEqual(got, []string{IssueHeadTiltLeft}) {
		t.Fatalf("tilt issues = %v, want [head_tilt_left]", got)
	}
}

func TestEngineNoPerson(t *testing.T) {
	e, _ := NewEngine(engineCfg())
	p := testPose(0, 0, 0, 0, 1, base)
	for i := range p.Landmarks {
		p.Landmarks[i].Visibility = 0
	}
	r := e.Evaluate(p)
	if r.Status != StatusNoPerson {
		t.Fatalf("status = %s, want no_person", r.Status)
	}
	if r.SeatedSeconds != 0 {
		t.Fatalf("seated = %.1f, want 0", r.SeatedSeconds)
	}
}

func TestEngineHipsHiddenSkipsTorso(t *testing.T) {
	e := newCalibratedEngine(engineCfg())
	// torso leaning forward but hips invisible: no torso issue, neck still works
	for i := 1; i <= engineCfg().HoldFrames+2; i++ {
		r := e.Evaluate(testPose(0, -0.15, 0, 0, 0.1, base.Add(time.Duration(i)*time.Second)))
		if r.Metrics.TorsoAvailable {
			t.Fatal("torso reported available despite hidden hips")
		}
		for _, iss := range r.Issues {
			if iss.Code == IssueTorsoLeanFwd || iss.Code == IssueTorsoLeanBack {
				t.Fatalf("torso issue raised without hips: %+v", iss)
			}
		}
	}
}

func TestEngineLongSittingAndReset(t *testing.T) {
	cfg := engineCfg()
	cfg.LongSittingMinutes = 1
	cfg.AbsenceResetSeconds = 120
	e, _ := NewEngine(cfg)

	// 13 frames 5s apart credit 60s of seated time; the reminder fires once
	// the 60s mark is reached (long_sitting_minutes = 1)
	var r Result
	for i := 0; i <= 12; i++ {
		r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(time.Duration(i)*5*time.Second)))
		if i < 12 && hasCode(r.Issues, IssueLongSitting) {
			t.Fatalf("long_sitting raised early at %.0fs", r.SeatedSeconds)
		}
	}
	if !hasCode(r.Issues, IssueLongSitting) {
		t.Fatalf("long_sitting not raised at %.0fs: %+v", r.SeatedSeconds, r.Issues)
	}
	if r.Status != StatusBad {
		t.Fatalf("status = %s, want bad", r.Status)
	}

	// short absence (< reset): timer paused, not reset
	absent := testPose(0, 0, 0, 0, 1, base)
	for i := range absent.Landmarks {
		absent.Landmarks[i].Visibility = 0
	}
	absent = absent.WithTime(base.Add(90 * time.Second))
	r = e.Evaluate(absent)
	if !hasCode(r.Issues, IssueLongSitting) || r.SeatedSeconds != 60 {
		t.Fatalf("short absence changed state: seated=%.0f issues=%v", r.SeatedSeconds, r.Issues)
	}
	r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(95*time.Second)))
	if !hasCode(r.Issues, IssueLongSitting) || r.SeatedSeconds != 60 {
		t.Fatalf("return from short absence should keep 60s: seated=%.0f issues=%v", r.SeatedSeconds, r.Issues)
	}

	// long absence (> reset): timer restarts, issue drops
	r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(95*time.Second+130*time.Second)))
	if hasCode(r.Issues, IssueLongSitting) {
		t.Fatalf("long_sitting survived a %ds absence", cfg.AbsenceResetSeconds)
	}
	if r.SeatedSeconds != 0 {
		t.Fatalf("seated = %.1f after reset, want 0", r.SeatedSeconds)
	}
}

func TestEngineSeatedGapNotCredited(t *testing.T) {
	e, _ := NewEngine(engineCfg())
	e.Evaluate(testPose(0, 0, 0, 0, 1, base))
	// 11s gap < absence reset but > maxFrameGap: no credit, no reset
	r := e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(11*time.Second)))
	if r.SeatedSeconds != 0 {
		t.Fatalf("seated = %.1f, want 0 (gap must not be credited)", r.SeatedSeconds)
	}
	// duplicate timestamp must not panic or credit
	r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(12*time.Second)))
	r = e.Evaluate(testPose(0, 0, 0, 0, 1, base.Add(12*time.Second)))
	if r.SeatedSeconds > 1.01 {
		t.Fatalf("seated = %.1f after duplicate frame, want ~1", r.SeatedSeconds)
	}
}

func TestEngineBadPoseFromStartUsesWallClock(t *testing.T) {
	e, _ := NewEngine(engineCfg())
	// no timestamp at all: must not panic and must stay consistent
	p := testPose(0, 0, 0, 0, 1, time.Time{})
	p.TimestampMs = 0
	r1 := e.Evaluate(p)
	r2 := e.Evaluate(p)
	if r1.Status == "" || r2.Status == "" {
		t.Fatal("empty status for timestamp-less frames")
	}
}

func TestEngineSetConfigValidates(t *testing.T) {
	e, _ := NewEngine(DefaultConfig())
	bad := DefaultConfig()
	bad.NeckForwardMildDeg = 50 // > severe
	if err := e.SetConfig(bad); err == nil {
		t.Fatal("expected validation error")
	}
	if got := e.Config().NeckForwardMildDeg; got != DefaultConfig().NeckForwardMildDeg {
		t.Fatalf("config changed despite error: %.0f", got)
	}
	good := DefaultConfig()
	good.HoldFrames = 2
	if err := e.SetConfig(good); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if e.Config().HoldFrames != 2 {
		t.Fatal("config not applied")
	}
}

func TestConfigValidation(t *testing.T) {
	if err := DefaultConfig().Validate(); err != nil {
		t.Fatalf("default config invalid: %v", err)
	}
	cases := []struct {
		name string
		mut  func(*Config)
	}{
		{"visibility_zero", func(c *Config) { c.MinVisibility = 0 }},
		{"visibility_above_one", func(c *Config) { c.MinVisibility = 1.5 }},
		{"neck_mild_negative", func(c *Config) { c.NeckForwardMildDeg = -1 }},
		{"neck_mild_above_severe", func(c *Config) { c.NeckForwardMildDeg = 99 }},
		{"torso_back_above_severe", func(c *Config) { c.TorsoBackMildDeg = 99 }},
		{"tilt_mild_above_severe", func(c *Config) { c.HeadTiltMildDeg = 99 }},
		{"long_sitting_zero", func(c *Config) { c.LongSittingMinutes = 0 }},
		{"hold_frames_zero", func(c *Config) { c.HoldFrames = 0 }},
		{"clear_frames_zero", func(c *Config) { c.ClearFrames = 0 }},
		{"absence_reset_zero", func(c *Config) { c.AbsenceResetSeconds = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := DefaultConfig()
			tc.mut(&c)
			if err := c.Validate(); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestNewEngineRejectsInvalidConfig(t *testing.T) {
	c := DefaultConfig()
	c.MinVisibility = 2
	if _, err := NewEngine(c); err == nil {
		t.Fatal("expected error from NewEngine")
	}
}

func TestNormalizeView(t *testing.T) {
	cases := map[string]string{
		"":             ViewDefault,
		"default":      ViewDefault,
		"front":        ViewFront,
		"side":         ViewSide,
		"webcam":       ViewDefault, // unknown → single-cam semantics
	}
	for in, want := range cases {
		if got := NormalizeView(in); got != want {
			t.Errorf("NormalizeView(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestViewRouting(t *testing.T) {
	cfg := engineCfg()
	e, _ := NewEngine(cfg)
	e.SetBaselineView(ViewFront, Baseline{})
	e.SetBaselineView(ViewSide, Baseline{})

	// front view: forward head invisible (masked check), tilt judged
	fwdTilt := testPose(-0.15, 0, 0, 0.03, 1, base) // 45° neck + tilt 26.6°
	front := fwdTilt
	front.View = ViewFront
	for i := 1; i <= cfg.HoldFrames; i++ {
		front.TimestampMs = base.Add(time.Duration(i) * time.Second).UnixMilli()
		r := e.Evaluate(front)
		for _, iss := range r.Issues {
			if iss.Code == IssueForwardHead {
				t.Fatalf("front view must not judge forward_head: %+v", iss)
			}
		}
	}
	// last frame: tilt issue should be active and tagged front
	var last Result
	for i := 1; i <= cfg.HoldFrames; i++ {
		last = e.Evaluate(front)
	}
	found := false
	for _, iss := range last.Issues {
		if iss.Code == IssueHeadTiltRight || iss.Code == IssueHeadTiltLeft {
			found = true
			if iss.View != ViewFront {
				t.Fatalf("tilt issue view = %q, want front", iss.View)
			}
		}
	}
	if !found {
		t.Fatalf("front view did not raise tilt issue: %+v", last.Issues)
	}

	// side view: tilt invisible, forward head judged
	e2, _ := NewEngine(cfg)
	e2.SetBaselineView(ViewSide, Baseline{})
	side := fwdTilt
	side.View = ViewSide
	var sideLast Result
	for i := 1; i <= cfg.HoldFrames; i++ {
		side.TimestampMs = base.Add(time.Duration(i) * time.Second).UnixMilli()
		sideLast = e2.Evaluate(side)
		for _, iss := range sideLast.Issues {
			if iss.Code == IssueHeadTiltLeft || iss.Code == IssueHeadTiltRight {
				t.Fatalf("side view must not judge head tilt: %+v", iss)
			}
		}
	}
	if !containsStr(issueCodes(sideLast.Issues), IssueForwardHead) {
		t.Fatalf("side view did not raise forward_head: %+v", sideLast.Issues)
	}
}

func containsStr(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func TestViewMetricsMasking(t *testing.T) {
	e, _ := NewEngine(engineCfg())
	e.SetBaseline(Baseline{})

	front := testPose(-0.15, 0, 0, 0.03, 1, base)
	front.View = ViewFront
	r := e.Evaluate(front)
	if r.Metrics.NeckForward != 0 || r.Metrics.TorsoForward != 0 {
		t.Errorf("front metrics must mask sagittal angles: %+v", r.Metrics)
	}
	if r.Metrics.HeadTilt == 0 {
		t.Error("front metrics must keep head tilt")
	}

	side := testPose(-0.15, 0, 0, 0.03, 1, base)
	side.View = ViewSide
	r = e.Evaluate(side)
	if r.Metrics.HeadTilt != 0 || r.Metrics.NeckLateral != 0 {
		t.Errorf("side metrics must mask frontal angles: %+v", r.Metrics)
	}
	if r.Metrics.NeckForward == 0 {
		t.Error("side metrics must keep neck forward")
	}

	def := testPose(-0.15, 0, 0, 0.03, 1, base)
	r = e.Evaluate(def)
	if r.Metrics.NeckForward == 0 || r.Metrics.HeadTilt == 0 {
		t.Error("default view must report all metrics")
	}
}

func TestInterleavedViewsSeatedNoDoubleCount(t *testing.T) {
	cfg := engineCfg()
	e, _ := NewEngine(cfg)
	e.SetBaseline(Baseline{})

	// front at 0,400,800…ms and side at 200,600,…ms: wall time 4s total
	t0 := base
	for i := 0; i < 10; i++ {
		front := testPose(0, 0, 0, 0, 1, t0.Add(time.Duration(i*400)*time.Millisecond))
		front.View = ViewFront
		e.Evaluate(front)
		side := testPose(0, 0, 0, 0, 1, t0.Add(time.Duration(i*400+200)*time.Millisecond))
		side.View = ViewSide
		r := e.Evaluate(side)
		_ = r
	}
	r := e.Evaluate(testPose(0, 0, 0, 0, 1, t0.Add(4*time.Second)))
	if r.SeatedSeconds > 4.5 || r.SeatedSeconds < 3.5 {
		t.Fatalf("seated = %.2fs, want ~4s (interleaved views must not double count)", r.SeatedSeconds)
	}
}

func TestPerViewCalibrationIndependence(t *testing.T) {
	cfg := engineCfg()
	e, _ := NewEngine(cfg)
	e.SetBaselineView(ViewFront, Baseline{})

	// front frame: good; side frame: needs calibration
	front := testPose(0, 0, 0, 0, 1, base)
	front.View = ViewFront
	if r := e.Evaluate(front); r.Status != StatusGood {
		t.Fatalf("front status = %s, want good", r.Status)
	}
	side := testPose(0, 0, 0, 0, 1, base.Add(time.Second))
	side.View = ViewSide
	r := e.Evaluate(side)
	if r.Status != StatusNeedsCalibration {
		t.Fatalf("global status = %s, want needs_calibration from uncalibrated side", r.Status)
	}
	if vs := r.Views[ViewSide]; !vs.Present || vs.Calibrated {
		t.Fatalf("side view state = %+v, want present && uncalibrated", vs)
	}
	if vs := r.Views[ViewFront]; !vs.Calibrated || vs.Status != StatusGood {
		t.Fatalf("front view state = %+v", vs)
	}

	// per-view baseline accessors
	if e.BaselineView(ViewSide) != nil {
		t.Fatal("side baseline should be nil")
	}
	if e.BaselineView(ViewFront) == nil {
		t.Fatal("front baseline should be set")
	}
	all := e.Baselines()
	if _, ok := all[ViewFront]; !ok {
		t.Fatalf("Baselines() = %v, want front entry", all)
	}
	if len(all) != 1 {
		t.Fatalf("Baselines() = %v, want exactly one view", all)
	}

	// clearing one view keeps the other
	e.ClearBaselineView(ViewFront)
	if e.BaselineView(ViewFront) != nil {
		t.Fatal("front baseline not cleared")
	}
	if e.BaselineView(ViewSide) != nil {
		t.Fatal("side baseline was set unexpectedly")
	}
	e.SetBaselineView(ViewSide, Baseline{})
	e.ClearBaselineView(ViewDefault)
	if e.BaselineView(ViewSide) == nil {
		t.Fatal("clearing default must not touch side")
	}
}

func TestGlobalIssuesUnionAcrossViews(t *testing.T) {
	cfg := engineCfg()
	e, _ := NewEngine(cfg)
	e.SetBaselineView(ViewFront, Baseline{})
	e.SetBaselineView(ViewSide, Baseline{})

	// raise tilt in front view
	front := testPose(0, 0, 0, 0.03, 1, base)
	front.View = ViewFront
	for i := 1; i <= cfg.HoldFrames; i++ {
		e.Evaluate(front.WithTime(base.Add(time.Duration(i) * time.Second)))
	}
	// raise forward head in side view
	side := testPose(-0.15, 0, 0, 0, 1, base)
	side.View = ViewSide
	for i := 1; i <= cfg.HoldFrames; i++ {
		side2 := side.WithTime(base.Add(time.Duration(100+i) * time.Second))
		side2.View = ViewSide
		r := e.Evaluate(side2)
		if i == cfg.HoldFrames {
			codes := issueCodes(r.Issues)
			if !containsStr(codes, IssueForwardHead) || !containsStr(codes, IssueHeadTiltLeft) && !containsStr(codes, IssueHeadTiltRight) {
				t.Fatalf("global issues = %v, want union of forward_head and tilt", codes)
			}
			if r.Status != StatusBad {
				t.Fatalf("status = %s, want bad", r.Status)
			}
		}
	}
}

func TestEngineNeedsCalibrationWhenUncalibrated(t *testing.T) {
	e, _ := NewEngine(engineCfg())
	// clearly bad pose without a baseline: no angle issue may be raised
	for i := 1; i <= engineCfg().HoldFrames*2; i++ {
		r := e.Evaluate(testPose(-0.15, -0.15, 0, 0.03, 1, base.Add(time.Duration(i)*time.Second)))
		if r.Status != StatusNeedsCalibration {
			t.Fatalf("frame %d: status = %s, want needs_calibration", i, r.Status)
		}
		if len(r.Issues) != 0 {
			t.Fatalf("frame %d: angle issues raised without baseline: %+v", i, r.Issues)
		}
	}
	// no person still wins over needs_calibration
	p := testPose(0, 0, 0, 0, 1, base)
	for i := range p.Landmarks {
		p.Landmarks[i].Visibility = 0
	}
	if r := e.Evaluate(p); r.Status != StatusNoPerson {
		t.Fatalf("status = %s, want no_person", r.Status)
	}
}

func TestEngineDeviationExactAngles(t *testing.T) {
	cfg := engineCfg()
	e, _ := NewEngine(cfg)
	e.SetBaseline(Baseline{NeckForward: 10})

	// ear drop that yields exactly ~30° absolute neck angle: atan2(0.087, 0.15)
	// choose z = -0.087 → 30.1°; deviation ≈ 20.1° → mild forward_head
	mild := testPose(-0.087, 0, 0, 0, 1, base)
	var r Result
	for i := 1; i <= cfg.HoldFrames; i++ {
		r = e.Evaluate(mild.WithTime(base.Add(time.Duration(i) * time.Second)))
	}
	if len(r.Issues) != 1 || r.Issues[0].Code != IssueForwardHead || r.Issues[0].Severity != SeverityMild {
		t.Fatalf("issues = %+v, want single mild forward_head", r.Issues)
	}
	if !degClose(r.Issues[0].Angle, 20, 0.5) {
		t.Fatalf("reported deviation = %.2f°, want ~20°", r.Issues[0].Angle)
	}

	// absolute ~46.9° (z=-0.16) with baseline 10° → deviation ~36.9° → severe
	e2, _ := NewEngine(cfg)
	e2.SetBaseline(Baseline{NeckForward: 10})
	for i := 1; i <= cfg.HoldFrames; i++ {
		r = e2.Evaluate(testPose(-0.16, 0, 0, 0, 1, base.Add(time.Duration(i)*time.Second)))
	}
	if len(r.Issues) != 1 || r.Issues[0].Severity != SeveritySevere {
		t.Fatalf("issues = %+v, want single severe forward_head", r.Issues)
	}
}

func TestEngineSetBaselineDropsActiveIssues(t *testing.T) {
	cfg := engineCfg()
	e := newCalibratedEngine(cfg) // zero baseline, absolute judging

	// raise an issue, then recalibrate: it must disappear immediately
	for i := 1; i <= cfg.HoldFrames; i++ {
		e.Evaluate(testPose(-0.15, 0, 0, 0, 1, base.Add(time.Duration(i)*time.Second)))
	}
	if r := e.Evaluate(testPose(-0.15, 0, 0, 0, 1, base.Add(100*time.Second))); len(r.Issues) == 0 {
		t.Fatal("precondition: issue should be active before recalibration")
	}
	e.SetBaseline(Baseline{NeckForward: 45}) // new reference covers the pose
	r := e.Evaluate(testPose(-0.15, 0, 0, 0, 1, base.Add(101*time.Second)))
	if len(r.Issues) != 0 || r.Status != StatusGood {
		t.Fatalf("stale issue survived recalibration: status=%s %+v", r.Status, r.Issues)
	}
	e.ClearBaseline()
	if e.Baseline() != nil {
		t.Fatal("ClearBaseline left a baseline behind")
	}
	r = e.Evaluate(testPose(-0.15, 0, 0, 0, 1, base.Add(102*time.Second)))
	if r.Status != StatusNeedsCalibration {
		t.Fatalf("status = %s after clear, want needs_calibration", r.Status)
	}
}

func TestCalibratePosesMedian(t *testing.T) {
	poses := make([]Pose, 0, 9)
	for i := 0; i < 9; i++ {
		// 8 upright frames + 1 with a 45° neck: median stays ~0
		earZ := 0.0
		if i == 4 {
			earZ = -0.15
		}
		poses = append(poses, testPose(earZ, 0, 0, 0, 1, base.Add(time.Duration(i)*time.Second)))
	}
	b, err := CalibratePoses(poses, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if !degClose(b.NeckForward, 0, 1) {
		t.Errorf("baseline neck = %.2f°, want ~0 (median must reject outliers)", b.NeckForward)
	}
	if !degClose(b.HeadTilt, 0, 0.01) || !degClose(b.TorsoForward, 0, 0.01) {
		t.Errorf("baseline = %+v, want zeros", b)
	}
}

func TestCalibratePosesTooFewFrames(t *testing.T) {
	poses := []Pose{testPose(0, 0, 0, 0, 1, base), testPose(0, 0, 0, 0, 1, base.Add(time.Second))}
	if _, err := CalibratePoses(poses, 0.5); err != ErrTooFewCalibrationFrames {
		t.Fatalf("err = %v, want ErrTooFewCalibrationFrames", err)
	}
	// invalid frames do not count either
	many := make([]Pose, 0, 10)
	for i := 0; i < 10; i++ {
		p := testPose(0, 0, 0, 0, 1, base)
		if i < 9 {
			for j := range p.Landmarks {
				p.Landmarks[j].Visibility = 0
			}
		}
		many = append(many, p)
	}
	if _, err := CalibratePoses(many, 0.5); err != ErrTooFewCalibrationFrames {
		t.Fatalf("err = %v, want ErrTooFewCalibrationFrames", err)
	}
}

func TestCalibratePosesHipsHiddenLeavesTorsoZero(t *testing.T) {
	poses := make([]Pose, 0, 8)
	for i := 0; i < 8; i++ {
		// upright posture, hips below the visibility threshold
		poses = append(poses, testPose(0, 0, 0, 0, 0.1, base.Add(time.Duration(i)*time.Second)))
	}
	b, err := CalibratePoses(poses, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if b.TorsoForward != 0 {
		t.Errorf("torso baseline = %.2f with hidden hips, want 0", b.TorsoForward)
	}
	if !degClose(b.NeckForward, 0, 0.01) {
		t.Errorf("neck baseline = %.2f, want 0", b.NeckForward)
	}
}

func TestSortedIssueCodes(t *testing.T) {
	codes := SortedIssueCodes()
	want := 6
	if len(codes) != want {
		t.Fatalf("got %d codes, want %d", len(codes), want)
	}
}

func issueCodes(issues []Issue) []string {
	out := make([]string, len(issues))
	for i, iss := range issues {
		out[i] = iss.Code
	}
	return out
}

func hasCode(issues []Issue, code string) bool {
	for _, iss := range issues {
		if iss.Code == code {
			return true
		}
	}
	return false
}

// WithTime returns a copy of the pose stamped with ts.
func (p Pose) WithTime(ts time.Time) Pose {
	p.Timestamp = ts
	p.TimestampMs = ts.UnixMilli()
	return p
}
