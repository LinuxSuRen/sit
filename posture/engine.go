package posture

import (
	"sort"
	"sync"
	"time"
)

// Status is the aggregated posture verdict for a frame.
const (
	StatusGood     = "good"     // no active issue
	StatusWarning  = "warning"  // at least one mild issue
	StatusBad      = "bad"      // at least one severe issue
	StatusNoPerson = "no_person" // no reliable body keypoints
	// StatusNeedsCalibration is reported while sitting in frame without a
	// calibrated baseline: angle checks are skipped until calibration.
	StatusNeedsCalibration = "needs_calibration"
)

// Severity grades a single issue.
const (
	SeverityMild   = "mild"
	SeveritySevere = "severe"
)

// Issue codes. These are stable contract values: clients map them to their
// own user-facing copy; the backend never sends display text.
const (
	IssueForwardHead   = "forward_head"
	IssueTorsoLeanFwd  = "torso_lean_forward"
	IssueTorsoLeanBack = "torso_lean_back"
	IssueHeadTiltLeft  = "head_tilt_left"
	IssueHeadTiltRight = "head_tilt_right"
	IssueLongSitting   = "long_sitting"
)

// View identifiers for pose frames. The frame's view decides which checks
// the engine runs on it:
//
//   - front: frontal-plane checks only (head tilt) — the side view cannot
//     see tilts, the front view sees them best
//   - side: sagittal-plane checks only (forward head, torso lean) — depth
//     from a frontal camera is too noisy for these
//   - default: single-camera mode, all checks (frontal depth is used as a
//     compromise, exactly like a single-camera setup)
const (
	ViewDefault = "default"
	ViewFront   = "front"
	ViewSide    = "side"
)

// NormalizeView maps a raw frame view label to a known view; unknown and
// empty labels fall back to single-camera semantics.
func NormalizeView(v string) string {
	switch v {
	case ViewFront:
		return ViewFront
	case ViewSide:
		return ViewSide
	default:
		return ViewDefault
	}
}

// viewOrder is the stable iteration order of views in snapshots.
var viewOrder = []string{ViewDefault, ViewFront, ViewSide}

// Issue is one active posture problem.
type Issue struct {
	Code     string  `json:"code"`
	Severity string  `json:"severity"`
	// Angle is the measured value behind the issue in degrees; 0 for
	// long_sitting, which is time-based.
	Angle float64 `json:"angle"`
	// View names the view whose check produced the issue; empty for the
	// view-independent long_sitting reminder.
	View string `json:"view,omitempty"`
}

// ViewState summarizes one view for the client.
type ViewState struct {
	Present    bool   `json:"present"`
	Calibrated bool   `json:"calibrated"`
	Status     string `json:"status"`
}

// Result is the engine verdict for one pose frame. It always carries the
// global snapshot across all views, not just the frame's own view.
type Result struct {
	Status        string               `json:"status"`
	View          string               `json:"view"`
	Metrics       Metrics              `json:"metrics"`
	Issues        []Issue              `json:"issues"`
	SeatedSeconds float64              `json:"seated_seconds"`
	Views         map[string]ViewState `json:"views,omitempty"`
}

// maxFrameGap caps the seated-time delta accumulated between two consecutive
// present frames. Larger gaps (throttled tab, hiccups) are not credited, and
// a view whose last frame is older than this is considered stale.
const maxFrameGap = 10 * time.Second

// issueState is the per-issue debounce bookkeeping.
type issueState struct {
	badStreak  int
	goodStreak int
	active     bool
	last       Issue
}

// viewEngine holds the per-view state: its baseline, debounce states and
// presence flag.
type viewEngine struct {
	baseline  *Baseline
	state     map[string]*issueState
	present   bool
	lastFrame time.Time
}

// Engine evaluates pose frames streamingly. Frames carry a view label; each
// view is judged by its own checks against its own baseline, while seated
// time and the long-sitting reminder are global (a person seen by any view
// counts as present). It is safe for concurrent use.
type Engine struct {
	mu    sync.Mutex
	cfg   Config
	views map[string]*viewEngine

	// global seated-time tracking
	lastPresentTime time.Time
	seated          time.Duration
	longRaised      bool
}

// NewEngine returns an engine using the given validated config.
func NewEngine(cfg Config) (*Engine, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Engine{cfg: cfg, views: make(map[string]*viewEngine)}, nil
}

// view returns the per-view engine, creating it on first use.
// Callers must hold the engine lock.
func (e *Engine) view(name string) *viewEngine {
	ve, ok := e.views[name]
	if !ok {
		ve = &viewEngine{state: make(map[string]*issueState)}
		e.views[name] = ve
	}
	return ve
}

// Baseline returns a copy of the default-view baseline, or nil if unset.
func (e *Engine) Baseline() *Baseline { return e.BaselineView(ViewDefault) }

// SetBaseline installs (or replaces) the default-view baseline.
func (e *Engine) SetBaseline(b Baseline) { e.SetBaselineView(ViewDefault, b) }

// ClearBaseline removes the default-view baseline.
func (e *Engine) ClearBaseline() { e.ClearBaselineView(ViewDefault) }

// BaselineView returns a copy of the given view's baseline, or nil if unset.
func (e *Engine) BaselineView(view string) *Baseline {
	e.mu.Lock()
	defer e.mu.Unlock()
	ve := e.views[NormalizeView(view)]
	if ve == nil {
		return nil
	}
	return ve.baselineCopy()
}

// SetBaselineView installs (or replaces) a view's baseline. Any active angle
// issues of that view are dropped: their meaning changed with the new
// reference.
func (e *Engine) SetBaselineView(view string, b Baseline) {
	e.mu.Lock()
	defer e.mu.Unlock()
	ve := e.view(NormalizeView(view))
	bas := b
	ve.baseline = &bas
	ve.resetIssues()
}

// ClearBaselineView removes a view's baseline and drops its active issues.
func (e *Engine) ClearBaselineView(view string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	ve := e.view(NormalizeView(view))
	ve.baseline = nil
	ve.resetIssues()
}

// Baselines returns copies of every set baseline keyed by view.
func (e *Engine) Baselines() map[string]Baseline {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make(map[string]Baseline, len(e.views))
	for name, ve := range e.views {
		if b := ve.baselineCopy(); b != nil {
			out[name] = *b
		}
	}
	return out
}

func (ve *viewEngine) baselineCopy() *Baseline {
	if ve.baseline == nil {
		return nil
	}
	b := *ve.baseline
	return &b
}

// resetIssues disables every active angle issue of the view.
func (ve *viewEngine) resetIssues() {
	for _, st := range ve.state {
		st.active = false
		st.badStreak = 0
		st.goodStreak = 0
	}
}

// Config returns a copy of the current configuration.
func (e *Engine) Config() Config {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.cfg
}

// SetConfig atomically replaces the configuration after validating it.
func (e *Engine) SetConfig(cfg Config) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.cfg = cfg
	return nil
}

// check describes one angle-based rule in the engine.
type check struct {
	code string
	// torso marks rules that depend on hip landmarks; they are skipped
	// (treated as no-hit) when the hips are not visible.
	torso  bool
	detect func(m Metrics, cfg Config) (hit bool, severity string, angle float64)
}

// checks contains every rule in reporting order. Lateral directions refer
// to the user's own left/right (the client mirrors camera coordinates
// before sending).
var checks = []check{
	{
		code: IssueForwardHead,
		detect: func(m Metrics, cfg Config) (bool, string, float64) {
			return grade(m.NeckForward, cfg.NeckForwardMildDeg, cfg.NeckForwardSevereDeg)
		},
	},
	{
		code:  IssueTorsoLeanFwd,
		torso: true,
		detect: func(m Metrics, cfg Config) (bool, string, float64) {
			return grade(m.TorsoForward, cfg.TorsoForwardMildDeg, cfg.TorsoSevereDeg)
		},
	},
	{
		code:  IssueTorsoLeanBack,
		torso: true,
		detect: func(m Metrics, cfg Config) (bool, string, float64) {
			return grade(-m.TorsoForward, cfg.TorsoBackMildDeg, cfg.TorsoSevereDeg)
		},
	},
	{
		code: IssueHeadTiltRight,
		detect: func(m Metrics, cfg Config) (bool, string, float64) {
			return grade(m.HeadTilt, cfg.HeadTiltMildDeg, cfg.HeadTiltSevereDeg)
		},
	},
	{
		code: IssueHeadTiltLeft,
		detect: func(m Metrics, cfg Config) (bool, string, float64) {
			return grade(-m.HeadTilt, cfg.HeadTiltMildDeg, cfg.HeadTiltSevereDeg)
		},
	},
}

var (
	tiltChecks     = []check{checks[3], checks[4]}
	sagittalChecks = []check{checks[0], checks[1], checks[2]}
)

// viewChecks returns the checks a view owns, in reporting order.
func viewChecks(view string) []check {
	switch view {
	case ViewFront:
		return tiltChecks
	case ViewSide:
		return sagittalChecks
	default:
		return checks
	}
}

// maskMetrics zeroes the metrics a view cannot measure reliably so clients
// never display misleading values.
func maskMetrics(view string, m Metrics) Metrics {
	switch view {
	case ViewFront:
		m.NeckForward = 0
		m.TorsoForward = 0
	case ViewSide:
		m.NeckLateral = 0
		m.TorsoLateral = 0
		m.HeadTilt = 0
	}
	return m
}

// grade classifies a signed angle against mild/severe thresholds. A negative
// angle never hits the thresholds (they are positive by validation).
func grade(angle, mildDeg, severeDeg float64) (bool, string, float64) {
	switch {
	case angle >= severeDeg:
		return true, SeveritySevere, angle
	case angle >= mildDeg:
		return true, SeverityMild, angle
	default:
		return false, "", angle
	}
}

// Evaluate processes one pose frame and returns the global snapshot.
func (e *Engine) Evaluate(p Pose) Result {
	now := p.Time()
	if p.Timestamp.IsZero() && p.TimestampMs == 0 {
		now = time.Now()
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	cfg := e.cfg
	view := NormalizeView(p.View)
	ve := e.view(view)

	present := p.personPresent(cfg.MinVisibility)
	ve.present = present
	ve.lastFrame = now
	e.updateSeated(now, present, cfg)

	var m Metrics
	if present {
		raw := computeMetrics(p, cfg.MinVisibility)
		if ve.baseline != nil {
			// judge deviations from the calibrated reference posture
			dev := Metrics{
				NeckForward:  raw.NeckForward - ve.baseline.NeckForward,
				TorsoForward: raw.TorsoForward - ve.baseline.TorsoForward,
				HeadTilt:     raw.HeadTilt - ve.baseline.HeadTilt,
			}
			for _, c := range viewChecks(view) {
				if c.torso && !raw.TorsoAvailable {
					// no evidence: feed no-hit so any active torso issue clears
					e.updateIssue(ve, c, false, "", 0, cfg)
					continue
				}
				hit, severity, angle := c.detect(dev, cfg)
				e.updateIssue(ve, c, hit, severity, angle, cfg)
			}
		} else {
			// no baseline yet: angle checks stay off; feed no-hit to drain
			// any stale active issues
			for _, c := range viewChecks(view) {
				e.updateIssue(ve, c, false, "", 0, cfg)
			}
		}
		m = maskMetrics(view, raw)
	}

	e.updateLongSitting(cfg)
	return e.snapshot(view, m, now)
}

// updateLongSitting raises the reminder when the seated time reaches the
// configured limit. Callers must hold the engine lock.
func (e *Engine) updateLongSitting(cfg Config) {
	limit := time.Duration(cfg.LongSittingMinutes) * time.Minute
	if !e.longRaised && limit > 0 && e.seated >= limit {
		e.longRaised = true
	}
}

// updateSeated accumulates seated time from consecutive present frames (any
// view) and resets the timer when the person has been away long enough.
func (e *Engine) updateSeated(now time.Time, present bool, cfg Config) {
	if !present {
		return
	}
	if e.lastPresentTime.IsZero() {
		e.lastPresentTime = now
		return
	}
	dt := now.Sub(e.lastPresentTime)
	switch {
	case dt <= 0:
		// out-of-order or duplicate frame: nothing to credit
	case dt <= maxFrameGap:
		e.seated += dt
	case dt > time.Duration(cfg.AbsenceResetSeconds)*time.Second:
		// away for longer than the reset window: start counting over
		e.seated = 0
		e.longRaised = false
	}
	// gaps between maxFrameGap and the reset window count as a break but do
	// not zero the timer
	e.lastPresentTime = now
}

// updateIssue runs the debounce state machine for one rule of one view.
// Callers must hold the engine lock.
func (e *Engine) updateIssue(ve *viewEngine, c check, hit bool, severity string, angle float64, cfg Config) {
	st, ok := ve.state[c.code]
	if !ok {
		st = &issueState{}
		ve.state[c.code] = st
	}
	if hit {
		st.badStreak++
		st.goodStreak = 0
		st.last = Issue{Code: c.code, Severity: severity, Angle: angle}
		if !st.active && st.badStreak >= cfg.HoldFrames {
			st.active = true
		}
		return
	}
	st.badStreak = 0
	if st.active {
		st.goodStreak++
		if st.goodStreak >= cfg.ClearFrames {
			st.active = false
			st.goodStreak = 0
		}
	} else {
		st.goodStreak = 0
	}
}

// snapshot builds the global Result: active issues of all (fresh) views,
// per-view states and the merged status. Callers must hold the engine lock.
func (e *Engine) snapshot(frameView string, m Metrics, now time.Time) Result {
	issues := make([]Issue, 0)
	viewStates := make(map[string]ViewState)
	anyPresent := false
	needsCalib := false
	worst := StatusGood

	for _, name := range viewOrder {
		ve, ok := e.views[name]
		if !ok {
			continue
		}
		// stale views (no frame for a while) count as absent
		fresh := now.Sub(ve.lastFrame) <= maxFrameGap
		present := ve.present && fresh

		vs := ViewState{
			Present:    present,
			Calibrated: ve.baseline != nil,
			Status:     viewStatus(ve, present),
		}
		viewStates[name] = vs

		for _, c := range viewChecks(name) {
			if st := ve.state[c.code]; st != nil && st.active {
				iss := st.last
				iss.View = name
				issues = append(issues, iss)
			}
		}

		if present {
			anyPresent = true
			switch vs.Status {
			case StatusBad:
				worst = StatusBad
			case StatusWarning:
				if worst != StatusBad {
					worst = StatusWarning
				}
			case StatusNeedsCalibration:
				needsCalib = true
			}
		}
	}

	if e.longRaised {
		issues = append(issues, Issue{Code: IssueLongSitting, Severity: SeveritySevere})
		// the reminder is severe-grade, but only matters while someone is
		// actually in frame
		if anyPresent && worst != StatusBad {
			worst = StatusBad
		}
	}

	status := worst
	switch {
	case worst == StatusGood && needsCalib:
		status = StatusNeedsCalibration
	case !anyPresent && worst == StatusGood && !needsCalib:
		status = StatusNoPerson
	}

	return Result{
		Status:        status,
		View:          frameView,
		Metrics:       m,
		Issues:        issues,
		SeatedSeconds: e.seated.Seconds(),
		Views:         viewStates,
	}
}

// viewStatus derives one view's own status from its state.
func viewStatus(ve *viewEngine, present bool) string {
	if !present {
		return StatusNoPerson
	}
	if ve.baseline == nil {
		return StatusNeedsCalibration
	}
	// worst active issue severity in this view
	worst := StatusGood
	for _, st := range ve.state {
		if !st.active {
			continue
		}
		switch {
		case st.last.Severity == SeveritySevere:
			return StatusBad
		case st.last.Severity == SeverityMild:
			worst = StatusWarning
		}
	}
	return worst
}

// SortedIssueCodes returns every issue code the engine can emit, sorted.
// Useful for clients building translation tables.
func SortedIssueCodes() []string {
	codes := []string{
		IssueForwardHead, IssueTorsoLeanFwd, IssueTorsoLeanBack,
		IssueHeadTiltLeft, IssueHeadTiltRight, IssueLongSitting,
	}
	sort.Strings(codes)
	return codes
}
