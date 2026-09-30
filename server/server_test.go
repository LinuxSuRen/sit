package server

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/linuxsuren/sitcoach/posture"
)

func newTestServer(t *testing.T, cfg posture.Config) *httptest.Server {
	t.Helper()
	s, err := New(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func TestIndexPageServed(t *testing.T) {
	srv := newTestServer(t, posture.DefaultConfig())
	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body := make([]byte, 4096)
	n, _ := resp.Body.Read(body)
	if !strings.Contains(string(body[:n]), "sit") && !strings.Contains(string(body[:n]), "<html") {
		t.Fatalf("unexpected index content: %q", string(body[:n]))
	}
}

// putConfig sends a PUT request with the given JSON body.
func putConfig(t *testing.T, url string, body []byte) *http.Response {
	t.Helper()
	req, err := http.NewRequest(http.MethodPut, url, bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestConfigGetPut(t *testing.T) {
	srv := newTestServer(t, posture.DefaultConfig())

	// GET returns the defaults
	resp, err := http.Get(srv.URL + "/api/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got posture.Config
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.HoldFrames != posture.DefaultConfig().HoldFrames {
		t.Fatalf("hold_frames = %d, want default %d", got.HoldFrames, posture.DefaultConfig().HoldFrames)
	}

	// PUT applies a new config
	updated := posture.DefaultConfig()
	updated.HoldFrames = 2
	updated.LongSittingMinutes = 30
	buf, _ := json.Marshal(updated)
	resp2 := putConfig(t, srv.URL+"/api/config", buf)
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("PUT status = %d, want 200", resp2.StatusCode)
	}
	var applied posture.Config
	if err := json.NewDecoder(resp2.Body).Decode(&applied); err != nil {
		t.Fatal(err)
	}
	if applied.HoldFrames != 2 || applied.LongSittingMinutes != 30 {
		t.Fatalf("config not applied: %+v", applied)
	}

	// GET reflects the update
	resp3, err := http.Get(srv.URL + "/api/config")
	if err != nil {
		t.Fatal(err)
	}
	defer resp3.Body.Close()
	var reloaded posture.Config
	_ = json.NewDecoder(resp3.Body).Decode(&reloaded)
	if reloaded.HoldFrames != 2 {
		t.Fatalf("hold_frames = %d after reload, want 2", reloaded.HoldFrames)
	}
}

func TestConfigPutRejectsInvalid(t *testing.T) {
	srv := newTestServer(t, posture.DefaultConfig())

	// inconsistent thresholds
	bad := posture.DefaultConfig()
	bad.NeckForwardMildDeg = 99
	buf, _ := json.Marshal(bad)
	resp := putConfig(t, srv.URL+"/api/config", buf)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}

	// unknown fields are rejected to keep the contract strict
	resp2 := putConfig(t, srv.URL+"/api/config", []byte(`{"hold_frames": 3, "bogus": 1}`))
	defer resp2.Body.Close()
	if resp2.StatusCode != http.StatusBadRequest {
		t.Fatalf("unknown-field status = %d, want 400", resp2.StatusCode)
	}
}

// dialWS opens a WebSocket to the pose endpoint.
func dialWS(t *testing.T, srv *httptest.Server) *websocket.Conn {
	t.Helper()
	url := "ws" + strings.TrimPrefix(srv.URL, "http") + "/ws/pose"
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

// sendPose writes one pose frame and returns the evaluated result.
func sendPose(t *testing.T, conn *websocket.Conn, p posture.Pose) posture.Result {
	t.Helper()
	if err := conn.WriteJSON(p); err != nil {
		t.Fatal(err)
	}
	var r posture.Result
	if err := conn.ReadJSON(&r); err != nil {
		t.Fatal(err)
	}
	return r
}

// wsPose mirrors the landmark construction of posture's tests with fast
// debounce settings baked in.
func wsPose(earZ float64, ts time.Time) posture.Pose {
	p := posture.Pose{Timestamp: ts, TimestampMs: ts.UnixMilli()}
	p.Landmarks = make([]posture.Landmark, 33)
	for i := range p.Landmarks {
		p.Landmarks[i] = posture.Landmark{Visibility: 1}
	}
	set := func(i int, x, y, z float64) {
		p.Landmarks[i] = posture.Landmark{X: x, Y: y, Z: z, Visibility: 1}
	}
	set(posture.LandmarkLeftEar, 0.47, 0.25, earZ)
	set(posture.LandmarkRightEar, 0.53, 0.25, earZ)
	set(posture.LandmarkLeftShoulder, 0.45, 0.40, 0)
	set(posture.LandmarkRightShoulder, 0.55, 0.40, 0)
	set(posture.LandmarkLeftHip, 0.46, 0.75, 0)
	set(posture.LandmarkRightHip, 0.54, 0.75, 0)
	return p
}

func TestPoseWebSocketRoundTrip(t *testing.T) {
	cfg := posture.DefaultConfig()
	cfg.HoldFrames = 2
	cfg.ClearFrames = 2
	cfg.LongSittingMinutes = 1000
	srv := newTestServer(t, cfg)
	conn := dialWS(t, srv)

	base := time.Now().Add(-time.Hour) // decouple from wall clock

	// calibrate an upright reference posture first (uncalibrated engines
	// skip angle checks by design)
	if !calibrateServer(t, srv, "", func(ts time.Time) posture.Pose { return wsPose(0, ts) }) {
		t.Fatal("calibration failed")
	}

	// upright posture: good immediately
	r := sendPose(t, conn, wsPose(0, base))
	if r.Status != posture.StatusGood {
		t.Fatalf("status = %s, want good", r.Status)
	}

	// forward head (ears 0.15 closer to camera ≈ 45°): raised after HoldFrames
	bad := wsPose(-0.15, base)
	for i := 1; i <= cfg.HoldFrames; i++ {
		bad.TimestampMs = base.Add(time.Duration(i) * time.Second).UnixMilli()
		r = sendPose(t, conn, bad)
	}
	if len(r.Issues) == 0 || r.Issues[0].Code != posture.IssueForwardHead {
		t.Fatalf("issues = %+v, want forward_head", r.Issues)
	}

	// recovering clears it after ClearFrames
	good := wsPose(0, base)
	for i := 1; i <= cfg.ClearFrames; i++ {
		good.TimestampMs = base.Add(time.Duration(i) * time.Second).UnixMilli()
		r = sendPose(t, conn, good)
	}
	if r.Status != posture.StatusGood || len(r.Issues) != 0 {
		t.Fatalf("not cleared: status=%s issues=%+v", r.Status, r.Issues)
	}
}

func TestPoseWebSocketNeedsCalibration(t *testing.T) {
	srv := newTestServer(t, posture.DefaultConfig())
	conn := dialWS(t, srv)

	// clearly bad pose without calibration: no angle issue, distinct status
	r := sendPose(t, conn, wsPose(-0.15, time.Now().Add(-time.Hour)))
	if r.Status != posture.StatusNeedsCalibration {
		t.Fatalf("status = %s, want needs_calibration", r.Status)
	}
	if len(r.Issues) != 0 {
		t.Fatalf("issues = %+v, want none", r.Issues)
	}
}

// calibrateServer POSTs a batch of poses for one view to /api/calibrate and
// reports whether the request succeeded. An empty view means default.
func calibrateServer(t *testing.T, srv *httptest.Server, view string, pose func(ts time.Time) posture.Pose) bool {
	t.Helper()
	frames := make([]posture.Pose, 0, 10)
	start := time.Now().Add(-time.Hour)
	for i := 0; i < 10; i++ {
		p := pose(start.Add(time.Duration(i) * time.Second))
		p.TimestampMs = p.Time().UnixMilli()
		frames = append(frames, p)
	}
	body, _ := json.Marshal(map[string]any{"view": view, "poses": frames})
	resp, err := http.Post(srv.URL+"/api/calibrate", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

func TestCalibrateFlow(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "calibration.json")

	cfg := posture.DefaultConfig()
	srvS, err := New(cfg, nil, CalibrationFile(file))
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(srvS.Handler())
	t.Cleanup(srv.Close)

	// uncalibrated at first
	resp, err := http.Get(srv.URL + "/api/calibration")
	if err != nil {
		t.Fatal(err)
	}
	var state calibrationResponse
	json.NewDecoder(resp.Body).Decode(&state)
	resp.Body.Close()
	if len(state.Views) != 0 {
		t.Fatal("should start uncalibrated")
	}

	// calibrating with a leaning reference posture
	if !calibrateServer(t, srv, "", func(ts time.Time) posture.Pose { return wsPose(-0.15, ts) }) {
		t.Fatal("calibrate request failed")
	}
	resp, err = http.Get(srv.URL + "/api/calibration")
	if err != nil {
		t.Fatal(err)
	}
	json.NewDecoder(resp.Body).Decode(&state)
	resp.Body.Close()
	vc, ok := state.Views[posture.ViewDefault]
	if !ok || !vc.Calibrated || vc.Baseline == nil {
		t.Fatalf("should be calibrated after POST: %+v", state.Views)
	}
	if !degCloseT(vc.Baseline.NeckForward, 45, 0.5) {
		t.Fatalf("baseline neck = %.2f°, want ~45°", vc.Baseline.NeckForward)
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("calibration not persisted: %v", err)
	}

	// DELETE clears state and file
	req, _ := http.NewRequest(http.MethodDelete, srv.URL+"/api/calibration", nil)
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("DELETE status = %d", resp2.StatusCode)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Fatalf("calibration file not removed: %v", err)
	}
	resp3, _ := http.Get(srv.URL + "/api/calibration")
	var cleared calibrationResponse
	json.NewDecoder(resp3.Body).Decode(&cleared)
	resp3.Body.Close()
	if len(cleared.Views) != 0 {
		t.Fatalf("should be uncalibrated after DELETE: %+v", cleared.Views)
	}
}

func TestCalibrateMultiView(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "calibration.json")

	cfg := posture.DefaultConfig()
	srvS, err := New(cfg, nil, CalibrationFile(file))
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(srvS.Handler())
	t.Cleanup(srv.Close)

	// calibrate both views
	if !calibrateServer(t, srv, "front", func(ts time.Time) posture.Pose { return wsPose(0, ts) }) {
		t.Fatal("front calibration failed")
	}
	if !calibrateServer(t, srv, "side", func(ts time.Time) posture.Pose { return wsPose(-0.15, ts) }) {
		t.Fatal("side calibration failed")
	}

	resp, _ := http.Get(srv.URL + "/api/calibration")
	var state calibrationResponse
	json.NewDecoder(resp.Body).Decode(&state)
	resp.Body.Close()
	if len(state.Views) != 2 {
		t.Fatalf("views = %v, want front+side", state.Views)
	}
	if vc := state.Views["side"]; vc.Baseline == nil || !degCloseT(vc.Baseline.NeckForward, 45, 0.5) {
		t.Fatalf("side baseline = %+v, want ~45° neck", vc.Baseline)
	}

	// persisted file holds both views
	views, err := loadBaselineFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := views["front"]; !ok {
		t.Fatalf("persisted views = %v, want front", views)
	}
	if _, ok := views["side"]; !ok {
		t.Fatalf("persisted views = %v, want side", views)
	}

	// DELETE only the front view: side must survive in engine and on disk
	req, _ := http.NewRequest(http.MethodDelete, srv.URL+"/api/calibration?view=front", nil)
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	resp3, _ := http.Get(srv.URL + "/api/calibration")
	var afterDelete calibrationResponse
	json.NewDecoder(resp3.Body).Decode(&afterDelete)
	resp3.Body.Close()
	if _, ok := afterDelete.Views["front"]; ok {
		t.Fatal("front view not cleared")
	}
	if _, ok := afterDelete.Views["side"]; !ok {
		t.Fatal("side view must survive deleting front")
	}
	views, err = loadBaselineFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := views["side"]; !ok {
		t.Fatalf("side baseline lost from disk: %v", views)
	}

	// per-view judging over WS: side view judges forward head, front does
	// not. The side baseline was calibrated on the leaning pose (-0.15 ≈
	// 45°), so send a much worse lean (-0.50 ≈ 73°): deviation ~28° → mild.
	conn := dialWS(t, srv)
	base := time.Now().Add(-time.Hour)
	var r posture.Result
	for i := 1; i <= cfg.HoldFrames; i++ {
		p := wsPose(-0.15, base.Add(time.Duration(i)*time.Second))
		p.View = posture.ViewFront
		r = sendPose(t, conn, p)
	}
	for _, iss := range r.Issues {
		if iss.Code == posture.IssueForwardHead {
			t.Fatalf("front view must not judge forward_head: %+v", iss)
		}
	}
	for i := 1; i <= cfg.HoldFrames; i++ {
		p := wsPose(-0.50, base.Add(time.Duration(100+i)*time.Second))
		p.View = posture.ViewSide
		r = sendPose(t, conn, p)
	}
	found := false
	for _, iss := range r.Issues {
		if iss.Code == posture.IssueForwardHead && iss.View == posture.ViewSide {
			found = true
		}
	}
	if !found {
		t.Fatalf("side view did not raise forward_head: %+v", r.Issues)
	}
}

func TestCalibrateRejectsTooFewFrames(t *testing.T) {
	srv := newTestServer(t, posture.DefaultConfig())
	one := []posture.Pose{wsPose(0, time.Now())}
	body, _ := json.Marshal(map[string]any{"poses": one})
	resp, err := http.Post(srv.URL+"/api/calibrate", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
}

func TestCalibrationLoadedOnStart(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "calibration.json")
	// zero baseline persisted: engine must judge absolute angles right away
	if err := os.WriteFile(file, []byte(`{"neck_forward":0,"torso_forward":0,"head_tilt":0}`), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := posture.DefaultConfig()
	cfg.HoldFrames = 2
	srvS, err := New(cfg, nil, CalibrationFile(file))
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(srvS.Handler())
	t.Cleanup(srv.Close)

	resp, _ := http.Get(srv.URL + "/api/calibration")
	var state calibrationResponse
	json.NewDecoder(resp.Body).Decode(&state)
	resp.Body.Close()
	if vc, ok := state.Views[posture.ViewDefault]; !ok || !vc.Calibrated {
		t.Fatalf("persisted baseline not loaded on start: %+v", state.Views)
	}

	// and it is active: a forward-head pose raises the issue over WS
	conn := dialWS(t, srv)
	base := time.Now().Add(-time.Hour)
	var r posture.Result
	for i := 1; i <= cfg.HoldFrames; i++ {
		r = sendPose(t, conn, wsPose(-0.15, base.Add(time.Duration(i)*time.Second)))
	}
	if len(r.Issues) == 0 || r.Issues[0].Code != posture.IssueForwardHead {
		t.Fatalf("issues = %+v, want forward_head", r.Issues)
	}
}

func degCloseT(got, want, tol float64) bool {
	return math.Abs(got-want) <= tol
}

func TestPoseWebSocketNoPerson(t *testing.T) {
	srv := newTestServer(t, posture.DefaultConfig())
	conn := dialWS(t, srv)

	p := posture.Pose{TimestampMs: time.Now().UnixMilli()}
	p.Landmarks = make([]posture.Landmark, 33) // all invisible
	r := sendPose(t, conn, p)
	if r.Status != posture.StatusNoPerson {
		t.Fatalf("status = %s, want no_person", r.Status)
	}
}
