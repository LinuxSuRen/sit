package server

import (
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"

	"github.com/linuxsuren/sitcoach/posture"
)

// Option configures optional Server behavior.
type Option func(*Server)

// CalibrationFile enables baseline persistence at the given path: loaded on
// startup, overwritten on every calibration, removed on reset.
func CalibrationFile(path string) Option {
	return func(s *Server) { s.calibFile = path }
}

// calibrateRequest carries the pose frames captured by the client while the
// user holds their reference posture, for one camera view.
type calibrateRequest struct {
	// View is the camera view being calibrated ("front", "side", or empty
	// for the single-camera default view).
	View  string          `json:"view"`
	Poses []posture.Pose `json:"poses"`
}

// viewCalibration is the per-view calibration state in API responses.
type viewCalibration struct {
	Calibrated bool              `json:"calibrated"`
	Baseline   *posture.Baseline `json:"baseline"`
}

// calibrationResponse is the GET/POST/DELETE calibration payload.
type calibrationResponse struct {
	Views map[string]viewCalibration `json:"views"`
}

// calibrationFileV1 is the persisted multi-view calibration document.
type calibrationFileV1 struct {
	Views map[string]posture.Baseline `json:"views"`
}

// loadBaselineFile reads a persisted calibration document. A missing file is
// not an error. The legacy single-baseline format (a flat Baseline object)
// is migrated to the "default" view.
func loadBaselineFile(path string) (map[string]posture.Baseline, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var doc calibrationFileV1
	if err := json.Unmarshal(data, &doc); err == nil && doc.Views != nil {
		return doc.Views, nil
	}
	// legacy flat format: a single baseline for the default view
	var legacy posture.Baseline
	if err := json.Unmarshal(data, &legacy); err != nil {
		return nil, err
	}
	return map[string]posture.Baseline{posture.ViewDefault: legacy}, nil
}

// saveBaselineFile persists every set baseline as pretty-printed JSON.
func saveBaselineFile(path string, views map[string]posture.Baseline) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(calibrationFileV1{Views: views}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// calibrationPayload snapshots the engine's per-view calibration state.
func (s *Server) calibrationPayload() calibrationResponse {
	out := calibrationResponse{Views: map[string]viewCalibration{}}
	for view, b := range s.engine.Baselines() {
		bas := b
		out.Views[view] = viewCalibration{Calibrated: true, Baseline: &bas}
	}
	return out
}

// persistCalibration writes the current per-view baselines to disk.
func (s *Server) persistCalibration() {
	if s.calibFile == "" {
		return
	}
	if err := saveBaselineFile(s.calibFile, s.engine.Baselines()); err != nil {
		s.logger.Error("persist calibration failed", "err", err, "path", s.calibFile)
	}
}

// handleCalibrate accepts a batch of pose frames for one view, derives the
// baseline from them, installs it in the engine and persists it.
func (s *Server) handleCalibrate(w http.ResponseWriter, r *http.Request) {
	var req calibrateRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	b, err := posture.CalibratePoses(req.Poses, s.engine.Config().MinVisibility)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	view := posture.NormalizeView(req.View)
	s.engine.SetBaselineView(view, b)
	s.persistCalibration()
	s.logger.Info("calibration updated", "view", view, "baseline", b)
	writeJSON(w, http.StatusOK, s.calibrationPayload())
}

// handleGetCalibration returns the per-view calibration state.
func (s *Server) handleGetCalibration(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.calibrationPayload())
}

// handleDeleteCalibration clears one view (?view=front) or every view.
func (s *Server) handleDeleteCalibration(w http.ResponseWriter, r *http.Request) {
	view := r.URL.Query().Get("view")
	if view != "" {
		s.engine.ClearBaselineView(posture.NormalizeView(view))
		s.rewriteOrRemoveCalibration()
	} else {
		for name := range s.engine.Baselines() {
			s.engine.ClearBaselineView(name)
		}
		if s.calibFile != "" {
			if err := os.Remove(s.calibFile); err != nil && !errors.Is(err, fs.ErrNotExist) {
				s.logger.Error("remove calibration file failed", "err", err, "path", s.calibFile)
			}
		}
	}
	s.logger.Info("calibration cleared", "view", view)
	writeJSON(w, http.StatusOK, s.calibrationPayload())
}

// rewriteOrRemoveCalibration persists the remaining baselines, removing the
// file entirely when the last baseline was just cleared.
func (s *Server) rewriteOrRemoveCalibration() {
	if s.calibFile == "" {
		return
	}
	if len(s.engine.Baselines()) == 0 {
		if err := os.Remove(s.calibFile); err != nil && !errors.Is(err, fs.ErrNotExist) {
			s.logger.Error("remove calibration file failed", "err", err, "path", s.calibFile)
		}
		return
	}
	s.persistCalibration()
}
