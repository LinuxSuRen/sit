// Package server exposes the posture engine over HTTP: a built-in web page
// that captures pose landmarks in the browser, a WebSocket endpoint that
// evaluates them, and a JSON API for the threshold configuration.
//
// The server owns one shared posture.Engine: it is designed for monitoring a
// single person (one camera / one browser tab), so the debounce and
// long-sitting state intentionally live in that single engine.
package server

import (
	"encoding/json"
	"io/fs"
	"log/slog"
	"net/http"

	"github.com/linuxsuren/sit/posture"
)

// Server wires the HTTP routes to the posture engine.
type Server struct {
	engine    *posture.Engine
	logger    *slog.Logger
	calibFile string
}

// New creates a Server with a fresh engine using the given config.
func New(cfg posture.Config, logger *slog.Logger, opts ...Option) (*Server, error) {
	engine, err := posture.NewEngine(cfg)
	if err != nil {
		return nil, err
	}
	if logger == nil {
		logger = slog.Default()
	}
	s := &Server{engine: engine, logger: logger}
	for _, opt := range opts {
		opt(s)
	}
	if s.calibFile != "" {
		views, err := loadBaselineFile(s.calibFile)
		if err != nil {
			logger.Warn("load calibration failed, starting uncalibrated",
				"err", err, "path", s.calibFile)
		} else {
			for view, b := range views {
				engine.SetBaselineView(view, b)
				logger.Info("calibration loaded", "view", view, "baseline", b, "path", s.calibFile)
			}
		}
	}
	return s, nil
}

// Handler returns the fully wired HTTP handler.
func (s *Server) Handler() http.Handler {
	content, err := fs.Sub(webFS, "web")
	if err != nil { // cannot happen with a valid embed directive
		panic(err)
	}
	mux := http.NewServeMux()
	mux.Handle("GET /", http.FileServer(http.FS(content)))
	mux.HandleFunc("GET /api/config", s.handleGetConfig)
	mux.HandleFunc("PUT /api/config", s.handlePutConfig)
	mux.HandleFunc("POST /api/calibrate", s.handleCalibrate)
	mux.HandleFunc("GET /api/calibration", s.handleGetCalibration)
	mux.HandleFunc("DELETE /api/calibration", s.handleDeleteCalibration)
	mux.HandleFunc("GET /ws/pose", s.handlePoseWS)
	return logRequests(s.logger, mux)
}

func logRequests(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		logger.Debug("http request", "method", r.Method, "path", r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

// handleGetConfig returns the current engine configuration as JSON.
func (s *Server) handleGetConfig(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.engine.Config())
}

// handlePutConfig replaces the whole configuration. The body must be a
// complete, valid Config document; unknown fields are rejected to keep the
// client/server contract strict.
func (s *Server) handlePutConfig(w http.ResponseWriter, r *http.Request) {
	var cfg posture.Config
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	if err := s.engine.SetConfig(cfg); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, s.engine.Config())
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// headers already sent; nothing left to do but report
		slog.Error("write json response", "err", err)
	}
}
