package server

import (
	"embed"
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	"github.com/linuxsuren/sit/posture"
)

//go:embed web
var webFS embed.FS

const (
	// wsReadWait is how long a connection may stay silent before it is
	// dropped; browser clients keep it alive with pings or pose frames.
	wsReadWait = 60 * time.Second
	// wsWriteWait bounds a single write.
	wsWriteWait = 10 * time.Second
)

var upgrader = websocket.Upgrader{
	// The UI is served from this server itself; no cross-origin use case.
	CheckOrigin: func(r *http.Request) bool { return true },
}

// handlePoseWS evaluates pose frames streamed over WebSocket.
//
// Protocol (JSON text frames):
//
//	client → server: {"landmarks": [{"x","y","z","visibility"}, ...],
//	                  "timestamp_ms": 1712345678901}
//	server → client: posture.Result (status/metrics/issues/seated_seconds)
//
// Malformed frames are skipped without closing the connection; dead
// connections are reaped by the read deadline.
func (s *Server) handlePoseWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		s.logger.Warn("ws upgrade failed", "err", err, "remote", r.RemoteAddr)
		return
	}
	defer conn.Close()

	log := s.logger.With("remote", r.RemoteAddr)
	log.Info("pose stream connected")

	conn.SetReadLimit(1 << 20) // 1 MiB: 33 landmarks are ~4 KiB
	if err := conn.SetReadDeadline(time.Now().Add(wsReadWait)); err != nil {
		return
	}
	conn.SetPongHandler(func(string) error {
		return conn.SetReadDeadline(time.Now().Add(wsReadWait))
	})

	// keepalive pings
	pingStop := make(chan struct{})
	defer close(pingStop)
	go func() {
		ticker := time.NewTicker(20 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := conn.WriteControl(websocket.PingMessage, nil,
					time.Now().Add(wsWriteWait)); err != nil {
					return
				}
			case <-pingStop:
				return
			}
		}
	}()

	for {
		var p posture.Pose
		if err := conn.ReadJSON(&p); err != nil {
			// normal close, deadline expiry or protocol error: all end here
			log.Info("pose stream disconnected", "err", err)
			return
		}
		result := s.engine.Evaluate(p)

		if err := conn.SetWriteDeadline(time.Now().Add(wsWriteWait)); err != nil {
			return
		}
		if err := conn.WriteJSON(result); err != nil {
			log.Info("pose stream write failed", "err", err)
			return
		}
	}
}
