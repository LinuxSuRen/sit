// sitcoach 是一个坐姿检测服务：浏览器用 MediaPipe 提取人体关键点，
// Go 后端基于几何角度判定坐姿是否标准，并提供久坐提醒。
package main

import (
	"flag"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/linuxsuren/sitcoach/posture"
	"github.com/linuxsuren/sitcoach/server"
)

// 构建时通过 -ldflags 注入（见 .github/workflows/release.yml）。
var (
	version   = "dev"
	commit    = "none"
	buildDate = "unknown"
)

func main() {
	addr := flag.String("addr", ":8080", "HTTP 监听地址")
	calibPath := flag.String("calibration", "", "坐姿标定基线文件路径；留空时使用 ~/.sitcoach-calibration.json")
	showVersion := flag.Bool("version", false, "输出版本信息后退出")
	flag.Parse()

	if *showVersion {
		slog.New(slog.NewTextHandler(os.Stdout, nil)).Info("sitcoach",
			"version", version, "commit", commit, "build_date", buildDate)
		return
	}

	path := *calibPath
	if path == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			slog.Warn("resolve home dir failed, calibration persistence disabled", "err", err)
		} else {
			path = filepath.Join(home, ".sitcoach-calibration.json")
		}
	}

	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	var opts []server.Option
	if path != "" {
		opts = append(opts, server.CalibrationFile(path))
	}
	srv, err := server.New(posture.DefaultConfig(), logger, opts...)
	if err != nil {
		logger.Error("init server failed", "err", err)
		os.Exit(1)
	}

	logger.Info("sitcoach server listening", "addr", *addr,
		"page", "http://localhost:8080/", "calibration_file", path,
		"version", version, "commit", commit)
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		logger.Error("server exited", "err", err)
		os.Exit(1)
	}
}
