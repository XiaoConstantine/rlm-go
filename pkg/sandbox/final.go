package sandbox

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

const finalMarkerPrefix = "__RLM_FINAL__"

func newFinalToken() string {
	var buf [16]byte
	if _, err := rand.Read(buf[:]); err == nil {
		return hex.EncodeToString(buf[:])
	}
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func formatFinalMarker(token, value string) string {
	encoded := base64.StdEncoding.EncodeToString([]byte(value))
	return fmt.Sprintf("\n%s%s__%s\n", finalMarkerPrefix, token, encoded)
}

func findFinalMarker(output, token string) (string, bool) {
	if token == "" {
		return "", false
	}
	marker := finalMarkerPrefix + token + "__"
	for _, line := range strings.Split(output, "\n") {
		if !strings.HasPrefix(line, marker) {
			continue
		}
		decoded, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(line, marker))
		if err != nil {
			continue
		}
		return string(decoded), true
	}
	return "", false
}

func stripFinalMarkers(output, token string) string {
	if token == "" {
		return output
	}
	marker := finalMarkerPrefix + token + "__"
	lines := strings.Split(output, "\n")
	dst := lines[:0]
	for _, line := range lines {
		if strings.HasPrefix(line, marker) {
			continue
		}
		dst = append(dst, line)
	}
	return strings.Join(dst, "\n")
}
