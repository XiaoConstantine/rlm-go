// Package interpreter provides shared Yaegi interpreter setup code.
package interpreter

import (
	"bytes"
	"fmt"

	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

// Base imports shared by all Yaegi interpreters.
const baseImports = `
import "fmt"
import "strings"
import "regexp"
import . "rlm/rlm"
`

// Additional imports needed by REPL for context loading.
const extendedImports = `
import "strconv"
import "encoding/json"
import "sort"
`

// MinMaxHelpers defines min/max since Yaegi doesn't support Go 1.21 builtins.
const MinMaxHelpers = `
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
`

// SetupCode is the common initialization code for all Yaegi interpreters.
// It pre-imports common packages and defines min/max helpers.
const SetupCode = baseImports + MinMaxHelpers

// SetupCodeExtended includes additional imports (strconv, encoding/json, sort).
// Used by REPL which needs these for context loading.
const SetupCodeExtended = baseImports + extendedImports + MinMaxHelpers

// Config holds configuration for creating a new interpreter.
type Config struct {
	Stdout *bytes.Buffer
	Stderr *bytes.Buffer
}

// New creates a new Yaegi interpreter with standard library loaded.
func New(cfg Config) (*interp.Interpreter, error) {
	i := interp.New(interp.Options{
		Stdout: cfg.Stdout,
		Stderr: cfg.Stderr,
	})

	if err := i.Use(stdlib.Symbols); err != nil {
		return nil, fmt.Errorf("failed to load stdlib: %w", err)
	}

	return i, nil
}

// RunSetup executes the setup code on an interpreter.
// Use SetupCode for basic setup or SetupCodeExtended for REPL usage.
func RunSetup(i *interp.Interpreter, setupCode string) error {
	_, err := i.Eval(setupCode)
	return err
}
