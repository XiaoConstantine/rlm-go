package sandbox

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

// MessageType identifies the type of IPC message.
type MessageType string

const (
	// MessageQuery is a Query() call from the sandbox to the host.
	MessageQuery MessageType = "query"

	// MessageQueryBatched is a QueryBatched() call from the sandbox.
	MessageQueryBatched MessageType = "query_batched"

	// MessageResponse is a response from the host to the sandbox.
	MessageResponse MessageType = "response"

	// MessageError is an error response from the host.
	MessageError MessageType = "error"

	// MessageReady indicates the sandbox is ready.
	MessageReady MessageType = "ready"

	// MessageExit indicates the sandbox is exiting.
	MessageExit MessageType = "exit"
)

// IPCMessage is the JSON message format for host-sandbox communication.
type IPCMessage struct {
	// Type identifies the message type.
	Type MessageType `json:"type"`

	// ID is a unique identifier for request-response correlation.
	ID string `json:"id,omitempty"`

	// Prompt is the query prompt (for Query messages).
	Prompt string `json:"prompt,omitempty"`

	// Prompts is a list of query prompts (for QueryBatched messages).
	Prompts []string `json:"prompts,omitempty"`

	// Response is the LLM response (for Response messages).
	Response string `json:"response,omitempty"`

	// Responses is a list of LLM responses (for batched Response messages).
	Responses []string `json:"responses,omitempty"`

	// Error is the error message (for Error messages).
	Error string `json:"error,omitempty"`

	// TokenUsage contains token usage metadata.
	TokenUsage *TokenUsage `json:"token_usage,omitempty"`

	// TokenUsages contains token usage for batched queries.
	TokenUsages []TokenUsage `json:"token_usages,omitempty"`

	// Duration is the execution time in seconds.
	Duration float64 `json:"duration,omitempty"`
}

// TokenUsage tracks token usage for an LLM call.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// IPCServer handles incoming IPC requests from the sandbox container.
type IPCServer struct {
	listener net.Listener
	client   LLMClient
	port     int
	mu       sync.RWMutex
	calls    []LLMCall
	running  bool
	ctx      context.Context
	cancel   context.CancelFunc
}

// NewIPCServer creates a new IPC server that handles Query() calls from the sandbox.
func NewIPCServer(client LLMClient, port int) (*IPCServer, error) {
	addr := fmt.Sprintf("127.0.0.1:%d", port)
	if port == 0 {
		addr = "127.0.0.1:0"
	}

	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("failed to start IPC server: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Get the assigned port if auto-assigned
	actualPort := listener.Addr().(*net.TCPAddr).Port

	return &IPCServer{
		listener: listener,
		client:   client,
		port:     actualPort,
		calls:    nil,
		running:  false,
		ctx:      ctx,
		cancel:   cancel,
	}, nil
}

// Port returns the port the server is listening on.
func (s *IPCServer) Port() int {
	return s.port
}

// Address returns the full address (host:port) the server is listening on.
func (s *IPCServer) Address() string {
	return s.listener.Addr().String()
}

// Start begins accepting connections.
func (s *IPCServer) Start() {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return
	}
	s.running = true
	s.mu.Unlock()

	go s.acceptLoop()
}

// acceptLoop accepts and handles incoming connections.
func (s *IPCServer) acceptLoop() {
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
		}

		// Set a deadline to periodically check for cancellation
		if tcpListener, ok := s.listener.(*net.TCPListener); ok {
			tcpListener.SetDeadline(time.Now().Add(1 * time.Second))
		}

		conn, err := s.listener.Accept()
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue // Timeout, check for cancellation
			}
			// Check if we're shutting down
			select {
			case <-s.ctx.Done():
				return
			default:
			}
			continue
		}

		go s.handleConnection(conn)
	}
}

// handleConnection processes messages from a single connection.
func (s *IPCServer) handleConnection(conn net.Conn) {
	defer conn.Close()

	reader := bufio.NewReader(conn)
	encoder := json.NewEncoder(conn)

	for {
		select {
		case <-s.ctx.Done():
			return
		default:
		}

		// Set read deadline
		conn.SetReadDeadline(time.Now().Add(5 * time.Minute))

		// Read a line (one JSON message per line)
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				return
			}
			return
		}

		// Parse the message
		var msg IPCMessage
		if err := json.Unmarshal(line, &msg); err != nil {
			s.sendError(encoder, "", fmt.Sprintf("invalid JSON: %v", err))
			continue
		}

		// Handle the message
		response := s.handleMessage(msg)
		if err := encoder.Encode(response); err != nil {
			return // Connection error
		}

		// Check for exit message
		if msg.Type == MessageExit {
			return
		}
	}
}

// handleMessage processes a single IPC message and returns the response.
func (s *IPCServer) handleMessage(msg IPCMessage) IPCMessage {
	switch msg.Type {
	case MessageQuery:
		return s.handleQuery(msg)
	case MessageQueryBatched:
		return s.handleQueryBatched(msg)
	case MessageReady:
		return IPCMessage{Type: MessageResponse, ID: msg.ID}
	case MessageExit:
		return IPCMessage{Type: MessageResponse, ID: msg.ID}
	default:
		return IPCMessage{
			Type:  MessageError,
			ID:    msg.ID,
			Error: fmt.Sprintf("unknown message type: %s", msg.Type),
		}
	}
}

// handleQuery processes a single Query() request.
func (s *IPCServer) handleQuery(msg IPCMessage) IPCMessage {
	start := time.Now()

	resp, err := s.client.Query(s.ctx, msg.Prompt)
	duration := time.Since(start).Seconds()

	if err != nil {
		s.recordCall(msg.Prompt, fmt.Sprintf("Error: %v", err), duration, 0, 0)
		return IPCMessage{
			Type:  MessageError,
			ID:    msg.ID,
			Error: err.Error(),
		}
	}

	s.recordCall(msg.Prompt, resp.Response, duration, resp.PromptTokens, resp.CompletionTokens)

	return IPCMessage{
		Type:     MessageResponse,
		ID:       msg.ID,
		Response: resp.Response,
		TokenUsage: &TokenUsage{
			PromptTokens:     resp.PromptTokens,
			CompletionTokens: resp.CompletionTokens,
		},
		Duration: duration,
	}
}

// handleQueryBatched processes a batched Query() request.
func (s *IPCServer) handleQueryBatched(msg IPCMessage) IPCMessage {
	start := time.Now()

	results, err := s.client.QueryBatched(s.ctx, msg.Prompts)
	duration := time.Since(start).Seconds()

	if err != nil {
		// Record each as failed
		for _, prompt := range msg.Prompts {
			s.recordCall(prompt, fmt.Sprintf("Error: %v", err), duration/float64(len(msg.Prompts)), 0, 0)
		}
		return IPCMessage{
			Type:  MessageError,
			ID:    msg.ID,
			Error: err.Error(),
		}
	}

	responses := make([]string, len(results))
	usages := make([]TokenUsage, len(results))
	for i, r := range results {
		responses[i] = r.Response
		usages[i] = TokenUsage{
			PromptTokens:     r.PromptTokens,
			CompletionTokens: r.CompletionTokens,
		}
		s.recordCall(msg.Prompts[i], r.Response, duration/float64(len(results)), r.PromptTokens, r.CompletionTokens)
	}

	return IPCMessage{
		Type:        MessageResponse,
		ID:          msg.ID,
		Responses:   responses,
		TokenUsages: usages,
		Duration:    duration,
	}
}

// recordCall records an LLM call for later retrieval.
func (s *IPCServer) recordCall(prompt, response string, duration float64, promptTokens, completionTokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	})
}

// GetCalls returns and clears the recorded LLM calls.
func (s *IPCServer) GetCalls() []LLMCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	calls := s.calls
	s.calls = nil
	return calls
}

// ClearCalls clears the recorded LLM calls.
func (s *IPCServer) ClearCalls() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = nil
}

// Stop shuts down the server.
func (s *IPCServer) Stop() error {
	s.cancel()
	return s.listener.Close()
}

// sendError sends an error response.
func (s *IPCServer) sendError(encoder *json.Encoder, id, errMsg string) {
	_ = encoder.Encode(IPCMessage{
		Type:  MessageError,
		ID:    id,
		Error: errMsg,
	})
}

// IPCClient is used by the sandbox to communicate with the host.
// This would be embedded in the container code.
type IPCClient struct {
	conn    net.Conn
	encoder *json.Encoder
	decoder *json.Decoder
	mu      sync.Mutex
	counter int
}

// NewIPCClient creates a new IPC client that connects to the host server.
func NewIPCClient(addr string) (*IPCClient, error) {
	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to IPC server: %w", err)
	}

	return &IPCClient{
		conn:    conn,
		encoder: json.NewEncoder(conn),
		decoder: json.NewDecoder(conn),
	}, nil
}

// nextID generates a unique message ID.
func (c *IPCClient) nextID() string {
	c.counter++
	return fmt.Sprintf("msg-%d", c.counter)
}

// Query sends a Query request and waits for the response.
func (c *IPCClient) Query(prompt string) (string, *TokenUsage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	id := c.nextID()
	msg := IPCMessage{
		Type:   MessageQuery,
		ID:     id,
		Prompt: prompt,
	}

	if err := c.encoder.Encode(msg); err != nil {
		return "", nil, fmt.Errorf("failed to send query: %w", err)
	}

	var resp IPCMessage
	if err := c.decoder.Decode(&resp); err != nil {
		return "", nil, fmt.Errorf("failed to receive response: %w", err)
	}

	if resp.Type == MessageError {
		return "", nil, fmt.Errorf("query error: %s", resp.Error)
	}

	return resp.Response, resp.TokenUsage, nil
}

// QueryBatched sends a batched Query request.
func (c *IPCClient) QueryBatched(prompts []string) ([]string, []TokenUsage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	id := c.nextID()
	msg := IPCMessage{
		Type:    MessageQueryBatched,
		ID:      id,
		Prompts: prompts,
	}

	if err := c.encoder.Encode(msg); err != nil {
		return nil, nil, fmt.Errorf("failed to send batched query: %w", err)
	}

	var resp IPCMessage
	if err := c.decoder.Decode(&resp); err != nil {
		return nil, nil, fmt.Errorf("failed to receive response: %w", err)
	}

	if resp.Type == MessageError {
		return nil, nil, fmt.Errorf("query error: %s", resp.Error)
	}

	return resp.Responses, resp.TokenUsages, nil
}

// Close closes the connection.
func (c *IPCClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Send exit message
	_ = c.encoder.Encode(IPCMessage{Type: MessageExit})

	return c.conn.Close()
}

// GenerateContainerRLMCode generates Go code for the container that provides
// Query() and QueryBatched() functions that communicate via IPC.
func GenerateContainerRLMCode(ipcAddr string) string {
	// Using backtick for struct tags
	bt := "`"
	return fmt.Sprintf(`package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"
)

type messageType string

const (
	messageQuery        messageType = "query"
	messageQueryBatched messageType = "query_batched"
	messageResponse     messageType = "response"
	messageError        messageType = "error"
)

type tokenUsage struct {
	PromptTokens     int %sjson:"prompt_tokens"%s
	CompletionTokens int %sjson:"completion_tokens"%s
}

type ipcMessage struct {
	Type        messageType   %sjson:"type"%s
	ID          string        %sjson:"id,omitempty"%s
	Prompt      string        %sjson:"prompt,omitempty"%s
	Prompts     []string      %sjson:"prompts,omitempty"%s
	Response    string        %sjson:"response,omitempty"%s
	Responses   []string      %sjson:"responses,omitempty"%s
	Error       string        %sjson:"error,omitempty"%s
	TokenUsage  *tokenUsage   %sjson:"token_usage,omitempty"%s
	TokenUsages []tokenUsage  %sjson:"token_usages,omitempty"%s
}

var (
	ipcConn    net.Conn
	ipcEncoder *json.Encoder
	ipcReader  *bufio.Reader
	ipcMu      sync.Mutex
	ipcCounter int
	ipcAddr    = %q
)

func init() {
	var err error
	ipcConn, err = net.DialTimeout("tcp", ipcAddr, 10*time.Second)
	if err != nil {
		panic(fmt.Sprintf("failed to connect to IPC server: %%v", err))
	}
	ipcEncoder = json.NewEncoder(ipcConn)
	ipcReader = bufio.NewReader(ipcConn)
}

func nextID() string {
	ipcCounter++
	return fmt.Sprintf("msg-%%d", ipcCounter)
}

// Query sends a query to the host LLM and returns the response.
func Query(prompt string) string {
	ipcMu.Lock()
	defer ipcMu.Unlock()

	msg := ipcMessage{
		Type:   messageQuery,
		ID:     nextID(),
		Prompt: prompt,
	}

	if err := ipcEncoder.Encode(msg); err != nil {
		return fmt.Sprintf("Error: failed to send query: %%v", err)
	}

	line, err := ipcReader.ReadBytes('\n')
	if err != nil {
		return fmt.Sprintf("Error: failed to receive response: %%v", err)
	}

	var resp ipcMessage
	if err := json.Unmarshal(line, &resp); err != nil {
		return fmt.Sprintf("Error: failed to parse response: %%v", err)
	}

	if resp.Type == messageError {
		return fmt.Sprintf("Error: %%s", resp.Error)
	}

	return resp.Response
}

// QueryBatched sends multiple queries concurrently and returns the responses.
func QueryBatched(prompts []string) []string {
	ipcMu.Lock()
	defer ipcMu.Unlock()

	msg := ipcMessage{
		Type:    messageQueryBatched,
		ID:      nextID(),
		Prompts: prompts,
	}

	if err := ipcEncoder.Encode(msg); err != nil {
		results := make([]string, len(prompts))
		for i := range results {
			results[i] = fmt.Sprintf("Error: failed to send query: %%v", err)
		}
		return results
	}

	line, err := ipcReader.ReadBytes('\n')
	if err != nil {
		results := make([]string, len(prompts))
		for i := range results {
			results[i] = fmt.Sprintf("Error: failed to receive response: %%v", err)
		}
		return results
	}

	var resp ipcMessage
	if err := json.Unmarshal(line, &resp); err != nil {
		results := make([]string, len(prompts))
		for i := range results {
			results[i] = fmt.Sprintf("Error: failed to parse response: %%v", err)
		}
		return results
	}

	if resp.Type == messageError {
		results := make([]string, len(prompts))
		for i := range results {
			results[i] = fmt.Sprintf("Error: %%s", resp.Error)
		}
		return results
	}

	return resp.Responses
}
`,
		bt, bt, bt, bt, // tokenUsage (2 fields x 2 backticks)
		bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, bt, // ipcMessage (9 fields x 2 backticks)
		ipcAddr)
}
