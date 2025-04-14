package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"go.uber.org/zap"
)

// Server represents the API server
type Server struct {
	router *mux.Router
	logger *zap.Logger
	srv    *http.Server

	// Service interfaces
	blockchain BlockchainService
	consensus  ConsensusService
	p2p        P2PService
}

// BlockchainService defines the interface for blockchain operations
type BlockchainService interface {
	GetBlock(height uint64) (interface{}, error)
	GetCurrentBlock() interface{}
	GetHeight() uint64
	AddTransaction(tx interface{}) error
}

// ConsensusService defines the interface for consensus operations
type ConsensusService interface {
	GetValidatorSet() interface{}
	IsValidator(address string) bool
	GetStake(address string) (interface{}, error)
}

// P2PService defines the interface for P2P network operations
type P2PService interface {
	GetPeers() interface{}
	GetPeerCount() int
	AddPeer(addr string) error
}

// Config represents API server configuration
type Config struct {
	ListenAddr string
	EnableTLS  bool
	CertFile   string
	KeyFile    string
}

// NewServer creates a new API server
func NewServer(cfg *Config, logger *zap.Logger) *Server {
	server := &Server{
		router: mux.NewRouter(),
		logger: logger,
	}

	// Initialize routes
	server.initRoutes()

	// Initialize HTTP server
	server.srv = &http.Server{
		Addr:         cfg.ListenAddr,
		Handler:      server.router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	return server
}

// Start starts the API server
func (s *Server) Start() error {
	s.logger.Info("Starting API server", zap.String("addr", s.srv.Addr))
	return s.srv.ListenAndServe()
}

// Stop stops the API server
func (s *Server) Stop(ctx context.Context) error {
	return s.srv.Shutdown(ctx)
}

// SetBlockchainService sets the blockchain service
func (s *Server) SetBlockchainService(service BlockchainService) {
	s.blockchain = service
}

// SetConsensusService sets the consensus service
func (s *Server) SetConsensusService(service ConsensusService) {
	s.consensus = service
}

// SetP2PService sets the P2P service
func (s *Server) SetP2PService(service P2PService) {
	s.p2p = service
}

// initRoutes initializes API routes
func (s *Server) initRoutes() {
	// Blockchain routes
	s.router.HandleFunc("/block/{height}", s.handleGetBlock).Methods("GET")
	s.router.HandleFunc("/block/latest", s.handleGetLatestBlock).Methods("GET")
	s.router.HandleFunc("/height", s.handleGetHeight).Methods("GET")
	s.router.HandleFunc("/tx", s.handleSubmitTransaction).Methods("POST")

	// Consensus routes
	s.router.HandleFunc("/validators", s.handleGetValidators).Methods("GET")
	s.router.HandleFunc("/validator/{address}", s.handleGetValidator).Methods("GET")
	s.router.HandleFunc("/stake/{address}", s.handleGetStake).Methods("GET")

	// P2P routes
	s.router.HandleFunc("/peers", s.handleGetPeers).Methods("GET")
	s.router.HandleFunc("/peer/count", s.handleGetPeerCount).Methods("GET")
	s.router.HandleFunc("/peer", s.handleAddPeer).Methods("POST")

	// Middleware
	s.router.Use(s.loggingMiddleware)
}

// Middleware for request logging
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		s.logger.Info("API request",
			zap.String("method", r.Method),
			zap.String("path", r.URL.Path),
			zap.Duration("duration", time.Since(start)),
		)
	})
}

// Response helpers
func (s *Server) sendJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		s.sendError(w, http.StatusInternalServerError, "Failed to encode response")
	}
}

func (s *Server) sendError(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": message})
}

// Route handlers

func (s *Server) handleGetBlock(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	height := vars["height"]
	// TODO: Parse height and get block
	s.sendJSON(w, map[string]string{"height": height})
}

func (s *Server) handleGetLatestBlock(w http.ResponseWriter, r *http.Request) {
	if s.blockchain == nil {
		s.sendError(w, http.StatusServiceUnavailable, "Blockchain service not available")
		return
	}
	block := s.blockchain.GetCurrentBlock()
	s.sendJSON(w, block)
}

func (s *Server) handleGetHeight(w http.ResponseWriter, r *http.Request) {
	if s.blockchain == nil {
		s.sendError(w, http.StatusServiceUnavailable, "Blockchain service not available")
		return
	}
	height := s.blockchain.GetHeight()
	s.sendJSON(w, map[string]uint64{"height": height})
}

func (s *Server) handleSubmitTransaction(w http.ResponseWriter, r *http.Request) {
	// TODO: Implement transaction submission
	s.sendError(w, http.StatusNotImplemented, "Not implemented")
}

func (s *Server) handleGetValidators(w http.ResponseWriter, r *http.Request) {
	if s.consensus == nil {
		s.sendError(w, http.StatusServiceUnavailable, "Consensus service not available")
		return
	}
	validators := s.consensus.GetValidatorSet()
	s.sendJSON(w, validators)
}

func (s *Server) handleGetValidator(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	address := vars["address"]
	if s.consensus == nil {
		s.sendError(w, http.StatusServiceUnavailable, "Consensus service not available")
		return
	}
	isValidator := s.consensus.IsValidator(address)
	s.sendJSON(w, map[string]bool{"is_validator": isValidator})
}

func (s *Server) handleGetStake(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	address := vars["address"]
	if s.consensus == nil {
		s.sendError(w, http.StatusServiceUnavailable, "Consensus service not available")
		return
	}
	stake, err := s.consensus.GetStake(address)
	if err != nil {
		s.sendError(w, http.StatusNotFound, fmt.Sprintf("Failed to get stake: %v", err))
		return
	}
	s.sendJSON(w, stake)
}

func (s *Server) handleGetPeers(w http.ResponseWriter, r *http.Request) {
	if s.p2p == nil {
		s.sendError(w, http.StatusServiceUnavailable, "P2P service not available")
		return
	}
	peers := s.p2p.GetPeers()
	s.sendJSON(w, peers)
}

func (s *Server) handleGetPeerCount(w http.ResponseWriter, r *http.Request) {
	if s.p2p == nil {
		s.sendError(w, http.StatusServiceUnavailable, "P2P service not available")
		return
	}
	count := s.p2p.GetPeerCount()
	s.sendJSON(w, map[string]int{"peer_count": count})
}

func (s *Server) handleAddPeer(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Address string `json:"address"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.sendError(w, http.StatusBadRequest, "Invalid request body")
		return
	}
	if s.p2p == nil {
		s.sendError(w, http.StatusServiceUnavailable, "P2P service not available")
		return
	}
	if err := s.p2p.AddPeer(req.Address); err != nil {
		s.sendError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to add peer: %v", err))
		return
	}
	s.sendJSON(w, map[string]string{"status": "success"})
}
