package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/ruinousarpan/PhantomX/consensus"
	"github.com/ruinousarpan/PhantomX/core"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

var (
	logger *zap.Logger
	cfg    *Config
)

type Config struct {
	// Network configuration
	NetworkID      uint64   `mapstructure:"network_id"`
	ListenAddr     string   `mapstructure:"listen_addr"`
	BootstrapNodes []string `mapstructure:"bootstrap_nodes"`

	// Consensus configuration
	ConsensusType string `mapstructure:"consensus_type"` // "pos" or "hybrid"
	MinStake      uint64 `mapstructure:"min_stake"`

	// Storage configuration
	DataDir string `mapstructure:"data_dir"`
}

func init() {
	// Initialize logger
	logger, _ = zap.NewDevelopment()
	defer logger.Sync()

	// Initialize config
	cfg = &Config{}
	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	viper.AddConfigPath(".")
	if err := viper.ReadInConfig(); err != nil {
		logger.Fatal("Failed to read config", zap.Error(err))
	}
	if err := viper.Unmarshal(cfg); err != nil {
		logger.Fatal("Failed to unmarshal config", zap.Error(err))
	}
}

func main() {
	rootCmd := &cobra.Command{
		Use:   "phantomx-node",
		Short: "PhantomX blockchain node",
		Run:   runNode,
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func runNode(cmd *cobra.Command, args []string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	logger.Info("Starting PhantomX node",
		zap.Uint64("networkID", cfg.NetworkID),
		zap.String("listenAddr", cfg.ListenAddr),
	)

	// Initialize node components
	node, err := NewNode(ctx, cfg)
	if err != nil {
		logger.Fatal("Failed to create node", zap.Error(err))
	}

	// Start the node
	if err := node.Start(); err != nil {
		logger.Fatal("Failed to start node", zap.Error(err))
	}

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	// Graceful shutdown
	logger.Info("Shutting down node...")
	if err := node.Stop(); err != nil {
		logger.Error("Error during shutdown", zap.Error(err))
	}
}

// Node represents the PhantomX blockchain node
type Node struct {
	ctx    context.Context
	config *Config
	logger *zap.Logger

	// Core components
	blockchain *core.Blockchain
	consensus  *consensus.Consensus
	// p2p       *P2P
	// api       *API
}

// NewNode creates a new blockchain node
func NewNode(ctx context.Context, cfg *Config) (*Node, error) {
	node := &Node{
		ctx:    ctx,
		config: cfg,
		logger: logger,
	}

	// Initialize components
	if err := node.initComponents(); err != nil {
		return nil, fmt.Errorf("failed to initialize node components: %w", err)
	}

	return node, nil
}

// initComponents initializes all node components
func (n *Node) initComponents() error {
	// Initialize blockchain
	blockchain, err := core.NewBlockchain(n.config.DataDir)
	if err != nil {
		return fmt.Errorf("failed to initialize blockchain: %w", err)
	}
	n.blockchain = blockchain

	// Initialize consensus
	n.consensus = consensus.NewConsensus(
		consensus.ConsensusType(n.config.ConsensusType),
		n.config.MinStake,
		n.logger,
	)

	// TODO: Initialize P2P and API components

	return nil
}

// Start starts the node and all its components
func (n *Node) Start() error {
	// Start blockchain
	if err := n.blockchain.Start(); err != nil {
		return fmt.Errorf("failed to start blockchain: %w", err)
	}

	// TODO: Start P2P and API components

	n.logger.Info("Node started successfully")
	return nil
}

// Stop stops the node and all its components
func (n *Node) Stop() error {
	// Stop blockchain
	if err := n.blockchain.Stop(); err != nil {
		n.logger.Error("Failed to stop blockchain", zap.Error(err))
	}

	// TODO: Stop P2P and API components

	n.logger.Info("Node stopped successfully")
	return nil
}
