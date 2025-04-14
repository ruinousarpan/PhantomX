package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
)

// Block represents a block in the blockchain
type Block struct {
	Header *BlockHeader `json:"header"`
	Body   *BlockBody   `json:"body"`
}

// BlockHeader contains metadata about the block
type BlockHeader struct {
	Version       uint32    `json:"version"`
	PrevHash      string    `json:"prevHash"`
	MerkleRoot    string    `json:"merkleRoot"`
	Timestamp     time.Time `json:"timestamp"`
	Height        uint64    `json:"height"`
	Nonce         uint64    `json:"nonce"`
	ValidatorAddr string    `json:"validatorAddr"`
}

// BlockBody contains the actual transactions
type BlockBody struct {
	Transactions []*Transaction `json:"transactions"`
}

// Transaction represents a transaction in the blockchain
type Transaction struct {
	Hash      string    `json:"hash"`
	From      string    `json:"from"`
	To        string    `json:"to"`
	Value     uint64    `json:"value"`
	Data      []byte    `json:"data"`
	Timestamp time.Time `json:"timestamp"`
	Signature []byte    `json:"signature"`
}

// Blockchain represents the main blockchain structure
type Blockchain struct {
	mu sync.RWMutex

	// Chain state
	currentBlock *Block
	genesisBlock *Block
	height       uint64

	// Storage
	db *leveldb.DB

	// Channels for block processing
	newBlockCh chan *Block
	quitCh     chan struct{}
}

// NewBlockchain creates a new blockchain instance
func NewBlockchain(dataDir string) (*Blockchain, error) {
	// Open the database
	db, err := leveldb.OpenFile(dataDir, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	bc := &Blockchain{
		db:         db,
		newBlockCh: make(chan *Block),
		quitCh:     make(chan struct{}),
	}

	// Initialize genesis block if needed
	genesis, err := bc.loadGenesisBlock()
	if err != nil {
		return nil, fmt.Errorf("failed to load genesis block: %w", err)
	}
	bc.genesisBlock = genesis
	bc.currentBlock = genesis
	bc.height = 0

	return bc, nil
}

// Start starts the blockchain processing
func (bc *Blockchain) Start() error {
	go bc.loop()
	return nil
}

// Stop stops the blockchain processing
func (bc *Blockchain) Stop() error {
	close(bc.quitCh)
	return bc.db.Close()
}

// loop is the main event loop for blockchain processing
func (bc *Blockchain) loop() {
	for {
		select {
		case block := <-bc.newBlockCh:
			if err := bc.processBlock(block); err != nil {
				// TODO: Handle error
				continue
			}
		case <-bc.quitCh:
			return
		}
	}
}

// processBlock processes a new block and adds it to the chain
func (bc *Blockchain) processBlock(block *Block) error {
	bc.mu.Lock()
	defer bc.mu.Unlock()

	// Validate block
	if err := bc.validateBlock(block); err != nil {
		return fmt.Errorf("invalid block: %w", err)
	}

	// Store block
	if err := bc.storeBlock(block); err != nil {
		return fmt.Errorf("failed to store block: %w", err)
	}

	// Update chain state
	bc.currentBlock = block
	bc.height = block.Header.Height

	return nil
}

// validateBlock validates a block before adding it to the chain
func (bc *Blockchain) validateBlock(block *Block) error {
	// Validate block header
	if block.Header.Height != bc.height+1 {
		return fmt.Errorf("invalid block height")
	}
	if block.Header.PrevHash != bc.currentBlock.getHash() {
		return fmt.Errorf("invalid previous hash")
	}

	// Validate transactions
	for _, tx := range block.Body.Transactions {
		if err := bc.validateTransaction(tx); err != nil {
			return fmt.Errorf("invalid transaction: %w", err)
		}
	}

	return nil
}

// validateTransaction validates a single transaction
func (bc *Blockchain) validateTransaction(tx *Transaction) error {
	// TODO: Implement transaction validation
	return nil
}

// storeBlock stores a block in the database
func (bc *Blockchain) storeBlock(block *Block) error {
	data, err := json.Marshal(block)
	if err != nil {
		return err
	}

	key := fmt.Sprintf("block_%d", block.Header.Height)
	return bc.db.Put([]byte(key), data, nil)
}

// loadGenesisBlock loads or creates the genesis block
func (bc *Blockchain) loadGenesisBlock() (*Block, error) {
	// Try to load existing genesis block
	data, err := bc.db.Get([]byte("block_0"), nil)
	if err == nil {
		var genesis Block
		if err := json.Unmarshal(data, &genesis); err != nil {
			return nil, err
		}
		return &genesis, nil
	}

	// Create new genesis block
	genesis := &Block{
		Header: &BlockHeader{
			Version:   1,
			Timestamp: time.Now(),
			Height:    0,
		},
		Body: &BlockBody{
			Transactions: []*Transaction{},
		},
	}

	// Store genesis block
	if err := bc.storeBlock(genesis); err != nil {
		return nil, err
	}

	return genesis, nil
}

// getHash returns the hash of a block
func (b *Block) getHash() string {
	header, _ := json.Marshal(b.Header)
	hash := sha256.Sum256(header)
	return hex.EncodeToString(hash[:])
}

// AddTransaction adds a new transaction to the pending pool
func (bc *Blockchain) AddTransaction(tx *Transaction) error {
	// TODO: Implement transaction pool
	return nil
}

// GetBlock retrieves a block by height
func (bc *Blockchain) GetBlock(height uint64) (*Block, error) {
	data, err := bc.db.Get([]byte(fmt.Sprintf("block_%d", height)), nil)
	if err != nil {
		return nil, err
	}

	var block Block
	if err := json.Unmarshal(data, &block); err != nil {
		return nil, err
	}

	return &block, nil
}

// GetCurrentBlock returns the current block
func (bc *Blockchain) GetCurrentBlock() *Block {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	return bc.currentBlock
}

// GetHeight returns the current blockchain height
func (bc *Blockchain) GetHeight() uint64 {
	bc.mu.RLock()
	defer bc.mu.RUnlock()
	return bc.height
}
