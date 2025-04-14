package consensus

import (
	"fmt"
	"math/big"
	"sort"
	"sync"
	"time"
)

// Validator represents a validator in the PoS system
type Validator struct {
	Address     string   `json:"address"`
	StakeAmount *big.Int `json:"stakeAmount"`
	LastActive  int64    `json:"lastActive"`
	IsActive    bool     `json:"isActive"`
}

// PoSConsensus implements the Proof of Stake consensus mechanism
type PoSConsensus struct {
	mu sync.RWMutex

	// Validator set
	validators map[string]*Validator
	minStake   *big.Int

	// Consensus parameters
	epochLength      uint64
	validatorSetSize uint32
	blockTime        time.Duration

	// Current state
	currentEpoch uint64
	lastBlock    time.Time
}

// NewPoSConsensus creates a new PoS consensus instance
func NewPoSConsensus(config *Config) (*PoSConsensus, error) {
	pos := &PoSConsensus{
		validators:       make(map[string]*Validator),
		minStake:         new(big.Int).SetUint64(config.MinStake),
		epochLength:      config.EpochLength,
		validatorSetSize: config.ValidatorSetSize,
		blockTime:        time.Duration(config.BlockTime) * time.Second,
		currentEpoch:     0,
		lastBlock:        time.Now(),
	}

	return pos, nil
}

// AddValidator adds a new validator to the set
func (pos *PoSConsensus) AddValidator(address string, stake *big.Int) error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	if stake.Cmp(pos.minStake) < 0 {
		return fmt.Errorf("stake amount below minimum requirement")
	}

	pos.validators[address] = &Validator{
		Address:     address,
		StakeAmount: new(big.Int).Set(stake),
		LastActive:  time.Now().Unix(),
		IsActive:    true,
	}

	return nil
}

// RemoveValidator removes a validator from the set
func (pos *PoSConsensus) RemoveValidator(address string) error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	if _, exists := pos.validators[address]; !exists {
		return fmt.Errorf("validator not found")
	}

	delete(pos.validators, address)
	return nil
}

// GetNextValidator returns the next validator to propose a block
func (pos *PoSConsensus) GetNextValidator() (*Validator, error) {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	if len(pos.validators) == 0 {
		return nil, fmt.Errorf("no validators available")
	}

	// Get active validators sorted by stake
	activeValidators := pos.getActiveValidatorsSorted()
	if len(activeValidators) == 0 {
		return nil, fmt.Errorf("no active validators")
	}

	// Select validator based on stake weight and time since last block
	selectedValidator := pos.selectValidatorByWeight(activeValidators)
	if selectedValidator == nil {
		return nil, fmt.Errorf("failed to select validator")
	}

	return selectedValidator, nil
}

// getActiveValidatorsSorted returns active validators sorted by stake
func (pos *PoSConsensus) getActiveValidatorsSorted() []*Validator {
	activeValidators := make([]*Validator, 0)
	for _, v := range pos.validators {
		if v.IsActive {
			activeValidators = append(activeValidators, v)
		}
	}

	sort.Slice(activeValidators, func(i, j int) bool {
		return activeValidators[i].StakeAmount.Cmp(activeValidators[j].StakeAmount) > 0
	})

	return activeValidators
}

// selectValidatorByWeight selects a validator based on stake weight
func (pos *PoSConsensus) selectValidatorByWeight(validators []*Validator) *Validator {
	if len(validators) == 0 {
		return nil
	}

	// Simple selection: return the validator with highest stake
	// TODO: Implement more sophisticated selection based on stake and time weights
	return validators[0]
}

// ValidateBlock validates a block according to PoS rules
func (pos *PoSConsensus) ValidateBlock(block interface{}) error {
	// TODO: Implement block validation
	// - Check if block proposer is a valid validator
	// - Verify block timing
	// - Verify signatures
	return nil
}

// UpdateEpoch updates the current epoch and validator set
func (pos *PoSConsensus) UpdateEpoch() error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	pos.currentEpoch++

	// Update validator states
	now := time.Now().Unix()
	for _, validator := range pos.validators {
		// Deactivate validators that haven't been active recently
		if now-validator.LastActive > int64(pos.epochLength*uint64(pos.blockTime.Seconds())) {
			validator.IsActive = false
		}
	}

	return nil
}

// GetValidatorSet returns the current set of validators
func (pos *PoSConsensus) GetValidatorSet() []*Validator {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	validators := make([]*Validator, 0, len(pos.validators))
	for _, v := range pos.validators {
		validators = append(validators, v)
	}

	return validators
}

// Config represents the configuration for PoS consensus
type Config struct {
	MinStake         uint64
	EpochLength      uint64
	ValidatorSetSize uint32
	BlockTime        uint64 // in seconds
}

// IsValidator checks if an address is a validator
func (pos *PoSConsensus) IsValidator(address string) bool {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	validator, exists := pos.validators[address]
	return exists && validator.IsActive
}

// GetStake returns the stake amount for a validator
func (pos *PoSConsensus) GetStake(address string) (*big.Int, error) {
	pos.mu.RLock()
	defer pos.mu.RUnlock()

	validator, exists := pos.validators[address]
	if !exists {
		return nil, fmt.Errorf("validator not found")
	}

	return new(big.Int).Set(validator.StakeAmount), nil
}

// UpdateStake updates the stake amount for a validator
func (pos *PoSConsensus) UpdateStake(address string, newStake *big.Int) error {
	pos.mu.Lock()
	defer pos.mu.Unlock()

	validator, exists := pos.validators[address]
	if !exists {
		return fmt.Errorf("validator not found")
	}

	if newStake.Cmp(pos.minStake) < 0 {
		return fmt.Errorf("stake amount below minimum requirement")
	}

	validator.StakeAmount = new(big.Int).Set(newStake)
	return nil
}
