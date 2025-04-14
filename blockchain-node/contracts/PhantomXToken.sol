// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/draft-ERC20Permit.sol";

/**
 * @title PhantomX Token
 * @dev Implementation of the PhantomX token with staking and governance features
 */
contract PhantomXToken is ERC20, ERC20Burnable, Pausable, AccessControl, ERC20Permit {
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant VALIDATOR_ROLE = keccak256("VALIDATOR_ROLE");

    // Staking configuration
    uint256 public constant MINIMUM_STAKE = 1000 * 10**18; // 1000 PHX
    uint256 public constant UNSTAKE_DELAY = 7 days;

    // Staking data structures
    struct Stake {
        uint256 amount;
        uint256 timestamp;
        uint256 unlockTime;
        bool isValidator;
    }

    mapping(address => Stake) public stakes;
    address[] public validators;
    uint256 public totalStaked;

    // Events
    event Staked(address indexed account, uint256 amount);
    event Unstaked(address indexed account, uint256 amount);
    event ValidatorAdded(address indexed account);
    event ValidatorRemoved(address indexed account);
    event RewardDistributed(address indexed validator, uint256 amount);

    /**
     * @dev Constructor that gives msg.sender all of the roles
     */
    constructor() ERC20("PhantomX", "PHX") ERC20Permit("PhantomX") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(PAUSER_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    /**
     * @dev Pauses all token transfers
     */
    function pause() public onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @dev Unpauses all token transfers
     */
    function unpause() public onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    /**
     * @dev Mints new tokens
     * @param to The address that will receive the minted tokens
     * @param amount The amount of tokens to mint
     */
    function mint(address to, uint256 amount) public onlyRole(MINTER_ROLE) {
        _mint(to, amount);
    }

    /**
     * @dev Stakes tokens to become a validator
     * @param amount The amount of tokens to stake
     */
    function stake(uint256 amount) public whenNotPaused {
        require(amount >= MINIMUM_STAKE, "Stake amount below minimum");
        require(stakes[msg.sender].amount == 0, "Already staked");
        require(balanceOf(msg.sender) >= amount, "Insufficient balance");

        _transfer(msg.sender, address(this), amount);

        stakes[msg.sender] = Stake({
            amount: amount,
            timestamp: block.timestamp,
            unlockTime: 0,
            isValidator: false
        });

        totalStaked += amount;
        emit Staked(msg.sender, amount);
    }

    /**
     * @dev Initiates unstaking process
     */
    function initiateUnstake() public whenNotPaused {
        Stake storage userStake = stakes[msg.sender];
        require(userStake.amount > 0, "No stake found");
        require(userStake.unlockTime == 0, "Unstake already initiated");

        userStake.unlockTime = block.timestamp + UNSTAKE_DELAY;
        if (userStake.isValidator) {
            _revokeRole(VALIDATOR_ROLE, msg.sender);
            _removeValidator(msg.sender);
        }
    }

    /**
     * @dev Completes unstaking process and returns staked tokens
     */
    function completeUnstake() public whenNotPaused {
        Stake storage userStake = stakes[msg.sender];
        require(userStake.amount > 0, "No stake found");
        require(userStake.unlockTime > 0, "Unstake not initiated");
        require(block.timestamp >= userStake.unlockTime, "Tokens still locked");

        uint256 amount = userStake.amount;
        totalStaked -= amount;

        delete stakes[msg.sender];
        _transfer(address(this), msg.sender, amount);

        emit Unstaked(msg.sender, amount);
    }

    /**
     * @dev Adds an address as a validator
     * @param account The address to add as validator
     */
    function addValidator(address account) public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(stakes[account].amount >= MINIMUM_STAKE, "Insufficient stake");
        require(!stakes[account].isValidator, "Already a validator");

        stakes[account].isValidator = true;
        validators.push(account);
        _grantRole(VALIDATOR_ROLE, account);

        emit ValidatorAdded(account);
    }

    /**
     * @dev Removes a validator
     * @param account The address to remove from validators
     */
    function removeValidator(address account) public onlyRole(DEFAULT_ADMIN_ROLE) {
        require(stakes[account].isValidator, "Not a validator");

        stakes[account].isValidator = false;
        _removeValidator(account);
        _revokeRole(VALIDATOR_ROLE, account);

        emit ValidatorRemoved(account);
    }

    /**
     * @dev Distributes rewards to a validator
     * @param validator The validator address
     * @param amount The reward amount
     */
    function distributeReward(address validator, uint256 amount) public onlyRole(MINTER_ROLE) {
        require(stakes[validator].isValidator, "Not a validator");
        _mint(validator, amount);
        emit RewardDistributed(validator, amount);
    }

    /**
     * @dev Returns the list of all validators
     */
    function getValidators() public view returns (address[] memory) {
        return validators;
    }

    /**
     * @dev Internal function to remove a validator from the array
     */
    function _removeValidator(address account) internal {
        for (uint256 i = 0; i < validators.length; i++) {
            if (validators[i] == account) {
                validators[i] = validators[validators.length - 1];
                validators.pop();
                break;
            }
        }
    }

    /**
     * @dev Hook that is called before any transfer of tokens
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
} 